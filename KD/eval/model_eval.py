from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KD.build_dataloader import (
    add_multi_resolution_batches,
    build_split_dataloader,
    get_selected_coco_image_ids,
    get_split_root,
    resolve_split_paths,
)
from KD.model.vit_small_patch14_reg4_dinov2 import build_vit_small_image_model
from KD.training.distributed import (
    destroy_distributed,
    initialize_distributed,
    is_distributed,
    is_main_process,
)
from sam3.eval.coco_eval_offline import CocoEvaluatorOfflineWithPredFileEvaluators
from sam3.eval.coco_writer import PredictionDumper
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "Config" / "Vit_Small_Distill.yaml"
DEFAULT_CHECKPOINT_PATH = Path(
    r"E:\reproduce\weights\sam3 distill\vit_small_patch14_reg4_dinov2 distill\best_student.pt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs" / "latest_student_val"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def move_to_student_device(batch, device: torch.device, cfg: Dict[str, Any]):
    batch = copy_data_to_device(batch, device, non_blocking=True)
    batch = add_multi_resolution_batches(batch, cfg)
    batch.img_batch = batch.student_img_batch
    return batch


def write_coco_annotation_subset(
    annotation_path: Path,
    image_ids: list[int],
    output_path: Path,
) -> Path:
    selected_image_ids = set(image_ids)
    with annotation_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)
    coco_data["images"] = [
        image for image in coco_data["images"] if image["id"] in selected_image_ids
    ]
    coco_data["annotations"] = [
        annotation
        for annotation in coco_data["annotations"]
        if annotation["image_id"] in selected_image_ids
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(coco_data, file)
    return output_path


def build_student_model(
    cfg: Dict[str, Any], checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Student checkpoint does not exist: {checkpoint_path}")

    student_cfg = cfg.get("Student", {})
    sam3_cfg = cfg.get("Sam3", {})
    model = build_vit_small_image_model(
        bpe_path=sam3_cfg.get("bpe_path"),
        checkpoint_path=sam3_cfg.get("checkpoint_path"),
        device=str(device),
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        vit_small_checkpoint_path=student_cfg.get("vit_small_checkpoint_path"),
        return_interm_layers=bool(cfg.get("train", {}).get("return_interm_layers", False)),
        vit_small_intermediate_layers=student_cfg.get(
            "vit_small_intermediate_layers", (2, 5, 8, 11)
        ),
        cfg=cfg,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "student_model" in checkpoint:
        checkpoint = checkpoint["student_model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported student checkpoint format: {type(checkpoint)}")

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    fusion_key = "backbone.vision_backbone.layer_fusion_logits"
    invalid_missing = [
        key
        for key in missing_keys
        if not key.startswith("segmentation_head.") and key != fusion_key
    ]
    invalid_unexpected = [key for key in unexpected_keys if key != fusion_key]
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(
            "Student checkpoint is incompatible with the current model. "
            f"Missing: {invalid_missing[:10]}; unexpected: {invalid_unexpected[:10]}."
        )
    if fusion_key in missing_keys:
        print("[checkpoint] fusion parameter uses its current zero initialization.")

    model.to(device)
    model.eval()
    return model


def build_evaluation_dataloader(cfg: Dict[str, Any]):
    dataset_cfg = cfg["dataset"]
    return build_split_dataloader(
        cfg,
        split=dataset_cfg.get("val_split", "valid"),
        training=False,
        limit_ids=dataset_cfg.get("val_num_images"),
        shuffle=False,
        drop_last=False,
        distributed=is_distributed(),
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, cfg: Dict[str, Any], output_dir: Path, use_amp: bool):
    dataset_cfg = cfg["dataset"]
    eval_cfg = cfg.get("eval", {})
    split = dataset_cfg.get("val_split", "valid")
    _, annotation_path = resolve_split_paths(
        get_split_root(dataset_cfg, training=False), split
    )
    val_loader = build_evaluation_dataloader(cfg)

    evaluation_annotations = annotation_path
    if dataset_cfg.get("val_num_images") is not None:
        selected_image_ids = get_selected_coco_image_ids(val_loader.dataset)
        evaluation_annotations = output_dir / "instances_val_subset.json"
        if is_main_process():
            write_coco_annotation_subset(
                annotation_path,
                selected_image_ids,
                evaluation_annotations,
            )
            print(
                f"[eval] evaluating {len(selected_image_ids)} source image(s)",
                flush=True,
            )
    elif is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        dist.barrier()

    max_dets = int(eval_cfg.get("max_dets", 100))
    postprocessor = PostProcessImage(
        max_dets_per_img=max_dets,
        iou_type="bbox",
        to_cpu=True,
        use_original_ids=True,
        use_original_sizes_box=True,
        use_presence=True,
    )
    evaluator = CocoEvaluatorOfflineWithPredFileEvaluators(
        gt_path=str(evaluation_annotations), tide=False, iou_type="bbox"
    )
    dumper = PredictionDumper(
        dump_dir=str(output_dir),
        postprocessor=postprocessor,
        maxdets=max_dets,
        iou_type="bbox",
        merge_predictions=True,
        pred_file_evaluators=[evaluator],
    )

    device = next(model.parameters()).device
    log_every = int(eval_cfg.get("log_every_batches", 20))
    total_batches = len(val_loader)
    for batch_index, batch in enumerate(val_loader, start=1):
        if is_main_process() and (
            batch_index == 1
            or batch_index % log_every == 0
            or batch_index == total_batches
        ):
            print(f"[eval] batch={batch_index}/{total_batches}", flush=True)
        batch = move_to_student_device(batch, device, cfg)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                outputs = model(batch)
        for stage in outputs:
            if not torch.isfinite(stage["pred_logits"]).all():
                raise FloatingPointError(
                    "Non-finite predictions detected. Run without --amp for a valid mAP."
                )
        dumper.update(find_stages=outputs, find_metadatas=batch.find_metadatas)

    metrics = dumper.compute_synced()
    if is_main_process():
        print(
            "[eval] "
            f"AP={float(metrics['coco_eval_bbox_AP']):.6f} "
            f"AP50={float(metrics['coco_eval_bbox_AP_50']):.6f} "
            f"AP75={float(metrics['coco_eval_bbox_AP_75']):.6f}",
            flush=True,
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the distilled ViT-Small SAM3 model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable multi-process evaluation; launch with torchrun.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use BF16 autocast. FP32 is the default for valid mAP evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")
    cfg = load_config(args.config)
    if args.distributed:
        cfg.setdefault("multi_gpu", {})["enabled"] = True
    device, _ = initialize_distributed(cfg)
    try:
        model = build_student_model(cfg, args.checkpoint, device)
        evaluate(model, cfg, args.output_dir, use_amp=args.amp)
    finally:
        destroy_distributed()
