from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Dict

import torch

from sam3.eval.coco_eval_offline import CocoEvaluatorOfflineWithPredFileEvaluators
from sam3.eval.coco_writer import PredictionDumper
from sam3.eval.postprocessors import PostProcessImage

from KD.build_dataloader import (
    build_val_dataloader,
    get_selected_coco_image_ids,
    get_split_root,
    resolve_split_paths,
)
from KD.data.SA_1B import is_sa1b_dataset
from KD.training.distributed import unwrap_model
from KD.training.runtime import move_to_device, should_log_batch


def format_metrics(metrics: Dict[str, Any]) -> str:
    keys = [
        "coco_eval_bbox_AP",
        "coco_eval_bbox_AP_50",
        "coco_eval_bbox_AP_75",
        "coco_eval_bbox_AP_small",
        "coco_eval_bbox_AP_medium",
        "coco_eval_bbox_AP_large",
        "coco_eval_bbox_AR_maxDets@1",
        "coco_eval_bbox_AR_maxDets@10",
        "coco_eval_bbox_AR_maxDets@100",
    ]
    aliases = {
        "coco_eval_bbox_AP": "mAP",
        "coco_eval_bbox_AP_50": "AP50",
        "coco_eval_bbox_AP_75": "AP75",
        "coco_eval_bbox_AP_small": "APs",
        "coco_eval_bbox_AP_medium": "APm",
        "coco_eval_bbox_AP_large": "APl",
        "coco_eval_bbox_AR_maxDets@1": "AR@1",
        "coco_eval_bbox_AR_maxDets@10": "AR@10",
        "coco_eval_bbox_AR_maxDets@100": "AR@100",
    }
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{aliases[key]}={float(metrics[key]):.6f}")
    return " ".join(parts)


def forward_student_for_eval(model, batch, cfg: Dict[str, Any]):
    model = unwrap_model(model)
    with torch.autocast(
            device_type=model.device.type,
            dtype=torch.bfloat16,
            enabled=bool(cfg.get("eval", {}).get("amp", False))
            and model.device.type == "cuda",
    ):
        return model.Student.model(batch)


def use_student_resolution(batch):
    if not hasattr(batch, "student_img_batch"):
        raise AttributeError("Eval batch is missing student_img_batch. Pass cfg to move_to_device first.")
    batch.img_batch = batch.student_img_batch
    return batch


def write_coco_annotation_subset(
    ann_file: Path,
    image_ids: list[int],
    output_path: Path,
) -> Path:
    selected_image_ids = set(image_ids)
    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    coco_data["images"] = [
        image for image in coco_data["images"] if image["id"] in selected_image_ids
    ]
    coco_data["annotations"] = [
        annotation
        for annotation in coco_data["annotations"]
        if annotation["image_id"] in selected_image_ids
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f)
    return output_path


@torch.no_grad()
def evaluate(model, cfg: Dict[str, Any], step: int):
    model = unwrap_model(model)
    if not cfg.get("eval", {}).get("enabled", True):
        return {}

    #获取验证集所在路径
    eval_cfg = cfg.get("eval", {})
    dataset_cfg = cfg["dataset"]
    if is_sa1b_dataset(dataset_cfg, training=False):
        print(
            "[eval] COCO mAP is skipped for direct SA-1B loading because "
            "no COCO validation annotation file is configured.",
            flush=True,
        )
        return {}
    split = dataset_cfg.get("val_split", "valid")
    _, ann_file = resolve_split_paths(
        get_split_root(dataset_cfg, training=False), split
    )
    dump_dir = Path(eval_cfg.get("output_dir", "KD/eval_outputs")) / f"step_{step}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    val_loader = build_val_dataloader(cfg)
    eval_ann_file = ann_file
    if dataset_cfg.get("val_num_images") is not None:
        selected_image_ids = get_selected_coco_image_ids(val_loader.dataset)
        eval_ann_file = write_coco_annotation_subset(
            ann_file,
            selected_image_ids,
            dump_dir / "instances_val_subset.json",
        )
        print(
            f"[eval] step={step} evaluating {len(selected_image_ids)} source image(s)",
            flush=True,
        )

    postprocessor = PostProcessImage(
        max_dets_per_img=int(eval_cfg.get("max_dets", 100)),
        iou_type="bbox",
        to_cpu=True,
        use_original_ids=True,
        use_original_sizes_box=True,
        use_presence=True,
    )

    #构造 COCO 离线评估器
    evaluator = CocoEvaluatorOfflineWithPredFileEvaluators(
        gt_path=str(eval_ann_file),
        tide=False,
        iou_type="bbox",
    )

    #构造预测结果 dumper
    dumper = PredictionDumper(
        dump_dir=str(dump_dir),
        postprocessor=postprocessor,
        maxdets=int(eval_cfg.get("max_dets", 100)),
        iou_type="bbox",
        merge_predictions=True,
        pred_file_evaluators=[evaluator],
    )

    was_training = model.Student.model.training
    model.Student.model.eval()
    log_every = int(cfg.get("eval", {}).get("log_every_batches", 20))
    total_val_batches = len(val_loader)

    for batch_idx, batch in enumerate(val_loader, start=1):
        if should_log_batch(batch_idx, total_val_batches, log_every):
            print(f"[eval] step={step} batch={batch_idx}/{total_val_batches}", flush=True)
        batch = move_to_device(batch, model.device, cfg)
        batch = use_student_resolution(batch)
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = forward_student_for_eval(model, batch, cfg)
        dumper.update(find_stages=outputs, find_metadatas=batch.find_metadatas)

    metrics = dumper.compute_synced()
    if was_training:
        model.Student.model.train()
    print(f"[eval] step={step} {format_metrics(metrics)}", flush=True)
    return metrics
