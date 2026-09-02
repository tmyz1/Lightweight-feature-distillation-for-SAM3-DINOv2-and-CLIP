import argparse
import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.eval.coco_eval_offline import CocoEvaluatorOfflineWithPredFileEvaluators
from sam3.eval.coco_writer import PredictionDumper
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device

from KD.build_dataloader import (
    add_multi_resolution_batches,
    build_dataloader,
    build_val_dataloader,
    get_selected_coco_image_ids,
    get_split_root,
    resolve_split_paths,
)
from KD.data.SA_1B import is_sa1b_dataset
from KD.distill_model import Distill

#下载配置文件
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

#创建蒸馏模型
def build_distiller(cfg: Dict[str, Any], device: torch.device):
    return Distill(cfg, device)

#将数据全部放在cuda上
def move_to_device(obj, device: torch.device, cfg: Dict[str, Any] = None):
    valid_boxes = getattr(obj, "kd_valid_boxes", None)
    obj = copy_data_to_device(obj, device, non_blocking=True)
    if valid_boxes is not None:
        obj.kd_valid_boxes = valid_boxes.to(device=device, non_blocking=True)
    if cfg is not None:
        obj = add_multi_resolution_batches(obj, cfg)#添加不同尺寸的分辨率
    return obj

#定义随机种子
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_log_batch(batch_idx: int, total_batches: int, log_every: int) -> bool:
    if log_every <= 0:
        return False
    return batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches


def format_logs(logs: Dict[str, torch.Tensor]) -> str:
    return " ".join(
        f"{key}={float(value.detach().float().cpu()):.6f}"
        for key, value in logs.items()
    )


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

#精度转换
def forward_with_amp(model: Distill, batch, cfg: Dict[str, Any]):
    with torch.autocast(
            device_type=model.device.type,
            dtype=torch.bfloat16,
            enabled=bool(cfg.get("train", {}).get("amp", True))
            and model.device.type == "cuda",
    ):
        return model(batch)


def forward_student_for_eval(model: Distill, batch, cfg: Dict[str, Any]):
    with torch.autocast(
            device_type=model.device.type,
            dtype=torch.bfloat16,
            enabled=bool(cfg.get("eval", {}).get("amp", False))
            and model.device.type == "cuda",
    ):
        return model.Student.model(batch)


def build_optimizer(model: Distill, cfg: Dict[str, Any]):
    optim_cfg = cfg.get("optim", {})
    learning_rates = {
        "trunk": float(optim_cfg.get("trunk_lr", 1.0e-5)),
        "neck": float(optim_cfg.get("neck_lr", 1.0e-4)),
        "adapter": float(optim_cfg.get("adapter_lr", 5.0e-5)),
    }
    weight_decay = float(optim_cfg.get("weight_decay", 0.05))
    groups = {
        name: {"decay": [], "no_decay": []}
        for name in learning_rates
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("Student.model.backbone.vision_backbone.trunk."):
            group_name = "trunk"
        elif name.startswith("Student.model.backbone.vision_backbone."):
            group_name = "neck"
        else:
            group_name = "adapter"
        decay_group = "no_decay" if parameter.ndim <= 1 or name.endswith(".bias") else "decay"
        groups[group_name][decay_group].append(parameter)

    parameter_groups = []
    for group_name, group in groups.items():
        if group["decay"]:
            parameter_groups.append(
                {
                    "params": group["decay"],
                    "lr": learning_rates[group_name],
                    "weight_decay": weight_decay,
                    "name": group_name,
                }
            )
        if group["no_decay"]:
            parameter_groups.append(
                {
                    "params": group["no_decay"],
                    "lr": learning_rates[group_name],
                    "weight_decay": 0.0,
                    "name": group_name,
                }
            )

    if not parameter_groups:
        raise RuntimeError("No trainable parameters were found.")
    return torch.optim.AdamW(
        parameter_groups,
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(optimizer, total_updates: int, cfg: Dict[str, Any]):
    scheduler_cfg = cfg.get("param_scheduler", {})
    warmup_updates = min(int(scheduler_cfg.get("warmup_updates", 500)), total_updates)
    warmup_start_factor = float(scheduler_cfg.get("warmup_start_factor", 0.1))
    min_lr = float(scheduler_cfg.get("min_lr", cfg.get("optim", {}).get("min_lr", 1.0e-6)))

    def make_lambda(base_lr: float):
        min_factor = min(min_lr / base_lr, 1.0)

        def schedule(update: int) -> float:
            if warmup_updates > 0 and update < warmup_updates:
                progress = update / warmup_updates
                return warmup_start_factor + (1.0 - warmup_start_factor) * progress
            if total_updates <= warmup_updates:
                return min_factor
            progress = (update - warmup_updates) / (total_updates - warmup_updates)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return schedule

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(group["lr"]) for group in optimizer.param_groups],
    )


def format_learning_rates(optimizer) -> str:
    values = {}
    for group in optimizer.param_groups:
        values[group["name"]] = group["lr"]
    return " ".join(f"lr_{name}={value:.8f}" for name, value in values.items())


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
def evaluate(model: Distill, cfg: Dict[str, Any], step: int):
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


def save_checkpoint(
    path: Path,
    model: Distill,
    optimizer,
    scheduler,
    scaler,
    cfg: Dict[str, Any],
    step: int,
    global_update: int,
    best_map: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    distillation_adapters = {
        name: getattr(model, name).state_dict()
        for name in (
            "sam3_adapter",
            "Dino_v2_adapter",
            "CLIP_adapter",
            "Dino_v2_cls_adapter",
            "CLIP_cls_adapter",
        )
        if hasattr(model, name)
    }
    torch.save(
        {
            "student_model": model.Student.model.state_dict(),
            "distillation_adapters": distillation_adapters,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "step": step,
            "global_update": global_update,
            "best_map": best_map,
        },
        path,
    )
    print(f"[checkpoint] saved: {path}", flush=True)


def save_student_model(path: Path, model: Distill) -> None:
    """Save the deployable student without teachers, adapters, or optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    student_state = model.Student.model.state_dict()
    segmentation_keys = [
        key for key in student_state if key.startswith("segmentation_head.")
    ]
    if not segmentation_keys:
        raise RuntimeError(
            "Student checkpoint has no segmentation_head parameters. Set "
            "Student.enable_segmentation to true before training."
        )
    torch.save(student_state, path)
    print(
        f"[student] saved: {path} "
        f"(segmentation parameters={len(segmentation_keys)})",
        flush=True,
    )


def load_checkpoint(
    path: Path,
    model: Distill,
    optimizer,
    scheduler,
    scaler,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=model.device, weights_only=False)
    if "student_model" not in checkpoint:
        raise ValueError(
            f"{path} is a student-only weight file. "
            "Use a *_distillation.pt checkpoint with --resume."
        )
    missing_keys, unexpected_keys = model.Student.model.load_state_dict(
        checkpoint["student_model"], strict=False
    )
    legacy_fusion_keys = {
        "backbone.vision_backbone.layer_fusion_logits",
    }
    non_segmentation_missing = [
        key
        for key in missing_keys
        if not key.startswith("segmentation_head.") and key not in legacy_fusion_keys
    ]
    unsupported_unexpected = [
        key for key in unexpected_keys if key not in legacy_fusion_keys
    ]
    if non_segmentation_missing or unsupported_unexpected:
        raise RuntimeError(
            "Student checkpoint is incompatible with the current model. "
            f"Missing keys: {non_segmentation_missing[:10]}; "
            f"unexpected keys: {unsupported_unexpected[:10]}."
        )
    if unexpected_keys:
        print(
            "[checkpoint] ignored removed ViT-Small fusion parameters.",
            flush=True,
        )
    restored_fusion_parameter = any(key in missing_keys for key in legacy_fusion_keys)
    if restored_fusion_parameter:
        print(
            "[checkpoint] added ViT-Small fusion parameters retain their "
            "current zero initialization.",
            flush=True,
        )
    if any(key.startswith("segmentation_head.") for key in missing_keys):
        print(
            "[checkpoint] segmentation head was absent and retains the "
            "SAM3-initialized parameters.",
            flush=True,
        )
    distillation_adapters = checkpoint.get("distillation_adapters")
    if distillation_adapters is None:
        print(
            "[checkpoint] warning: legacy checkpoint has no distillation adapter "
            "state; adapters keep their newly initialized weights.",
            flush=True,
        )
    else:
        for name, state_dict in distillation_adapters.items():
            if not hasattr(model, name):
                raise ValueError(
                    f"Checkpoint contains adapter {name!r}, but the current "
                    "distillation configuration does not create it."
                )
            getattr(model, name).load_state_dict(state_dict, strict=True)
    scheduler.load_state_dict(checkpoint["scheduler"])
    if restored_fusion_parameter:
        # The new trainable parameter changes the AdamW parameter-group layout.
        # Keep its zero initialization and resume with fresh optimizer moments.
        resumed_update = int(checkpoint["global_update"])
        scheduler.last_epoch = resumed_update - 1
        scheduler._step_count = resumed_update
        scheduler.step()
        print(
            "[checkpoint] optimizer moments were reset because the fusion "
            "parameter did not exist in this checkpoint; scheduler progress "
            "was preserved.",
            flush=True,
        )
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    print(
        f"[checkpoint] resumed: {path} step={checkpoint['step']} "
        f"update={checkpoint['global_update']}",
        flush=True,
    )
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KD Training")
    parser.add_argument(
        "--config",
        default=r"E:\pycharm\Vision Distillation\Config\Vit_Small_Distill.yaml",
        help="config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"E:\pycharm\Vision Distillation\output",
        help="output directory",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("train", {}).get("seed", 123)))


    #加载设别
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this training run.")

    #构建数据集
    print(f'begin building dataloader')
    dataloader = build_dataloader(cfg)
    print(f"dataloader built: batches={len(dataloader)}")

    #构建训练模型
    model = build_distiller(cfg,device)
    model.to(device)
    model.train()

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(cfg.get("train", {}).get("amp", True)) and model.device.type == "cuda",
    )

    max_steps = args.max_steps or int(cfg.get("train", {}).get("max_steps", 1))
    log_every_batches = int(cfg.get("train", {}).get("log_every_batches", 20))
    total_batches = len(dataloader)
    if total_batches == 0:
        raise RuntimeError("Training dataloader is empty.")

    eval_every = int(cfg.get("train", {}).get("eval_every_steps", 0))
    total_updates = total_batches * max_steps
    scheduler = build_scheduler(optimizer, total_updates, cfg)
    max_grad_norm = float(cfg.get("optim", {}).get("max_grad_norm", 1.0))
    output_dir = Path(args.output)
    best_map = float("-inf")
    start_step = 1
    global_update = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler
        )
        start_step = int(checkpoint["step"]) + 1
        global_update = int(checkpoint["global_update"])
        best_map = float(checkpoint.get("best_map", best_map))

    if args.eval_only:
        if args.resume is None:
            raise ValueError("--eval-only requires --resume with a student checkpoint.")
        evaluate(model, cfg, start_step - 1)
        raise SystemExit(0)

    for step in range(start_step, max_steps + 1):
        running_loss = 0.0
        running_batches = 0
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = move_to_device(batch, model.device, cfg)
            optimizer.zero_grad(set_to_none=True)

            total_loss, logs = forward_with_amp(model, batch, cfg)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_update += 1
            running_loss += float(total_loss.detach().float().cpu())
            running_batches += 1

            if should_log_batch(batch_idx, total_batches, log_every_batches):
                print(
                    f"[train] step={step}/{max_steps} "
                    f"batch={batch_idx}/{total_batches} "
                    f"update={global_update} grad_norm={float(grad_norm):.6f} "
                    f"{format_learning_rates(optimizer)} {format_logs(logs)}",
                    flush=True,
                )

        avg_loss = running_loss / max(running_batches, 1)
        print(
            f"[train] step={step}/{max_steps} done "
            f"batches={running_batches} avg_loss_total={avg_loss:.6f}",
            flush=True,
        )

        metrics = {}
        if eval_every > 0 and step % eval_every == 0:
            metrics = evaluate(model, cfg, step)
            current_map = float(metrics.get("coco_eval_bbox_AP", float("-inf")))
            if current_map > best_map:
                best_map = current_map
                save_checkpoint(
                    output_dir / "best_distillation.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    cfg,
                    step,
                    global_update,
                    best_map,
                )
                save_student_model(output_dir / "best_student.pt", model)

        save_checkpoint(
            output_dir / "latest_distillation.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            step,
            global_update,
            best_map,
        )
        save_student_model(output_dir / "latest_student.pt", model)
