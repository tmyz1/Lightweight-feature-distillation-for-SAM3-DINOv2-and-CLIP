from __future__ import annotations

import argparse
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

from KD.build_dataloader import (
    add_multi_resolution_batches,
    build_dataloader,
    build_val_dataloader,
)
from KD.decoder_mask_distill import MaskDistill, NoValidAnnotationsError
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.loss.loss_fns import segment_miou


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch, device: torch.device, cfg: Dict[str, Any]):
    valid_boxes = getattr(batch, "kd_valid_boxes", None)
    batch = copy_data_to_device(batch, device, non_blocking=True)
    if valid_boxes is not None:
        batch.kd_valid_boxes = valid_boxes.to(device=device, non_blocking=True)
    return add_multi_resolution_batches(batch, cfg)


def format_logs(logs: Dict[str, float | torch.Tensor]) -> str:
    parts = []
    for key, value in logs.items():
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().cpu())
        parts.append(f"{key}={float(value):.6f}")
    return " ".join(parts)


def should_log(batch_index: int, total: int, every: int) -> bool:
    return batch_index == 1 or batch_index == total or (every > 0 and batch_index % every == 0)


def forward_with_amp(
    model: MaskDistill,
    batch,
    cfg: Dict[str, Any],
    prompt_type: str | None = None,
):
    enabled = bool(cfg.get("train", {}).get("amp", True)) and model.device.type == "cuda"
    with torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
    ):
        return model(batch, prompt_type=prompt_type)


def predict_student_with_amp(model, batch, cfg, prompt_type):
    enabled = bool(cfg.get("eval", {}).get("amp", True)) and model.device.type == "cuda"
    with torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
    ):
        return model.predict_student_masks(batch, prompt_type)


def build_optimizer(model: MaskDistill, cfg: Dict[str, Any]):
    optim_cfg = cfg.get("optim", {})
    weight_decay = float(optim_cfg.get("weight_decay", 0.05))
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay
        target.append(parameter)
    if not decay and not no_decay:
        raise RuntimeError("No trainable parameters were found.")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(
        groups,
        lr=float(optim_cfg.get("lr", 1.0e-5)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(optimizer, total_updates: int, cfg: Dict[str, Any]):
    scheduler_cfg = cfg.get("param_scheduler", {})
    warmup = min(int(scheduler_cfg.get("warmup_updates", 100)), total_updates)
    warmup_factor = float(scheduler_cfg.get("warmup_start_factor", 0.1))
    min_lr = float(scheduler_cfg.get("min_lr", 1.0e-6))
    base_lr = float(cfg.get("optim", {}).get("lr", 1.0e-5))
    min_factor = min(min_lr / max(base_lr, 1.0e-12), 1.0)

    def schedule(update: int) -> float:
        if warmup > 0 and update < warmup:
            return warmup_factor + (1.0 - warmup_factor) * update / warmup
        if total_updates <= warmup:
            return min_factor
        progress = min(max((update - warmup) / (total_updates - warmup), 0.0), 1.0)
        return min_factor + (1.0 - min_factor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(
    path: Path,
    model: MaskDistill,
    optimizer,
    scheduler,
    scaler,
    cfg: Dict[str, Any],
    epoch: int,
    global_update: int,
    best_miou: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_model": model.get_student_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "global_update": global_update,
            "best_miou": best_miou,
        },
        path,
    )
    print(f"[checkpoint] saved: {path}", flush=True)


def save_student(path: Path, model: MaskDistill) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.get_student_state_dict(), path)
    print(f"[student] saved: {path}", flush=True)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(path, map_location=model.device, weights_only=False)
    model.student_model.load_state_dict(checkpoint["student_model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    print(
        f"[checkpoint] resumed: {path} epoch={checkpoint['epoch']} "
        f"update={checkpoint['global_update']}",
        flush=True,
    )
    return checkpoint


@torch.no_grad()
def evaluate(
    model: MaskDistill,
    cfg: Dict[str, Any],
    epoch: int,
    max_batches_override: int | None = None,
) -> Dict[str, float]:
    """Evaluate prompted student masks against validation annotation masks."""
    if not bool(cfg.get("eval", {}).get("enabled", True)):
        return {}
    model.eval()
    loader = build_val_dataloader(cfg)
    eval_cfg = cfg.get("eval", {})
    prompt_types = eval_cfg.get(
        "prompt_types", cfg.get("mask_distill", {}).get("prompt_types", ["box"])
    )
    max_batches = (
        max_batches_override
        if max_batches_override is not None
        else int(eval_cfg.get("max_batches", 0))
    )
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    iou_sums = {str(prompt_type): 0.0 for prompt_type in prompt_types}
    mask_counts = {str(prompt_type): 0 for prompt_type in prompt_types}
    skipped = 0
    valid_batches = 0
    threshold = float(eval_cfg.get("mask_threshold", 0.5))

    for batch_index, batch in enumerate(loader, start=1):
        if int(batch.find_targets[0].num_boxes.sum()) == 0:
            skipped += 1
            continue
        valid_batches += 1
        if max_batches > 0 and valid_batches > max_batches:
            break
        batch = move_to_device(batch, model.device, cfg)
        for prompt_type in prompt_types:
            prompt_type = str(prompt_type)
            try:
                output = predict_student_with_amp(
                    model, batch, cfg, prompt_type
                )
            except NoValidAnnotationsError:
                skipped += 1
                continue
            predictions = output["pred_masks"].sigmoid() > threshold
            targets = output["gt_masks"].bool()
            count = len(targets)
            batch_miou = segment_miou(predictions, targets)
            iou_sums[prompt_type] += float(batch_miou.cpu()) * count
            mask_counts[prompt_type] += count
        if should_log(valid_batches, total_batches, int(eval_cfg.get("log_every_batches", 20))):
            running = {
                f"{name}_mIoU": iou_sums[name] / max(mask_counts[name], 1)
                for name in iou_sums
            }
            print(
                f"[val] epoch={epoch} valid_batch={valid_batches}/{total_batches} "
                f"source_batch={batch_index}/{len(loader)} {format_logs(running)}",
                flush=True,
            )

    metrics = {
        f"{name}_mIoU": iou_sums[name] / max(mask_counts[name], 1)
        for name in iou_sums
    }
    total_iou = sum(iou_sums.values())
    total_masks = sum(mask_counts.values())
    metrics["mIoU"] = total_iou / max(total_masks, 1)
    metrics["evaluated_masks"] = float(total_masks)
    metrics["skipped_batches"] = float(skipped)
    print(f"[val] epoch={epoch} {format_logs(metrics)}", flush=True)
    model.train()
    return metrics


def train(args, cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg.get("train", {}).get("seed", 123)))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM3 mask distillation.")
    device = torch.device("cuda")

    if args.smoke_test:
        cfg["mask_distill"]["max_prompts_per_batch"] = 1
        cfg["train"]["max_train_batches"] = 1
        cfg["eval"]["max_batches"] = 1
        cfg["dataset"]["train_num_images"] = 1
        cfg["dataset"]["val_num_images"] = 1

    print("[data] building training dataloader", flush=True)
    train_loader = build_dataloader(cfg)
    print(f"[data] training batches={len(train_loader)}", flush=True)
    model = MaskDistill(cfg, device).to(device)
    model.train()
    optimizer = build_optimizer(model, cfg)

    train_cfg = cfg.get("train", {})
    epochs = 1 if args.smoke_test else int(args.epochs or train_cfg.get("epochs", 1))
    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
    configured_max_batches = int(train_cfg.get("max_train_batches", 0))
    train_batches = (
        min(len(train_loader), configured_max_batches)
        if configured_max_batches > 0
        else len(train_loader)
    )
    updates_per_epoch = math.ceil(train_batches / accumulation_steps)
    scheduler = build_scheduler(optimizer, max(1, updates_per_epoch * epochs), cfg)
    amp_enabled = bool(train_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    output_dir = Path(args.output)
    start_epoch = 1
    global_update = 0
    best_miou = float("-inf")
    if args.resume:
        checkpoint = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_update = int(checkpoint["global_update"])
        best_miou = float(checkpoint.get("best_miou", best_miou))

    if args.eval_only:
        evaluate(model, cfg, start_epoch - 1)
        return

    max_grad_norm = float(cfg.get("optim", {}).get("max_grad_norm", 1.0))
    log_every = int(train_cfg.get("log_every_batches", 10))
    eval_every = int(train_cfg.get("eval_every_epochs", 1))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        processed = 0
        accumulated = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            if configured_max_batches > 0 and batch_index > configured_max_batches:
                break
            batch = move_to_device(batch, device, cfg)
            try:
                total_loss, logs = forward_with_amp(model, batch, cfg)
            except NoValidAnnotationsError:
                continue
            scaler.scale(total_loss / accumulation_steps).backward()
            accumulated += 1
            processed += 1
            running_loss += float(total_loss.detach().float().cpu())

            is_last = batch_index == train_batches
            if accumulated == accumulation_steps or is_last:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1
                accumulated = 0
            else:
                grad_norm = total_loss.new_zeros(())

            if should_log(batch_index, train_batches, log_every):
                print(
                    f"[train] epoch={epoch}/{epochs} batch={batch_index}/{train_batches} "
                    f"update={global_update} lr={optimizer.param_groups[0]['lr']:.8f} "
                    f"grad_norm={float(grad_norm):.6f} {format_logs(logs)}",
                    flush=True,
                )

        if accumulated > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_update += 1
        if processed == 0:
            raise RuntimeError("No training batches contained valid annotations.")
        print(
            f"[train] epoch={epoch}/{epochs} done batches={processed} "
            f"avg_loss={running_loss / processed:.6f}",
            flush=True,
        )

        metrics = {}
        if eval_every > 0 and epoch % eval_every == 0:
            metrics = evaluate(
                model,
                cfg,
                epoch,
                max_batches_override=1 if args.smoke_test else None,
            )
        current_miou = float(metrics.get("mIoU", float("-inf")))
        if current_miou > best_miou:
            best_miou = current_miou
            save_checkpoint(
                output_dir / "best_mask_distillation.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                epoch,
                global_update,
                best_miou,
            )
            save_student(output_dir / "best_mask_student.pt", model)

        save_checkpoint(
            output_dir / "latest_mask_distillation.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            epoch,
            global_update,
            best_miou,
        )
        save_student(output_dir / "latest_mask_student.pt", model)


def parse_args():
    parser = argparse.ArgumentParser(description="Run second-stage SAM3 mask distillation")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "Config" / "mask_distill_cfg" / "vit_small_mask_distill.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "mask_distill"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args, load_config(cli_args.config))
