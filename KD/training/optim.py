from __future__ import annotations

import math
from typing import Any, Dict

import torch

from KD.training.distributed import unwrap_model


def build_optimizer(model, cfg: Dict[str, Any]):
    model = unwrap_model(model)
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
