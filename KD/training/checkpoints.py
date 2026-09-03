from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from KD.training.distributed import unwrap_model


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    cfg: Dict[str, Any],
    step: int,
    global_update: int,
    best_map: float,
) -> None:
    model = unwrap_model(model)
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


def save_student_model(path: Path, model) -> None:
    """Save the deployable student without teachers, adapters, or optimizer state."""
    model = unwrap_model(model)
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
    model,
    optimizer,
    scheduler,
    scaler,
) -> Dict[str, Any]:
    model = unwrap_model(model)
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
