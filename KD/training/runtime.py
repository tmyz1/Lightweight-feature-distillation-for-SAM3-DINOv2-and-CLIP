from __future__ import annotations

import random
from typing import Any, Dict

import torch

from sam3.model.utils.misc import copy_data_to_device

from KD.build_dataloader import add_multi_resolution_batches
from KD.training.distributed import unwrap_model


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


#精度转换
def forward_with_amp(model, batch, cfg: Dict[str, Any]):
    raw_model = unwrap_model(model)
    with torch.autocast(
            device_type=raw_model.device.type,
            dtype=torch.bfloat16,
            enabled=bool(cfg.get("train", {}).get("amp", True))
            and raw_model.device.type == "cuda",
    ):
        return model(batch)
