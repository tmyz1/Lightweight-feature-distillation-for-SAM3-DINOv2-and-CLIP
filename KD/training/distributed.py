from __future__ import annotations

import os
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def initialize_distributed(cfg: Dict[str, Any]) -> tuple[torch.device, int]:
    multi_gpu_cfg = cfg.get("multi_gpu", {})
    if not bool(multi_gpu_cfg.get("enabled", False)):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training run.")
        return torch.device("cuda"), 0

    if not torch.cuda.is_available():
        raise RuntimeError("multi_gpu.enabled requires CUDA.")
    required_env = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    missing_env = [name for name in required_env if name not in os.environ]
    if missing_env:
        raise RuntimeError(
            "multi_gpu.enabled requires torchrun, for example: "
            "torchrun --standalone --nproc_per_node=2 KD/main.py. "
            f"Missing environment variables: {', '.join(missing_env)}."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError(
            "multi_gpu.enabled requires at least two processes. Set "
            "--nproc_per_node to the number of GPUs to use."
        )
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} has no visible CUDA device. "
            f"torch.cuda.device_count()={torch.cuda.device_count()}."
        )
    torch.cuda.set_device(local_rank)
    default_backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(
        backend=str(multi_gpu_cfg.get("backend", default_backend))
    )
    return torch.device("cuda", local_rank), local_rank


def destroy_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def distributed_mean(value: float, device: torch.device) -> float:
    if not is_distributed():
        return value
    value_tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    return float(value_tensor.item() / dist.get_world_size())
