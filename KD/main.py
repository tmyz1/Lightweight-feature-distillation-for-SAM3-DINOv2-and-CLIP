import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KD.build_dataloader import build_dataloader
from KD.distill_model import Distill
from KD.eval.coco_evaluation import evaluate
from KD.training.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    save_student_model,
)
from KD.training.distributed import (
    destroy_distributed,
    distributed_mean,
    initialize_distributed,
    is_distributed,
    is_main_process,
)
from KD.training.optim import (
    build_optimizer,
    build_scheduler,
    format_learning_rates,
)
from KD.training.runtime import (
    format_logs,
    forward_with_amp,
    move_to_device,
    seed_everything,
    should_log_batch,
)

#下载配置文件
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

#创建蒸馏模型
def build_distiller(cfg: Dict[str, Any], device: torch.device):
    return Distill(cfg, device)

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
    device, local_rank = initialize_distributed(cfg)
    rank = dist.get_rank() if is_distributed() else 0
    seed_everything(int(cfg.get("train", {}).get("seed", 123)) + rank)

    #构建数据集
    if is_main_process():
        print(f'begin building dataloader')
    dataloader = build_dataloader(cfg)
    if is_main_process():
        print(f"dataloader built: batches={len(dataloader)}")

    #构建训练模型
    raw_model = build_distiller(cfg, device)
    raw_model.to(device)
    if is_distributed():
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(
                cfg.get("multi_gpu", {}).get("find_unused_parameters", False)
            ),
        )
    else:
        model = raw_model
    model.train()

    optimizer = build_optimizer(raw_model, cfg)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(cfg.get("train", {}).get("amp", True)) and raw_model.device.type == "cuda",
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
            Path(args.resume), raw_model, optimizer, scheduler, scaler
        )
        start_step = int(checkpoint["step"]) + 1
        global_update = int(checkpoint["global_update"])
        best_map = float(checkpoint.get("best_map", best_map))

    if args.eval_only:
        if args.resume is None:
            raise ValueError("--eval-only requires --resume with a student checkpoint.")
        if is_main_process():
            evaluate(raw_model, cfg, start_step - 1)
        if is_distributed():
            dist.barrier()
        destroy_distributed()
        raise SystemExit(0)

    #begin training
    for step in range(start_step, max_steps + 1):
        if is_distributed() and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(step)
        running_loss = 0.0
        running_batches = 0
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = move_to_device(batch, raw_model.device, cfg)
            optimizer.zero_grad(set_to_none=True)

            total_loss, logs = forward_with_amp(model, batch, cfg)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.trainable_parameters(), max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_update += 1
            running_loss += float(total_loss.detach().float().cpu())
            running_batches += 1

            if is_main_process() and should_log_batch(
                batch_idx, total_batches, log_every_batches
            ):
                print(
                    f"[train] step={step}/{max_steps} "
                    f"batch={batch_idx}/{total_batches} "
                    f"update={global_update} grad_norm={float(grad_norm):.6f} "
                    f"{format_learning_rates(optimizer)} {format_logs(logs)}",
                    flush=True,
                )

        avg_loss = distributed_mean(
            running_loss / max(running_batches, 1), raw_model.device
        )
        if is_main_process():
            print(
                f"[train] step={step}/{max_steps} done "
                f"batches={running_batches} avg_loss_total={avg_loss:.6f}",
                flush=True,
            )

        metrics = {}
        if eval_every > 0 and step % eval_every == 0:
            if is_main_process():
                metrics = evaluate(raw_model, cfg, step)
                current_map = float(metrics.get("coco_eval_bbox_AP", float("-inf")))
                if current_map > best_map:
                    best_map = current_map
                    save_checkpoint(
                        output_dir / "best_distillation.pt",
                        raw_model,
                        optimizer,
                        scheduler,
                        scaler,
                        cfg,
                        step,
                        global_update,
                        best_map,
                    )
                    save_student_model(output_dir / "best_student.pt", raw_model)
            if is_distributed():
                dist.barrier()

        if is_main_process():
            save_checkpoint(
                output_dir / "latest_distillation.pt",
                raw_model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                step,
                global_update,
                best_map,
            )
            save_student_model(output_dir / "latest_student.pt", raw_model)

    destroy_distributed()
