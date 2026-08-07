"""Benchmark SAM3 and a distilled image backbone on one prompted image.

The measured interval starts at ``Sam3Processor.set_image`` and ends after
``Sam3Processor.set_text_prompt`` returns. Image decoding and model/checkpoint
loading are intentionally excluded from the measurements.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from KD.model.vit_small_patch14_reg4_dinov2 import (
    build_vit_small_image_model,
    replace_vit_small_memory_efficient_attention,
)


DEFAULT_SAM3_CHECKPOINT = Path(r"E:\reproduce\weights\sam3\sam3.pt")
DEFAULT_VIT_SMALL_CHECKPOINT = Path(
    r"E:\reproduce\weights\sam3 distill\vit_small_patch14_reg4_dinov2 distill\best_student.pt"
)
DEFAULT_IMAGE_PATH = Path(r"E:\my_data\COCO\train\train2017\000000000328.jpg")


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    resolution: int
    image_encoder_ms: float
    prompt_and_output_ms: float
    total_ms: float
    output: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full SAM3 against a distilled image-backbone model."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument(
        "--vit-small-checkpoint", type=Path, default=DEFAULT_VIT_SMALL_CHECKPOINT
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Display threshold for prompted detections. COCO mAP itself ranks all boxes.",
    )
    parser.add_argument(
        "--sam3-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BF16 autocast for original SAM3. Required by this local SAM3 implementation.",
    )
    parser.add_argument(
        "--vit-small-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BF16 autocast for ViT-Small after matching its training-time attention implementation.",
    )
    return parser.parse_args()


def validate_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def run_prompted_inference(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    amp: bool,
) -> tuple[float, float, dict[str, Any]]:
    """Run image encoding and prompted prediction with CUDA-accurate timing."""
    synchronize(device)
    image_start = time.perf_counter()
    with autocast_context(device, enabled=amp):
        state = processor.set_image(image)
    synchronize(device)
    image_encoder_ms = (time.perf_counter() - image_start) * 1000.0

    synchronize(device)
    prompt_start = time.perf_counter()
    # Sam3Processor prints all raw proposal scores; suppress it during timing.
    with contextlib.redirect_stdout(io.StringIO()), autocast_context(
        device, enabled=amp
    ):
        state = processor.set_text_prompt(state=state, prompt=prompt)
    synchronize(device)
    prompt_and_output_ms = (time.perf_counter() - prompt_start) * 1000.0
    return image_encoder_ms, prompt_and_output_ms, state


def load_distilled_vit_small(
    sam3_checkpoint: Path,
    student_checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Build the ViT-Small SAM3 variant and load its distilled student weights.

    The distillation checkpoint contains ``student_model``. Its segmentation
    head was disabled during distillation, so the original SAM3 checkpoint
    supplies that unchanged head for normal prompted inference.
    """
    model = build_vit_small_image_model(
        checkpoint_path=str(sam3_checkpoint),
        vit_small_checkpoint_path=None,
        device=str(device),
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    checkpoint = torch.load(student_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("student_model", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("The ViT-Small checkpoint does not contain a state dictionary.")

    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_segmentation_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("segmentation_head.")
    ]
    if unexpected or non_segmentation_missing:
        raise RuntimeError(
            "Distilled checkpoint is incompatible with build_vit_small_image_model. "
            f"Unexpected keys: {unexpected[:8]}; "
            f"missing non-segmentation keys: {non_segmentation_missing[:8]}."
        )
    # Match Vit_Small_feature_extractor: the distilled model was trained with
    # native attention instead of xFormers MemEffAttention.
    replace_vit_small_memory_efficient_attention(model)
    model.to(device)
    print(
        "Loaded distilled ViT-Small checkpoint: "
        f"{len(incompatible.missing_keys)} inherited segmentation-head key(s)."
    )
    return model


def benchmark_model(
    name: str,
    resolution: int,
    model_builder: Callable[[], torch.nn.Module],
    image: Image.Image,
    prompt: str,
    warmup_runs: int,
    measured_runs: int,
    device: torch.device,
    confidence_threshold: float,
    amp: bool,
) -> BenchmarkResult:
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup must be non-negative and runs must be at least 1.")

    # Model construction and checkpoint loading happen before any timer starts.
    model = model_builder().eval()
    processor = Sam3Processor(
        model,
        resolution=resolution,
        device=str(device),
        confidence_threshold=confidence_threshold,
    )

    for _ in range(warmup_runs):
        run_prompted_inference(processor, image, prompt, device, amp)

    image_times: list[float] = []
    prompt_times: list[float] = []
    output: dict[str, Any] | None = None
    for _ in range(measured_runs):
        image_time, prompt_time, output = run_prompted_inference(
            processor, image, prompt, device, amp
        )
        image_times.append(image_time)
        prompt_times.append(prompt_time)

    assert output is not None
    result = BenchmarkResult(
        name=name,
        resolution=resolution,
        image_encoder_ms=sum(image_times) / len(image_times),
        prompt_and_output_ms=sum(prompt_times) / len(prompt_times),
        total_ms=(sum(image_times) + sum(prompt_times)) / len(image_times),
        output=output,
    )

    del processor, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def print_output_summary(result: BenchmarkResult, top_k: int = 5) -> None:
    output = result.output
    boxes = output.get("boxes")
    scores = output.get("scores")
    masks = output.get("masks")
    print(f"\n[{result.name}] output")
    print(f"  boxes: {tuple(boxes.shape) if isinstance(boxes, torch.Tensor) else boxes}")
    print(f"  scores: {tuple(scores.shape) if isinstance(scores, torch.Tensor) else scores}")
    print(f"  masks: {tuple(masks.shape) if isinstance(masks, torch.Tensor) else masks}")

    if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
        return
    if scores.numel() == 0:
        print("  detections above confidence threshold: 0")
        return

    k = min(top_k, scores.numel())
    top_scores, top_indices = torch.topk(scores.detach().float().cpu(), k=k)
    boxes_cpu = boxes.detach().float().cpu()
    print(f"  detections above confidence threshold: {scores.numel()}")
    for rank, (score, index) in enumerate(zip(top_scores, top_indices), start=1):
        box = [round(value, 2) for value in boxes_cpu[index].tolist()]
        print(f"  top-{rank}: score={score.item():.4f}, box_xyxy={box}")


def print_timing_summary(results: list[BenchmarkResult]) -> None:
    print("\nAverage inference time per image (model loading and image decoding excluded):")
    for result in results:
        print(
            f"  {result.name} ({result.resolution}x{result.resolution}): "
            f"image_encoder={result.image_encoder_ms:.2f} ms, "
            f"prompt_and_output={result.prompt_and_output_ms:.2f} ms, "
            f"total={result.total_ms:.2f} ms"
        )
    sam3, vit_small = results
    speedup = sam3.total_ms / vit_small.total_ms
    reduction = (1.0 - vit_small.total_ms / sam3.total_ms) * 100.0
    print(f"  ViT-Small speedup: {speedup:.2f}x; latency reduction: {reduction:.1f}%")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    validate_paths([args.image, args.sam3_checkpoint, args.vit_small_checkpoint])

    # Decoding the file is deliberately outside the timed inference interval.
    with Image.open(args.image) as source_image:
        image = source_image.convert("RGB")

    print(f"device={device}, image={args.image}, prompt={args.prompt!r}")
    print(
        f"warmup={args.warmup}, measured_runs={args.runs}, "
        f"confidence_threshold={args.confidence_threshold}, "
        f"sam3_amp={args.sam3_amp}, vit_small_amp={args.vit_small_amp}"
    )

    sam3_result = benchmark_model(
        name="SAM3",
        resolution=1008,
        model_builder=lambda: build_sam3_image_model(
            checkpoint_path=str(args.sam3_checkpoint),
            device=str(device),
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        ),
        image=image,
        prompt=args.prompt,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        device=device,
        confidence_threshold=args.confidence_threshold,
        amp=args.sam3_amp,
    )
    vit_small_result = benchmark_model(
        name="Distilled ViT-Small",
        resolution=518,
        model_builder=lambda: load_distilled_vit_small(
            args.sam3_checkpoint, args.vit_small_checkpoint, device
        ),
        image=image,
        prompt=args.prompt,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        device=device,
        confidence_threshold=args.confidence_threshold,
        amp=args.vit_small_amp,
    )

    print_timing_summary([sam3_result, vit_small_result])
    print_output_summary(sam3_result)
    print_output_summary(vit_small_result)


if __name__ == "__main__":
    main()
