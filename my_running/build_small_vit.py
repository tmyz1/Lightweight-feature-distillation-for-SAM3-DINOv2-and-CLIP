from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KD.model.vit_small_patch14_reg4_dinov2 import build_vit_small_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "Config" / "Vit_Small_Distill.yaml"
DEFAULT_STUDENT_CHECKPOINT = Path(
    r"E:\reproduce\weights\sam3 distill\vit_small_patch14_reg4_dinov2 distill\best_student.pt"
)
DEFAULT_IMAGE_PATH = Path(r"E:\my_data\rf100-vl\apex-videogame\train\4_png_jpg.rf.23cd233ad6ae68291c2de556f76263b0.jpg")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "my_running" / "outputs" / "small_vit_result.png"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_student_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Student checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "student_model" in checkpoint:
        checkpoint = checkpoint["student_model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported student checkpoint format: {type(checkpoint)}")
    return checkpoint


def build_distilled_vit_small(
    cfg: Dict[str, Any],
    student_checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    student_cfg = cfg.get("Student", {})
    sam3_cfg = cfg.get("Sam3", {})

    model = build_vit_small_image_model(
        bpe_path=sam3_cfg.get("bpe_path"),
        checkpoint_path=sam3_cfg.get("checkpoint_path"),
        device=str(device),
        eval_mode=True,
        enable_segmentation=bool(student_cfg.get("enable_segmentation", True)),
        enable_inst_interactivity=False,
        vit_small_checkpoint_path=student_cfg.get("vit_small_checkpoint_path"),
        return_interm_layers=bool(
            cfg.get("train", {}).get("return_interm_layers", False)
        ),
        vit_small_intermediate_layers=student_cfg.get(
            "vit_small_intermediate_layers", (2, 5, 8, 11)
        ),
        cfg=cfg,
    )

    student_state = load_student_state_dict(student_checkpoint_path)
    missing_keys, unexpected_keys = model.load_state_dict(student_state, strict=False)

    allowed_missing = {
        "backbone.vision_backbone.layer_fusion_logits",
    }
    invalid_missing = [
        key
        for key in missing_keys
        if not key.startswith("segmentation_head.") and key not in allowed_missing
    ]
    invalid_unexpected = [
        key for key in unexpected_keys if key not in allowed_missing
    ]
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(
            "Student checkpoint was not loaded cleanly. "
            f"Missing: {invalid_missing[:10]}; "
            f"unexpected: {invalid_unexpected[:10]}."
        )
    if missing_keys:
        print(f"[checkpoint] allowed missing keys: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"[checkpoint] ignored unexpected keys: {unexpected_keys[:10]}")

    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    image_path: Path,
    prompt: str,
    resolution: int,
    confidence_threshold: float,
    output_path: Path,
    use_amp: bool,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    processor = Sam3Processor(
        model,
        resolution=resolution,
        device="cuda" if next(model.parameters()).is_cuda else "cpu",
        confidence_threshold=confidence_threshold,
    )

    amp_enabled = use_amp and next(model.parameters()).is_cuda
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        inference_state = processor.set_image(image)
        inference_state = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )

    scores = inference_state.get("scores")
    num_objects = 0 if scores is None else int(scores.numel())
    print(f"found {num_objects} object(s)")
    if scores is not None and scores.numel() > 0:
        print(f"scores: {scores.detach().float().cpu().tolist()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(image, inference_state)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distilled ViT-Small SAM3 demo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", default="people")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run inference with CUDA bfloat16 autocast. Disabled by default.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")

    device = torch.device("cuda")
    resolution = int(
        args.resolution
        or cfg.get("Student", {}).get(
            "kd_resolution",
            cfg.get("dataset", {}).get("resolution", 1008),
        )
    )

    model = build_distilled_vit_small(cfg, args.checkpoint, device)
    run_inference(
        model=model,
        image_path=args.image,
        prompt=args.prompt,
        resolution=resolution,
        confidence_threshold=args.threshold,
        output_path=args.output,
        use_amp=args.amp,
    )


if __name__ == "__main__":
    main()
