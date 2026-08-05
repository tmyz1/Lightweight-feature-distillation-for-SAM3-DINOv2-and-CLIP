from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import v2

from sam3.model_builder import build_sam3_image_model


DEFAULT_CHECKPOINT = r"E:\reproduce\weights\sam3\sam3.pt"
DEFAULT_IMAGE = r"E:\pycharm\sam3-main\sam3-main\sam3-main\assets\images\test_image.jpg"


def build_image_tensor(image_path: str, device: torch.device, resolution: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(v2.functional.to_image(image).to(device)).unsqueeze(0)


def save_feature_heatmap(feature: torch.Tensor, save_path: Path) -> None:
    feature_map = feature.detach().float().cpu()[0]
    heatmap = feature_map.abs().mean(dim=0)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max().clamp_min(1e-6)

    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap.numpy(), cmap="viridis")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract four intermediate SAM3 ViT features.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device=str(device),
        eval_mode=True,
        enable_segmentation=False,
    )

    vit = model.backbone.vision_backbone.trunk
    vit.return_interm_layers = True
    vit.eval()

    image_tensor = build_image_tensor(args.image, device=device, resolution=args.resolution)

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        features = vit(image_tensor)

    if len(features) != 4:
        raise RuntimeError(f"Expected 4 ViT features, got {len(features)}.")

    feature_shapes = []
    for index, feature in enumerate(features):
        feature = getattr(feature, "tensors", feature)
        feature_shapes.append(tuple(feature.shape))

    print("Extracted 4 SAM3 ViT intermediate features.")
    for index, shape in enumerate(feature_shapes):
        print(f"feature_{index}: shape={shape}")


if __name__ == "__main__":
    main()
