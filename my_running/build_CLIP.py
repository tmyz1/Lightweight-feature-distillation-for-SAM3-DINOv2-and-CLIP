import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clip  # noqa: E402


PRETRAINED_MODEL_PATH = r"E:\reproduce\weights\CLIP\ViT-L-14.pt"


def build_clip_model(
    model_path: str = PRETRAINED_MODEL_PATH,
    input_resolution: int | None = None,
    device: torch.device | str | None = None,
):
    """Build CLIP and load the local pretrained ViT-L/14 weights."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Pretrained CLIP weights not found: {model_path}")

    model, preprocess = clip.load(model_path, device=device, jit=False)
    if input_resolution is not None:
        resize_vit_positional_embedding(model, input_resolution)
        preprocess = clip.clip._transform(input_resolution)

    model.eval()
    return model, preprocess, torch.device(device)


def resize_vit_positional_embedding(model: torch.nn.Module, input_resolution: int) -> None:
    """Resize pretrained ViT positional embeddings for a new image resolution."""
    visual = model.visual

    if not hasattr(visual, "positional_embedding"):
        raise TypeError("model.visual is not a VisionTransformer.")

    patch_size = visual.conv1.kernel_size[0]
    if input_resolution % patch_size != 0:
        raise ValueError(
            f"input_resolution={input_resolution} must be divisible by patch_size={patch_size}."
        )

    old_pos = visual.positional_embedding.detach()
    cls_pos = old_pos[:1]
    patch_pos = old_pos[1:]

    old_grid = int(patch_pos.shape[0] ** 0.5)
    new_grid = input_resolution // patch_size
    if old_grid == new_grid:
        visual.input_resolution = input_resolution
        return

    patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(),
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=old_pos.dtype, device=old_pos.device)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid * new_grid, -1)

    new_pos = torch.cat([cls_pos, patch_pos], dim=0)
    visual.positional_embedding = torch.nn.Parameter(new_pos)
    visual.input_resolution = input_resolution


@torch.no_grad()
def extract_vit_xy(
    model: torch.nn.Module,
    image: torch.Tensor,
    out_indices: Sequence[int] = (5, 11, 17, 23),
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """
    Return image features from VisionTransformer.forward:
    x: CLS/global image feature after ln_post and proj, shape [B, output_dim]
    y: patch feature map after ViT transformer, shape [B, C, grid, grid]
    multi_layer_features: patch feature maps from out_indices, each [B, C, grid, grid]
    multi_layer_cls: CLS tokens from out_indices, each [B, C]
    """
    visual = model.visual

    if not hasattr(visual, "transformer"):
        raise TypeError("model.visual is not a VisionTransformer.")

    image = image.to(next(model.parameters()).device)
    x = image.type(model.dtype)

    x = visual.conv1(x)
    batch_size, channels, grid_h, grid_w = x.shape

    x = x.reshape(batch_size, channels, -1)
    x = x.permute(0, 2, 1)
    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        batch_size, 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)
    if x.shape[1] != visual.positional_embedding.shape[0]:
        raise ValueError(
            "Image token count does not match positional embedding length. "
            f"Got {x.shape[1]} image tokens, but positional_embedding has "
            f"{visual.positional_embedding.shape[0]} tokens. "
            "Call build_clip_model(input_resolution=...) or "
            "resize_vit_positional_embedding(model, input_resolution) first."
        )
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    num_blocks = len(visual.transformer.resblocks)
    invalid_indices = [idx for idx in out_indices if idx < 0 or idx >= num_blocks]
    if invalid_indices:
        raise ValueError(
            f"out_indices contains invalid layer indices {invalid_indices}; "
            f"valid range is 0 to {num_blocks - 1}."
        )

    def tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.permute(1, 0, 2)
        feat = tokens[:, 1:, :]
        feat = feat.view(batch_size, grid_h, grid_w, channels)
        return feat.permute(0, 3, 1, 2)

    out_indices = tuple(out_indices)
    multi_layer_features = []
    multi_layer_cls = []
    x = x.permute(1, 0, 2)
    for block_index, block in enumerate(visual.transformer.resblocks):
        x = block(x)
        if block_index in out_indices:
            multi_layer_features.append(tokens_to_feature_map(x))
            multi_layer_cls.append(x[0])
    return tuple(multi_layer_features), tuple(multi_layer_cls)


if __name__ == "__main__":
    input_resolution = 1008
    model, preprocess, device = build_clip_model(input_resolution=input_resolution)

    image = preprocess(Image.open(r"E:\pycharm\CLIP-main\CLIP-main\CLIP.png")).unsqueeze(0).to(device)
    print(image.shape)
    multi_layer_features, multi_layer_cls = extract_vit_xy(model, image)
    for i, (feature, cls) in enumerate(zip(multi_layer_features, multi_layer_cls)):
        print(f"feature_{i}: {feature.shape}, cls_{i}: {cls.shape}")
