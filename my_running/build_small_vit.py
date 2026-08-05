from pathlib import Path
import sys
from typing import Tuple

import torch
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dinov2.models.vision_transformer import vit_small  # noqa: E402
from sam3.model_builder import build_vit_small_image_model  # noqa: E402


WEIGHT_PATH = Path(r"E:\reproduce\weights\DINO-V2\vit_small_patch14_reg4_dinov2.safetensors")


def build_dinov2_small_reg(
    weight_path: Path = WEIGHT_PATH,
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.nn.Module:
    model = vit_small(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    state_dict = load_file(str(weight_path), device="cpu")
    if "reg_token" in state_dict and "register_tokens" not in state_dict:
        state_dict["register_tokens"] = state_dict.pop("reg_token")
    if (
        "pos_embed" in state_dict
        and state_dict["pos_embed"].shape[1] + 1 == model.pos_embed.shape[1]
    ):
        cls_pos_embed = model.pos_embed[:, :1].detach().clone()
        state_dict["pos_embed"] = torch.cat((cls_pos_embed, state_dict["pos_embed"]), dim=1)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}")
        if missing_keys:
            print(f"missing examples: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"unexpected examples: {unexpected_keys[:5]}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def output_last_cls_reg_pos(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = model.forward_features(image)
    cls_token = features["x_norm_clstoken"]
    reg_token = features["x_norm_regtokens"]

    patch_tokens = model.patch_embed(image)
    token_input = torch.cat((model.cls_token.expand(image.shape[0], -1, -1), patch_tokens), dim=1)
    pos_embed = model.interpolate_pos_encoding(token_input, image.shape[-2], image.shape[-1])
    return cls_token, reg_token, pos_embed


@torch.no_grad()
def output_backbone_feature_map(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    features = model.forward_features(image)
    patch_tokens = features["x_norm_patchtokens"]
    batch_size, num_patches, channels = patch_tokens.shape
    patch_h = image.shape[-2] // model.patch_size
    patch_w = image.shape[-1] // model.patch_size
    if num_patches != patch_h * patch_w:
        raise ValueError(
            f"patch token number {num_patches} does not match image grid {patch_h}x{patch_w}"
        )
    return patch_tokens.reshape(batch_size, patch_h, patch_w, channels).permute(0, 3, 1, 2).contiguous()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dinov2_small_reg(device=device)
    image = torch.randn(1, 3, 518, 518, device=device)

    cls_token, reg_token, pos_embed = output_last_cls_reg_pos(model, image)
    backbone_feature = output_backbone_feature_map(model, image)

    print(f"image: {tuple(image.shape)}")
    print(f"cls_token: {tuple(cls_token.shape)}")
    print(f"reg_token: {tuple(reg_token.shape)}")
    print(f"pos_embed: {tuple(pos_embed.shape)}")
    print(f"backbone_feature: {tuple(backbone_feature.shape)}")

    sam3_model = build_vit_small_image_model(
        device=str(device),
        eval_mode=True,
        checkpoint_path=None,
        enable_segmentation=False,
    )
    with torch.no_grad():
        raw_features, _, _, _ = sam3_model.backbone.vision_backbone(image)
        sam3_backbone_out = sam3_model.backbone.forward_image(image)
    raw_fpn_shapes = [tuple(feature.shape) for feature in raw_features]
    fpn_shapes = [tuple(feature.shape) for feature in sam3_backbone_out["backbone_fpn"]]
    print(f"sam3_vit_small_raw_neck_fpn: {raw_fpn_shapes}")
    print(f"sam3_vit_small_backbone_fpn: {fpn_shapes}")
    print(f"sam3_vit_small_vision_features: {tuple(sam3_backbone_out['vision_features'].shape)}")


if __name__ == "__main__":
    main()
