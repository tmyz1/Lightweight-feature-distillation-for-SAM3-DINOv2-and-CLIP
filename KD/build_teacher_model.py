import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Sequence

import torchvision
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import BatchedDatapoint
from dinov2.hub.backbones import dinov2_vitl14_reg
from dinov2.layers.attention import Attention, MemEffAttention
import clip


def replace_dinov2_memory_efficient_attention(model: nn.Module) -> None:
    """Use native PyTorch attention so DINOv2 remains stable under BF16 AMP."""
    for block in model.blocks:
        attention = block.attn
        if not isinstance(attention, MemEffAttention):
            continue
        replacement = Attention(
            dim=attention.dim,
            num_heads=attention.num_heads,
            qkv_bias=attention.qkv.bias is not None,
            proj_bias=attention.proj.bias is not None,
            attn_drop=getattr(attention.attn_drop, "p", attention.attn_drop),
            proj_drop=getattr(attention.proj_drop, "p", attention.proj_drop),
        )
        replacement.load_state_dict(attention.state_dict())
        block.attn = replacement


def extract_backbone_and_neck_features(
    vision_backbone: torch.nn.Module,
    img: torch.Tensor,
) :
    backbone_features = vision_backbone.trunk(img)
    if len(backbone_features) != 4:
        raise RuntimeError(f"Expected 4 backbone features, got {len(backbone_features)}.")

    if all(
        hasattr(vision_backbone, name)
        for name in (
            "lateral_convs",
            "top_down_convs",
            "bottom_up_downsample_convs",
            "bottom_up_convs",
            "vit_feature_proj",
            "_fuse_swin_features",
        )
    ):
        #用于适配Swin-Sam3的neck部分
        neck_features = [vision_backbone._fuse_swin_features(
            backbone_features,
            vision_backbone.lateral_convs,
            vision_backbone.top_down_convs,
            vision_backbone.bottom_up_downsample_convs,
            vision_backbone.bottom_up_convs,
            vision_backbone.vit_feature_proj,
        )]
    elif hasattr(vision_backbone, "convs"):
        #用于适配原本Sam3的neck部分
        x = backbone_features[-1]
        neck_features = [conv(x) for conv in vision_backbone.convs]
    else:
        raise TypeError(f"Unsupported vision_backbone type: {type(vision_backbone)}")

    return backbone_features, neck_features


def select_feature_layers(
    features: Sequence[torch.Tensor],
    layer_indices: Optional[int | Sequence[int]] = None,
) -> list[torch.Tensor]:
    """Return configured layers, defaulting to the final feature layer."""
    features = [getattr(feature, "tensors", feature) for feature in features]
    if not features:
        raise RuntimeError("Feature extractor returned no feature maps.")

    if layer_indices is None:
        return [features[-1]]
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]

    invalid_indices = [
        index for index in layer_indices if index < 0 or index >= len(features)
    ]
    if invalid_indices:
        raise ValueError(
            f"Invalid feature layer indices {invalid_indices}; valid range is "
            f"0 to {len(features) - 1}."
        )
    return [features[index] for index in layer_indices]


def select_layer_indices(
    layer_indices: Optional[int | Sequence[int]],
    num_layers: int,
) -> list[int]:
    """Validate transformer layer indices, defaulting to the final block."""
    if layer_indices is None:
        return [num_layers - 1]
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
    invalid_indices = [
        index for index in layer_indices if index < 0 or index >= num_layers
    ]
    if invalid_indices:
        raise ValueError(
            f"Invalid transformer layer indices {invalid_indices}; valid range is "
            f"0 to {num_layers - 1}."
        )
    return list(layer_indices)


def extract_distillation_features(
    vision_backbone: torch.nn.Module,
    img: torch.Tensor,
    layer_indices: Optional[int | Sequence[int]] = None,
) -> list[torch.Tensor]:
    """Extract Swin's fused ViT-like map or selected ViT trunk feature maps."""
    backbone_features, swin_features = extract_backbone_and_neck_features(
        vision_backbone, img
    )
    if hasattr(vision_backbone, "_fuse_swin_features"):
        if layer_indices not in (None, 0, [0], (0,)):
            raise ValueError(
                "Sam3SwinFPNDetNeck exposes one fused ViT-like distillation feature; "
                "its only valid layer index is 0."
            )
        return swin_features
    return select_feature_layers(backbone_features, layer_indices)


def aggregate_detector_features(
    detector_features: Sequence[torch.Tensor],
    target_size: Sequence[int],
) -> list[torch.Tensor]:
    if not detector_features:
        raise RuntimeError("Detector neck returned no feature maps.")
    reference = getattr(detector_features[0], "tensors", detector_features[0])
    aggregated_feature = reference.new_zeros(
        reference.shape[0], reference.shape[1], *target_size
    )
    for detector_feature in detector_features:
        detector_feature = getattr(detector_feature, "tensors", detector_feature)
        if detector_feature.shape[-2:] != target_size:
            detector_feature = F.interpolate(
                detector_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        aggregated_feature = aggregated_feature + detector_feature
    return [aggregated_feature / len(detector_features)]


def extract_sam3_detector_features(
    vision_backbone: torch.nn.Module,
    img: torch.Tensor,
) -> list[torch.Tensor]:
    return aggregate_detector_features(
        extract_sam3_neck_features(vision_backbone, img), target_size=(72, 72)
    )


def extract_sam3_neck_features(
    vision_backbone: torch.nn.Module,
    img: torch.Tensor,
) -> list[torch.Tensor]:
    sam3_out, _, _, _ = vision_backbone(img)
    return [getattr(feature, "tensors", feature) for feature in sam3_out]


def extract_sam3_backbone_and_neck_features(
    vision_backbone: torch.nn.Module,
    img: torch.Tensor,
    layer_indices: Optional[int | Sequence[int]] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    backbone_features = vision_backbone.trunk(img)
    backbone_features = [
        getattr(feature, "tensors", feature) for feature in backbone_features
    ]
    x = backbone_features[-1]
    neck_features = [conv(x) for conv in vision_backbone.convs]
    return select_feature_layers(backbone_features, layer_indices), neck_features


class Sam3_feature_extractor(torch.nn.Module):
    def __init__(self,
        checkpoint_path: str | nn.Module,
        device: torch.device,
        cfg: Optional[Dict[str, Any]] = None,
        feature_source: str = "backbone",
        ):
        super(Sam3_feature_extractor, self).__init__()
        self.device = device
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
            enable_segmentation=False,
            enable_inst_interactivity=False,
        )
        self.model.to(device)
        self.model.eval()
        self.return_interm_layers = bool(
            (cfg or {}).get("train", {}).get("return_interm_layers", False)
        )
        vision_backbone = self.model.backbone.vision_backbone
        num_blocks = len(vision_backbone.trunk.blocks)
        if self.return_interm_layers:
            available_layers = tuple(vision_backbone.trunk.full_attn_ids)
            self.sam3_intermediate_layers = tuple(
                (cfg or {}).get("Sam3", {}).get(
                    "sam3_intermediate_layers", available_layers
                )
            )
            invalid_layers = [
                layer
                for layer in self.sam3_intermediate_layers
                if layer not in available_layers
            ]
            if invalid_layers:
                raise ValueError(
                    f"SAM3 can return only global-attention layers "
                    f"{list(available_layers)}, got {invalid_layers}."
                )
            self.sam3_feature_indices = tuple(
                available_layers.index(layer)
                for layer in self.sam3_intermediate_layers
            )
        else:
            self.sam3_intermediate_layers = (num_blocks - 1,)
            self.sam3_feature_indices = (0,)
        self.num_feature_layers = len(self.sam3_feature_indices)
        self.feature_source = feature_source

    def forward(self,batch):
        img = getattr(batch, "sam3_img_batch", batch.img_batch).to(device=self.device, dtype=torch.float32)
        vision_backbone = self.model.backbone.vision_backbone
        vision_backbone.trunk.return_interm_layers = self.return_interm_layers
        vision_backbone.eval()
        with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
        ):
            if self.feature_source == "backbone":
                return extract_distillation_features(
                    vision_backbone, img, self.sam3_feature_indices
                )
            if self.feature_source == "detector":
                return extract_sam3_detector_features(vision_backbone, img)
            if self.feature_source == "both":
                return extract_sam3_backbone_and_neck_features(
                    vision_backbone, img, self.sam3_feature_indices
                )
            raise ValueError(
                "feature_source must be 'backbone', 'detector', or 'both', got "
                f"{self.feature_source!r}."
            )

class DINO_V2_feature_extractor(torch.nn.Module):
    def __init__(self,
        checkpoint_path: str | nn.Module,
        device: torch.device,
        cfg: Optional[Dict[str, Any]],
    ):
        super(DINO_V2_feature_extractor, self).__init__()
        self.device = device
        self.checkpoint = checkpoint_path
        self.model = self.load_model(
            weights_path=self.checkpoint,
            device=self.device,
        )
        len_blocks = len(self.model.blocks)
        return_interm_layers = bool(
            (cfg or {}).get("train", {}).get("return_interm_layers", False)
        )
        if return_interm_layers:
            self.dino_v2_intermediate_layers = tuple((cfg or {}).get(
                "DINO_V2", {}
            ).get("dino_v2_intermediate_layers", (len_blocks - 1,)))
        else:
            self.dino_v2_intermediate_layers = (len_blocks - 1,)
        self.num_feature_layers = len(self.dino_v2_intermediate_layers)

    #加载预训练模型
    def load_model(self,weights_path: str, device: torch.device):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Cannot find weights file: {weights_path}")

        model = dinov2_vitl14_reg(pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        replace_dinov2_memory_efficient_attention(model)
        model.eval()
        model.to(device)
        return model

    @torch.inference_mode()
    def _infer_legacy_multiscale(self,model, image_tensor: torch.Tensor, device: torch.device):
        image_tensor = image_tensor.to(device=device, dtype=torch.float32)
        return model.get_intermediate_layers(
            image_tensor,
            n=[5, 11, 17, 23],  # 返回的中间层
            reshape=True,
            return_class_token=True,
        )

    @torch.inference_mode()
    def infer(self, model, image_tensor: torch.Tensor, device: torch.device):
        image_tensor = image_tensor.to(device=device, dtype=torch.float32)
        return model.get_intermediate_layers(
            image_tensor,
            n=select_layer_indices(self.dino_v2_intermediate_layers, len(model.blocks)),
            reshape=True,
            return_class_token=True,
        )

    def forward(self,batch):
        img = getattr(batch, "dino_v2_img_batch", batch.img_batch)
        out = self.infer(
            model=self.model,
            image_tensor=img,
            device=self.device, )

        features = []
        cls_tokens = []
        for feature, cls_token in out:
            features.append(feature)
            cls_tokens.append(cls_token)

        return features,cls_tokens

class CLIP_feature_extractor(torch.nn.Module):
    def __init__(self,
        checkpoint_path: str | nn.Module,
        device: torch.device,
        cfg: Optional[Dict[str, Any]],
                 ):
        super(CLIP_feature_extractor, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.cfg = cfg or {}
        self.input_resolution = self.cfg.get("CLIP", {}).get("resolution", 224)
        self.return_interm_layers = bool(
            self.cfg.get("train", {}).get("return_interm_layers", False)
        )
        self.model = self.build_clip_model(
            model_path=self.checkpoint_path,
            device=self.device,
            input_resolution=self.input_resolution,
        )
        len_blocks = len(self.model.visual.transformer.resblocks)
        if self.return_interm_layers:
            self.out_indices = tuple(self.cfg.get("CLIP", {}).get(
                "clip_intermediate_layers", (len_blocks - 1,)
            ))
        else:
            self.out_indices = (len_blocks - 1,)
        self.num_feature_layers = len(self.out_indices)

    def build_clip_model(
            self,
            model_path: str,
            device: torch.device,
            input_resolution: int | None = None,
    ):
        """Build CLIP and load the local pretrained ViT-L/14 weights."""

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Pretrained CLIP weights not found: {model_path}")

        model, preprocess = clip.load(model_path, device=device, jit=False)
        if input_resolution is not None:
            self.resize_vit_positional_embedding(model, input_resolution)

        model.eval()
        return model

    def resize_vit_positional_embedding(self,model: torch.nn.Module, input_resolution: int) -> None:
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

    @staticmethod
    @torch.no_grad()
    def extract_vit_xy(
            model: torch.nn.Module,
            image: torch.Tensor,
            out_indices: Sequence[int],
    ):
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
        return multi_layer_features, multi_layer_cls

    def forward(self,batch):
        img = getattr(batch, "clip_img_batch", batch.img_batch)
        out_indices = select_layer_indices(
            self.out_indices, len(self.model.visual.transformer.resblocks)
        )
        features , cls = self.extract_vit_xy(self.model, img, out_indices)
        features = [x.float() for x in features]
        cls = [x.float() for x in cls]
        return features, cls

class Data:
    def __init__(self,img):
        self.img_batch = img


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_feature_shapes(name: str, features, cls_tokens=None) -> None:
    print(f"{name} features:")
    for index, feature in enumerate(features):
        feature = getattr(feature, "tensors", feature)
        print(f"  feature_{index}: shape={tuple(feature.shape)}, dtype={feature.dtype}, device={feature.device}")

    if cls_tokens is not None:
        print(f"{name} cls tokens:")
        for index, cls_token in enumerate(cls_tokens):
            print(f"  cls_{index}: shape={tuple(cls_token.shape)}, dtype={cls_token.dtype}, device={cls_token.device}")


if __name__ == "__main__":
    config_path = r"E:\pycharm\Vision Distillation\Config\KD_config.yaml"
    cfg = load_config(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolution = cfg.get("dataset", {}).get("resolution", 1008)

    img_root = r"E:\my_data\rf100-vl\apex-videogame\train\7--2-_png_jpg.rf.72266a0209766e9f48a3d2d2688ea594.jpg"
    img = Image.open(img_root).convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resolution, resolution)),
        torchvision.transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0).to(device=device, dtype=torch.float32)
    data = BatchedDatapoint(
        img_batch=img,
        find_text_batch=[],
        find_inputs=[],
        find_targets=[],
        find_metadatas=[],
    )
    print(f"input: shape={tuple(img.shape)}, dtype={img.dtype}, device={img.device}")
    print(type(cfg["Sam3"]["checkpoint_path"]))
    sam3 = Sam3_feature_extractor(
        checkpoint_path=cfg["Sam3"]["checkpoint_path"],
        device=device,
        cfg=cfg,
    )
    sam3_features = sam3(data)
    print_feature_shapes("SAM3", sam3_features)

    dino_v2 = DINO_V2_feature_extractor(
        checkpoint_path=cfg["DINO_V2"]["checkpoint"],
        device=device,
        cfg=cfg,
    )
    dino_features, dino_cls = dino_v2(data)
    print_feature_shapes("DINO_V2", dino_features, dino_cls)

    clip_model = CLIP_feature_extractor(
        checkpoint_path=cfg["CLIP"]["checkpoint_path"],
        device=device,
        cfg=cfg,
    )
    clip_features, clip_cls = clip_model(data)
    print_feature_shapes("CLIP", clip_features, clip_cls)
