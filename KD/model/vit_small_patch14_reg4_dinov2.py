"""
https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m 预训练权重下载
"""
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict, Any

import pkg_resources
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

from dinov2.layers.attention import Attention, MemEffAttention
from dinov2.models.vision_transformer import vit_small as dinov2_vit_small
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model_builder import _create_position_encoding, _create_text_encoder, _create_vl_backbone, \
    _create_sam3_transformer, _create_dot_product_scoring, _create_segmentation_head, _create_geometry_encoder, \
    build_tracker, _create_sam3_model, _setup_device_and_mode, _load_checkpoint
def save_debug_batch(
    batch,
    output_dir: Path,
    step: int,
    batch_idx: int,
    max_images: int = 4,
):
    """Save original images and normalized model inputs for debugging."""

    vis_dir = output_dir / "debug_images"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 保存实际输入模型的图片
    images = batch.img_batch.detach().float().cpu()
    images = (images * 0.5 + 0.5).clamp(0, 1)
    images = images[:max_images]

    save_image(
        images,
        str(vis_dir / f"step_{step}_batch_{batch_idx}_input.png"),
        nrow=min(images.shape[0], 4),
    )

#预训练权重加载函数
def _load_vit_small_pretrained(model: nn.Module, checkpoint_path: str):
    state_dict = load_file(checkpoint_path, device="cpu")
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
        print(
            f"loaded {checkpoint_path} with {len(missing_keys)} missing keys and "
            f"{len(unexpected_keys)} unexpected keys."
        )
        if missing_keys:
            print(f"missing examples: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"unexpected examples: {unexpected_keys[:5]}")


def replace_vit_small_memory_efficient_attention(model: nn.Module) -> None:
    """Match training-time native attention for stable BF16 inference."""
    trunk = model.backbone.vision_backbone.trunk.trunk
    for block in trunk.blocks:
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

"""
创建backbone部分
"""

class DinoV2SmallBackbone(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        return_interm_layers: bool = False,
        intermediate_layers: Sequence[int] = (2, 5, 8, 11),
    ):
        super().__init__()
        self.trunk = trunk
        self.patch_size = trunk.patch_size
        num_blocks = len(trunk.blocks)
        if return_interm_layers:
            self.intermediate_layers = tuple(intermediate_layers)
        else:
            self.intermediate_layers = (num_blocks-1,)
        invalid_layers = [
            index
            for index in self.intermediate_layers
            if index < 0 or index >= num_blocks
        ]
        if invalid_layers:
            raise ValueError(
                f"Invalid ViT-Small intermediate layers {invalid_layers}; "
                f"valid range is 0 to {num_blocks - 1}."
            )
        if not self.intermediate_layers:
            raise ValueError("At least one ViT-Small intermediate layer is required.")
        self.channel_list = [trunk.embed_dim] * len(self.intermediate_layers)

    def get_intermediate_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        outputs = self.trunk.get_intermediate_layers(
            x,
            n=self.intermediate_layers,
            reshape=True,
            return_class_token=True,
        )
        feature_maps = [feature_map for feature_map, _ in outputs]
        cls_tokens = [cls_token for _, cls_token in outputs]
        return feature_maps, cls_tokens

    def forward(self, tensor_list):
        x = getattr(tensor_list, "tensors", tensor_list)
        feature_maps, _ = self.get_intermediate_features(x)
        return feature_maps

def _create_dinov2_vit_small_backbone(
    checkpoint_path: str = r"E:\reproduce\weights\DINO-V2\vit_small_patch14_reg4_dinov2.safetensors",
    return_interm_layers: bool = False,
    intermediate_layers: Sequence[int] = (2, 5, 8, 11),
    image_size: int = 518,
):
    vit_backbone = dinov2_vit_small(
        img_size=image_size,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    if checkpoint_path is not None:
        _load_vit_small_pretrained(vit_backbone, checkpoint_path)
    return DinoV2SmallBackbone(
        vit_backbone,
        intermediate_layers=intermediate_layers,
        return_interm_layers = return_interm_layers
    )

"""
创建Vit Small neck 部分
"""
class Sam3ViTSmallFPNDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int = 256,
        in_channels: int = 384,
        vit_feature_size: Tuple[int, int] = (72, 72),
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
        fuse_neck_features: bool = True,
    ):
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.vit_feature_size = tuple(vit_feature_size)
        self.scale_factors = tuple(scale_factors)
        self.fuse_neck_features = fuse_neck_features
        self.num_input_features = len(trunk.channel_list)
        self.layer_fusion_logits = nn.Parameter(
            torch.zeros(self.num_input_features, dtype=torch.float32)
        )

        self.feature_proj = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                d_model,
                kernel_size=2,
                stride=2,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True),
        )
        self.output_convs = self._make_output_convs(d_model)

        self.sam2_feature_proj = None
        self.sam2_output_convs = None
        if add_sam2_neck:
            self.sam2_feature_proj = deepcopy(self.feature_proj)
            self.sam2_output_convs = deepcopy(self.output_convs)

    def _make_output_convs(self, d_model: int) -> nn.ModuleList:
        output_convs = nn.ModuleList()
        for scale in self.scale_factors:
            current = nn.Sequential()
            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=2, stride=2),
                )
                current.add_module("gelu", nn.GELU())
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(
                        d_model // 2, d_model // 4, kernel_size=2, stride=2
                    ),
                )
                out_dim = d_model // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=2, stride=2),
                )
                out_dim = d_model // 2
            elif scale == 1.0:
                out_dim = d_model
            elif scale == 0.5:
                current.add_module("maxpool_2x2", nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = d_model
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(out_dim, d_model, kernel_size=1, bias=True),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True),
            )
            output_convs.append(current)
        return output_convs

    #与 sam3 的 neck 尺寸对齐
    def _project_to_vit_feature(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        x = proj(x)
        if x.shape[-2:] != self.vit_feature_size:
            x = F.interpolate(
                x,
                size=self.vit_feature_size,
                mode="bilinear",
                align_corners=False,
            )
        return x

    #与sam3相同的neck部分
    def _build_simple_fpn_outputs(
        self,
        vit_like: torch.Tensor,
        output_convs: nn.ModuleList,
    ) -> List[torch.Tensor]:
        return [output_conv(vit_like) for output_conv in output_convs]

    def _fuse_intermediate_features(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(features) != self.num_input_features:
            raise ValueError(
                f"Expected {self.num_input_features} ViT-Small feature maps, "
                f"got {len(features)}."
            )
        reference_shape = features[0].shape
        if any(feature.shape != reference_shape for feature in features[1:]):
            raise ValueError("ViT-Small intermediate feature maps must share a shape.")
        weights = self.layer_fusion_logits.softmax(dim=0)
        return sum(weight * feature for weight, feature in zip(weights, features))

    def forward(
        self, tensor_list: List[torch.Tensor]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        xs = self.trunk(tensor_list)
        #x = self._fuse_intermediate_features(xs) if len(xs) > 1 else xs[0]
        feature = (
            self._fuse_intermediate_features(xs)
            if self.fuse_neck_features and len(xs) > 1
            else xs[-1]
        )
        sam3_vit_like = self._project_to_vit_feature(feature, self.feature_proj)
        sam3_out = self._build_simple_fpn_outputs(sam3_vit_like, self.output_convs)
        sam3_pos = [
            self.position_encoding(feature).to(feature.dtype) for feature in sam3_out
        ]

        sam2_out, sam2_pos = None, None
        if self.sam2_feature_proj is not None and self.sam2_output_convs is not None:
            sam2_vit_like = self._project_to_vit_feature(
                feature, self.sam2_feature_proj
            )
            sam2_out = self._build_simple_fpn_outputs(
                sam2_vit_like, self.sam2_output_convs
            )
            sam2_pos = [
                self.position_encoding(feature).to(feature.dtype)
                for feature in sam2_out
            ]

        return sam3_out, sam3_pos, sam2_out, sam2_pos

def _create_vit_small_neck(
    position_encoding,
    vit_small_backbone,
    enable_inst_interactivity=False,
    fuse_neck_features=True,
):
    return Sam3ViTSmallFPNDetNeck(
        position_encoding=position_encoding,
        trunk=vit_small_backbone,
        d_model=256,
        in_channels=vit_small_backbone.channel_list[-1],
        vit_feature_size=(72, 72),
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck=enable_inst_interactivity,
        fuse_neck_features=fuse_neck_features,
    )

def _create_vit_small_backbone(
    enable_inst_interactivity=True,
    vit_small_checkpoint_path: str = r"E:\reproduce\weights\DINO-V2\vit_small_patch14_reg4_dinov2.safetensors",
    return_interm_layers: bool = False,
    vit_small_intermediate_layers: Sequence[int] = (2, 5, 8, 11),
    image_size: int = 518,
    position_encoding_resolution: int = 1008,
    fuse_neck_features: bool = True,
) -> Sam3ViTSmallFPNDetNeck:
    position_encoding = _create_position_encoding(
        precompute_resolution=position_encoding_resolution
    )
    vit_backbone = _create_dinov2_vit_small_backbone(
        checkpoint_path=vit_small_checkpoint_path,
        return_interm_layers = return_interm_layers,
        intermediate_layers=vit_small_intermediate_layers,
        image_size=image_size,
    )
    vit_neck = _create_vit_small_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
        fuse_neck_features=fuse_neck_features,
    )
    return vit_neck

"""
整体vit-small Sam3模型创建
"""
def build_vit_small_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=r"E:\reproduce\weights\sam3\sam3.pt",
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    vit_small_checkpoint_path: str = r"E:\reproduce\weights\DINO-V2\vit_small_patch14_reg4_dinov2.safetensors",
    return_interm_layers: bool = False,
    vit_small_intermediate_layers: Sequence[int] = (2, 5, 8, 11),
    cfg: Optional[Dict[str, Any]] = None,
):
    student_cfg = (cfg or {}).get("Student", {})
    vit_small_checkpoint_path = student_cfg.get(
        "vit_small_checkpoint_path", vit_small_checkpoint_path
    )
    # Keep the learned ViT position table at its pretrained grid. The trunk
    # interpolates it at runtime when kd_resolution uses a different size.
    pos_embed_image_size = int(
        student_cfg.get("vit_small_pos_embed_resolution", 518)
    )
    position_encoding_resolution = int(
        student_cfg.get("position_encoding_resolution", 1008)
    )
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )
    compile_mode = "default" if compile else None
    vision_encoder = _create_vit_small_backbone(
        enable_inst_interactivity=enable_inst_interactivity,
        vit_small_checkpoint_path=vit_small_checkpoint_path,
        vit_small_intermediate_layers=vit_small_intermediate_layers,
        return_interm_layers = return_interm_layers,
        image_size=pos_embed_image_size,
        position_encoding_resolution=position_encoding_resolution,
        fuse_neck_features=bool(student_cfg.get("fuse_neck_features", True)),
    )
    if compile_mode is not None:
        vision_encoder = torch.compile(vision_encoder)

    text_encoder = _create_text_encoder(bpe_path)
    backbone = _create_vl_backbone(vision_encoder, text_encoder)
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None

    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if checkpoint_path is not None:
        _load_checkpoint(
            model,
            checkpoint_path,
            skip_prefixes=("backbone.vision_backbone.",),
        )

    model = _setup_device_and_mode(model, device, eval_mode)
    return model

"""
特征提取器，获取模型的backbone和neck输出
"""
class Vit_Small_feature_extractor(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        student_cfg = self.cfg.get("Student", {})
        self.model = build_vit_small_image_model(
            bpe_path=self.cfg.get("Sam3", {}).get("bpe_path"),
            checkpoint_path=self.cfg.get("Sam3", {}).get("checkpoint_path"),
            device=str(self.device),
            eval_mode=False,
            enable_segmentation=bool(student_cfg.get("enable_segmentation", True)),
            enable_inst_interactivity=False,
            vit_small_checkpoint_path=student_cfg.get("vit_small_checkpoint_path"),
            return_interm_layers=cfg.get('train').get('return_interm_layers'),
            vit_small_intermediate_layers=student_cfg.get(
                "vit_small_intermediate_layers", (2, 5, 8, 11)
            ),
            cfg=self.cfg,
        )
        self.replace_vit_small_memory_efficient_attention(self.model)
        self.model.to(self.device)
        self.num_feature_layers = len(
            self.model.backbone.vision_backbone.trunk.channel_list
        )

    def replace_vit_small_memory_efficient_attention(self,model: torch.nn.Module) -> None:
        replace_vit_small_memory_efficient_attention(model)

    def extract_vit_small_feature_neck_and_cls(self,
            vision_backbone: torch.nn.Module,
            img: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        required_attributes = (
            "trunk",
            "feature_proj",
            "_project_to_vit_feature",
        )
        if not all(hasattr(vision_backbone, name) for name in required_attributes):
            raise TypeError(
                "Expected Sam3ViTSmallFPNDetNeck, got "
                f"{type(vision_backbone)}."
        )

        student_trunk = vision_backbone.trunk
        backbone_features, cls_tokens = student_trunk.get_intermediate_features(img)
        feature = (
            vision_backbone._fuse_intermediate_features(backbone_features)
            if vision_backbone.fuse_neck_features and len(backbone_features) > 1
            else backbone_features[-1]
        )
        vit_like = vision_backbone._project_to_vit_feature(
            feature, vision_backbone.feature_proj
        )
        neck_features = vision_backbone._build_simple_fpn_outputs(
            vit_like, vision_backbone.output_convs
        )
        return backbone_features, neck_features, cls_tokens

    def extract_vit_small_feature_and_neck(self,
            vision_backbone: torch.nn.Module,
            img: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features, neck_features, _ = self.extract_vit_small_feature_neck_and_cls(
            vision_backbone, img
        )
        return features, neck_features

    def extract_vit_small_features(self,
            vision_backbone: torch.nn.Module,
            img: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return the ViT-Small neck's single ViT-like 256-channel feature map."""
        features, _ = self.extract_vit_small_feature_and_neck(vision_backbone, img)
        return features

    def forward(self, batch: BatchedDatapoint):
        img = getattr(batch, "student_img_batch", batch.img_batch).to(
            device=self.device, dtype=torch.float32
        )
        return self.extract_vit_small_feature_neck_and_cls(
            self.model.backbone.vision_backbone, img
        )
