from copy import deepcopy
from typing import Optional, Dict, Any, Sequence, Tuple, List

import pkg_resources
import torch
from torch import nn
from torch.nn import functional as F
from sam3.model.Swin_Transformer import SwinTransformer
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model_builder import _as_tuple, _resolve_nn_layer, _create_position_encoding, _create_text_encoder, \
    _create_vl_backbone, _create_sam3_transformer, _create_dot_product_scoring, _create_segmentation_head, \
    _create_geometry_encoder, build_tracker, _create_sam3_model, _load_checkpoint, _setup_device_and_mode

"""
backbone部分
"""
def _create_swin_backbone(
    compile_mode=None,
    use_fa3=False,
    use_rope_real=False,
    swin_cfg: Optional[Dict[str, Any]] = None,
):
    cfg = {
        "pretrain_img_size": 384,
        "in_chans": 3,
        "embed_dim": 192,
        "patch_size": 4,
        "window_size": 12,
        "mlp_ratio": 4.0,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
        "strides": (4, 2, 2, 2),
        "out_indices": (0, 1, 2, 3),
        "qkv_bias": True,
        "qk_scale": None,
        "patch_norm": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "use_abs_pos_embed": False,
        "norm_layer": nn.LayerNorm,
        "act_layer": nn.GELU,
        "use_act_checkpoint": False,
        "frozen_stages": -1,
        "pretrained": r"E:\reproduce\weights\SwinTransformer\swin_large_384.pth",
    }
    cfg.update(swin_cfg or {})
    for key in ("depths", "num_heads", "strides", "out_indices", "pretrain_img_size"):
        if key in cfg:
            cfg[key] = _as_tuple(cfg[key])
    cfg["norm_layer"] = _resolve_nn_layer(cfg["norm_layer"])
    cfg["act_layer"] = _resolve_nn_layer(cfg["act_layer"])
    return SwinTransformer(**cfg)

"""
neck部分
"""
class Sam3SwinFPNDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int = 256,
        in_channels: Optional[Sequence[int]] = None,
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        target_sizes: Optional[Sequence[Tuple[int, int]]] = None,
        add_sam2_neck: bool = False,
    ):
        """
        FPN/PAN neck for Swin-style multi-stage backbones.

        Swin returns a four-level pyramid. This adapter first fuses those levels
        top-down, then fuses them bottom-up, converts the highest-resolution
        fused map to a ViT-like 72x72 feature, and finally emits the same four
        feature-map sizes expected from SAM3's ViT SimpleFPN neck.
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.scale_factors = tuple(scale_factors)
        self.vit_feature_size = (72, 72)

        if in_channels is None:
            in_channels = getattr(trunk, "channel_list", None)
        if in_channels is None:
            raise ValueError(
                "in_channels must be provided when trunk has no channel_list attribute"
            )

        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=channel,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=True,
                )
                for channel in in_channels
            ]
        )

        self.top_down_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                for _ in in_channels
            ]
        )

        self.bottom_up_downsample_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                )
                for _ in range(len(in_channels) - 1)
            ]
        )
        self.bottom_up_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                for _ in in_channels
            ]
        )
        self.vit_feature_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True),
        )
        self.output_convs = nn.ModuleList()
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
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=True,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
            )
            self.output_convs.append(current)

        self.sam2_lateral_convs = None
        self.sam2_top_down_convs = None
        self.sam2_bottom_up_downsample_convs = None
        self.sam2_bottom_up_convs = None
        self.sam2_vit_feature_proj = None
        self.sam2_output_convs = None
        if add_sam2_neck:
            self.sam2_lateral_convs = deepcopy(self.lateral_convs)
            self.sam2_top_down_convs = deepcopy(self.top_down_convs)
            self.sam2_bottom_up_downsample_convs = deepcopy(
                self.bottom_up_downsample_convs
            )
            self.sam2_bottom_up_convs = deepcopy(self.bottom_up_convs)
            self.sam2_vit_feature_proj = deepcopy(self.vit_feature_proj)
            self.sam2_output_convs = deepcopy(self.output_convs)

    @staticmethod
    def _as_tensor(x):
        return getattr(x, "tensors", x)

    def _fuse_swin_features(
        self,
        xs: Sequence[torch.Tensor],
        lateral_convs: nn.ModuleList,
        top_down_convs: nn.ModuleList,
        bottom_up_downsample_convs: nn.ModuleList,
        bottom_up_convs: nn.ModuleList,
        vit_feature_proj: nn.Module,
    ) -> torch.Tensor:
        laterals = [
            lateral_conv(self._as_tensor(x))
            for lateral_conv, x in zip(lateral_convs, xs)
        ]

        # Top-down FPN: deep semantic features enrich shallower high-res maps.
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        top_down = [
            conv(feature) for conv, feature in zip(top_down_convs, laterals)
        ]

        # Bottom-up PAN: refined high-res detail is propagated back down.
        pan = [top_down[0]]
        for i in range(1, len(top_down)):
            down = bottom_up_downsample_convs[i - 1](pan[i - 1])
            if down.shape[-2:] != top_down[i].shape[-2:]:
                down = F.interpolate(
                    down,
                    size=top_down[i].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            pan.append(top_down[i] + down)
        pan = [conv(feature) for conv, feature in zip(bottom_up_convs, pan)]

        vit_like = pan[0]
        if vit_like.shape[-2:] != self.vit_feature_size:
            vit_like = F.interpolate(
                vit_like,
                size=self.vit_feature_size,
                mode="bilinear",
                align_corners=False,
            )
        return vit_feature_proj(vit_like)

    def _build_simple_fpn_outputs(
        self,
        vit_like: torch.Tensor,
        output_convs: nn.ModuleList,
    ) -> List[torch.Tensor]:
        outs = []
        for output_conv in output_convs:
            outs.append(output_conv(vit_like))
        return outs

    def forward(
        self, tensor_list: List[torch.Tensor]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        xs = self.trunk(tensor_list)
        if len(xs) != len(self.lateral_convs):
            raise ValueError(
                "Swin trunk must return one feature map per lateral conv, got "
                f"{len(xs)} features for {len(self.lateral_convs)} lateral convs"
            )

        sam3_vit_like = self._fuse_swin_features(
            xs,
            self.lateral_convs,
            self.top_down_convs,
            self.bottom_up_downsample_convs,
            self.bottom_up_convs,
            self.vit_feature_proj,
        )
        sam3_out = self._build_simple_fpn_outputs(
            sam3_vit_like, self.output_convs
        )
        sam3_pos = [
            self.position_encoding(feature).to(feature.dtype) for feature in sam3_out
        ]

        sam2_out, sam2_pos = None, None
        if (
            self.sam2_lateral_convs is not None
            and self.sam2_top_down_convs is not None
            and self.sam2_bottom_up_downsample_convs is not None
            and self.sam2_bottom_up_convs is not None
            and self.sam2_vit_feature_proj is not None
            and self.sam2_output_convs is not None
        ):
            sam2_vit_like = self._fuse_swin_features(
                xs,
                self.sam2_lateral_convs,
                self.sam2_top_down_convs,
                self.sam2_bottom_up_downsample_convs,
                self.sam2_bottom_up_convs,
                self.sam2_vit_feature_proj,
            )
            sam2_out = self._build_simple_fpn_outputs(
                sam2_vit_like, self.sam2_output_convs
            )
            sam2_pos = [
                self.position_encoding(feature).to(feature.dtype)
                for feature in sam2_out
            ]

        return sam3_out, sam3_pos, sam2_out, sam2_pos

def _create_swin_neck(
    position_encoding,
    Swin_backbone,
    enable_inst_interactivity=False,
    neck_cfg: Optional[Dict[str, Any]] = None,
):
    cfg = {
        "d_model": 256,
        "scale_factors": (4.0, 2.0, 1.0, 0.5),
        "in_channels": None,
    }
    cfg.update(neck_cfg or {})
    cfg.pop("target_sizes", None)
    cfg["scale_factors"] = _as_tuple(cfg["scale_factors"])
    if cfg.get("in_channels") is not None:
        cfg["in_channels"] = _as_tuple(cfg["in_channels"])
    return Sam3SwinFPNDetNeck(
        position_encoding=position_encoding,
        trunk=Swin_backbone,
        add_sam2_neck=enable_inst_interactivity,
        **cfg,
    )

"""整体模型创建部分"""
def _create_Swin_vision_backbone(
    compile_mode=None,
    enable_inst_interactivity=True,
    swin_backbone_cfg: Optional[Dict[str, Any]] = None,
    swin_neck_cfg: Optional[Dict[str, Any]] = None,
    position_encoding_resolution: int = 1008,
) -> Sam3SwinFPNDetNeck:
    # Position encoding
    position_encoding = _create_position_encoding(
        precompute_resolution=position_encoding_resolution
    )
    # 主要改动部分
    Swin_backbone: SwinTransformer = _create_swin_backbone(
        compile_mode=compile_mode,
        swin_cfg=swin_backbone_cfg,
    )
    Swin_neck: Sam3SwinFPNDetNeck = _create_swin_neck(
        position_encoding,
        Swin_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
        neck_cfg=swin_neck_cfg,
    )
    # Swin Neck
    return Swin_neck

def build_swin_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    swin_backbone_cfg: Optional[Dict[str, Any]] = None,
    swin_neck_cfg: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
):
    student_cfg = (cfg or {}).get("Student", {})
    swin_backbone_cfg = student_cfg.get("swin_backbone", swin_backbone_cfg)
    swin_neck_cfg = student_cfg.get("swin_neck", swin_neck_cfg)
    position_encoding_resolution = int(
        student_cfg.get("position_encoding_resolution", 1008)
    )
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )
    compile_mode = "default" if compile else None
    #主要修改部分
    vision_encoder = _create_Swin_vision_backbone(
        compile_mode=compile_mode,
        enable_inst_interactivity=enable_inst_interactivity,
        swin_backbone_cfg=swin_backbone_cfg,
        swin_neck_cfg=swin_neck_cfg,
        position_encoding_resolution=position_encoding_resolution,
    )
    text_encoder = _create_text_encoder(bpe_path)

    #主要修改部分
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
    # Create the Swin_SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    # 如果权重文件只有一部分适配，skip_prefixes表示需要跳过的地方
    if checkpoint_path is not None:
        _load_checkpoint(
            model,
            checkpoint_path,
            skip_prefixes=("backbone.vision_backbone.",),
        )
    #权重文件完全适配
    # if checkpoint_path is not None:
    #     _load_checkpoint(
    #         model,
    #         checkpoint_path,
    #     )

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model

"""
特征提取部分
"""
class Swin_Sam3_feature_extractor(nn.Module):
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        self.model = build_swin_sam3_image_model(
            bpe_path=self.cfg.get("Sam3", {}).get("bpe_path"),
            checkpoint_path=self.cfg.get("Sam3", {}).get("checkpoint_path"),
            device=str(self.device),
            eval_mode=False,
            enable_segmentation=bool(self.cfg.get("Sam3", {}).get("enable_segmentation", False)),
            enable_inst_interactivity=False,
            swin_backbone_cfg=self.cfg.get("Student", {}).get("swin_backbone"),
            swin_neck_cfg=self.cfg.get("Student", {}).get("swin_neck"),
            cfg=self.cfg,
        )
        self.model.to(self.device)
        self.num_feature_layers = 1

    def select_feature_layers(self,
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

    def extract_backbone_and_neck_features(self,
            vision_backbone: torch.nn.Module,
            img: torch.Tensor,
    ):
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
            # 用于适配Swin-Sam3的neck部分
            neck_features = [vision_backbone._fuse_swin_features(
                backbone_features,
                vision_backbone.lateral_convs,
                vision_backbone.top_down_convs,
                vision_backbone.bottom_up_downsample_convs,
                vision_backbone.bottom_up_convs,
                vision_backbone.vit_feature_proj,
            )]
        elif hasattr(vision_backbone, "convs"):
            # 用于适配原本Sam3的neck部分
            x = backbone_features[-1]
            neck_features = [conv(x) for conv in vision_backbone.convs]
        else:
            raise TypeError(f"Unsupported vision_backbone type: {type(vision_backbone)}")

        return backbone_features, neck_features

    def extract_distillation_features(self,
            vision_backbone: torch.nn.Module,
            img: torch.Tensor,
            layer_indices: Optional[int | Sequence[int]] = None,
    ) -> list[torch.Tensor]:
        """Extract Swin's fused ViT-like map or selected ViT trunk feature maps."""
        backbone_features, swin_features = self.extract_backbone_and_neck_features(
            vision_backbone, img
        )
        if hasattr(vision_backbone, "_fuse_swin_features"):
            if layer_indices not in (None, 0, [0], (0,)):
                raise ValueError(
                    "Sam3SwinFPNDetNeck exposes one fused ViT-like distillation feature; "
                    "its only valid layer index is 0."
                )
            return swin_features
        return self.select_feature_layers(backbone_features, layer_indices)

    def forward(self, batch: BatchedDatapoint):
        img = getattr(batch, "student_img_batch", batch.img_batch).to(device=self.device, dtype=torch.float32)
        vision_backbone = self.model.backbone.vision_backbone
        features = self.extract_distillation_features(vision_backbone, img)
        if not hasattr(vision_backbone, "_build_simple_fpn_outputs"):
            raise TypeError(
                "Student vision backbone must expose _build_simple_fpn_outputs "
                "for neck distillation."
            )
        neck_features = vision_backbone._build_simple_fpn_outputs(
            features[0], vision_backbone.output_convs
        )
        return features, neck_features
