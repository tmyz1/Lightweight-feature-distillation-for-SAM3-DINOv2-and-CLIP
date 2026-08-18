import sys

import pkg_resources
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sam3.model.data_misc import BatchedDatapoint
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_position_encoding,
    _create_sam3_model,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_text_encoder,
    _create_vl_backbone,
    _load_checkpoint,
    _setup_device_and_mode,
    build_tracker,
)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, rd_ratio: float = 0.25):
        super().__init__()
        rd_channels = max(8, int(channels * rd_ratio + 4) // 8 * 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.fc2(self.act(self.fc1(scale)))
        return x * self.gate(scale)


class Conv2dBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ):
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=False,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, drop: float = 0.0):
        super().__init__()
        self.m = module
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop > 0:
            keep = torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop)
            return x + self.m(x) * keep.div(1 - self.drop).detach()
        return x + self.m(x)


class RepVGGDW(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2dBN(channels, channels, 3, 1, 1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x) + self.conv1(x) + x)


class RepViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ):
        super().__init__()
        assert stride in (1, 2)
        self.identity = stride == 1 and in_channels == out_channels
        act = nn.GELU

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2dBN(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=in_channels,
                ),
                SqueezeExcite(in_channels, 0.25) if use_se else nn.Identity(),
                Conv2dBN(in_channels, out_channels, kernel_size=1),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2dBN(out_channels, 2 * out_channels, kernel_size=1),
                    act(),
                    Conv2dBN(
                        2 * out_channels,
                        out_channels,
                        kernel_size=1,
                        bn_weight_init=0,
                    ),
                )
            )
        else:
            if not self.identity:
                raise ValueError("RepViT stride-1 blocks require identity shape.")
            self.token_mixer = nn.Sequential(
                RepVGGDW(in_channels),
                SqueezeExcite(in_channels, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2dBN(in_channels, hidden_channels, kernel_size=1),
                    act(),
                    Conv2dBN(
                        hidden_channels,
                        out_channels,
                        kernel_size=1,
                        bn_weight_init=0,
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


def _make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


REPVIT_M1_1_CFGS = [
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 128, 0, 0, 2],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 256, 0, 1, 2],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 512, 0, 1, 2],
    [3, 2, 512, 1, 1, 1],
    [3, 2, 512, 0, 1, 1],
]

REPVIT_M2_3_CFGS = [
    [3, 2, 80, 1, 0, 1],
    [3, 2, 80, 0, 0, 1],
    [3, 2, 80, 1, 0, 1],
    [3, 2, 80, 0, 0, 1],
    [3, 2, 80, 1, 0, 1],
    [3, 2, 80, 0, 0, 1],
    [3, 2, 80, 0, 0, 1],
    [3, 2, 160, 0, 0, 2],
    [3, 2, 160, 1, 0, 1],
    [3, 2, 160, 0, 0, 1],
    [3, 2, 160, 1, 0, 1],
    [3, 2, 160, 0, 0, 1],
    [3, 2, 160, 1, 0, 1],
    [3, 2, 160, 0, 0, 1],
    [3, 2, 160, 0, 0, 1],
    [3, 2, 320, 0, 1, 2],
    *[[3, 2, 320, 1 if i % 2 == 0 else 0, 1, 1] for i in range(34)],
    [3, 2, 320, 0, 1, 1],
    [3, 2, 640, 0, 1, 2],
    [3, 2, 640, 1, 1, 1],
    [3, 2, 640, 0, 1, 1],
]

REPVIT_ARCH_CFGS = {
    "m1_1": REPVIT_M1_1_CFGS,
    "m2_3": REPVIT_M2_3_CFGS,
}
"""
RepVit backbone
"""
class RepViTBackbone(nn.Module):
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        strict_pretrained: bool = False,
        arch: str = "m1_1",
    ):
        super().__init__()
        if arch not in REPVIT_ARCH_CFGS:
            raise ValueError(
                f"Unsupported RepViT arch {arch!r}; available: {sorted(REPVIT_ARCH_CFGS)}"
            )
        self.arch = arch
        self.cfgs = REPVIT_ARCH_CFGS[arch]
        input_channel = self.cfgs[0][2]
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2dBN(3, input_channel // 2, 3, 2, 1),
                    nn.GELU(),
                    Conv2dBN(input_channel // 2, input_channel, 3, 2, 1),
                )
            ]
        )
        self.stage_idx: List[int] = []
        self.stage_channels: List[int] = []
        previous_c = input_channel
        for index, (kernel, expansion, channels, use_se, use_hs, stride) in enumerate(
            self.cfgs
        ):
            out_channels = _make_divisible(channels, 8)
            hidden_channels = _make_divisible(input_channel * expansion, 8)
            if channels != previous_c:
                self.stage_idx.append(index - 1)
                self.stage_channels.append(_make_divisible(previous_c, 8))
                previous_c = channels
            self.features.append(
                RepViTBlock(
                    input_channel,
                    hidden_channels,
                    out_channels,
                    kernel,
                    stride,
                    bool(use_se),
                    bool(use_hs),
                )
            )
            input_channel = out_channels
        self.stage_idx.append(index)
        self.stage_channels.append(input_channel)
        self.channel_list = [self.stage_channels[-1]]
        self.last_two_channels = self.stage_channels[-2:]

        if checkpoint_path:
            self.load_pretrained(checkpoint_path, strict=strict_pretrained)

    @staticmethod
    def _as_tensor(x):
        return getattr(x, "tensors", x)

    def _normalize_state_dict(self, checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("RepViT checkpoint must be a state dict or contain model/state_dict.")

        state_dict = {}
        for key, value in checkpoint.items():
            new_key = key
            for prefix in ("module.", "backbone.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            if new_key.startswith("head.") or new_key.startswith("classifier."):
                continue
            state_dict[new_key] = value
        return state_dict

    def load_pretrained(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._normalize_state_dict(checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        print(
            f"loaded RepViT checkpoint {checkpoint_path} with "
            f"{len(missing)} missing keys and {len(unexpected)} unexpected keys."
        )
        if missing:
            print(f"missing examples: {missing[:5]}")
        if unexpected:
            print(f"unexpected examples: {unexpected[:5]}")

    def get_stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self._as_tensor(x)
        outputs = []
        x = self.features[0](x)
        stage_counter = 0
        for index, block in enumerate(self.features[1:]):
            x = block(x)
            if index in self.stage_idx:
                outputs.append(x)
                stage_counter += 1
        if len(outputs) < 2:
            raise RuntimeError("RepViT must expose at least two stage features.")
        return outputs

    def get_last_two_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.get_stage_features(x)
        return outputs[-2], outputs[-1]

    def forward(self, tensor_list) -> List[torch.Tensor]:
        return [self.get_stage_features(tensor_list)[-1]]


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

"""
RepVit neck to Sam3
"""
class Sam3RepViTFPNDetNeck(nn.Module):
    def __init__(
        self,
        trunk: RepViTBackbone,
        position_encoding: nn.Module,
        d_model: int = 256,
        vit_feature_size: Optional[Tuple[int, int]] = (72, 72),
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.scale_factors = tuple(scale_factors)
        self.vit_feature_size = tuple(vit_feature_size) if vit_feature_size is not None else None
        stage2_channels, stage3_channels = trunk.last_two_channels
        self.channel_list = [d_model]

        self.fuse_stage2 = nn.Conv2d(stage2_channels, d_model, kernel_size=1, bias=False)
        self.fuse_stage3 = nn.Sequential(
            nn.Conv2d(stage3_channels, d_model, kernel_size=1, bias=False),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.neck = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            LayerNorm2d(d_model),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(d_model),
        )
        self.output_convs = self._make_output_convs(d_model)

        self.sam2_fuse_stage2 = None
        self.sam2_fuse_stage3 = None
        self.sam2_neck = None
        self.sam2_output_convs = None
        if add_sam2_neck:
            self.sam2_fuse_stage2 = deepcopy(self.fuse_stage2)
            self.sam2_fuse_stage3 = deepcopy(self.fuse_stage3)
            self.sam2_neck = deepcopy(self.neck)
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
                    nn.ConvTranspose2d(d_model // 2, d_model // 4, kernel_size=2, stride=2),
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

            current.add_module("conv_1x1", nn.Conv2d(out_dim, d_model, kernel_size=1, bias=True))
            current.add_module("conv_3x3", nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True))
            output_convs.append(current)
        return output_convs

    """
    参考论文 EdgeSam，将最后两层进行融合，之后送入neck中与sam3的neck部分进行对齐
    """
    def _fuse_repvit_features(
        self,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        fuse_stage2: nn.Module,
        fuse_stage3: nn.Module,
        neck: nn.Module,
    ) -> torch.Tensor:
        stage2 = fuse_stage2(stage2)
        stage3 = fuse_stage3(stage3)
        if stage3.shape[-2:] != stage2.shape[-2:]:
            stage3 = F.interpolate(
                stage3,
                size=stage2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = stage2 + stage3
        x = neck(x)
        if self.vit_feature_size is not None and x.shape[-2:] != self.vit_feature_size:
            x = F.interpolate(
                x,
                size=self.vit_feature_size,
                mode="bilinear",
                align_corners=False,
            )
        return x

    def _build_simple_fpn_outputs(
        self,
        vit_like: torch.Tensor,
        output_convs: nn.ModuleList,
    ) -> List[torch.Tensor]:
        return [output_conv(vit_like) for output_conv in output_convs]

    def get_repvit_embedding(self, img: torch.Tensor) -> torch.Tensor:
        stage2, stage3 = self.trunk.get_last_two_features(img)
        return self._fuse_repvit_features(
            stage2,
            stage3,
            self.fuse_stage2,
            self.fuse_stage3,
            self.neck,
        )

    def forward(
        self,
        tensor_list,
    ):
        stage2, stage3 = self.trunk.get_last_two_features(tensor_list)
        sam3_vit_like = self._fuse_repvit_features(
            stage2,
            stage3,
            self.fuse_stage2,
            self.fuse_stage3,
            self.neck,
        )
        sam3_out = self._build_simple_fpn_outputs(sam3_vit_like, self.output_convs)
        sam3_pos = [self.position_encoding(feature).to(feature.dtype) for feature in sam3_out]

        sam2_out, sam2_pos = None, None
        if (
            self.sam2_fuse_stage2 is not None
            and self.sam2_fuse_stage3 is not None
            and self.sam2_neck is not None
            and self.sam2_output_convs is not None
        ):
            sam2_vit_like = self._fuse_repvit_features(
                stage2,
                stage3,
                self.sam2_fuse_stage2,
                self.sam2_fuse_stage3,
                self.sam2_neck,
            )
            sam2_out = self._build_simple_fpn_outputs(sam2_vit_like, self.sam2_output_convs)
            sam2_pos = [self.position_encoding(feature).to(feature.dtype) for feature in sam2_out]

        return sam3_out, sam3_pos, sam2_out, sam2_pos

"""
以RepVit作为backbone，来创建整体结构
"""
def _create_repvit_backbone(
    checkpoint_path: str = r"E:\reproduce\weights\RepVit\repvit_m1_1_distill_450e.pth",
    strict_pretrained: bool = False,
    arch: str = "m1_1",
) -> RepViTBackbone:
    return RepViTBackbone(
        checkpoint_path=checkpoint_path,
        strict_pretrained=strict_pretrained,
        arch=arch,
    )


def _create_repvit_neck(
    position_encoding: nn.Module,
    repvit_backbone: RepViTBackbone,
    enable_inst_interactivity: bool = False,
    neck_cfg: Optional[Dict[str, Any]] = None,
) -> Sam3RepViTFPNDetNeck:
    cfg = {
        "d_model": 256,
        "vit_feature_size": (72, 72),
        "scale_factors": (4.0, 2.0, 1.0, 0.5),
    }
    cfg.update(neck_cfg or {})
    if cfg.get("vit_feature_size") is not None:
        cfg["vit_feature_size"] = tuple(cfg["vit_feature_size"])
    cfg["scale_factors"] = tuple(cfg["scale_factors"])
    return Sam3RepViTFPNDetNeck(
        trunk=repvit_backbone,
        position_encoding=position_encoding,
        add_sam2_neck=enable_inst_interactivity,
        **cfg,
    )


def _create_repvit_vision_backbone(
    enable_inst_interactivity: bool = True,
    repvit_checkpoint_path: str = r"E:\reproduce\weights\RepVit\repvit_m1_1_distill_450e.pth",
    repvit_neck_cfg: Optional[Dict[str, Any]] = None,
    position_encoding_resolution: int = 1008,
    strict_pretrained: bool = False,
    repvit_arch: str = "m1_1",
) -> Sam3RepViTFPNDetNeck:
    position_encoding = _create_position_encoding(
        precompute_resolution=position_encoding_resolution
    )
    repvit_backbone = _create_repvit_backbone(
        checkpoint_path=repvit_checkpoint_path,
        strict_pretrained=strict_pretrained,
        arch=repvit_arch,
    )
    return _create_repvit_neck(
        position_encoding=position_encoding,
        repvit_backbone=repvit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
        neck_cfg=repvit_neck_cfg,
    )


def build_RepVit_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=r"E:\reproduce\weights\sam3\sam3.pt",
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    repvit_checkpoint_path: str = r"E:\reproduce\weights\RepVit\repvit_m1_1_distill_450e.pth",
    repvit_neck_cfg: Optional[Dict[str, Any]] = None,
    repvit_arch: str = "m1_1",
    cfg: Optional[Dict[str, Any]] = None,
):
    student_cfg = (cfg or {}).get("Student", {})
    repvit_checkpoint_path = student_cfg.get(
        "repvit_checkpoint_path", repvit_checkpoint_path
    )
    repvit_neck_cfg = student_cfg.get("repvit_neck", repvit_neck_cfg)
    position_encoding_resolution = int(
        student_cfg.get("position_encoding_resolution", 1008)
    )
    strict_pretrained = bool(student_cfg.get("strict_pretrained", False))
    repvit_arch = student_cfg.get("repvit_arch", repvit_arch)

    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )
    compile_mode = "default" if compile else None
    vision_encoder = _create_repvit_vision_backbone(
        enable_inst_interactivity=enable_inst_interactivity,
        repvit_checkpoint_path=repvit_checkpoint_path,
        repvit_neck_cfg=repvit_neck_cfg,
        position_encoding_resolution=position_encoding_resolution,
        strict_pretrained=strict_pretrained,
        repvit_arch=repvit_arch,
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
    return _setup_device_and_mode(model, device, eval_mode)

"""
特征提取器
"""
class RepVit_feature_extractor(nn.Module):
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        student_cfg = self.cfg.get("Student", {})
        self.model = build_RepVit_image_model(
            bpe_path=self.cfg.get("Sam3", {}).get("bpe_path"),
            checkpoint_path=self.cfg.get("Sam3", {}).get("checkpoint_path"),
            device=str(self.device),
            eval_mode=False,
            enable_segmentation=bool(
                self.cfg.get("Sam3", {}).get("enable_segmentation", False)
            ),
            enable_inst_interactivity=False,
            repvit_checkpoint_path=student_cfg.get("repvit_checkpoint_path"),
            repvit_neck_cfg=student_cfg.get("repvit_neck"),
            cfg=self.cfg,
        )
        self.model.to(self.device)
        self.num_feature_layers = 1

    def extract_repvit_feature_and_neck(
        self,
        vision_backbone: Sam3RepViTFPNDetNeck,
        img: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        required = ("get_repvit_embedding", "_build_simple_fpn_outputs", "output_convs")
        if not all(hasattr(vision_backbone, name) for name in required):
            raise TypeError(
                "Expected Sam3RepViTFPNDetNeck, got " f"{type(vision_backbone)}."
            )
        repvit_embedding = vision_backbone.get_repvit_embedding(img)
        neck_features = vision_backbone._build_simple_fpn_outputs(
            repvit_embedding,
            vision_backbone.output_convs,
        )
        return [repvit_embedding], neck_features

    def extract_repvit_features(
        self,
        vision_backbone: Sam3RepViTFPNDetNeck,
        img: torch.Tensor,
    ) -> List[torch.Tensor]:
        features, _ = self.extract_repvit_feature_and_neck(vision_backbone, img)
        return features

    def forward(self, batch: BatchedDatapoint):
        img = getattr(batch, "student_img_batch", batch.img_batch).to(
            device=self.device,
            dtype=torch.float32,
        )
        return self.extract_repvit_feature_and_neck(
            self.model.backbone.vision_backbone,
            img,
        )





