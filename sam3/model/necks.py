# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Necks are the interface between a vision backbone and the rest of the detection model"""
import time
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam3.model.data_misc import NestedTensor


def _sync_cuda_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)
        It supports a "dual neck" setting, where we have two identical necks (for SAM3 and SAM2), with different weights

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes sam2 neck is just a clone of the original neck
            self.sam2_convs = deepcopy(self.convs)

    def forward(
        self, tensor_list: List[torch.Tensor]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        xs = self.trunk(tensor_list)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []
        x = xs[-1]  # simpleFPN
        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)
        return sam3_out, sam3_pos, sam2_out, sam2_pos


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
    ):
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.vit_feature_size = tuple(vit_feature_size)
        self.scale_factors = tuple(scale_factors)

        # vit_small 对齐 sam3 的 backbone 输出的尺寸
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

    #单尺寸上采样与下采样变成多尺寸
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

    def _build_simple_fpn_outputs(
        self,
        vit_like: torch.Tensor,
        output_convs: nn.ModuleList,
    ) -> List[torch.Tensor]:
        return [output_conv(vit_like) for output_conv in output_convs]

    def forward(
        self, tensor_list: List[torch.Tensor]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        xs = self.trunk(tensor_list)
        x = xs[-1] if isinstance(xs, (list, tuple)) else xs

        sam3_vit_like = self._project_to_vit_feature(x, self.feature_proj)
        sam3_out = self._build_simple_fpn_outputs(sam3_vit_like, self.output_convs)
        sam3_pos = [
            self.position_encoding(feature).to(feature.dtype) for feature in sam3_out
        ]

        sam2_out, sam2_pos = None, None
        if self.sam2_feature_proj is not None and self.sam2_output_convs is not None:
            sam2_vit_like = self._project_to_vit_feature(x, self.sam2_feature_proj)
            sam2_out = self._build_simple_fpn_outputs(
                sam2_vit_like, self.sam2_output_convs
            )
            sam2_pos = [
                self.position_encoding(feature).to(feature.dtype)
                for feature in sam2_out
            ]

        return sam3_out, sam3_pos, sam2_out, sam2_pos


class Sam3TriViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        neck_norm=None,
        scale_factors=(4.0, 2.0, 1.0),
    ):
        """
        SimpleFPN neck with three heads (sam3, interactive, propagation).
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = neck_norm is None
        dim = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        # 三套neck，其中convs用于目标检测和分割，interactive 给交互式分割用， propagation_convs 给视频用
        self.interactive_convs = deepcopy(self.convs)
        self.propagation_convs = deepcopy(self.convs)

    def forward(
        self,
        tensor_list,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
        xs = self.trunk(tensor_list)
        sam3_out = []
        interactive_out = []
        propagation_out = []

        sam3_pos = []
        interactive_pos = []
        propagation_pos = []
        x = xs[-1]  # simpleFPN
        # OSS trunk returns plain tensors; onevision trunk returns NestedTensors.
        # Use getattr to handle both in a torch.compile-friendly way.
        x_data = getattr(x, "tensors", x)
        x_mask = getattr(x, "mask", None)
        for _, (conv, interactive_conv, propagation_conv) in enumerate(
            zip(self.convs, self.interactive_convs, self.propagation_convs)
        ):
            if need_sam3_out:
                sam3_conv_out = conv(x_data)
                sam3_x_out = NestedTensor(sam3_conv_out, x_mask)
                sam3_out.append(sam3_x_out)
                sam3_pos.append(
                    self.position_encoding(sam3_conv_out).to(sam3_conv_out.dtype)
                )

            if need_interactive_out:
                interactive_conv_out_t = interactive_conv(x_data)
                interactive_conv_out = NestedTensor(interactive_conv_out_t, x_mask)
                interactive_out.append(interactive_conv_out)
                interactive_pos.append(
                    self.position_encoding(interactive_conv_out_t).to(
                        interactive_conv_out_t.dtype
                    )
                )

            if need_propagation_out:
                propagation_conv_out = propagation_conv(x_data)
                propagation_x_out = NestedTensor(propagation_conv_out, x_mask)
                propagation_out.append(propagation_x_out)
                propagation_pos.append(
                    self.position_encoding(propagation_conv_out).to(
                        propagation_conv_out.dtype
                    )
                )
        return (
            sam3_out,
            sam3_pos,
            interactive_out,
            interactive_pos,
            propagation_out,
            propagation_pos,
        )
