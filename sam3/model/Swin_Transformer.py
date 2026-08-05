# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Copyright (c) OpenMMLab. All rights reserved.

# pyre-unsafe

"""
Swin Transformer backbone adapted for SAM3.

This implementation follows the mmdetection Swin backbone structure, but avoids
mmcv/mmengine/mmdet dependencies so it can run inside the SAM3 package. The Swin
patch embedding/merging layers are implemented locally because Swin uses flattened
tokens plus explicit spatial shapes, while ``vitdet.PatchEmbed`` returns NHWC
feature maps for ViTDet blocks.
"""

import math
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

try:
    from timm.layers import DropPath, trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import DropPath, trunc_normal_

from sam3.model.data_misc import NestedTensor


def to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        assert len(value) == 2
        return value
    return (value, value)


class AdaptivePadding(nn.Module):
    """Pad an image/tensor so a kernel covers the full spatial extent."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding: str = "corner",
    ) -> None:
        super().__init__()
        assert padding in ("same", "corner")
        self.padding = padding
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)

    def get_pad_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max(
            (output_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - input_h, 0
        )
        pad_w = max(
            (output_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - input_w, 0
        )
        return pad_h, pad_w

    def forward(self, x: Tensor) -> Tensor:
        pad_h, pad_w = self.get_pad_shape(x.shape[-2:])
        if pad_h == 0 and pad_w == 0:
            return x
        if self.padding == "corner":
            return F.pad(x, (0, pad_w, 0, pad_h))
        return F.pad(
            x,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )


class PatchEmbed(nn.Module):
    """Swin patch embedding returning flattened tokens and spatial shape."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        kernel_size: Union[int, Tuple[int, int]] = 4,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int], str] = "corner",
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size, stride, dilation, padding)
            conv_padding = (0, 0)
        else:
            self.adap_padding = None
            conv_padding = to_2tuple(padding)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        if self.adap_padding is not None:
            x = self.adap_padding(x)
        x = self.proj(x)
        hw_shape = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape


class PatchMerging(nn.Module):
    """Merge neighboring patches and project to the next stage width."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int], str] = "corner",
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size, stride, dilation, padding)
            unfold_padding = (0, 0)
        else:
            self.adap_padding = None
            unfold_padding = to_2tuple(padding)

        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=unfold_padding,
            stride=stride,
        )
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        self.norm = norm_layer(sample_dim) if norm_layer is not None else None
        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)
        self.out_channels = out_channels

    def forward(
        self, x: Tensor, input_size: Tuple[int, int]
    ) -> Tuple[Tensor, Tuple[int, int]]:
        B, L, C = x.shape
        H, W = input_size
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        if self.adap_padding is not None:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        out_h = (
            H
            + 2 * self.sampler.padding[0]
            - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1)
            - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            W
            + 2 * self.sampler.padding[1]
            - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1)
            - 1
        ) // self.sampler.stride[1] + 1

        x = x.transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)
        return x, (out_h, out_w)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowMSA(nn.Module):
    """单窗口的多头自注意力计算"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def double_step_seq(step1: int, len1: int, step2: int, len2: int) -> Tensor:
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ShiftWindowMSA(nn.Module):
    """移动窗口多头注意力计算"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        """
        shift_size:需要移动的距离
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size
        self.w_msa = WindowMSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.drop = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self._attn_mask_cache = {}

    def window_partition(self, x: Tensor) -> Tensor:
        """转化成小窗口的形式"""
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, window_size, window_size, C)

    def window_reverse(self, windows: Tensor, H: int, W: int) -> Tensor:
        """小窗口转化回整幅图像"""
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(
            B, H // window_size, W // window_size, window_size, window_size, -1
        )
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def get_attn_mask(self, H_pad: int, W_pad: int, device: torch.device) -> Tensor:
        cache_key = (H_pad, W_pad, device.type, device.index)
        cached_mask = self._attn_mask_cache.get(cache_key)
        if cached_mask is not None and cached_mask.device == device:
            return cached_mask

        img_mask = torch.zeros((1, H_pad, W_pad, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = self.window_partition(img_mask).view(-1, self.window_size**2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        self._attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def forward(self, query: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, "input feature has wrong size"
        query = query.view(B, H, W, C)

        # 如果窗口尺寸不能整除整体图像尺寸，添加padding
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # 如果窗口需要移动
        if self.shift_size > 0:
            # 沿着W和H的方向分别移动shift_size个距离
            shifted_query = torch.roll(
                query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            # 掩码只由特征尺寸、window_size 和 shift_size 决定，重复推理时直接复用。
            attn_mask = self.get_attn_mask(H_pad, W_pad, query.device)
        else:
            shifted_query = query
            attn_mask = None

        # 单窗口注意力计算
        query_windows = self.window_partition(shifted_query).view(
            -1, self.window_size**2, C
        )
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        return self.drop(x)


class SwinBlock(nn.Module):
    """单个SwinBlock块"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        use_act_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_act_checkpoint = use_act_checkpoint
        self.norm1 = norm_layer(embed_dim)
        self.attn = ShiftWindowMSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop_rate,
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        def inner_forward(x_inner: Tensor) -> Tensor:
            identity = x_inner
            x_inner = self.norm1(x_inner)
            x_inner = identity + self.attn(x_inner, hw_shape)
            x_inner = x_inner + self.drop_path(self.mlp(self.norm2(x_inner)))
            return x_inner

        if self.use_act_checkpoint and self.training and x.requires_grad:
            return checkpoint.checkpoint(inner_forward, x, use_reentrant=False)
        return inner_forward(x)


class SwinBlockSequence(nn.Module):
    """Swin-Transformer中的一层,一层中包含多个Swin-Transformer块以及一个下采样块"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        window_size: int = 7,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: Union[float, List[float]] = 0.0,
        downsample: Optional[nn.Module] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        use_act_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [drop_path_rate for _ in range(depth)]

        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    window_size=window_size,
                    shift=i % 2 == 1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_act_checkpoint=use_act_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample

    def forward(
        self, x: Tensor, hw_shape: Tuple[int, int]
    ) -> Tuple[Tensor, Tuple[int, int], Tensor, Tuple[int, int]]:
        for block in self.blocks:
            x = block(x, hw_shape)

        out = x
        out_hw_shape = hw_shape
        if self.downsample is not None:
            x, hw_shape = self.downsample(x, hw_shape)
        return x, hw_shape, out, out_hw_shape


class SwinTransformer(nn.Module):
    """Swin-Transformer 总体框架"""

    def __init__(
        self,
        pretrain_img_size: Union[int, Tuple[int, int]] = 224,
        in_chans: int = 3,
        in_channels: Optional[int] = None,
        embed_dim: int = 96,
        embed_dims: Optional[int] = None,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        strides: Tuple[int, ...] = (4, 2, 2, 2),
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_abs_pos_embed: bool = False,
        norm_layer: Union[Callable[..., nn.Module], str] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_act_checkpoint: bool = False,
        frozen_stages: int = -1,
        pretrained: Optional[str] = None,
    ) -> None:
        super().__init__()
        if in_channels is not None:
            in_chans = in_channels
        if embed_dims is not None:
            embed_dim = embed_dims
        if isinstance(norm_layer, str):
            norm_layer = getattr(nn, norm_layer)

        pretrain_img_size = to_2tuple(pretrain_img_size)
        assert strides[0] == patch_size, "Use non-overlapping patch embed."
        assert len(depths) == len(num_heads) == len(strides)

        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.frozen_stages = frozen_stages
        self.num_features = [int(embed_dim * 2**i) for i in range(len(depths))]
        self.channel_list = [self.num_features[i] for i in out_indices]

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            kernel_size=patch_size,
            stride=strides[0],
            norm_layer=norm_layer if patch_norm else None,
        )

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, patch_row * patch_col, embed_dim)
            )
        else:
            self.absolute_pos_embed = None

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        in_dim = embed_dim
        for i, depth in enumerate(depths):
            downsample = (
                PatchMerging(
                    in_channels=in_dim,
                    out_channels=2 * in_dim,
                    stride=strides[i + 1],
                    norm_layer=norm_layer if patch_norm else None,
                )
                if i < len(depths) - 1
                else None
            )
            stage = SwinBlockSequence(
                embed_dim=in_dim,
                num_heads=num_heads[i],
                depth=depth,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=downsample,
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_act_checkpoint=use_act_checkpoint,
            )
            self.stages.append(stage)
            if downsample is not None:
                in_dim = downsample.out_channels

        for i in out_indices:
            self.add_module(f"norm{i}", norm_layer(self.num_features[i]))

        self.apply(self._init_weights)
        self.pretrained_load_info = None
        if pretrained:
            self.pretrained_load_info = self.load_pretrained(pretrained)
        self._freeze_stages()

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> OrderedDict:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                state = checkpoint.get(key)
                if isinstance(state, dict):
                    return OrderedDict(state)
        if isinstance(checkpoint, dict):
            return OrderedDict(checkpoint)
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    @staticmethod
    def _convert_mmdet_state_dict(state_dict: OrderedDict) -> OrderedDict:
        converted = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                key = key[len("backbone.") :]
            if key.startswith("patch_embed.projection."):
                key = key.replace("patch_embed.projection.", "patch_embed.proj.", 1)

            key = key.replace(".ffn.layers.0.0.", ".mlp.fc1.")
            key = key.replace(".ffn.layers.1.", ".mlp.fc2.")

            # Classification/FPN/dropout bookkeeping keys do not belong to this backbone.
            if (
                key.startswith("head.")
                or key.startswith("neck.")
                or ".ffn." in key
                or key.endswith(".attn.attn_mask")
            ):
                continue
            converted[key] = value
        return converted

    def _convert_pretrained_state_dict(self, state_dict: OrderedDict) -> OrderedDict:
        first_key = next(iter(state_dict.keys()), "")
        if first_key.startswith("backbone."):
            converted = self._convert_mmdet_state_dict(state_dict)
        else:
            converted = swin_converter(state_dict)

        last_norm = f"norm{self.out_indices[-1]}"
        if "norm.weight" in converted and f"{last_norm}.weight" not in converted:
            converted[f"{last_norm}.weight"] = converted.pop("norm.weight")
        if "norm.bias" in converted and f"{last_norm}.bias" not in converted:
            converted[f"{last_norm}.bias"] = converted.pop("norm.bias")

        return OrderedDict(
            (key, value)
            for key, value in converted.items()
            if not (
                key.startswith("head.")
                or key.startswith("neck.")
                or key.endswith(".attn_mask")
            )
        )

    def _filter_loadable_state_dict(self, state_dict: OrderedDict) -> OrderedDict:
        model_state = self.state_dict()
        loadable = OrderedDict()
        skipped_shape = []
        for key, value in state_dict.items():
            if key not in model_state:
                loadable[key] = value
                continue
            if model_state[key].shape == value.shape:
                loadable[key] = value
            else:
                skipped_shape.append(
                    (key, tuple(value.shape), tuple(model_state[key].shape))
                )
        self._last_skipped_shape = skipped_shape
        return loadable

    def load_pretrained(self, checkpoint_path: str) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._convert_pretrained_state_dict(state_dict)
        state_dict = self._filter_loadable_state_dict(state_dict)
        incompatible = self.load_state_dict(state_dict, strict=False)
        return {
            "checkpoint_path": checkpoint_path,
            "loaded_keys": len(state_dict),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "skipped_shape": getattr(self, "_last_skipped_shape", []),
        }

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def train(self, mode: bool = True) -> "SwinTransformer":
        super().train(mode)
        self._freeze_stages()
        return self

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.absolute_pos_embed is not None:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f"norm{i - 1}")
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False
            stage = self.stages[i - 1]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def _get_abs_pos(self, hw_shape: Tuple[int, int]) -> Tensor:
        assert self.absolute_pos_embed is not None
        H, W = hw_shape
        B, L, C = self.absolute_pos_embed.shape
        src_size = int(math.sqrt(L))
        if src_size * src_size != L:
            raise ValueError("absolute_pos_embed must describe a square grid")
        if src_size == H and src_size == W:
            return self.absolute_pos_embed
        abs_pos = self.absolute_pos_embed.reshape(B, src_size, src_size, C)
        abs_pos = abs_pos.permute(0, 3, 1, 2)
        abs_pos = F.interpolate(
            abs_pos, size=(H, W), mode="bicubic", align_corners=False
        )
        return abs_pos.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def forward(
        self, tensor_list: Union[Tensor, NestedTensor]
    ) -> List[Union[Tensor, NestedTensor]]:
        if isinstance(tensor_list, NestedTensor):
            x = tensor_list.tensors
            mask = tensor_list.mask
        else:
            x = tensor_list
            mask = None

        x, hw_shape = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self._get_abs_pos(hw_shape).to(dtype=x.dtype, device=x.device)
        x = self.drop_after_pos(x)

        outs: List[Union[Tensor, NestedTensor]] = []
        resized_masks: dict[Tuple[int, int], Optional[Tensor]] = {}
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i not in self.out_indices:
                continue

            out = getattr(self, f"norm{i}")(out)
            out = out.view(-1, *out_hw_shape, self.num_features[i])
            out = out.permute(0, 3, 1, 2).contiguous()

            if isinstance(tensor_list, NestedTensor):
                out_mask = None
                if mask is not None and mask.any():
                    if out_hw_shape not in resized_masks:
                        resized_masks[out_hw_shape] = F.interpolate(
                            mask[None].float(), size=out_hw_shape
                        ).to(torch.bool)[0]
                    out_mask = resized_masks[out_hw_shape]
                outs.append(NestedTensor(out, out_mask))
            else:
                outs.append(out)

        return outs

    def get_layer_id(self, layer_name: str) -> int:
        num_layers = self.get_num_layers()
        if layer_name.find("absolute_pos_embed") != -1:
            return 0
        if layer_name.find("patch_embed") != -1:
            return 0
        if layer_name.find("stages") != -1:
            return int(layer_name.split("stages")[1].split(".")[1]) + 1
        return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.stages)


def swin_converter(ckpt: OrderedDict) -> OrderedDict:
    """Convert official Swin checkpoint keys toward this SAM3 module layout."""

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x: Tensor) -> Tensor:
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x: Tensor) -> Tensor:
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith("head"):
            continue
        new_v = v
        if k.startswith("layers"):
            if "attn." in k:
                new_k = k.replace("attn.", "attn.w_msa.")
            elif "mlp." in k:
                if "mlp.fc1." in k:
                    new_k = k.replace("mlp.fc1.", "mlp.fc1.")
                elif "mlp.fc2." in k:
                    new_k = k.replace("mlp.fc2.", "mlp.fc2.")
                else:
                    new_k = k
            elif "downsample" in k:
                new_k = k
                if "reduction." in k:
                    new_v = correct_unfold_reduction_order(v)
                elif "norm." in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace("layers", "stages", 1)
        elif k.startswith("patch_embed"):
            new_k = k
            if "projection" in new_k:
                new_k = new_k.replace("projection", "proj")
        else:
            new_k = k
        new_ckpt[new_k] = new_v

    return new_ckpt


def main() -> None:
    def count_parameters(module: nn.Module) -> Tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    warmup_iters = 10
    benchmark_iters = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model = SwinTransformer(
        pretrain_img_size = 384,
        embed_dim= 192,
        patch_size= 4,
        window_size= 12,
        depths= (2, 2, 18, 2),
        num_heads= (6, 12, 24, 48),
        strides= (4, 2, 2, 2),
        pretrained=r"E:\reproduce\weights\SwinTransformer\swin_large_384.pth"
    ).to(device).eval()
    total_params, trainable_params = count_parameters(model)
    x = torch.randn(1, 3, 1008, 1008, device=device)

    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for _ in range(warmup_iters):
            outs = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_times = []
        for _ in range(benchmark_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outs = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_times.append(time.perf_counter() - start)

    print(f"input_shape: {tuple(x.shape)}")
    print(f"total_params: {total_params} ({total_params / 1e6:.2f}M)")
    print(f"trainable_params: {trainable_params} ({trainable_params / 1e6:.2f}M)")
    print(f"autocast_dtype: {torch.bfloat16 if device.type == 'cuda' else None}")
    print(f"warmup_iters: {warmup_iters}")
    print(f"benchmark_iters: {benchmark_iters}")
    print(f"mean_elapsed_sec: {sum(elapsed_times) / len(elapsed_times):.6f}")
    print(f"min_elapsed_sec: {min(elapsed_times):.6f}")
    print(f"max_elapsed_sec: {max(elapsed_times):.6f}")
    print(f"out_shapes: {[tuple(out.shape) for out in outs]}")


if __name__ == "__main__":
    main()
