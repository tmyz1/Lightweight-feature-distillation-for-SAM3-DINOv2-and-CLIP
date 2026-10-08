import sys
from typing import List, Optional, Sequence
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()

class Linear2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)       # BCHW -> BHWC
        x = self.linear(x)               # 对每个空间位置变换通道
        return x.permute(0, 3, 1, 2).contiguous()


class FeatureAlignAdapter(nn.Module):
    """
    用于处理单层的学生与教师网络的特征空间和通道的对齐
    学生网络特征图像向教师网络特征图像看齐
    返回和教师网络特征图像大小相同的学生网络特征
    """

    def __init__(
        self,
        student_channel: int,
        teacher_channel: int,
        use_norm: bool = True,
        use_gelu: bool = False,
        resize_first: bool = False,
        align_corners: bool = False,
    ):
        """
        student_channel:学生特征图的通道维数
        teacher_channel:教师特征图的通道维数
        hidden_channel: 隐藏层通道维数，默认和教师维数相同
        use_norm: 标准化
        use_gelu：激活函数
        resize_first: 先进行空间对齐
        align_corners: 做双线性插值时的一个坐标对齐方式参数
        """
        super().__init__()
        self.resize_first = resize_first
        self.align_corners = align_corners

        layers = [
            Linear2d(student_channel, teacher_channel, bias=True),
        ]

        if use_norm:
            layers.append(LayerNorm2d(teacher_channel))

        if use_gelu:
            layers.append(nn.GELU())

        self.channel_proj = nn.Sequential(*layers)

    def _resize(self, x: torch.Tensor, target_hw):
        if tuple(x.shape[-2:]) == tuple(target_hw):
            return x

        return F.interpolate(
            x,
            size=target_hw,
            mode="bilinear",
            align_corners=self.align_corners,
        )

    def forward(
        self,
        student: torch.Tensor,
        teacher: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        student: 学生网络特征图像
        teacher: 教师网络特征图像
        """
        if student.ndim != 4:
            raise ValueError(f"student must be [B, C, H, W], got {student.shape}")

        if teacher is not None:
            if teacher.ndim != 4:
                raise ValueError(f"teacher must be [B, C, H, W], got {teacher.shape}")

        if self.resize_first:
            student = self._resize(student, teacher.shape[-2:])
            student = self.channel_proj(student)
        else:
            student = self.channel_proj(student)
            student = self._resize(student, teacher.shape[-2:])

        return student


class MultiScaleFeatureAlignAdapter(nn.Module):
    """
    处理多层的特征图像
    将学生网络和教师网络的所有特征图层进行统一对齐
    """

    def __init__(
        self,
        student_features_channel: Sequence[int],
        teacher_features_channel: Sequence[int],
    ):
        """
        student_features:所有学生网络的特征图层
        teacher_features:所有教师网络的特征图层
        target_sizes: 每一层统一设置的目标尺寸
        """
        super().__init__()
        if not len(student_features_channel) == len(teacher_features_channel):
            assert (f'length of student_features, teacher_features, target_sizes are different'
                    f'\nstudent_features length: {len(student_features_channel)}, '
                    f'\nteacher_features length: {len(teacher_features_channel)}, '
                    )

        self.adapters = nn.ModuleList(
            FeatureAlignAdapter(in_ch, out_ch)
            for in_ch, out_ch in zip(student_features_channel, teacher_features_channel)
        )


    def forward(self, features: Sequence[torch.Tensor],teacher_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.adapters):
            assert f'length of features {len(features)} != length of adapter {len(self.adapters)}'

        return [
            adapter(feature, teacher_feature)
            for adapter, feature, teacher_feature in zip(self.adapters, features, teacher_features)
        ]


class MultiScaleClsTokenAlignAdapter(nn.Module):
    """
    功能类似于MultiScaleFeatureAlignAdapter，主要用于dino和clip中提取的cls_tokens
    """
    def __init__(
        self,
        student_channels: Sequence[int],
        teacher_channels: Sequence[int],
    ):
        super().__init__()
        if len(student_channels) != len(teacher_channels):
            raise ValueError(
                "student_channels and teacher_channels must have the same length, "
                f"got {len(student_channels)} and {len(teacher_channels)}."
            )

        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(student_channel, teacher_channel),
                nn.LayerNorm(teacher_channel),
                nn.GELU(),
                nn.Linear(teacher_channel, teacher_channel),
            )
            for student_channel, teacher_channel in zip(student_channels, teacher_channels)
        )

    def forward(self, cls_tokens: Sequence[torch.Tensor],teacher_cls_tokens: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(cls_tokens) != len(self.adapters):
            raise ValueError(
                f"length of cls_tokens {len(cls_tokens)} != length of adapter {len(self.adapters)}"
            )

        return [
            adapter(cls_token, teacher_cls_token)
            for adapter, cls_token, teacher_cls_token in zip(self.adapters, cls_tokens, teacher_cls_tokens)
        ]



if __name__ == "__main__":
    img_root = r"E:\my_data\rf100-vl\apex-videogame\train\7--2-_png_jpg.rf.72266a0209766e9f48a3d2d2688ea594.jpg"
    img = Image.open(img_root).convert("RGB")
    img_tensor = torchvision.transforms.ToTensor()(img)


