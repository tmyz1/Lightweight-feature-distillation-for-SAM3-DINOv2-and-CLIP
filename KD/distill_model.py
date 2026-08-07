import sys
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from KD.model.vit_small_patch14_reg4_dinov2 import Vit_Small_feature_extractor
from KD.model.swin_large_384 import Swin_Sam3_feature_extractor
from KD.adapters import MultiScaleFeatureAlignAdapter,MultiScaleClsTokenAlignAdapter
from KD.KD_Loss import cls_token_total_loss, features_total_loss
from KD.build_teacher_model import (
    CLIP_feature_extractor,
    DINO_V2_feature_extractor,
    Sam3_feature_extractor,
)
from sam3.model.data_misc import BatchedDatapoint

def swin_features_to_pseudo_cls_tokens(features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [feature.mean(dim=(-2, -1)) for feature in features]


def detach_teacher_tensors(tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [tensor.detach().clone().float() for tensor in tensors]


def expand_per_layer_config(
    values: Sequence[Any] | Any,
    num_layers: int,
    name: str,
) -> list[Any]:
    """Expand one shared config value or validate explicit per-layer values."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        values = [values]
    else:
        values = list(values)
    if not values:
        raise ValueError(f"Adapter.{name} must contain at least one value.")
    if len(values) == 1:
        return values * num_layers
    if len(values) != num_layers:
        raise ValueError(
            f"Adapter.{name} has {len(values)} entries, but the configured "
            f"student and teachers return {num_layers} feature layers. Use one "
            f"shared entry or exactly {num_layers} entries."
        )
    return values


STUDENT_FEATURE_EXTRACTORS = {
    "swin_sam3": Swin_Sam3_feature_extractor,
    "vit_small": Vit_Small_feature_extractor,
}


class Distill(nn.Module):

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]],
        device: torch.device,
    ):
        super().__init__()
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        #加载设备
        self.device = torch.device(device)
        self.cfg = cfg
        self.student_model_name = self.cfg.get("Student", {}).get(
            "model", "swin_sam3"
        ).lower()
        try:
            self.student_feature_extractor_cls = STUDENT_FEATURE_EXTRACTORS[
                self.student_model_name
            ]
        except KeyError as error:
            supported_models = ", ".join(sorted(STUDENT_FEATURE_EXTRACTORS))
            raise ValueError(
                f"Unsupported Student.model={self.student_model_name!r}. "
                f"Supported values: {supported_models}."
            ) from error

        #获取教师模型的预训练权重
        self.sam3_checkpoint = self.cfg.get("Sam3").get("checkpoint_path")
        self.DINO_V2_checkpoint = self.cfg.get("DINO_V2").get("checkpoint")
        self.CLIP_checkpoint = self.cfg.get("CLIP").get("checkpoint_path")

        #构建学生模型和三个教师模型的特征提取器
        self.Student = self.student_feature_extractor_cls(self.cfg, self.device)
        self.Sam3 = Sam3_feature_extractor(
            checkpoint_path=self.sam3_checkpoint,
            device=self.device,
            cfg=self.cfg,
            feature_source="both",
        )
        self.Dino_v2 = DINO_V2_feature_extractor(
            checkpoint_path=self.DINO_V2_checkpoint,
            device=self.device,
            cfg=self.cfg,
        )
        self.CLIP = CLIP_feature_extractor(
            checkpoint_path=self.CLIP_checkpoint,
            device=self.device,
            cfg=self.cfg,
        )
        self.freeze_parameters()

        #构建adapter，修改学生特征与各个教师模型的特征空间通道尺寸相同
        adapter_cfg = self.cfg.get("Adapter")
        student_feature_layers = self.Student.num_feature_layers
        teacher_feature_layers = {
            "SAM3": self.Sam3.num_feature_layers,
            "DINO-V2": self.Dino_v2.num_feature_layers,
            "CLIP": self.CLIP.num_feature_layers,
        }
        if any(count != student_feature_layers for count in teacher_feature_layers.values()):
            counts = ", ".join(
                f"{name}={count}" for name, count in teacher_feature_layers.items()
            )
            raise ValueError(
                "Feature-layer counts must match for distillation: "
                f"student={student_feature_layers}, {counts}."
            )

        self.student_channels: List[int] = expand_per_layer_config(
            adapter_cfg.get("student_channel"), student_feature_layers, "student_channel"
        )
        self.student_cls_channels: List[int] = expand_per_layer_config(
            adapter_cfg.get("student_cls_channel", self.student_channels),
            student_feature_layers,
            "student_cls_channel",
        )
        self.teacher_channels: List[int] = expand_per_layer_config(
            adapter_cfg.get("teacher_channel"), student_feature_layers, "teacher_channel"
        )
        self.target_sizes: List[List[int]] = expand_per_layer_config(
            adapter_cfg.get("target_sizes"), student_feature_layers, "target_sizes"
        )
        dino_v2_grid_size = int(self.cfg.get("DINO_V2", {}).get("resolution", 224)) // 14
        clip_grid_size = int(self.cfg.get("CLIP", {}).get("resolution", 224)) // 14
        self.dino_v2_target_sizes: List[List[int]] = [[dino_v2_grid_size, dino_v2_grid_size] for _ in self.teacher_channels]
        self.clip_target_sizes: List[List[int]] = [[clip_grid_size, clip_grid_size] for _ in self.teacher_channels]

        self.sam3_adapter = MultiScaleFeatureAlignAdapter(
            student_features_channel=self.student_channels,
            teacher_features_channel=self.teacher_channels,
            target_sizes=self.target_sizes,
        ).to(self.device)
        self.Dino_v2_adapter = MultiScaleFeatureAlignAdapter(
            student_features_channel=self.student_channels,
            teacher_features_channel=self.teacher_channels,
            target_sizes=self.dino_v2_target_sizes,
        ).to(self.device)
        self.CLIP_adapter = MultiScaleFeatureAlignAdapter(
            student_features_channel=self.student_channels,
            teacher_features_channel=self.teacher_channels,
            target_sizes=self.clip_target_sizes,
        ).to(self.device)

        self.Dino_v2_cls_adapter = MultiScaleClsTokenAlignAdapter(
            student_channels=self.student_cls_channels,
            teacher_channels=self.teacher_channels,
        ).to(self.device)
        self.CLIP_cls_adapter = MultiScaleClsTokenAlignAdapter(
            student_channels=self.student_cls_channels,
            teacher_channels=self.teacher_channels,
        ).to(self.device)

    def get_student_checkpoint(self):
        student_cfg = self.cfg.get("Student", {})
        if self.student_model_name == "swin_sam3":
            return student_cfg.get("swin_backbone", {}).get("pretrained")
        if self.student_model_name == "vit_small":
            return student_cfg.get("vit_small_checkpoint_path")
        return None

    def freeze_parameters(self) -> None:
        # 冻结教师网络所有参数
        for teacher in (self.Sam3, self.Dino_v2, self.CLIP):
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

        # 学生网络冻结部分
        for name, param in self.Student.model.named_parameters():
            param.requires_grad_(name.startswith("backbone.vision_backbone."))

    def train(self, mode: bool = True):
        super().train(mode)
        for teacher in (self.Sam3, self.Dino_v2, self.CLIP):
            teacher.eval()
        return self

    def trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

    def forward(self, batch: BatchedDatapoint):
        #获取学生模型和教师模型的backbone,neck and cls_token
        student_outputs = self.Student(batch)
        #针对模型有返回cls_token的情况
        if len(student_outputs) == 3:
            student_features, student_necks, student_cls_tokens = student_outputs
        #针对模型没有cls_token的情况
        else:
            student_features, student_necks = student_outputs
            student_cls_tokens = None
        with torch.no_grad():
            sam3_features, sam3_necks = self.Sam3(batch)
            dino_features, dino_cls_tokens = self.Dino_v2(batch)
            clip_features, clip_cls_tokens = self.CLIP(batch)

        #学生模型与三个教师模型backbone特征对齐
        student_features_float = [feature.float() for feature in student_features]
        student_to_sam3_features = self.sam3_adapter(student_features_float)
        student_to_dino_v2_features = self.Dino_v2_adapter(student_features_float)
        student_to_clip_features = self.CLIP_adapter(student_features_float)

        #学生模型与DINO-V2，CLIP cls_token 对齐
        if float(self.cfg.get("loss", {}).get("cls_tokens_loss_weight", 0.0)) > 0:
            #如果模型没有cls_token,以最后一层平均池化作为cls_token
            if student_cls_tokens is None:
                student_cls_tokens = swin_features_to_pseudo_cls_tokens(student_features)
            student_cls_tokens = [cls_token.float() for cls_token in student_cls_tokens]
            student_to_dino_cls_tokens = self.Dino_v2_cls_adapter(student_cls_tokens)
            student_to_clip_cls_tokens = self.CLIP_cls_adapter(student_cls_tokens)

        #获取各部分损失值的权重系数
        loss_cfg = self.cfg.get("loss")
        feature_loss_weight = loss_cfg.get("features_loss_weight")
        necks_loss_weight = loss_cfg.get("necks_loss_weight")
        cls_tokens_loss_weight = loss_cfg.get("cls_tokens_loss_weight")
        losses_type = loss_cfg.get("losses_type")

        #计算backbone层面上的损失
        sam3_loss = features_total_loss(
            student_features=student_to_sam3_features,
            teacher_features=detach_teacher_tensors(sam3_features),
            losses_type=losses_type,
            cfg=self.cfg,
            avg=True,
        )
        dino_v2_loss = features_total_loss(
            student_features=student_to_dino_v2_features,
            teacher_features=detach_teacher_tensors(dino_features),
            losses_type=losses_type,
            cfg=self.cfg,
            avg=True,
        )
        clip_loss = features_total_loss(
            student_features=student_to_clip_features,
            teacher_features=detach_teacher_tensors(clip_features),
            losses_type=losses_type,
            cfg=self.cfg,
            avg=True,
        )

        #计算neck部分的损失值
        neck_loss = features_total_loss(
            student_features=student_necks,
            teacher_features=detach_teacher_tensors(sam3_necks),
            losses_type=losses_type,
            cfg=self.cfg,
            avg=True,
        )

        #计算cls_token层面上的损失
        dino_v2_cls_loss = cls_token_total_loss(
            student_cls_token=student_to_dino_cls_tokens,
            teacher_cls_token=detach_teacher_tensors(dino_cls_tokens),
            losses_type=losses_type,
            cfg=self.cfg,
        )
        clip_cls_loss = cls_token_total_loss(
            student_cls_token=student_to_clip_cls_tokens,
            teacher_cls_token=detach_teacher_tensors(clip_cls_tokens),
            losses_type=losses_type,
            cfg=self.cfg,
        )

        teacher_feature_weights = {
            "sam3": float(loss_cfg.get("sam3_features_weight", 1.0)),
            "dino_v2": float(loss_cfg.get("dino_v2_features_weight", 0.15)),
            "clip": float(loss_cfg.get("clip_features_weight", 0.15)),
        }
        feature_weight_sum = sum(teacher_feature_weights.values())
        feature_loss = (
            sam3_loss["total_loss"] * teacher_feature_weights["sam3"]
            + dino_v2_loss["total_loss"] * teacher_feature_weights["dino_v2"]
            + clip_loss["total_loss"] * teacher_feature_weights["clip"]
        ) / feature_weight_sum
        cls_tokens_loss = (
            dino_v2_cls_loss["cls_total_loss"]
            + clip_cls_loss["cls_total_loss"]
        ) / 2
        kd_loss = (
            feature_loss * feature_loss_weight
            + neck_loss["total_loss"] * necks_loss_weight
            + cls_tokens_loss * cls_tokens_loss_weight
        )
        total_loss = kd_loss

        logs = {
            "total_loss": total_loss,
            "avg_feature_loss": feature_loss,
            "neck_loss": neck_loss["total_loss"],
            "cls_token_loss" : cls_tokens_loss,
            "sam3_features_loss" : sam3_loss["total_loss"],
            "dino_v2_features_loss" : dino_v2_loss["total_loss"],
            "clip_features_loss" : clip_loss["total_loss"],
            "dino_v2_cls_token_loss" : dino_v2_cls_loss["cls_total_loss"],
            "clip_cls_token_loss" : clip_cls_loss['cls_total_loss'],
        }
        return total_loss, logs
