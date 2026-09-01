import sys
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from KD.model.vit_small_patch14_reg4_dinov2 import Vit_Small_feature_extractor
from KD.model.swin_large_384 import Swin_Sam3_feature_extractor
from KD.model.RepVit import RepVit_feature_extractor
from KD.adapters import MultiScaleFeatureAlignAdapter,MultiScaleClsTokenAlignAdapter
from KD.KD_Loss import (
    build_feature_valid_masks,
    cls_token_total_loss,
    features_total_loss,
)
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
    "repvit": RepVit_feature_extractor,
}


class _BaseDistill(nn.Module):

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]],
        device: torch.device,
        with_auxiliary_teachers: bool,
    ):
        super().__init__()
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)
        self.cfg = cfg
        self.with_auxiliary_teachers = with_auxiliary_teachers
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

        self.sam3_checkpoint = self.cfg.get("Sam3", {}).get("checkpoint_path")

        #build Student model and teacher model
        self.Student = self.student_feature_extractor_cls(self.cfg, self.device)
        self.Sam3 = Sam3_feature_extractor(
            checkpoint_path=self.sam3_checkpoint,
            device=self.device,
            cfg=self.cfg,
            feature_source="both",
        )
        self.Dino_v2 = None
        self.CLIP = None
        if self.with_auxiliary_teachers:
            dino_cfg = self.cfg.get("DINO_V2", {})
            clip_cfg = self.cfg.get("CLIP", {})
            if not dino_cfg.get("is_use", False) or not clip_cfg.get("is_use", False):
                raise ValueError(
                    "distillation.teacher_mode='sam3_dino_v2_clip' requires "
                    "DINO_V2.is_use: true and CLIP.is_use: true."
                )
            self.Dino_v2 = DINO_V2_feature_extractor(
                checkpoint_path=dino_cfg.get("checkpoint"),
                device=self.device,
                cfg=self.cfg,
            )
            self.CLIP = CLIP_feature_extractor(
                checkpoint_path=clip_cfg.get("checkpoint_path"),
                device=self.device,
                cfg=self.cfg,
            )
        self.freeze_parameters()

        #build adapter
        adapter_cfg = self.cfg.get("Adapter")
        student_feature_layers = self.Student.num_feature_layers
        teacher_feature_layers = {
            "SAM3": self.Sam3.num_feature_layers,
            "DINO-V2": self.Dino_v2.num_feature_layers if self.Dino_v2 else None,
            "CLIP": self.CLIP.num_feature_layers if self.CLIP else None,
        }

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

        if self.Dino_v2:
            dino_v2_grid_size = int(self.cfg.get("DINO_V2", {}).get("resolution", 224)) // 14
            self.dino_v2_target_sizes: List[List[int]] = [[dino_v2_grid_size, dino_v2_grid_size] for _ in self.teacher_channels]
        if self.CLIP:
            clip_grid_size = int(self.cfg.get("CLIP", {}).get("resolution", 224)) // 14
            self.clip_target_sizes: List[List[int]] = [[clip_grid_size, clip_grid_size] for _ in self.teacher_channels]

        self.sam3_adapter = MultiScaleFeatureAlignAdapter(
            student_features_channel=self.student_channels,
            teacher_features_channel=self.teacher_channels,
        ).to(self.device)
        if self.Dino_v2:
            self.Dino_v2_adapter = MultiScaleFeatureAlignAdapter(
                student_features_channel=self.student_channels,
                teacher_features_channel=self.teacher_channels,
            ).to(self.device)

            self.Dino_v2_cls_adapter = MultiScaleClsTokenAlignAdapter(
                student_channels=self.student_cls_channels,
                teacher_channels=self.teacher_channels,
            ).to(self.device)
        if self.CLIP:
            self.CLIP_adapter = MultiScaleFeatureAlignAdapter(
                student_features_channel=self.student_channels,
                teacher_features_channel=self.teacher_channels,
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
        if self.student_model_name == "repvit":
            return student_cfg.get("repvit_checkpoint_path")
        return None

    def freeze_parameters(self) -> None:
        for teacher in (self.Sam3, self.Dino_v2, self.CLIP):
            if teacher is not None:
                teacher.eval()
                for param in teacher.parameters():
                    param.requires_grad_(False)

        for name, param in self.Student.model.named_parameters():
            param.requires_grad_(name.startswith("backbone.vision_backbone."))

    def train(self, mode: bool = True):
        super().train(mode)
        self.Sam3.eval()
        if self.Dino_v2:
            self.Dino_v2.eval()
        if self.CLIP:
            self.CLIP.eval()
        return self

    def trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

    def forward(self, batch: BatchedDatapoint):
        #Student network feature output
        student_outputs = self.Student(batch)
        if len(student_outputs) == 3:
            student_features, student_necks, student_cls_tokens = student_outputs
        else:
            student_features, student_necks = student_outputs
            student_cls_tokens = None

        #Teacher model feature output
        with torch.no_grad():
            sam3_features, sam3_necks = self.Sam3(batch)
            if self.Dino_v2:
                dino_features, dino_cls_tokens = self.Dino_v2(batch)
            if self.CLIP:
                clip_features, clip_cls_tokens = self.CLIP(batch)

        student_features_float = [feature.float() for feature in student_features]
        loss_cfg = self.cfg.get("loss")
        sam3_teacher_features = detach_teacher_tensors(sam3_features)
        sam3_teacher_necks = detach_teacher_tensors(sam3_necks)

        #Align student model features with teacher model features
        student_to_sam3_features = self.sam3_adapter(student_features_float,sam3_teacher_features)
        if self.Dino_v2:
            student_to_dino_v2_features = self.Dino_v2_adapter(student_features_float,dino_features)
        if self.CLIP:
            student_to_clip_features = self.CLIP_adapter(student_features_float,clip_features)

        #Loss function calculation section
        feature_loss_weight = loss_cfg.get("features_loss_weight")
        necks_loss_weight = loss_cfg.get("necks_loss_weight")
        cls_tokens_loss_weight = loss_cfg.get("cls_tokens_loss_weight")
        losses_type = loss_cfg.get("losses_type")
        cls_losses_type = loss_cfg.get(
            "cls_losses_type",
            [
                loss_type for loss_type in losses_type
                if loss_type in loss_cfg.get("cls_token_loss", {})
            ],
        )
        use_cls_loss = float(cls_tokens_loss_weight) > 0

        valid_boxes = getattr(batch, "kd_valid_boxes", None)
        if not bool(loss_cfg.get("mask_sam3_padding", True)):
            valid_boxes = None
        sam3_feature_valid_masks = build_feature_valid_masks(
            valid_boxes, sam3_teacher_features
        )
        sam3_neck_valid_masks = build_feature_valid_masks(
            valid_boxes, sam3_teacher_necks
        )

        if use_cls_loss:
            if student_cls_tokens is None:
                student_cls_tokens = swin_features_to_pseudo_cls_tokens(student_features)
            student_cls_tokens = [cls_token.float() for cls_token in student_cls_tokens]
            if self.Dino_v2:
                student_to_dino_cls_tokens = self.Dino_v2_cls_adapter(student_cls_tokens, dino_cls_tokens)
            if self.CLIP:
                student_to_clip_cls_tokens = self.CLIP_cls_adapter(student_cls_tokens, clip_cls_tokens)

        sam3_loss = features_total_loss(
            student_features=student_to_sam3_features,
            teacher_features=sam3_teacher_features,
            losses_type=losses_type,
            cfg=self.cfg,
            avg=True,
            valid_masks=sam3_feature_valid_masks,
        )
        if self.Dino_v2:
            dino_v2_loss = features_total_loss(
                student_features=student_to_dino_v2_features,
                teacher_features=detach_teacher_tensors(dino_features),
                losses_type=losses_type,
                cfg=self.cfg,
                avg=True,
            )
        if self.CLIP:
            clip_loss = features_total_loss(
                student_features=student_to_clip_features,
                teacher_features=detach_teacher_tensors(clip_features),
                losses_type=losses_type,
                cfg=self.cfg,
                avg=True,
            )
        neck_loss = features_total_loss(
            student_features=student_necks,
            teacher_features=sam3_teacher_necks,
            losses_type=losses_type,
            cfg=self.cfg,
            avg=True,
            valid_masks=sam3_neck_valid_masks,
        )

        if use_cls_loss:
            cls_tokens_loss = torch.zeros((), device=self.device)
            if self.Dino_v2:
                dino_v2_cls_loss = cls_token_total_loss(
                    student_cls_token=student_to_dino_cls_tokens,
                    teacher_cls_token=detach_teacher_tensors(dino_cls_tokens),
                    losses_type=cls_losses_type,
                    cfg=self.cfg,
                )
                cls_tokens_loss += dino_v2_cls_loss['cls_total_loss']
            if self.CLIP:
                clip_cls_loss = cls_token_total_loss(
                    student_cls_token=student_to_clip_cls_tokens,
                    teacher_cls_token=detach_teacher_tensors(clip_cls_tokens),
                    losses_type=cls_losses_type,
                    cfg=self.cfg,
                )
                cls_tokens_loss += clip_cls_loss['cls_total_loss']
        else:
            cls_tokens_loss = torch.zeros((), device=self.device)

        teacher_feature_weights = {
            "sam3": float(loss_cfg.get("sam3_features_weight", 1.0)),
            "dino_v2": float(loss_cfg.get("dino_v2_features_weight", 0.15)) if self.Dino_v2 else 0,
            "clip": float(loss_cfg.get("clip_features_weight", 0.15)) if self.CLIP else 0,
        }
        feature_weight_sum = sum(teacher_feature_weights.values())
        feature_loss = sam3_loss["total_loss"] * teacher_feature_weights["sam3"]
        if self.Dino_v2:
            feature_loss += dino_v2_loss["total_loss"] * teacher_feature_weights["dino_v2"]
        if self.CLIP:
            feature_loss += clip_loss["total_loss"] * teacher_feature_weights["clip"]
        feature_loss = feature_loss / feature_weight_sum
        total_loss = (
            feature_loss * feature_loss_weight
            + neck_loss["total_loss"] * necks_loss_weight
            + cls_tokens_loss * cls_tokens_loss_weight
        )

        logs = {
            "total_loss": total_loss,
            "avg_feature_loss": feature_loss,
            "neck_loss": neck_loss["total_loss"],
            "cls_token_loss": cls_tokens_loss,
            "sam3_features_loss": sam3_loss["total_loss"],
            "sam3_mse_loss": sam3_loss.get("mse", torch.zeros((), device=self.device)),
            "neck_mse_loss": neck_loss.get("mse", torch.zeros((), device=self.device)),
        }
        if self.Dino_v2:
            logs["dino_v2_features_loss"] = dino_v2_loss["total_loss"]
            if use_cls_loss:
                logs["dino_v2_cls_loss"] = dino_v2_cls_loss["cls_total_loss"]
        if self.CLIP:
            logs["clip_features_loss"] = clip_loss["total_loss"]
            if use_cls_loss:
                logs["clip_cls_token_loss"] = clip_cls_loss["cls_total_loss"]
        return total_loss, logs


class Sam3Distill(_BaseDistill):
    """Distill the student only from SAM3 backbone and neck features."""

    def __init__(self, cfg: Optional[Dict[str, Any]], device: torch.device):
        super().__init__(cfg, device, with_auxiliary_teachers=False)


class MultiTeacherDistill(_BaseDistill):
    """Jointly distill the student from SAM3, DINOv2, and CLIP teachers."""

    def __init__(self, cfg: Optional[Dict[str, Any]], device: torch.device):
        super().__init__(cfg, device, with_auxiliary_teachers=True)


class Distill(nn.Module):
    """Select and expose the configured SAM3-only or multi-teacher distiller."""

    def __init__(self, cfg: Optional[Dict[str, Any]], device: torch.device):
        super().__init__()
        self.cfg = cfg
        teacher_mode = str(
            self.cfg.get("distillation", {}).get("teacher_mode", "sam3")
        ).lower()
        if teacher_mode == "sam3":
            self.distiller = Sam3Distill(cfg, device)
        elif teacher_mode in {"sam3_dino_v2_clip", "multi_teacher"}:
            self.distiller = MultiTeacherDistill(cfg, device)
        else:
            raise ValueError(
                "Unsupported distillation.teacher_mode="
                f"{teacher_mode!r}. Use 'sam3' or 'sam3_dino_v2_clip'."
            )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as error:
            if name == "distiller":
                raise error
            distiller = super().__getattr__("distiller")
            return getattr(distiller, name)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self.distiller.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def trainable_parameters(self):
        return self.distiller.trainable_parameters()

    def forward(self, batch: BatchedDatapoint):
        return self.distiller(batch)
