from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from KD.KD_Loss import mask_distillation_loss, strict_valid_region_mask
from KD.model.vit_small_patch14_reg4_dinov2 import (
    build_vit_small_image_model,
    replace_vit_small_memory_efficient_attention,
)
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model


class NoValidAnnotationsError(ValueError):
    pass


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type in {path}: {type(checkpoint)!r}")
    for key in ("student_model", "model", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key.removeprefix("module."): value for key, value in checkpoint.items()
        }
    return checkpoint

"""Distill annotation-prompted SAM3 masks from teacher to student."""
class MaskDistill(nn.Module):

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.distill_cfg = cfg.get("mask_distill", {})
        sam3_cfg = cfg.get("Sam3", {})
        student_cfg = cfg.get("Student", {})
        teacher_checkpoint = sam3_cfg.get("checkpoint_path")
        if not teacher_checkpoint:
            raise ValueError("Sam3.checkpoint_path is required.")

        common = {
            "bpe_path": sam3_cfg.get("bpe_path"),
            "device": str(self.device),
            "eval_mode": True,
            "checkpoint_path": teacher_checkpoint,
            "enable_segmentation": True,
            "enable_inst_interactivity": False,
        }
        self.teacher_model = build_sam3_image_model(
            **common, load_from_HF=False
        )
        self.student_model = build_vit_small_image_model(
            **common,
            vit_small_checkpoint_path=student_cfg.get("vit_small_checkpoint_path"),
            return_interm_layers=False,
            vit_small_intermediate_layers=student_cfg.get(
                "vit_small_intermediate_layers", (2, 5, 8, 11)
            ),
            cfg=cfg,
        )
        replace_vit_small_memory_efficient_attention(self.student_model)
        self._load_student(student_cfg.get("initial_checkpoint"))
        self._configure_trainable_parameters()

    def _load_student(self, checkpoint_path: Optional[str]) -> None:
        if not checkpoint_path:
            raise ValueError("Student.initial_checkpoint is required.")
        missing, unexpected = self.student_model.load_state_dict(
            _load_state_dict(checkpoint_path), strict=False
        )
        invalid_missing = [
            key for key in missing if not key.startswith("segmentation_head.")
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                f"Incompatible student checkpoint: missing={invalid_missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
        if missing:
            print(
                f"[model] retained {len(missing)} teacher-initialized mask-head parameters",
                flush=True,
            )
        print(f"[model] loaded student: {checkpoint_path}", flush=True)

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)
        prefixes = tuple(
            self.distill_cfg.get(
                "trainable_modules",
                ("geometry_encoder", "transformer.decoder", "segmentation_head"),
            )
        )
        for name, parameter in self.student_model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        trainable = sum(
            parameter.numel()
            for parameter in self.student_model.parameters()
            if parameter.requires_grad
        )
        if trainable == 0:
            raise RuntimeError("mask_distill.trainable_modules selected no parameters.")
        print(f"[model] trainable student parameters: {trainable:,}", flush=True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Disable SAM3's training-only matcher and DAC; gradients remain enabled.
        self.teacher_model.eval()
        self.student_model.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def get_student_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.student_model.state_dict()

    def _select_annotations(
        self, batch: BatchedDatapoint, max_prompts: Optional[int] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        if len(batch.find_inputs) != 1 or len(batch.find_targets) != 1:
            raise ValueError("MaskDistill expects one find stage per batch.")
        find_input, target = batch.find_inputs[0], batch.find_targets[0]
        counts = target.num_boxes.long()
        parent = torch.repeat_interleave(
            torch.arange(len(counts), device=counts.device), counts
        )
        boxes = target.boxes.reshape(-1, 4).float()
        valid = (boxes[:, 2:] > 0).all(dim=1)
        packed_indices = torch.arange(len(boxes), device=boxes.device)[valid]
        boxes, parent = boxes[valid].clamp(0.0, 1.0), parent[valid]
        if len(boxes) == 0:
            raise NoValidAnnotationsError("The batch has no valid annotated boxes.")

        if max_prompts is None:
            max_prompts = int(self.distill_cfg.get("max_prompts_per_batch", 4))
        if max_prompts > 0 and len(boxes) > max_prompts:
            chosen = (
                torch.randperm(len(boxes), device=boxes.device)[:max_prompts]
                if self.training
                else torch.arange(max_prompts, device=boxes.device)
            )
            boxes, parent = boxes[chosen], parent[chosen]
            packed_indices = packed_indices[chosen]

        segments = None
        segment_valid = None
        if target.segments is not None and len(target.segments) > 0:
            segments = target.segments[packed_indices]
            segment_valid = (
                torch.ones(len(segments), device=boxes.device, dtype=torch.bool)
                if target.is_valid_segment is None
                else target.is_valid_segment[packed_indices].bool()
            )
        return {
            "boxes": boxes,
            "img_ids": find_input.img_ids[parent],
            "text_ids": find_input.text_ids[parent],
            "segments": segments,
            "segment_valid": segment_valid,
        }

    @staticmethod
    def _index_annotations(
        annotations: Dict[str, Optional[torch.Tensor]], index
    ) -> Dict[str, Optional[torch.Tensor]]:
        return {
            key: None if value is None else value[index]
            for key, value in annotations.items()
        }

    def _points_from_annotations(
        self,
        annotations: Dict[str, Optional[torch.Tensor]],
        sampling: str,
    ) -> torch.Tensor:
        boxes = annotations["boxes"]
        points = boxes[:, :2].clone()
        segments = annotations["segments"]
        segment_valid = annotations["segment_valid"]
        if segments is None:
            return points
        for index, segment in enumerate(segments):
            if segment_valid is not None and not bool(segment_valid[index]):
                continue
            candidates = segment.bool().nonzero(as_tuple=False)
            if len(candidates) == 0:
                continue
            if sampling == "random":
                point_yx = candidates[
                    torch.randint(len(candidates), (1,), device=candidates.device).item()
                ].float()
            elif sampling == "center":
                centroid = candidates.float().mean(dim=0)
                point_yx = candidates[
                    (candidates.float() - centroid).square().sum(dim=1).argmin()
                ].float()
            else:
                raise ValueError(f"Unsupported point sampling mode: {sampling}")
            height, width = segment.shape[-2:]
            points[index] = torch.stack(
                ((point_yx[1] + 0.5) / width, (point_yx[0] + 0.5) / height)
            )
        return points.clamp(0.0, 1.0)

    def _build_prompt(
        self,
        annotations: Dict[str, Optional[torch.Tensor]],
        prompt_type: str,
        point_sampling: str,
    ) -> Prompt:
        boxes = annotations["boxes"]
        count = len(boxes)
        labels = torch.ones(1, count, device=boxes.device, dtype=torch.long)
        padding = torch.zeros(count, 1, device=boxes.device, dtype=torch.bool)
        if prompt_type == "box":
            return Prompt(
                box_embeddings=boxes.unsqueeze(0),
                box_labels=labels,
                box_mask=padding,
            )
        if prompt_type == "point":
            points = self._points_from_annotations(annotations, point_sampling)
            return Prompt(
                point_embeddings=points.unsqueeze(0),
                point_labels=labels,
                point_mask=padding,
            )
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

    def _choose_prompt_type(self, requested: Optional[str]) -> str:
        choices = [
            str(value).lower()
            for value in self.distill_cfg.get("prompt_types", ["box"])
        ]
        if requested is not None:
            if requested not in choices:
                raise ValueError(f"Prompt type {requested!r} is not enabled.")
            return requested
        return choices[torch.randint(len(choices), (1,)).item()] if self.training else choices[0]

    @staticmethod
    def _encode_inputs(model, images, text_batch):
        output = {"img_batch_all_stages": images}
        output.update(model.backbone.forward_image(images.float()))
        output.update(model.backbone.forward_text(text_batch, device=model.device))
        return output

    @staticmethod
    def _decode_prompts(model, backbone_output, annotations, prompt):
        find_input = SimpleNamespace(
            img_ids=annotations["img_ids"], text_ids=annotations["text_ids"]
        )
        return model.forward_grounding(
            backbone_out=backbone_output,
            find_input=find_input,
            find_target=None,
            geometric_prompt=prompt,
        )

    @staticmethod
    def _quality(logits: torch.Tensor) -> torch.Tensor:
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits.max(-1).values

    @classmethod
    def _gather_output(cls, output, query_indices=None):
        if query_indices is None:
            query_indices = cls._quality(output["pred_logits"]).argmax(dim=1)
        batch_indices = torch.arange(len(query_indices), device=query_indices.device)
        return {
            "masks": output["pred_masks"][batch_indices, query_indices].unsqueeze(1),
            "scores": output["pred_logits"][batch_indices, query_indices],
            "queries": output["queries"][batch_indices, query_indices],
            "indices": query_indices,
        }

    @staticmethod
    def _valid_mask(batch, img_ids, size):
        valid_boxes = getattr(batch, "kd_valid_boxes", None)
        if valid_boxes is None:
            return None
        return strict_valid_region_mask(valid_boxes[img_ids], size)

    def forward(self, batch: BatchedDatapoint, prompt_type: Optional[str] = None):
        annotations = self._select_annotations(batch)
        prompt_type = self._choose_prompt_type(prompt_type)
        prompt = self._build_prompt(
            annotations,
            prompt_type,
            str(self.distill_cfg.get("point_sampling", "random")),
        )
        student_backbone = self._encode_inputs(
            self.student_model,
            getattr(batch, "student_img_batch", batch.img_batch),
            batch.find_text_batch,
        )
        student = self._decode_prompts(
            self.student_model, student_backbone, annotations, prompt
        )
        with torch.no_grad():
            teacher_backbone = self._encode_inputs(
                self.teacher_model,
                getattr(batch, "sam3_img_batch", batch.img_batch),
                batch.find_text_batch,
            )
            teacher = self._decode_prompts(
                self.teacher_model, teacher_backbone, annotations, prompt
            )

        teacher_best = self._quality(teacher["pred_logits"]).argmax(dim=1)
        student = self._gather_output(student, teacher_best)
        teacher = self._gather_output(teacher, teacher_best)
        if teacher["masks"].shape[-2:] != student["masks"].shape[-2:]:
            teacher["masks"] = F.interpolate(
                teacher["masks"].float(),
                size=student["masks"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        valid = self._valid_mask(
            batch, annotations["img_ids"], student["masks"].shape[-2:]
        )
        total_loss, logs = mask_distillation_loss(
            student["masks"],
            teacher["masks"],
            student["scores"],
            teacher["scores"],
            student["queries"],
            teacher["queries"],
            self.cfg,
            valid,
        )
        logs["prompt_count"] = total_loss.new_tensor(float(len(annotations["boxes"])))
        logs["box_prompt"] = total_loss.new_tensor(float(prompt_type == "box"))
        return total_loss, logs

    @torch.no_grad()
    def predict_student_masks(self, batch: BatchedDatapoint, prompt_type: str):
        """Run prompted student-only inference and return masks aligned with GT masks."""
        annotations = self._select_annotations(batch, max_prompts=0)
        if annotations["segments"] is None or annotations["segment_valid"] is None:
            raise NoValidAnnotationsError("Validation annotations have no masks.")
        keep = annotations["segment_valid"].bool()
        if not bool(keep.any()):
            raise NoValidAnnotationsError("Validation annotations have no valid masks.")
        annotations = self._index_annotations(annotations, keep)

        images = getattr(batch, "student_img_batch", batch.img_batch)
        backbone = self._encode_inputs(
            self.student_model, images, batch.find_text_batch
        )
        eval_cfg = self.cfg.get("eval", {})
        chunk_size = int(eval_cfg.get("prompt_chunk_size", 8))
        chunk_size = len(annotations["boxes"]) if chunk_size <= 0 else chunk_size
        point_sampling = str(eval_cfg.get("point_sampling", "center"))
        predictions, targets = [], []

        for start in range(0, len(annotations["boxes"]), chunk_size):
            chunk = self._index_annotations(
                annotations, slice(start, start + chunk_size)
            )
            prompt = self._build_prompt(chunk, prompt_type, point_sampling)
            output = self._decode_prompts(
                self.student_model, backbone, chunk, prompt
            )
            selected = self._gather_output(output)
            masks = selected["masks"].float()
            gt_masks = F.interpolate(
                chunk["segments"].float().unsqueeze(1),
                size=masks.shape[-2:],
                mode="nearest",
            ).bool()
            valid = self._valid_mask(
                batch, chunk["img_ids"], masks.shape[-2:]
            )
            if valid is not None:
                masks = masks.masked_fill(valid < 0.5, -20.0)
                gt_masks = gt_masks & valid.bool()
            predictions.append(masks.squeeze(1))
            targets.append(gt_masks.squeeze(1))

        return {
            "pred_masks": torch.cat(predictions),
            "gt_masks": torch.cat(targets),
        }


mask_Distill = MaskDistill
