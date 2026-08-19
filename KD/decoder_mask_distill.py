from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from KD.model.vit_small_patch14_reg4_dinov2 import (
    build_vit_small_image_model,
    replace_vit_small_memory_efficient_attention,
)
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type in {path}: {type(checkpoint)!r}")
    for key in ("student_model", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    return checkpoint


def _masked_mean(value: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    if valid is None:
        return value.mean()
    valid = valid.to(dtype=value.dtype)
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(1)
    valid = valid.expand_as(value)
    return (value * valid).sum() / valid.sum().clamp_min(1.0)


class NoValidAnnotationsError(ValueError):
    pass

"""
Distillation of the masks output by the student and teacher models
"""
class MaskDistill(nn.Module):
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.distill_cfg = cfg.get("mask_distill", {})
        self.loss_cfg = cfg.get("loss", {})

        sam3_cfg = cfg.get("Sam3", {})
        student_cfg = cfg.get("Student", {})
        bpe_path = sam3_cfg.get("bpe_path")
        teacher_checkpoint = sam3_cfg.get("checkpoint_path")
        if not teacher_checkpoint:
            raise ValueError("Sam3.checkpoint_path is required for mask distillation.")

        #Loading teacher and student models
        self.teacher_model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=str(self.device),
            eval_mode=True,
            checkpoint_path=teacher_checkpoint,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        self.student_model = build_vit_small_image_model(
            bpe_path=bpe_path,
            device=str(self.device),
            eval_mode=True,
            checkpoint_path=teacher_checkpoint,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            vit_small_checkpoint_path=student_cfg.get("vit_small_checkpoint_path"),
            return_interm_layers=False,
            vit_small_intermediate_layers=student_cfg.get(
                "vit_small_intermediate_layers", (2, 5, 8, 11)
            ),
            cfg=cfg,
        )
        replace_vit_small_memory_efficient_attention(self.student_model)

        initial_checkpoint = student_cfg.get("initial_checkpoint")
        if not initial_checkpoint:
            raise ValueError(
                "Student.initial_checkpoint must point to the feature-distilled "
                "ViT-Small SAM3 checkpoint."
            )
        student_state = _load_state_dict(initial_checkpoint)
        missing, unexpected = self.student_model.load_state_dict(student_state, strict=False)
        allowed_missing_prefixes = ("segmentation_head.",)
        disallowed_missing = [
            key for key in missing if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                f"Student checkpoint is incompatible: {len(disallowed_missing)} "
                f"disallowed missing and {len(unexpected)} unexpected keys. "
                f"Missing examples: {disallowed_missing[:5]}; unexpected examples: "
                f"{unexpected[:5]}"
            )
        if missing:
            print(
                f"[model] feature checkpoint has no segmentation head; retained "
                f"{len(missing)} teacher-initialized segmentation parameters",
                flush=True,
            )
        print(f"[model] loaded feature-distilled student: {initial_checkpoint}", flush=True)

        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)
        self.teacher_model.eval()

        trainable_modules = tuple(
            self.distill_cfg.get(
                "trainable_modules",
                (
                    "geometry_encoder",
                    "transformer",
                    "segmentation_head",
                    "dot_prod_scoring",
                ),
            )
        )
        for name, parameter in self.student_model.named_parameters():
            parameter.requires_grad_(any(name.startswith(prefix) for prefix in trainable_modules))

        trainable = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student_model.parameters())
        if trainable == 0:
            raise RuntimeError("mask_distill.trainable_modules selected no parameters.")
        print(
            f"[model] trainable student parameters: {trainable:,}/{total:,} "
            f"({100.0 * trainable / total:.2f}%)",
            flush=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep teacher and student on the same deterministic grounding path. Gradients
        # still flow through every selected student parameter while SAM3 target matching
        # and training-only DAC outputs remain disabled.
        self.teacher_model.eval()
        self.student_model.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def get_student_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.student_model.state_dict()

    def _select_annotations(self, batch: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        if len(batch.find_inputs) != 1 or len(batch.find_targets) != 1:
            raise ValueError("MaskDistill currently expects one SAM3 find stage per batch.")

        find_input = batch.find_inputs[0]
        target = batch.find_targets[0]
        counts = target.num_boxes.long()
        parent_queries = torch.repeat_interleave(
            torch.arange(len(counts), device=counts.device), counts
        )
        boxes = target.boxes.reshape(-1, 4).float()
        if len(boxes) != len(parent_queries):
            raise RuntimeError(
                f"Packed target mismatch: {len(boxes)} boxes for "
                f"{len(parent_queries)} query assignments."
            )

        valid = (boxes[:, 2:] > 0).all(dim=1)
        packed_indices = torch.arange(len(boxes), device=boxes.device)[valid]
        boxes = boxes[valid].clamp(0.0, 1.0)
        parent_queries = parent_queries[valid]
        if boxes.numel() == 0:
            raise NoValidAnnotationsError("The batch contains no valid annotated boxes.")

        max_prompts = int(self.distill_cfg.get("max_prompts_per_batch", 4))
        if max_prompts > 0 and len(boxes) > max_prompts:
            if self.training:
                chosen = torch.randperm(len(boxes), device=boxes.device)[:max_prompts]
            else:
                chosen = torch.arange(max_prompts, device=boxes.device)
            boxes = boxes[chosen]
            parent_queries = parent_queries[chosen]
            packed_indices = packed_indices[chosen]

        segments = target.segments
        selected_segments = None
        valid_segments = None
        if segments is not None and len(segments) > 0:
            selected_segments = segments[packed_indices]
            if target.is_valid_segment is None:
                valid_segments = torch.ones(
                    len(selected_segments), device=boxes.device, dtype=torch.bool
                )
            else:
                valid_segments = target.is_valid_segment[packed_indices].bool()

        return {
            "boxes": boxes,
            "parent_queries": parent_queries,
            "img_ids": find_input.img_ids[parent_queries],
            "text_ids": find_input.text_ids[parent_queries],
            "packed_indices": packed_indices,
            "segments": selected_segments,
            "valid_segments": valid_segments,
        }

    def _points_from_annotations(
        self,
        boxes: torch.Tensor,
        segments: Optional[torch.Tensor],
        valid_segments: Optional[torch.Tensor],
    ) -> torch.Tensor:
        points = boxes[:, :2].clone()
        if segments is None:
            return points

        sampling = str(self.distill_cfg.get("point_sampling", "random")).lower()
        for index, segment in enumerate(segments):
            if valid_segments is not None and not bool(valid_segments[index]):
                continue
            candidates = segment.bool().nonzero(as_tuple=False)
            if candidates.numel() == 0:
                continue
            if sampling == "center":
                center_yx = candidates.float().mean(dim=0)
            elif sampling == "random":
                selected = torch.randint(len(candidates), (1,), device=candidates.device)
                center_yx = candidates[selected.item()].float()
            else:
                raise ValueError(
                    f"Unsupported mask_distill.point_sampling={sampling!r}; "
                    "use 'random' or 'center'."
                )
            height, width = segment.shape[-2:]
            points[index, 0] = (center_yx[1] + 0.5) / width
            points[index, 1] = (center_yx[0] + 0.5) / height
        return points.clamp(0.0, 1.0)

    def _choose_prompt_type(self, forced: Optional[str]) -> str:
        prompt_types = [
            str(value).lower()
            for value in self.distill_cfg.get("prompt_types", ["box"])
        ]
        invalid = set(prompt_types) - {"box", "point"}
        if not prompt_types or invalid:
            raise ValueError(
                f"mask_distill.prompt_types must contain box and/or point; got {prompt_types}."
            )
        if forced is not None:
            forced = forced.lower()
            if forced not in prompt_types:
                raise ValueError(f"Prompt type {forced!r} is not enabled in {prompt_types}.")
            return forced
        if len(prompt_types) == 1 or not self.training:
            return prompt_types[0]
        index = int(torch.randint(len(prompt_types), (1,)).item())
        return prompt_types[index]

    def _build_prompt(
        self, annotations: Dict[str, torch.Tensor], prompt_type: str
    ) -> Prompt:
        boxes = annotations["boxes"]
        num_prompts = len(boxes)
        labels = torch.ones(1, num_prompts, device=boxes.device, dtype=torch.long)
        mask = torch.zeros(num_prompts, 1, device=boxes.device, dtype=torch.bool)
        if prompt_type == "box":
            return Prompt(
                box_embeddings=boxes.unsqueeze(0),
                box_labels=labels,
                box_mask=mask,
            )

        points = self._points_from_annotations(
            boxes, annotations["segments"], annotations["valid_segments"]
        )
        return Prompt(
            point_embeddings=points.unsqueeze(0),
            point_labels=labels,
            point_mask=mask,
        )

    @staticmethod
    def _forward_grounding(
        model: nn.Module,
        images: torch.Tensor,
        text_batch,
        img_ids: torch.Tensor,
        text_ids: torch.Tensor,
        prompt: Prompt,
    ) -> Dict[str, torch.Tensor]:
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(model.backbone.forward_image(images.float()))
        backbone_out.update(model.backbone.forward_text(text_batch, device=model.device))
        prompt_input = SimpleNamespace(img_ids=img_ids, text_ids=text_ids)
        return model.forward_grounding(
            backbone_out=backbone_out,
            find_input=prompt_input,
            find_target=None,
            geometric_prompt=prompt,
        )

    @staticmethod
    def _gather_teacher_aligned(
        student_out: Dict[str, torch.Tensor], teacher_out: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        teacher_logits = teacher_out["pred_logits"].float()
        teacher_quality = (
            teacher_logits.squeeze(-1)
            if teacher_logits.shape[-1] == 1
            else teacher_logits.max(dim=-1).values
        )
        best_query = teacher_quality.argmax(dim=1)
        batch_index = torch.arange(len(best_query), device=best_query.device)

        student_masks = student_out["pred_masks"][batch_index, best_query].unsqueeze(1)
        teacher_masks = teacher_out["pred_masks"][batch_index, best_query].unsqueeze(1)
        if teacher_masks.shape[-2:] != student_masks.shape[-2:]:
            teacher_masks = F.interpolate(
                teacher_masks.float(),
                size=student_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        student_scores = student_out["pred_logits"][batch_index, best_query]
        teacher_scores = teacher_out["pred_logits"][batch_index, best_query]
        return {
            "student_masks": student_masks.float(),
            "teacher_masks": teacher_masks.float(),
            "student_scores": student_scores.float(),
            "teacher_scores": teacher_scores.float(),
            "student_queries": student_out["queries"][batch_index, best_query].float(),
            "teacher_queries": teacher_out["queries"][batch_index, best_query].float(),
        }

    @staticmethod
    def _valid_mask(
        batch: BatchedDatapoint,
        img_ids: torch.Tensor,
        size: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        valid_boxes = getattr(batch, "kd_valid_boxes", None)
        if valid_boxes is None:
            return None
        boxes = valid_boxes[img_ids]
        height, width = size
        valid = torch.zeros(len(boxes), 1, height, width, device=boxes.device)
        for index, box in enumerate(boxes):
            x0 = max(0, min(width, int(torch.floor(box[0] * width).item())))
            y0 = max(0, min(height, int(torch.floor(box[1] * height).item())))
            x1 = max(0, min(width, int(torch.ceil(box[2] * width).item())))
            y1 = max(0, min(height, int(torch.ceil(box[3] * height).item())))
            valid[index, :, y0:y1, x0:x1] = 1
        return valid

    def _compute_losses(
        self,
        aligned: Dict[str, torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        temperature = float(self.loss_cfg.get("temperature", 1.0))
        student_logits = aligned["student_masks"] / temperature
        teacher_prob = (aligned["teacher_masks"] / temperature).sigmoid().detach()
        student_prob = student_logits.sigmoid()
        losses: Dict[str, torch.Tensor] = {}

        bce_weight = float(self.loss_cfg.get("mask_bce_weight", 1.0))
        if bce_weight > 0:
            bce = F.binary_cross_entropy_with_logits(
                student_logits, teacher_prob, reduction="none"
            )
            losses["mask_bce_loss"] = (
                _masked_mean(bce, valid) * bce_weight * temperature**2
            )

        focal_weight = float(self.loss_cfg.get("mask_focal_weight", 0.0))
        if focal_weight > 0:
            gamma = float(self.loss_cfg.get("focal_gamma", 2.0))
            alpha = float(self.loss_cfg.get("focal_alpha", 0.25))
            ce = F.binary_cross_entropy_with_logits(
                student_logits, teacher_prob, reduction="none"
            )
            modulation = (student_prob - teacher_prob).abs().pow(gamma)
            alpha_factor = teacher_prob * alpha + (1.0 - teacher_prob) * (1.0 - alpha)
            losses["mask_focal_loss"] = (
                _masked_mean(ce * modulation * alpha_factor, valid)
                * focal_weight
                * temperature**2
            )

        dice_weight = float(self.loss_cfg.get("mask_dice_weight", 1.0))
        if dice_weight > 0:
            valid_float = 1.0 if valid is None else valid.to(student_prob.dtype)
            student_flat = (student_prob * valid_float).flatten(1)
            teacher_flat = (teacher_prob * valid_float).flatten(1)
            numerator = 2.0 * (student_flat * teacher_flat).sum(dim=1) + 1.0
            denominator = student_flat.sum(dim=1) + teacher_flat.sum(dim=1) + 1.0
            losses["mask_dice_loss"] = (1.0 - numerator / denominator).mean() * dice_weight

        score_weight = float(self.loss_cfg.get("score_mse_weight", 0.1))
        if score_weight > 0:
            losses["score_mse_loss"] = F.mse_loss(
                aligned["student_scores"], aligned["teacher_scores"].detach()
            ) * score_weight

        query_weight = float(self.loss_cfg.get("query_mse_weight", 0.0))
        if query_weight > 0:
            student_query = F.normalize(aligned["student_queries"], dim=-1)
            teacher_query = F.normalize(aligned["teacher_queries"].detach(), dim=-1)
            losses["query_mse_loss"] = F.mse_loss(student_query, teacher_query) * query_weight

        if not losses:
            raise ValueError("At least one mask distillation loss weight must be positive.")
        return losses

    @staticmethod
    def _binary_iou(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = prediction > 0
        tgt = target > 0
        if valid is not None:
            valid_bool = valid.bool()
            pred = pred & valid_bool
            tgt = tgt & valid_bool
        intersection = (pred & tgt).flatten(1).sum(dim=1).float()
        union = (pred | tgt).flatten(1).sum(dim=1).float()
        return (intersection / union.clamp_min(1.0)).mean()

    def _ground_truth_metrics(
        self,
        aligned: Dict[str, torch.Tensor],
        annotations: Dict[str, torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        segments = annotations["segments"]
        segment_valid = annotations["valid_segments"]
        if segments is None or segment_valid is None or not bool(segment_valid.any()):
            zero = aligned["student_masks"].new_zeros(())
            return {"student_gt_iou": zero, "teacher_gt_iou": zero, "gt_mask_count": zero}

        keep = segment_valid.bool()
        size = aligned["student_masks"].shape[-2:]
        gt = F.interpolate(
            segments[keep].float().unsqueeze(1), size=size, mode="nearest"
        )
        selected_valid = None if valid is None else valid[keep]
        return {
            "student_gt_iou": self._binary_iou(
                aligned["student_masks"][keep], gt, selected_valid
            ),
            "teacher_gt_iou": self._binary_iou(
                aligned["teacher_masks"][keep], gt, selected_valid
            ),
            "gt_mask_count": gt.new_tensor(float(len(gt))),
        }

    def forward(
        self,
        batch: BatchedDatapoint,
        prompt_type: Optional[str] = None,
    ):
        annotations = self._select_annotations(batch)
        selected_prompt_type = self._choose_prompt_type(prompt_type)
        prompt = self._build_prompt(annotations, selected_prompt_type)

        student_images = getattr(batch, "student_img_batch", batch.img_batch)
        teacher_images = getattr(batch, "sam3_img_batch", batch.img_batch)
        student_out = self._forward_grounding(
            self.student_model,
            student_images,
            batch.find_text_batch,
            annotations["img_ids"],
            annotations["text_ids"],
            prompt,
        )
        with torch.no_grad():
            teacher_out = self._forward_grounding(
                self.teacher_model,
                teacher_images,
                batch.find_text_batch,
                annotations["img_ids"],
                annotations["text_ids"],
                prompt,
            )

        aligned = self._gather_teacher_aligned(student_out, teacher_out)
        valid = self._valid_mask(
            batch, annotations["img_ids"], aligned["student_masks"].shape[-2:]
        )
        losses = self._compute_losses(aligned, valid)
        total_loss = sum(losses.values())
        logs = {"total_loss": total_loss, **losses}
        logs["teacher_student_iou"] = self._binary_iou(
            aligned["student_masks"], aligned["teacher_masks"], valid
        )
        logs.update(self._ground_truth_metrics(aligned, annotations, valid))
        logs["prompt_count"] = total_loss.new_tensor(float(len(annotations["boxes"])))
        logs["box_prompt"] = total_loss.new_tensor(float(selected_prompt_type == "box"))
        return total_loss, logs


# Keep the original draft's public name usable by older imports.
mask_Distill = MaskDistill
