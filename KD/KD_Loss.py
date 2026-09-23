import sys
from typing import List, Dict, Optional, Any, Sequence
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from sam3.train.loss.loss_fns import dice_loss, sigmoid_focal_loss

class Generation_adapter(nn.Module):
    def __init__(self,high_s,high_t):
        super().__init__()
        self.high_s = high_s
        self.high_t = high_t
        self.teacher_dims = self.high_t.shape[1]
        self.mask_ratio = 0.5
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.teacher_dims))
        self.generation = nn.Sequential(
            nn.Conv2d(self.teacher_dims, self.teacher_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.teacher_dims, self.teacher_dims, kernel_size=3, padding=1))

    def forward(self, x, high_t):
        batch_size, channels, height, width = x.shape
        num_tokens = height * width

        # [B,C,H,W] -> [B,N,C]
        x = x.flatten(2).transpose(1, 2)
        high_t = high_t.flatten(2).transpose(1, 2)

        x, mask, ids_restore, ids_masked = self.random_masking(
            x, self.mask_ratio
        )

        mask_tokens = self.mask_token.to(
            device=x.device,
            dtype=x.dtype,
        ).expand(batch_size, num_tokens - x.shape[1], -1)

        x = torch.cat([x, mask_tokens], dim=1)

        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(
                -1, -1, channels
            ),
        )
        x = x.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )

        # [B,C,H,W] -> [B,N,C]
        x = self.generation(x).flatten(2).transpose(1, 2)

        # mask: [B,N] -> [B,N,1]
        token_mask = mask.unsqueeze(-1).to(
            device=x.device,
            dtype=x.dtype,
        )

        x = x * token_mask
        high_t = high_t.to(x.dtype) * token_mask

        # [B,N,C] -> [B,C,H,W]
        x = x.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        high_t = high_t.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )


        return x, high_t


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:L]

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked

def strict_valid_region_mask(
        valid_boxes: torch.Tensor,
        feature_size: Sequence[int],
) -> torch.Tensor:
    """Return feature cells fully covered by each normalized valid image box."""
    if valid_boxes.ndim != 2 or valid_boxes.shape[-1] != 4:
        raise ValueError(
            "valid_boxes must have shape [B, 4] as [left, top, right, bottom]."
        )

    height, width = feature_size
    device = valid_boxes.device
    dtype = valid_boxes.dtype
    x0, y0, x1, y1 = valid_boxes.unbind(dim=1)
    x_begin = torch.arange(width, device=device, dtype=dtype) / width
    x_end = (torch.arange(width, device=device, dtype=dtype) + 1) / width
    y_begin = torch.arange(height, device=device, dtype=dtype) / height
    y_end = (torch.arange(height, device=device, dtype=dtype) + 1) / height

    # A cell contributes only when its whole area belongs to the unpadded image.
    # Comparing cell boundaries avoids numerical errors from coverage products.
    epsilon = 1e-6
    x_valid = (x_begin[None] >= x0[:, None] - epsilon) & (
        x_end[None] <= x1[:, None] + epsilon
    )
    y_valid = (y_begin[None] >= y0[:, None] - epsilon) & (
        y_end[None] <= y1[:, None] + epsilon
    )
    return (x_valid[:, None, None, :] & y_valid[:, None, :, None]).to(dtype)


def build_feature_valid_masks(
        valid_boxes: Optional[torch.Tensor],
        features: Sequence[torch.Tensor],
) -> Optional[list[torch.Tensor]]:
    if valid_boxes is None:
        return None
    return [
        strict_valid_region_mask(valid_boxes, feature.shape[-2:])
        for feature in features
    ]


def _masked_mean(value: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return value.mean()
    valid_mask = valid_mask.to(device=value.device, dtype=value.dtype)
    return (value * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def mask_distillation_loss(
    student_masks: torch.Tensor,
    teacher_masks: torch.Tensor,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    student_queries: torch.Tensor,
    teacher_queries: torch.Tensor,
    cfg: Dict[str, Any],
    valid_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute all decoder-level distillation losses on aligned SAM3 outputs."""
    loss_cfg = cfg.get("loss", {})
    temperature = float(loss_cfg.get("temperature", 1.0))
    student_logits = student_masks.float() / temperature
    teacher_targets = (teacher_masks.float() / temperature).sigmoid().detach()
    num_masks = max(student_logits.shape[0], 1)
    losses: Dict[str, torch.Tensor] = {}

    if valid_mask is not None:
        valid_mask = valid_mask.to(device=student_logits.device, dtype=student_logits.dtype)

    bce_weight = float(loss_cfg.get("mask_bce_weight", 1.0))
    if bce_weight > 0:
        bce = F.binary_cross_entropy_with_logits(
            student_logits, teacher_targets, reduction="none"
        )
        losses["mask_bce_loss"] = (
            _masked_mean(bce, valid_mask) * bce_weight * temperature**2
        )

    focal_weight = float(loss_cfg.get("mask_focal_weight", 0.0))
    if focal_weight > 0:
        focal = sigmoid_focal_loss(
            student_logits,
            teacher_targets,
            num_boxes=num_masks,
            alpha=float(loss_cfg.get("focal_alpha", 0.25)),
            gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            reduce=False,
            triton=False,
        )
        losses["mask_focal_loss"] = (
            _masked_mean(focal, valid_mask) * focal_weight * temperature**2
        )

    dice_weight = float(loss_cfg.get("mask_dice_weight", 1.0))
    if dice_weight > 0:
        dice_inputs = student_logits
        dice_targets = teacher_targets
        if valid_mask is not None:
            dice_inputs = dice_inputs.masked_fill(valid_mask < 0.5, -20.0)
            dice_targets = dice_targets * valid_mask
        losses["mask_dice_loss"] = dice_loss(
            dice_inputs.flatten(1),
            dice_targets.flatten(1),
            num_boxes=num_masks,
        ) * dice_weight

    score_weight = float(loss_cfg.get("score_mse_weight", 0.1))
    if score_weight > 0:
        losses["score_mse_loss"] = F.mse_loss(
            student_scores.float(), teacher_scores.detach().float()
        ) * score_weight

    query_weight = float(loss_cfg.get("query_mse_weight", 0.0))
    if query_weight > 0:
        student_queries = F.normalize(student_queries.float(), dim=-1)
        teacher_queries = F.normalize(teacher_queries.detach().float(), dim=-1)
        losses["query_mse_loss"] = F.mse_loss(
            student_queries, teacher_queries
        ) * query_weight

    if not losses:
        raise ValueError("At least one mask distillation loss weight must be positive.")
    total_loss = sum(losses.values())
    return total_loss, {"total_loss": total_loss, **losses}

def student_and_teacher_features_loss(
        student_features: List,
        teacher_features: List,
        loss_type:str = 'cosine',
        avg:bool = True,
        valid_masks: Optional[Sequence[torch.Tensor]] = None,
):
    """
    用于计算学生网络和教师网络在特征层面上的损失
    student_feature: 学生网络特征
    teacher_feature: 教师网络特征
    loss_type: 采用什么类型的方法来计算损失函数[cosine,L1,L2,SmoothL1,MSE]
    avg:在计算完成每一层的损失之后是采用每层的损失相加还是平均,True表示相加
    vaild_masks: mask用来表示哪一些部分包含了图像的padding
    """
    if len(student_features) != len(teacher_features):
        raise ValueError(
            f"student features length {len(student_features)} is different from "
            f"teacher features length {len(teacher_features)}"
        )

    if valid_masks is not None and len(valid_masks) != len(student_features):
        raise ValueError("valid_masks must match the number of feature layers.")

    losses = 0

    for i in range(len(student_features)):
        student_feature = student_features[i]
        teacher_feature = teacher_features[i]
        if student_feature.shape != teacher_feature.shape:
            raise ValueError(
                f"student_feature {student_feature.shape} is different from "
                f"teacher_feature {teacher_feature.shape}"
            )

        valid_mask = None if valid_masks is None else valid_masks[i]
        if valid_mask is not None:
            expected_mask_shape = (
                student_feature.shape[0], 1, *student_feature.shape[-2:]
            )
            if tuple(valid_mask.shape) != expected_mask_shape:
                raise ValueError(
                    f"valid mask shape {tuple(valid_mask.shape)} does not match "
                    f"feature shape {tuple(student_feature.shape)}."
                )
        if i == len(student_features) - 1 or i == len(student_features) - 2:
            generation = Generation_adapter(
                high_s = student_feature,
                high_t = teacher_feature,
            ).to(device=student_feature.device,dtype=student_feature.dtype)
            student_feature, teacher_feature = generation(student_feature, teacher_feature)

        # Keep the channel dimension reduction independent from the valid-area
        # reduction, so every loss ignores the same padded feature cells.
        if loss_type == 'cosine':
            per_pixel_loss = 1.0 - F.cosine_similarity(
                student_feature, teacher_feature, dim=1
            )
        elif loss_type == 'l1':
            per_pixel_loss = F.l1_loss(
                student_feature, teacher_feature, reduction='none'
            ).mean(dim=1)
        elif loss_type in ('l2', 'mse'):
            per_pixel_loss = F.mse_loss(
                student_feature, teacher_feature, reduction='none'
            ).mean(dim=1)
        elif loss_type == 'smooth_l1':
            per_pixel_loss = F.smooth_l1_loss(
                student_feature, teacher_feature, reduction='none'
            ).mean(dim=1)
        else:
            raise ValueError(f'{loss_type} is invalid loss type')

        if valid_mask is not None:
            valid_mask = valid_mask[:, 0].to(
                device=per_pixel_loss.device, dtype=per_pixel_loss.dtype
            )
            per_image_loss = (per_pixel_loss * valid_mask).flatten(1).sum(dim=1)
            per_image_loss = per_image_loss / valid_mask.flatten(1).sum(dim=1).clamp_min(1)
            losses += per_image_loss.mean()
            continue

        losses += per_pixel_loss.mean()

    if avg:
        return losses / len(student_features)
    else:
        return losses

def student_and_teacher_cls_token_loss(
        student_cls_tokens: torch.Tensor,
        teacher_cls_tokens: torch.Tensor,
        loss_type:str = "cosine",
        avg:bool = True
):
    """
    用于计算学生网络和教师网络之间的cls_token损失，cls_token：特殊token用于表示全局语义向量
    student_cls_tokens: 学生网络特征
    teacher_cls_tokens: 教师网络特征
    loss_type: 采用什么类型的方法来计算损失函数[cosine,L1,L2,SmoothL1]
    avg:在计算完成每一层的损失之后是采用每层的损失相加还是平均,True表示相加
    """
    if len(student_cls_tokens) != len(teacher_cls_tokens):
        assert f'student_cls_tokens len({len(student_cls_tokens)}) is different from  teacher_cls_tokens len{len(teacher_cls_tokens)}'

    losses = 0

    for i in range(len(student_cls_tokens)):
        student_cls_token = student_cls_tokens[i]
        teacher_cls_token = teacher_cls_tokens[i]
        if student_cls_token.shape != teacher_cls_token.shape:
            assert f'student_cls_token shape{student_cls_token.shape} is different from teacher_cls_token shape {teacher_cls_token.shape} '

        loss = -1
        if loss_type == 'cosine':
            loss = 1.0 - F.cosine_similarity(student_cls_token, teacher_cls_token, dim=1).mean()
        if loss_type == 'l1':
            loss = F.l1_loss(student_cls_token, teacher_cls_token)
        if loss_type == 'l2':
            loss = F.mse_loss(student_cls_token, teacher_cls_token)
        if loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(student_cls_token, teacher_cls_token)

        if loss == -1:
            assert f'{loss_type} is invalid loss type'

        losses += loss

    if avg:
        return losses / len(student_cls_tokens)
    return losses


def features_total_loss(
        student_features: List,
        teacher_features: List,
        losses_type:List[str],
        cfg:Optional[Dict[str, Any]],
        avg:bool = True,
        valid_masks: Optional[Sequence[torch.Tensor]] = None,
        feature_type: str = 'backbone'
) -> Dict[str, float]:
    """
    用于计算不同方法下得到的损失值，传入的losses_type是List类型，最后返回Dict类型
    student_feature: 学生网络特征
    teacher_feature: 教师网络特征
    losses_type: 采用那些类型的方法来计算损失函数，目前定义了[cosine,L1,L2,SmoothL1,MSE]五类
    cfg:配置文件，用于取配置文件中各个不同方法的权重系数
    avg:在计算完成每一层的损失之后是采用每层的损失相加还是平均,True表示相加
    """
    losses = {}
    features_total_loss = 0
    if len(losses_type) == 0:
        assert f'loss_type with the length of 0'
    if feature_type == 'neck':
        s_list = []
        t_list = []
        m_list = []
        for i in range(len(student_features)):
            s_f = student_features[i]
            t_f = teacher_features[i]
            m = valid_masks[i]
            B,C,H,W = s_f.shape
            if H == 72 and W == 72:
                s_list.append(s_f)
                t_list.append(t_f)
                m_list.append(m)
                break
        student_features = s_list
        teacher_features = t_list
        valid_masks = m_list

    for loss_type in losses_type:
        loss = student_and_teacher_features_loss(
            student_features,
            teacher_features,
            loss_type,
            avg,
            valid_masks,
        )
        losses[loss_type] = loss
        weight = cfg.get('loss',{}).get('features_loss', {}).get(loss_type,{})
        if weight == {}:
            assert f'cfg do not have weight {loss_type}'
        features_total_loss = features_total_loss + loss * weight

    losses['total_loss'] = features_total_loss
    return losses

def cls_token_total_loss(
        student_cls_token: torch.Tensor,
        teacher_cls_token: torch.Tensor,
        losses_type:List[str],
        cfg:Optional[Dict[str, Any]],
)-> Dict[str, float]:
    """
    用于计算学生网络和教师网络之间不同方法的cls_token总损失,返回字典类型
    student_cls_token: 学生网络特征
    teacher_cls_token: 教师网络特征
    loss_type: 采用那些类型的方法来计算损失函数[cosine,L1,L2,SmoothL1]
    cfg:配置文件,用于取配置文件中各个不同方法的权重系数
    """
    losses = {}
    cls_total_loss = 0
    if len(losses_type) == 0:
        assert f'loss_type with the length of 0'

    for loss_type in losses_type:
        loss = student_and_teacher_cls_token_loss(student_cls_token,teacher_cls_token,loss_type)
        losses[loss_type] = loss
        weight = cfg.get('loss',{}).get('cls_token_loss', {}).get(loss_type,{})
        if weight == {}:
            assert f'cfg do not have weight {loss_type}'
        cls_total_loss = cls_total_loss + loss * weight

    losses['cls_total_loss'] = cls_total_loss

    return losses

if __name__ == '__main__':
    a = torch.arange(10).float()
    print(a)
