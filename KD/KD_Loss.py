from typing import List, Dict, Optional, Any, Sequence
from dataclasses import dataclass
import torch
from torch.nn import functional as F


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


def warp_feature_to_valid_region(
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        student_valid_boxes: torch.Tensor,
        teacher_valid_boxes: torch.Tensor,
) -> torch.Tensor:
    """Warp a student's valid feature region into the teacher's canvas coordinates."""
    if student_feature.shape[0] != teacher_feature.shape[0]:
        raise ValueError("Student and teacher feature batches must have the same size.")

    student_feature = student_feature.float()
    _, _, _, _ = student_feature.shape
    target_height, target_width = teacher_feature.shape[-2:]
    device = student_feature.device
    dtype = student_feature.dtype
    student_boxes = student_valid_boxes.to(device=device, dtype=dtype)
    teacher_boxes = teacher_valid_boxes.to(device=device, dtype=dtype)

    target_x = (
        (torch.arange(target_width, device=device, dtype=dtype) + 0.5)
        / target_width
    )
    target_y = (
        (torch.arange(target_height, device=device, dtype=dtype) + 0.5)
        / target_height
    )
    student_left, student_top, student_right, student_bottom = student_boxes.unbind(1)
    teacher_left, teacher_top, teacher_right, teacher_bottom = teacher_boxes.unbind(1)
    target_widths = (teacher_right - teacher_left).clamp_min(torch.finfo(dtype).eps)
    target_heights = (teacher_bottom - teacher_top).clamp_min(torch.finfo(dtype).eps)

    u = (target_x[None, :] - teacher_left[:, None]) / target_widths[:, None]
    v = (target_y[None, :] - teacher_top[:, None]) / target_heights[:, None]
    source_x = student_left[:, None] + u * (student_right - student_left)[:, None]
    source_y = student_top[:, None] + v * (student_bottom - student_top)[:, None]
    grid_x = source_x[:, None, :].expand(-1, target_height, -1) * 2 - 1
    grid_y = source_y[:, :, None].expand(-1, -1, target_width) * 2 - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)

    warped = F.grid_sample(
        student_feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    target_mask = strict_valid_region_mask(
        teacher_boxes, (target_height, target_width)
    ).to(dtype=warped.dtype)
    return warped * target_mask


def warp_feature_list_to_teacher_regions(
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
        student_valid_boxes: torch.Tensor,
        teacher_valid_boxes: torch.Tensor,
) -> list[torch.Tensor]:
    if len(student_features) != len(teacher_features):
        raise ValueError(
            "Student and teacher feature lists must have the same number of layers."
        )
    return [
        warp_feature_to_valid_region(
            student_feature,
            teacher_feature,
            student_valid_boxes,
            teacher_valid_boxes,
        )
        for student_feature, teacher_feature in zip(student_features, teacher_features)
    ]

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
    loss_type: 采用什么类型的方法来计算损失函数[cosine,L1,L2,SmoothL1]
    avg:在计算完成每一层的损失之后是采用每层的损失相加还是平均,True表示相加
    vaild_masks: mask用来表示哪一些部分包含了图像的padding
    """
    if len(student_features) != len(teacher_features):
        assert f'student features length {len(student_features)} is different from teacher features length {len(teacher_features)} '

    if valid_masks is not None and len(valid_masks) != len(student_features):
        raise ValueError("valid_masks must match the number of feature layers.")

    losses = 0

    for i in range(len(student_features)):
        student_feature = student_features[i]
        teacher_feature = teacher_features[i]
        if student_feature.shape != teacher_feature.shape:
            assert f'student_feature {student_feature.shape} is different from teacher_feature {teacher_feature.shape} '

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
            if loss_type == 'cosine':
                per_pixel_loss = 1.0 - F.cosine_similarity(
                    student_feature, teacher_feature, dim=1
                )
            elif loss_type == 'l1':
                per_pixel_loss = F.l1_loss(
                    student_feature, teacher_feature, reduction='none'
                ).mean(dim=1)
            elif loss_type == 'l2':
                per_pixel_loss = F.mse_loss(
                    student_feature, teacher_feature, reduction='none'
                ).mean(dim=1)
            elif loss_type == 'smooth_l1':
                per_pixel_loss = F.smooth_l1_loss(
                    student_feature, teacher_feature, reduction='none'
                ).mean(dim=1)
            else:
                raise ValueError(f'{loss_type} is invalid loss type')
            valid_mask = valid_mask[:, 0].to(
                device=per_pixel_loss.device, dtype=per_pixel_loss.dtype
            )
            losses += (
                (per_pixel_loss * valid_mask).sum()
                / valid_mask.sum().clamp_min(1)
            )
            continue

        loss = -1
        if loss_type == 'cosine':
            loss = 1.0 - F.cosine_similarity(student_feature, teacher_feature, dim=1).mean() #dim = 1 -> [B,C,H,W] 中的C
        if loss_type == 'l1':
            loss = F.l1_loss(student_feature, teacher_feature)
        if loss_type == 'l2':
            loss = F.mse_loss(student_feature, teacher_feature)
        if loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(student_feature, teacher_feature)
        if loss == -1:
            assert f'{loss_type} is invalid loss type'

        losses += loss

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
) -> Dict[str, float]:
    """
    用于计算不同方法下得到的损失值，传入的losses_type是List类型，最后返回Dict类型
    student_feature: 学生网络特征
    teacher_feature: 教师网络特征
    losses_type: 采用那些类型的方法来计算损失函数，目前只定义了[cosine,L1,L2,SmoothL1]四类
    cfg:配置文件，用于取配置文件中各个不同方法的权重系数
    avg:在计算完成每一层的损失之后是采用每层的损失相加还是平均,True表示相加
    """
    losses = {}
    features_total_loss = 0
    if len(losses_type) == 0:
        assert f'loss_type with the length of 0'

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
