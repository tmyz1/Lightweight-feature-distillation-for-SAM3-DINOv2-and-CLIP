from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch
from torch.nn import functional as F

def student_and_teacher_features_loss(
        student_features: List,
        teacher_features: List,
        loss_type:str = 'cosine',
        avg:bool = True
):
    """
    用于计算学生网络和教师网络在特征层面上的损失
    student_feature: 学生网络特征
    teacher_feature: 教师网络特征
    loss_type: 采用什么类型的方法来计算损失函数[cosine,L1,L2,SmoothL1]
    avg:在计算完成每一层的损失之后是采用每层的损失相加还是平均,True表示相加
    """
    if len(student_features) != len(teacher_features):
        assert f'student features length {len(student_features)} is different from teacher features length {len(teacher_features)} '

    losses = 0

    for i in range(len(student_features)):
        student_feature = student_features[i]
        teacher_feature = teacher_features[i]
        if student_feature.shape != teacher_feature.shape:
            assert f'student_feature {student_feature.shape} is different from teacher_feature {teacher_feature.shape} '

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
        loss = student_and_teacher_features_loss(student_features, teacher_features, loss_type, avg)
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