from typing import Sequence, Dict

import torch
import argparse
from torch import nn
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.geometry_encoders import Prompt

def forward_from_neck(
    model: nn.Module,
    images: torch.Tensor,
    neck_features: Sequence[torch.Tensor],
    batch: BatchedDatapoint,
) -> Dict[str, torch.Tensor]:
    if len(batch.find_inputs) != 1:
        raise ValueError(
            "Logit distillation currently expects one find stage per batch."
        )

    vision_backbone = model.backbone.vision_backbone

    # 根据已经得到的neck特征，补充SAM3下游所需的位置编码
    position_features = [
        vision_backbone.position_encoding(feature).to(feature.dtype)
        for feature in neck_features
    ]

    backbone_out = {
        "img_batch_all_stages": images,
        "vision_features": neck_features[-1],
        "vision_pos_enc": position_features,
        "backbone_fpn": list(neck_features),
        "sam2_backbone_out": None,
    }

    # 从batch.find_text_batch中编码文字
    text_out = model.backbone.forward_text(
        batch.find_text_batch,
        device=model.device,
    )
    backbone_out.update(text_out)

    find_input = batch.find_inputs[0]
    find_target = batch.find_targets[0]

    geometric_prompt = Prompt(
        box_embeddings=find_input.input_boxes,
        box_mask=find_input.input_boxes_mask,
        box_labels=find_input.input_boxes_label,
    )

    return model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        find_target=find_target,
        geometric_prompt=geometric_prompt,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference from neck")
    parser.add_argument('--neck_file', type=str, default='neck.pth')