import timm

model = timm.create_model(
    "hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m",
    pretrained=True,
    num_classes=0
)