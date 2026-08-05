from pathlib import Path
from torchvision import transforms
import torch
from PIL import Image

from dinov2.hub.backbones import dinov2_vitl14_reg

def load_model(weights_path: str, device: torch.device):
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Cannot find weights file: {weights_path}")

    model = dinov2_vitl14_reg(pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model

def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

def load_image(image_path: str, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return build_transform(image_size)(image).unsqueeze(0)


@torch.inference_mode()
def infer(model, image_tensor: torch.Tensor, device: torch.device):
    image_tensor = image_tensor.to(device)
    return model.get_intermediate_layers(
        image_tensor,
        n=[5, 11, 17, 23],#返回的中间层
        reshape=True,
        return_class_token=True,
    )

if __name__ == "__main__":
    pretrained_root = r"E:\reproduce\weights\DINO-V2\dinov2_vitl14_reg4_pretrain.pth"
    img_root = r"E:\my_data\rf100-vl\apex-videogame\train\5--4-_png_jpg.rf.3d8e99f81717ebacbb8eaa6b3278251e.jpg"
    img_size = 1008
    img = load_image(img_root, img_size)
    pitch_size = 14
    model = load_model(pretrained_root,torch.device("cuda:0"))
    features = infer(model,img,torch.device("cuda:0"))
    print(len(features))
    for feature,cls in features:
        print(feature.shape)
        print(cls.shape)



