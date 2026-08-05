import torch
from PIL import Image
from matplotlib import pyplot as plt

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_swin_sam3_image_model
from sam3.visualization_utils import plot_results

model = build_swin_sam3_image_model()
processor = Sam3Processor(model,resolution=1008)

img_root = r"E:\my_data\rf100-vl\apex-videogame\train\5--4-_png_jpg.rf.3d8e99f81717ebacbb8eaa6b3278251e.jpg"
img = Image.open(img_root)

with torch.autocast("cuda",dtype=torch.bfloat16):
    inference_state = processor.set_image(img)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="people")

plot_results(img, inference_state)
plt.show()