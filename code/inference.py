import os
import json
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from dataset import Agtron_Dataset
from train import Non_BN_resnet_regression
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

model = Non_BN_resnet_regression()
ckpt = torch.load("/cmlscratch/xyu054/Agtron/checkpoints/mild_jitter_checkpoint_epoch_full_model_98.pth", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
image_path = "/cmlscratch/xyu054/Agtron/Sample_1/Bean_6.JPG" # Replace with your image path
#image_path = "/path/to/your/image.jpg" # Replace with your image path
img = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
            transforms.Resize((224, 224)),     
            transforms.ToTensor()
            ])
img = transform(img).unsqueeze(0) 

img = img.to("cuda")
model.to("cuda")
model.eval()

inferred_agtron = float(model(img).detach().cpu().squeeze(-1))

print(inferred_agtron)
