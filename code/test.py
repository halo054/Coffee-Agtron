import os
import json
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from dataset import Agtron_Dataset
from train import Non_BN_resnet_regression
from tqdm import tqdm
model = Non_BN_resnet_regression()
ckpt = torch.load("/cmlscratch/xyu054/Agtron/checkpoints/mild_jitter_checkpoint_epoch_full_model_98.pth", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
dataset_json_path = "../data.json"
split_json_path = "../split.json"
dataset = Agtron_Dataset(dataset_json_path, split_json_path, data_type ='Bean', split='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1,shuffle=True)
model.to("cuda")
model.eval()
running_loss = 0.0
MSEloss = nn.MSELoss()
for batch in tqdm(dataloader):
    images_data,true_agtron = batch
    images_data = images_data.to("cuda")
    true_agtron = true_agtron.to("cuda")
    inferred_agtron = model(images_data).squeeze(-1)  
    loss = MSEloss(inferred_agtron,true_agtron)
    print(true_agtron)
    print(inferred_agtron)
    running_loss += loss.item()
full_loss = running_loss / len(dataloader)
print(full_loss)
