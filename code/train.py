import os
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import Agtron_Dataset
import torch.optim as optim

class Non_BN_resnet_regression(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet18(weights = "DEFAULT")

        def replace_bn(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module,name,nn.Identity())
                else:
                    replace_bn(child)
                
        replace_bn(model)
        model.fc = nn.Linear(512 , 1)
        self.model = model
    def forward(self, image):
        return self.model(image)


class Trainer:
    def __init__(self,ckpt_path = '',batch_size = 32):
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.setup_environment()
        self.build_dataloaders()
        self.build_model()
        self.setup_optimization()
        
    def setup_environment(self):
        """Setup random seeds and computing device"""
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
    def build_model(self):
        
        self.model = Non_BN_resnet_regression()
        if len(self.ckpt_path) >=1:
            ckpt = torch.load(self.ckpt_path, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            print("loading "+ self.ckpt_path)
        #self.model.requires_grad_(False)
        #self.model.model.fc.requires_grad_(True)
        self.model.requires_grad_(True)
        self.model = self.model.to("cuda")
        
        
    def build_dataloaders(self):
        dataset_json_path = "../data.json"
        split_json_path = "../split.json"
        dataset = Agtron_Dataset(dataset_json_path, split_json_path, data_type ='Bean', split='train')
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size,shuffle=True)
        
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler"""
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
                trainable,
                lr=5e-5,
                weight_decay=1e-2
            )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50, #self.args.epochs
            eta_min=1e-6
        )
        self.MSELoss = nn.MSELoss(reduction = "sum")
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        
        for batch in tqdm(self.dataloader):
            images_data,true_agtron = batch
            
            images_data = images_data.to("cuda")
            true_agtron = true_agtron.to("cuda")
            # forward propagation
            self.optimizer.zero_grad()
            inferred_agtron = self.model(images_data).squeeze(-1)  
            
            loss = self.MSELoss(inferred_agtron,true_agtron)
            loss.backward()
            self.optimizer.step()
            print(loss.item()/len(batch[0]))
            print("len(batch)",len(batch[0]))
            running_loss += loss.item()

        # calculate average loss
        epoch_loss = running_loss / len(self.dataloader.dataset)
        self.train_losses.append(epoch_loss)

        return epoch_loss
    

    def train(self):
        """Main training loop"""
        #self.train_start = time.time()
        self.train_losses = []
        for epoch in range(500):
            # Training
            #self.epoch_start = time.time() 
            #print(f"Start of epoch {epoch}:",self.epoch_start - self.train_start)
            train_loss = self.train_epoch(epoch)

            # Update learning rate
            self.scheduler.step()

            print(f"Epoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}")
            print()
            self.save_checkpoint(epoch,last_time_epoch_saved = 0)
            
    def save_checkpoint(self, epoch,last_time_epoch_saved):
        """Save model checkpoint"""
        # Save checkpoint for current epoch
        
        checkpoint = {
            'epoch': epoch + last_time_epoch_saved+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }
        

        torch.save(
            checkpoint,
            os.path.join("../checkpoints/", f'mild_jitter_checkpoint_epoch_full_model_{epoch + last_time_epoch_saved+1}.pth')
            )
        print(f'mild_jitter_checkpoint_epoch_full_model_{epoch + last_time_epoch_saved+1}.pth saved')
        '''
        if get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.ego_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
        

            torch.save(
                checkpoint,
                osp.join("/cmlscratch/xyu054/Imagebind/ImageBind-LoRA/ckpt/", f'checkpoint_epoch_{epoch}.pth')
                )
                '''
def main():
    
    exo_dir = "embeddings/exo-training/"
    # Initialize trainer
    trainer = Trainer(ckpt_path = "/cmlscratch/xyu054/Agtron/checkpoints/no_jitter_checkpoint_epoch_full_model_67.pth",batch_size = 32)
    #trainer.save_checkpoint(0)
    # Run training or evaluation

    trainer.train()


if __name__ == '__main__':
    main()