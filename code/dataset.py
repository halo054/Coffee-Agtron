from torch.utils.data import Dataset
from torchvision import transforms
import json
from PIL import Image
import torch

class Agtron_Dataset(Dataset):
    # dataset_json_path is path to data.json
    # split_json_path is path to split.json
    # data_type can be either "Bean" or "Powder"
    def __init__(self, dataset_json_path, split_json_path, data_type ='Bean', split='test'):
        self.split = split
        with open(dataset_json_path) as f:
            data = json.load(f)
        with open(split_json_path) as f:
            split_json = json.load(f)
        image_paths  = split_json[data_type][split]
        which_sample = []
        for path in image_paths:
            sample = path.split("/")[0]
            which_sample.append(sample)
        
        # Average the agtron reads for a sample to generate a single value.
        sample_number_to_agtron_dict = {}
        for sample in data.keys():
            sample_number_to_agtron_dict[sample] = data[sample]["Agtron_value"][data_type]
            sample_number_to_agtron_dict[sample] = round(sum(sample_number_to_agtron_dict[sample]) / len(sample_number_to_agtron_dict[sample]),2)
        
        self.entries = []
        for index in range(len(image_paths)):
            agtron_value = sample_number_to_agtron_dict[which_sample[index]]
            data_point = (image_paths[index],agtron_value)
            self.entries.append(data_point)
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),   
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.RandomRotation(5),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,hue=0.02),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05,hue=0.01),
            transforms.Resize((224, 224)),     
            transforms.ToTensor()
            ])
        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),   
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.Resize((224, 224)),     
            transforms.ToTensor()
            ])
        self.train_transform = train_transform
        self.test_transform = test_transform
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        path = '../' + self. entries[idx][0]
        img = Image.open(path).convert("RGB")
        if "train" in self.split:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        label = torch.tensor(self.entries[idx][1], dtype=torch.float32)
        return (img,label)
