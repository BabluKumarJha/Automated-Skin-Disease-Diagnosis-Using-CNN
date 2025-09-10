import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

# data path
train_dir = Path("C:\\Users\\BKJST\\Desktop\\python\\Project\\skin disease\\output_dataset\\train")
test_dir = Path("C:\\Users\\BKJST\\Desktop\\python\\Project\\skin disease\\output_dataset\\val")


#######################
# Creating class for transformation
######################
class transform:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir

    # function for train transform.
    def train_transform(self):
        train_transform = transforms.Compose(
            [
                transforms.Resize((128,128)),
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast =0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.22, 0.22, 0.22]),
            ]
        )
        return train_transform

    # create test_transfrom function
    def test_transform(self):
        test_transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.22, 0.22, 0.22])
            ]
        )
        return  test_transform

    # create train_dataset function.
    def train_dataset(self):
        train_transform = self.train_transform()
        train_dataset = torchvision.datasets.ImageFolder(
            train_dir, train_transform
        )
        return train_dataset

    # function for test_dataset
    def test_dataset(self):
        test_transform = self.test_transform()
        test_dataset = torchvision.datasets.ImageFolder(test_dir,
                                                         transform = test_transform)
        return test_dataset

    # function for train dataloader.
    def train_dataloader(self):
        train_dataset = self.train_dataset()
        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle= True, num_workers = os.cpu_count())
        return train_dataloader

    # create function for test dataloder.
    def test_dataloader(self):
        test_dataset = self.test_dataset()
        test_dataloader = DataLoader(test_dataset, batch_size= 256, shuffle = False, num_workers = os.cpu_count())
        return test_dataloader





# Instantiate the class
data = transform(train_dir, test_dir)

# Create datasets and dataloaders
train_dataset = data.train_dataset()
test_dataset = data.test_dataset()
train_dataloader = data.train_dataloader()
test_dataloader = data.test_dataloader()

# ✅ Get class info
class_names = train_dataset.classes
class_to_idx = train_dataset.class_to_idx


# print(f"Class Names: {class_names} \n Class index: { class_to_idx}")