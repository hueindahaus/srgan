# Imports
from os.path import exists
from pathlib import Path

import math

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn # Includes all modules, nn.Linear, nn.Conv2d, BatchNorm etc
import torch.optim as optim # Is used for otimization algorithms such as Adam, SGD ...
from torch.utils.data import DataLoader # Helps with managing datasets in mini batches
from torch.utils.data import Dataset

import torchvision
import torchvision.datasets as datasets # Has standard datasets
import torchvision.transforms as transforms # Transformations to be used on images
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.models import vgg19
from itertools import chain
from PIL import Image


class DataHandler(Dataset):
    
    def __init__(self, datapath, high_res_size, low_res_size):
        datapath = Path(datapath)
        if not (datapath.exists() and datapath.is_dir()):
            raise ValueError(f"Data root '{root}' is invalid")
            
        self.datapath = datapath
        
        # Set transforms
        self.transform_both = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomRotation((-180, 180)), T.RandomCrop((high_res_size, high_res_size), pad_if_needed=True)])
        self.transform_low = T.Compose([T.Resize((low_res_size, low_res_size), Image.BICUBIC), T.ToTensor(), T.Normalize([0,0,0] ,[1,1,1])])
        self.transform_high = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) # NOTE: Normalization to range [-1,1] is only done for hr images according to paper
        
        # Collect samples
        self.samples = self.collect_samples()
            
    def __getitem__(self, index):
        # Access the stored path and label for the correct index
        img_path = self.samples[index]
        # Load the image into memory
        
        # Convert to image data with RGB (3 channels)
        high_res_img = Image.open(img_path).convert('RGB')
        high_res_img = self.transform_both(high_res_img)
        low_res_img = high_res_img.copy()
          
        # Perform transforms, if any.
        high_res_img = self.transform_high(high_res_img)
        low_res_img = self.transform_low(low_res_img)
        return high_res_img, low_res_img
    
    def __len__(self):
        return len(self.samples)
        
    
    def collect_samples(self):
        if not self.datapath.exists():
            raise ValueError(f"Data root '{self.datapath}' must contain sub dir '{self.datapath.name}'")
        
        # Finds all pathnames matching a specified pattern
        file_extensions = ['jpg', 'jpeg', 'png']
        img_paths = []
        for file_extension in file_extensions:
            img_paths += list(self.datapath.rglob("*." + file_extension))
            
        return img_paths
     
    
    def get_sample_by_name(self, name, img_size=24):
        try:
            img_path = next(path for path in self.samples if path.stem == name)
        except StopIteration:
            print("No image with specified name found. Returning a random image")
            img_path = self.samples[1]
        img = Image.open(img_path).convert('RGB')
        
        return F.to_tensor(F.resize(img ,size = [img_size, img_size], interpolation = Image.BICUBIC))