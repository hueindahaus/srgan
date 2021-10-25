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


class Generator(nn.Module):
    def __init__(self, num_channels=64, scale_factor=2, use_inception_blocks = False):
        
        super(Generator, self).__init__()

        self.init_layers = nn.Sequential(
            nn.Conv2d(3, num_channels, 3 , padding=1),
            nn.LeakyReLU(0.2, True),
        )

        if use_inception_blocks:
            self.convolutional_blocks = nn.Sequential(
                *[ResidualInceptionBlock() for _ in range(24)]
            )
        else:
            self.convolutional_blocks = nn.Sequential(
                *[ResidualBlock(num_channels,num_channels) for _ in range(18)]
            )
        
        
        self.final_residual_block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
        )
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(num_channels,scale_factor),
            UpsampleBlock(num_channels,scale_factor),
        )
        self.output_layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, 3, 3, padding=1),
        )
    
    def forward(self, x):
        out_init_layers = self.init_layers(x)
        out_convolutional_blocks= self.convolutional_blocks(out_init_layers)
        out_final_residual_block = self.final_residual_block(out_convolutional_blocks) + out_init_layers
        out_upsample_blocks = self.upsample_blocks(out_final_residual_block)
        return self.output_layers(out_upsample_blocks)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, num_channels=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
        )
        
    def forward(self, x):
        return self.block(x) + x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# https://arxiv.org/ftp/arxiv/papers/1810/1810.13169.pdf
class ResidualInceptionBlock(nn.Module):
    def __init__(self):
        super(ResidualInceptionBlock, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),   # Reduction block
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),   # Reduction block
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x)], 1) + x # concatenate and add the skip connection   
        

class Discriminator(nn.Module):
    
    def __init__(self, img_size, in_channels=3):
        super(Discriminator, self).__init__()
        
        self.feature_layers = nn.Sequential(
            *[
            nn.Conv2d(in_channels, 64, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, 3, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 3, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 128, 3, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 3, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 256, 3, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 3, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, 3, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True) 
        ]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024,1),
        )
    
    def forward(self, x):
        return self.classifier(self.feature_layers(x))