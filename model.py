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
            nn.Conv2d(3, num_channels, 9, padding=4),
            nn.PReLU(num_parameters = num_channels),
        )

        if use_inception_blocks:
            self.convolutional_blocks = nn.Sequential(
                InceptionBlock(num_channels, red_3x3=32, red_5x5=16, out_1x1 = 64, out_3x3 = 64, out_5x5 = 32, out_1x1pool = 32),
                InceptionBlock(192, red_3x3=96, red_5x5=16, out_1x1 = 64, out_3x3 = 128, out_5x5 = 32, out_1x1pool = 32),
                InceptionBlock(256, red_3x3=128, red_5x5=32, out_1x1 = 128, out_3x3 = 192, out_5x5 = 96, out_1x1pool = 64),
                
                InceptionBlock(480, red_3x3 = 96, red_5x5 = 16, out_1x1 = 192, out_3x3 = 208, out_5x5 = 48, out_1x1pool = 64),
                InceptionBlock(512, red_3x3 = 112, red_5x5 = 24, out_1x1 = 160, out_3x3 =224, out_5x5 = 64, out_1x1pool = 64),
                InceptionBlock(512, red_3x3 = 128, red_5x5 = 24, out_1x1 = 128, out_3x3 =256, out_5x5 = 64, out_1x1pool = 64),
                InceptionBlock(512, red_3x3 = 112, red_5x5 = 24, out_1x1 = 160, out_3x3 =224, out_5x5 = 64, out_1x1pool = 64),
                InceptionBlock(512, red_3x3 = 128, red_5x5 = 24, out_1x1 = 128, out_3x3 =256, out_5x5 = 64, out_1x1pool = 64),
                InceptionBlock(512, red_3x3 = 144, red_5x5 = 32, out_1x1 = 112, out_3x3 =288, out_5x5 = 64, out_1x1pool = 64),
                InceptionBlock(528, red_3x3 = 128, red_5x5 = 24, out_1x1 = 128, out_3x3 =256, out_5x5 = 64, out_1x1pool = 64),
                
                InceptionBlock(512, red_3x3=96, red_5x5=16, out_1x1 = 64, out_3x3 = 128, out_5x5 = 32, out_1x1pool = 32),
                InceptionBlock(256, red_3x3=32, red_5x5=16, out_1x1 = 64, out_3x3 = 64, out_5x5 = 32, out_1x1pool = 32),
                InceptionBlock(192, red_3x3 = 32, red_5x5 = 16, out_1x1 = 16, out_3x3 =32, out_5x5 = 8, out_1x1pool = 8),
            )
        else:
            self.convolutional_blocks = nn.Sequential(
                *[ResidualBlock(num_channels,num_channels) for _ in range(18)]
            )
        
        
        self.final_residual_block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
        )
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(num_channels,scale_factor),
            UpsampleBlock(num_channels,scale_factor),
        )
        self.output_layer = nn.Conv2d(num_channels, 3, 9, padding=4)
    
    def forward(self, x):
        # Save the output from initial conv layers so that we can add it to our skip connection before upsampling
        out_init_layers = self.init_layers(x)
        out_convolutional_blocks= self.convolutional_blocks(out_init_layers)
        out_final_residual_block = self.final_residual_block(out_convolutional_blocks) + out_init_layers
        out_upsample_blocks = self.upsample_blocks(out_final_residual_block)
        return self.output_layer(out_upsample_blocks)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, red_3x3, red_5x5, out_1x1, out_3x3, out_5x5, out_1x1pool):
        super(InceptionBlock, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_1x1),
            nn.PReLU(num_parameters=out_1x1)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1, bias=False),
            nn.BatchNorm2d(red_3x3),
            nn.PReLU(num_parameters=red_3x3),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_3x3),
            nn.PReLU(num_parameters=out_3x3)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1, bias=False),
            nn.BatchNorm2d(red_5x5),
            nn.PReLU(num_parameters=red_5x5),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_5x5),
            nn.PReLU(num_parameters=out_5x5)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_1x1pool, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_1x1pool),
            nn.PReLU(num_parameters=out_1x1pool),
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)        
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, num_channels=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(num_parameters = num_channels),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )
        
    def forward(self, x):
        return self.block(x) + x
        
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            # The number of output filters is altered to be in_channels * sf^2 so that pixel shuffle (next layer) can upscale image
            nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, padding=1, bias=False),
            nn.PixelShuffle(scale_factor), #  in_channels * scale_factor^2, height, width -> in_channels, height*2, width*2
            nn.PReLU(num_parameters=in_channels),
        )
    def forward(self, x):
        return self.block(x)




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
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.classifier(self.feature_layers(x))