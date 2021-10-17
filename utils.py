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
from torchvision.utils import save_image

from itertools import chain
from PIL import Image


def display_image(axis, image_tensor, reverse_normalization = False):
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("The `display_image` function expects a `torch.Tensor` " +
                        "use the `ToTensor` transformation to convert the images to tensors.")
        
    # The imshow commands expects a `numpy array` with shape (3, width, height)
    # We rearrange the dimensions with `permute` and then convert it to `numpy`
    image_data = image_tensor.permute(1, 2, 0).numpy()
    
    # Assuming our normalization on the image is mean=0.5 and std=0.5 for all channels we can do this to more accurately display the image
    if reverse_normalization:
        image_data = 0.5 * image_data + 0.5
    
    height, width, _ = image_data.shape
    axis.imshow(image_data)
    axis.set_xlim(0, width)
    # By convention when working with images, the origin is at the top left corner.
    # Therefore, we switch the order of the y limits.
    axis.set_ylim(height, 0)

# Very specific and bad designed function to generate resulting images.
def save_result_images(generator, datahandler, folder_path, reverse_normalization = True):
    if not datahandler or not generator or not folder_path:
        print("Make sure that datahandler, generator and path are provided correctly")
        return

    with torch.no_grad():

        img_lr_1 = datahandler.get_sample_by_name('baboon', 128)
        img_sr_1 = generator.forward(torch.unsqueeze(img_lr_1.cuda().detach(), 0))[-1].cpu()
        img_hr_1 = datahandler.get_sample_by_name('baboon', 512)

        img_lr_2 = datahandler.get_sample_by_name('eagle', 128)
        img_sr_2 = generator.forward(torch.unsqueeze(img_lr_2.cuda().detach(), 0))[-1].cpu()
        img_hr_2 = datahandler.get_sample_by_name('eagle', 512)

        img_lr_3 = datahandler.get_sample_by_name('bee', 128)
        img_sr_3 = generator.forward(torch.unsqueeze(img_lr_3.cuda().detach(), 0))[-1].cpu()
        img_hr_3 = datahandler.get_sample_by_name('bee', 512)

        if reverse_normalization:
            img_sr_1 = 0.5 * img_sr_1 + 0.5
            img_sr_2 = 0.5 * img_sr_2 + 0.5
            img_sr_3 = 0.5 * img_sr_3 + 0.5
        try:
            save_image(img_lr_1, folder_path + '/baboon-lr.jpg')
            save_image(img_sr_1, folder_path + '/baboon-sr.jpg')
            save_image(img_hr_1, folder_path + '/baboon-hr.jpg')
            
            save_image(img_lr_2, folder_path + '/eagle-lr.jpg')
            save_image(img_sr_2, folder_path + '/eagle-sr.jpg')
            save_image(img_hr_2, folder_path + '/eagle- hr.jpg')

            save_image(img_lr_3, folder_path + '/bee-lr.jpg')
            save_image(img_sr_3, folder_path + '/bee-sr.jpg')
            save_image(img_hr_3, folder_path + '/bee-hr.jpg')
            print("Succesfully saved result images in ")
        except:
            print("ERROR when saving result images")
        

        
