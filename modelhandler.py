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

from model import Generator, Discriminator
from loss import VGGContentLoss
import config


class ModelHandler:
    
    def __init__(self):
        self.saved_model_path = './saved-models'
    
    def save_models(self, generator, discriminator, disc_opti, gen_opti, gen_opti_pretrain, config_dict = {}):
        print('Saving procedure initialized..')
        stem = input("Enter stem of path (leave empty if you don't want to save): ")

        # if no configs provided, get them from config file
        if not len(config_dict.keys()):
            config_dict = config.export_config_dict()
        
        if stem:
            PATH = self.saved_model_path + '/' + str(stem) + '.pt'
            
            if exists(PATH):
                overwrite = input('There already exist a checkpoint with this name. Do you want to overwrite? (Write YES)')
                if overwrite != 'YES':
                    print('Interrupting saving models (no overwrite)')
                    return
            
            torch.save({
                'generator_state_dict':generator.state_dict(),
                'discriminator_state_dict':discriminator.state_dict(),
                'gen_opti_state_dict': gen_opti.state_dict(),
                'disc_opti_state_dict': disc_opti.state_dict(),
                'gen_opti_pretrain_state_dict': gen_opti_pretrain.state_dict(),
                **config_dict
            }, PATH)
            if exists(PATH):
                print('Saved succesfully')
        else:
            print('Interrupting saving models (no path specified)')
            return
            
    def load_models(self):
        print('Load procedure initialized..')
        stem = input('Enter stem of path (leave empty for new models): ')
        if stem:
            PATH = self.saved_model_path + '/' + str(stem) + '.pt'
            
            if exists(PATH):
                print('Loading existing models')
                checkpoint = torch.load(PATH)
                # Load generator
                generator = Generator(use_inception_blocks = checkpoint['USE_INCEPTION_BLOCKS'])
                generator.load_state_dict(checkpoint['generator_state_dict'])
                generator.to(config.DEVICE)

                # Load discriminator
                discriminator = Discriminator(checkpoint['HIGH_RES_SIZE'])
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                discriminator.to(config.DEVICE)

                # Load optimizers
                disc_opti = optim.Adam(discriminator.parameters(), lr=checkpoint['DISC_LR'], betas=(0.9, 0.999))
                disc_opti.load_state_dict(checkpoint['disc_opti_state_dict']) 
                gen_opti = optim.Adam(generator.parameters(), lr=checkpoint['GEN_LR'], betas=(0.9, 0.999))
                gen_opti.load_state_dict(checkpoint['gen_opti_state_dict'])
                gen_opti_pretrain = optim.Adam(generator.parameters(), lr=checkpoint['GEN_LR_PRETRAIN'], betas=(0.9, 0.999))
                gen_opti_pretrain.load_state_dict(checkpoint['gen_opti_pretrain_state_dict'])     
                
                # Initialize loss functions that are used for perceptual and adversarial loss
                adversarial_criterion = nn.BCELoss().to(config.DEVICE)
                content_criterion = VGGContentLoss()
                pixel_criterion = nn.MSELoss().to(config.DEVICE)

                # Training parameters
                config_dict = checkpoint
    
                print_loaded_args(config_dict)
                return generator, discriminator, disc_opti, gen_opti, gen_opti_pretrain, config_dict
            else:
                print('Path ' + PATH + '  does not exist')
        
        print('Loading models from scratch')
        config_dict = config.export_config_dict()

        generator = Generator().to(config.DEVICE)
        discriminator = Discriminator(config_dict['HIGH_RES_SIZE']).to(config.DEVICE)

        disc_opti = optim.Adam(discriminator.parameters(), lr=config_dict['DISC_LR'], betas=(0.9, 0.999))
        gen_opti = optim.Adam(generator.parameters(), lr=config_dict['GEN_LR'], betas=(0.9, 0.999))
        gen_opti_pretrain = optim.Adam(generator.parameters(), lr=config_dict['GEN_LR_PRETRAIN'], betas=(0.9, 0.999))
        
        # Initialize loss functions that are used for perceptual and adversarial loss
        adversarial_criterion = nn.BCELoss().to(config.DEVICE)
        content_criterion = VGGContentLoss()
        pixel_criterion = nn.MSELoss().to(config.DEVICE)
        print_loaded_args(config_dict)
        return generator, discriminator, disc_opti, gen_opti, gen_opti_pretrain, config_dict

def print_loaded_args(config_dict):
    print('high_res_size: ' + str(config_dict['HIGH_RES_SIZE']))
    print('low_res_size: ' + str(config_dict['LOW_RES_SIZE'] ))
    print('scaling_factor: ' + str(config_dict['SCALING_FACTOR']))
    print('batch_size: ' + str(config_dict['BATCH_SIZE']))
    print('num_epochs_train: ' + str(config_dict['NUM_EPOCHS_TRAIN']))
    print('num_epochs_pretrain: ' + str(config_dict['NUM_EPOCHS_PRETRAIN']))
    print('gen_lr: ' + str(config_dict['GEN_LR']))
    print('gen_lr_pretrain: ' + str(config_dict['GEN_LR_PRETRAIN']))
    print('disc_lr: ' + str(config_dict['DISC_LR']))
    print('use_inception_blocks: ' + str(config_dict['USE_INCEPTION_BLOCKS']))
    print('pixel_weight: ' + str(config_dict['PIXEL_WEIGHT']))
    print('content_weight: ' + str(config_dict['CONTENT_WEIGHT']))
    print('adversarial_weight: ' + str(config_dict['ADVERSARIAL_WEIGHT']))