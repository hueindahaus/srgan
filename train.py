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

from model import Generator, Discriminator # Models
from loss import VGGContentLoss
from datahandler import DataHandler
from modelhandler import ModelHandler
from utils import display_image
import config


def train_loop(dataloader,  generator,discriminator, disc_opti, gen_opti, gen_opti_pretrain, config_dict, adversarial=True):
    print('##################\n%s\n##################' % ('Starting ADVERSARIAL training' if adversarial else 'Starting GENERATOR training'))
    
    adversarial_criterion = config.ADVERSARIAL_CRITERION
    content_criterion = config.CONTENT_CRITERION
    pixel_criterion = config.PIXEL_CRITERION
    adversarial_weight = config_dict['ADVERSARIAL_WEIGHT']
    content_weight = config_dict['CONTENT_WEIGHT']
    pixel_weight = config_dict['PIXEL_WEIGHT']
    
    num_epochs = config_dict['NUM_EPOCHS_TRAIN'] if adversarial else config_dict['NUM_EPOCHS_PRETRAIN']
    
    # Set training mode
    discriminator.train()
    generator.train()
    
    # Initialize loss-lists
    disc_losses = []
    gen_losses = []
    
    # Matplotlib to visualize training progression
    try:
        import IPython
        shell = IPython.get_ipython()
        shell.enable_matplotlib(gui='qt')
    except:
        pass   

    fig = plt.figure(figsize=(15, 12))
    ax_lr_dynamic = plt.subplot2grid((3, 6), (0, 0), colspan=2)
    ax_sr_dynamic = plt.subplot2grid((3, 6), (0, 2), colspan=2)
    ax_hr_dynamic = plt.subplot2grid((3, 6), (0, 4), colspan=2)
    
    ax_lr_fixed = plt.subplot2grid((3, 6), (1, 0), colspan=2)
    ax_sr_fixed = plt.subplot2grid((3, 6), (1, 2), colspan=2)
    ax_hr_fixed = plt.subplot2grid((3, 6), (1, 4), colspan=2)

    ax_disc_loss = plt.subplot2grid((3, 6), (2, 0), colspan=3)
    ax_gen_loss = plt.subplot2grid((3, 6), (2, 3), colspan=3)
    
    memory_allocated_for_tensors = 0
    
    for epoch in range(1, num_epochs + 1):
        for batch_index, (high_res_batch, low_res_batch) in enumerate(dataloader, 1):
            
            # Transfer high res and low res image tensors to device
            high_res_batch = high_res_batch.to(config.DEVICE)
            low_res_batch = low_res_batch.to(config.DEVICE)
            
            # Generate fake images with the generator
            fake_batch = generator(low_res_batch)
            
            # If training with adversarial include discriminator and update it
            if adversarial:
                # [1] Update Discriminator network: maximize log(D(x)) + log(1-D(G(Z)))

                # Initialize the gradient of the discriminator model.
                discriminator.zero_grad()

                # [1.1] Train discriminator with real images
                real_preds = discriminator(high_res_batch)
                real_labels = torch.ones_like(real_preds, device=config.DEVICE)
                disc_real_loss = adversarial_criterion(real_preds, real_labels)

                # [1.2] Train discriminator with fake generated images
                fake_preds = discriminator(fake_batch.detach())
                fake_labels = torch.zeros_like(fake_preds, device=config.DEVICE)
                disc_fake_loss = adversarial_criterion(fake_preds, fake_labels)

                # Discriminator loss
                disc_loss = disc_real_loss + disc_fake_loss
                # Calculate discriminator gradient
                disc_loss.backward()
                # Update discriminator parameters
                disc_opti.step()
                # Append loss
                #disc_losses.append(disc_loss.detach().item())
                
                D_x = real_preds.mean().detach().item()
                D_G_z1 = fake_preds.mean().detach().item()


            # [2] Update Genertaor network: maximize log(D(G(z)))
            
            # Initialize the gradient of the generator model.
            generator.zero_grad()
            
            gen_loss = None #Initialize gen loss for batch
            
            if adversarial:
                # Generator wants the discriminator to output 1 on its generated images
                new_fake_preds = discriminator(fake_batch)
                gen_desirable_labels = torch.ones_like(new_fake_preds, device=config.DEVICE)

                # Generator loss is based on a weighted sum of adversarial loss and content loss
                adversarial_loss = adversarial_criterion(new_fake_preds, gen_desirable_labels)
                content_loss = content_criterion(fake_batch, high_res_batch.detach())
                pixel_loss = pixel_criterion(fake_batch, high_res_batch.detach())

                # Adversarial training
                gen_loss = pixel_weight * pixel_loss + content_weight * content_loss + adversarial_weight * adversarial_loss
                
                # Calculate gradients for Generator
                gen_loss.backward()

                # Update Generator
                gen_opti.step()
                
                 # Save mean of new fake preds
                D_G_z2 = new_fake_preds.mean().detach().item()
            else:
                # Exclusive generator training
                gen_loss = pixel_criterion(fake_batch, high_res_batch.detach())
                
                # Calculate gradients for Generator
                gen_loss.backward()
                
                # Update Generator
                gen_opti_pretrain.step()
                
            #gen_losses.append(gen_loss.detach().item())
                
            # plot and print progression every x:th batch
            if batch_index % 50 == 0:
                if adversarial:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, batch_index, len(dataloader),
                             disc_loss.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))
                else:
                    print('[%d/%d][%d/%d]\tLoss_G: %.4f'
                          % (epoch, num_epochs, batch_index, len(dataloader), gen_loss.item()))
                    
                # plot images
                with torch.no_grad():
                     # Display images to visualize current state in training
                    display_image(ax_lr_dynamic, low_res_batch[-1].cpu().detach())
                    display_image(ax_sr_dynamic, fake_batch[-1].cpu().detach(), reverse_normalization = True)
                    display_image(ax_hr_dynamic, high_res_batch[-1].cpu().detach(), reverse_normalization = True)
                    display_image(ax_lr_fixed, dataloader.dataset.get_sample_by_name('baboon', 128))
                    display_image(ax_sr_fixed, generator.forward(torch.unsqueeze(dataloader.dataset.get_sample_by_name('baboon', 128).cuda().detach(), 0))[-1].cpu(), reverse_normalization = True)
                    display_image(ax_hr_fixed, dataloader.dataset.get_sample_by_name('baboon', 512))
                    ax_disc_loss.plot(disc_losses)
                    ax_disc_loss.set_title('Discriminator loss')
                    ax_gen_loss.plot(gen_losses)
                    ax_gen_loss.set_title('Generator loss')
                    plt.pause(0.5)
                    fig.show()
                
                # Detect memory increases for potential memory leaks
                if memory_allocated_for_tensors == 0:
                    memory_allocated_for_tensors = torch.cuda.memory_allocated()
                elif memory_allocated_for_tensors != torch.cuda.memory_allocated():
                    print('[Memory increase alert] If this is a reoccuring print statement. Program might be subject to memory leak!')
                    print('Current memory is: ' + str(torch.cuda.memory_allocated()))
                    memory_allocated_for_tensors = torch.cuda.memory_allocated()            

    try:
        import IPython
        shell = IPython.get_ipython()
        shell.enable_matplotlib(gui='inline')
    except:
        pass 

    torch.cuda.empty_cache()
    return generator, discriminator, disc_opti, gen_opti, gen_opti_pretrain, config_dict
    
