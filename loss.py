import torch
import torch.nn as nn # Includes all modules, nn.Linear, nn.Conv2d, BatchNorm etc
from torchvision.models import vgg19
import config

# class for content loss function to penalize if features don't match between generated and target image
class VGGContentLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Extract the features of vgg19 network
        self.feature_layers = vgg19(pretrained=True).features[:18].eval().to(config.DEVICE)
        # Use mean squared error loss for feature diffs
        self.mse = nn.MSELoss().to(config.DEVICE)
        
        for param in self.feature_layers.parameters():
            param.requires_grad = False
            
        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, gen_img, target_img):
        gen_img_features = self.feature_layers(gen_img)
        target_img_features = self.feature_layers(target_img)

        return self.mse(gen_img_features, target_img_features)