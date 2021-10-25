import torch
import torch.nn as nn
from torchvision.models import vgg19

# class for content loss function to penalize if features don't match between generated and target image
class VGGContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Extract the features layers of vgg19 network
        self.feature_layers = vgg19(pretrained=True).features[:35].eval()
        # Use l1 loss for feature (content) loss
        self.criterion = nn.L1Loss()
        
        for param in self.feature_layers.parameters():
            param.requires_grad = False
            
        # This is the VGG model normalization values of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, gen_img, target_img):
        gen_img = (gen_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std

        gen_img_features = self.feature_layers(gen_img)
        target_img_features = self.feature_layers(target_img)

        return self.criterion(gen_img_features, target_img_features)

    def to(self, device):
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
        self.feature_layers.to(device)
        self.criterion.to(device)
        return self