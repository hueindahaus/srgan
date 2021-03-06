# ==============================================================================
# Original config file for initializing training from scratch
# ==============================================================================

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from loss import VGGContentLoss

# Flexible args
BATCH_SIZE              = 32
NUM_EPOCHS_TRAIN        = 300 #150
NUM_EPOCHS_PRETRAIN     = 50 #25
HIGH_RES_SIZE           = 96
SCALING_FACTOR          = 2
LOW_RES_SIZE            = HIGH_RES_SIZE // 4

# Perceptual loss function weights
ADVERSARIAL_WEIGHT      = 5e-3
CONTENT_WEIGHT          = 1
PIXEL_WEIGHT            = 1e-1 # 1e-2 in esrgan paper

# Learning rates
GEN_LR_PRETRAIN         = 1e-3
GEN_LR                  = 1e-4
DISC_LR                 = 1e-4

# Fixed args (don't touch these if it isn't necessary)
DEVICE                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ADVERSARIAL_CRITERION   = nn.BCEWithLogitsLoss().to(DEVICE)
CONTENT_CRITERION       = VGGContentLoss().to(DEVICE)
PIXEL_CRITERION         = nn.L1Loss().to(DEVICE)

def export_config_dict():
    dict = {
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_EPOCHS_TRAIN': NUM_EPOCHS_TRAIN,
        'NUM_EPOCHS_PRETRAIN': NUM_EPOCHS_PRETRAIN,
        'HIGH_RES_SIZE': HIGH_RES_SIZE,
        'SCALING_FACTOR': SCALING_FACTOR,
        'LOW_RES_SIZE': LOW_RES_SIZE,
        'PIXEL_WEIGHT': PIXEL_WEIGHT,
        'CONTENT_WEIGHT': CONTENT_WEIGHT,
        'ADVERSARIAL_WEIGHT': ADVERSARIAL_WEIGHT,
        'GEN_LR_PRETRAIN': GEN_LR_PRETRAIN,
        'GEN_LR': GEN_LR,
        'DISC_LR': DISC_LR,
    }
    return dict


