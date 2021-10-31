# Super resolution GAN :sparkles:

This repository presents a generative adversarial network model to super resolute imaging systems inspired by [SRGAN](https://arxiv.org/abs/1609.04802) and [ESRGAN](https://arxiv.org/abs/1809.00219). The implementation is done in pytorch.

![thumbnail](https://user-images.githubusercontent.com/45295311/139563914-d50ef174-a98c-4ecf-8db2-91c7b3545f96.png)

## Super resolution GAN concept
- Generator inputs low resolution images and outputs upscaled super resolution images
- Discriminator inputs super resolution images and ground truth high resolution images and outputs real/fake predictions
- Discriminator's learnable parameters are updated based on prediction accuracy
- Generator's learnable parameters are updated based on perceptual similarity between sr and hr images, and also discriminator's prediction accuracy

![gan-concept](https://user-images.githubusercontent.com/45295311/139563228-ae41f960-03ee-439d-af28-4aa910ca1853.png)


## Model architecture
- Model architecture are highly inspired by [SRGAN](https://arxiv.org/abs/1609.04802) with modifications brought by [ESRGAN](https://arxiv.org/abs/1809.00219).

![architecture](https://user-images.githubusercontent.com/45295311/139563153-4064d5af-f965-496a-a3c3-cdf8072efa21.png)


## Inception-residual block
- In this project, inception-residual blocks are considered instead of regular residual blocks (from SRGAN) or residual in residual dense block (from ESRGAN)

![inception-residual-block (2)](https://user-images.githubusercontent.com/45295311/139563156-970bbf47-e071-4feb-87ad-c2ece40bd13c.png)


## Other details
- Relativistic discriminator is used for a more stable and contextual training phase
- Generator perceptual loss considers pixel wise loss, feature extracted loss from a pretrained VGG19-network and adversarial loss based on discriminator's performance


## Results
- Result are presented in (PSNR/SSIM) where PSNR is peak signal to noise ratio and SSIM is structural similarity index measure.

![gan-results](https://user-images.githubusercontent.com/45295311/139563229-abbe62c1-d619-4a03-be34-7c61fd70c904.png)

> This project is a part of the course **Deep Machine Learning (SSY340)** from Chalmers University of Technology.
