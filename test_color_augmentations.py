# this is a placeholder for diffusion policy training data augmentation
# IMPORTANT: probably want to apply the same randomization to the whole trajectory b/c real world doesnt flicker
# make sure it's not applying a different transform to each image in the trajectory

import torchvision.transforms as T

# Define a reasonable range for each parameter
# These values are factors, e.g., 0.8 is 80% brightness
b_factor = 0.3  # Brightness jitter
c_factor = 0.3  # Contrast jitter
s_factor = 0.3  # Saturation jitter
h_factor = 0.1  # Hue jitter

color_augment = T.ColorJitter(
    brightness=(1 - b_factor, 1 + b_factor),
    contrast=(1 - c_factor, 1 + c_factor),
    saturation=(1 - s_factor, 1 + s_factor),
    hue=(-h_factor, h_factor)
)

# You can add this directly to your transform pipeline
# pipeline = T.Compose([
#     ...
#     color_augment,
#     ...
# ])



import torch
import torch.nn as nn
import random

class RandomGamma(nn.Module):
    """
    Applies a random gamma correction to an image.
    Assumes image is in [0, 1] range.
    """
    def __init__(self, min_gamma=0.7, max_gamma=1.5):
        super().__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def forward(self, img):
        # Sample a random gamma value
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        
        # Apply gamma correction
        # We add a small epsilon to avoid log(0) or pow(0, <1) issues
        return torch.pow(img + 1e-5, gamma)

# Usage in pipeline
# pipeline = T.Compose([
#     ...
#     RandomGamma(min_gamma=0.7, max_gamma=1.5),
#     ...
# ])

class RandomVignette(nn.Module):
    """
    Applies a random vignette effect to an image.
    Assumes image is in [B, C, H, W] format and [0, 1] range.
    """
    def __init__(self, min_strength=0.1, max_strength=0.8):
        super().__init__()
        self.min_strength = min_strength
        self.max_strength = max_strength

    def forward(self, img):
        b, c, h, w = img.shape
        device = img.device

        # Create coordinate grids
        x = torch.linspace(-1, 1, w, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        xx, yy = torch.meshgrid(y, x, indexing='ij')

        # Calculate radial distance (squared)
        # We can also randomize the center of the vignette slightly
        center_x = random.uniform(-0.2, 0.2)
        center_y = random.uniform(-0.2, 0.2)
        r_squared = (xx - center_y)**2 + (xx - center_x)**2

        # Create Gaussian-like mask
        strength = random.uniform(self.min_strength, self.max_strength)
        
        # A simple quadratic falloff: 1.0 - (r^2 * strength)
        # We use max_r_squared to normalize the effect
        max_r_squared = 1.0**2 + 1.0**2 # Furthest corner
        mask = 1.0 - (r_squared / max_r_squared) * strength
        
        # Ensure mask is in [0, 1]
        mask = torch.clamp(mask, 0.0, 1.0)
        
        # Reshape to [1, 1, H, W] for broadcasting
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Apply mask
        return img * mask
    


# Assumes your tensor is on the 'cuda' device and in [0, 1] range
# Note: Vignette needs a [B, C, H, W] tensor. 
# ColorJitter and Gamma work on [C, H, W].

# 1. Define your main pipeline for individual images [C, H, W]
train_transforms = T.Compose([
    T.ToTensor(),  # Example: if starting from PIL
    T.RandomHorizontalFlip(),
    # ... other spatial augmentations ...
    
    # Apply color grading and gamma
    color_augment,
    RandomGamma(min_gamma=0.7, max_gamma=1.5),
    
    # Clamp values to ensure they are still valid [0, 1]
    T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
])

# 2. Define your vignette (which must run on the BATCH)
vignette_augment = RandomVignette(min_strength=0.1, max_strength=0.8)

# 3. In your training loop:
# for batch in data_loader:
#     images, labels = batch
#     images = images.to('cuda')
#
#     # Apply spatial, color, and gamma transforms
#     # (If transforms are not already in the DataLoader)
#
#     # Apply vignette on the batch
#     images = vignette_augment(images)
#
#     # ... proceed with training ...