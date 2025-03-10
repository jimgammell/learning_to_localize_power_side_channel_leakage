import torch
from torch import nn

class AdditiveNoise(nn.Module):
    def __init__(self, std=1.0):
        super().__init__()
        self.std = std
    
    def forward(self, x):
        return x + self.std*torch.randn_like(x)