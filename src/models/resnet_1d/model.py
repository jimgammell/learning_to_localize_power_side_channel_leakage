from typing import *
import numpy as np
import torch
from torch import nn

from .building_blocks import ResNetBlock

class ResNet1d(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256,
        base_channels: int = 64,
        head_count: int = 1,
        noise_conditional: bool = False
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.base_channels = base_channels
        self.head_count = head_count
        self.noise_conditional = noise_conditional
        
        if self.noise_conditional:
            self.noise_embedding = nn.Sequential(
                nn.Linear(np.prod(self.input_shape), 4*self.base_channels),
                nn.ReLU()
            )
            noise_shape = (4*self.base_channels,)
        else:
            noise_shape = None
        self.blocks = nn.ModuleDict([
            ('block_1', ResNetBlock(2*self.input_shape[0] if self.noise_conditional else self.input_shape[0], self.base_channels, noise_shape=noise_shape)),
            ('block_2', ResNetBlock(self.base_channels, 2*self.base_channels, noise_shape=noise_shape)),
            ('block_3', ResNetBlock(2*self.base_channels, 2*self.base_channels, noise_shape=noise_shape))
        ])
        self.out_layer = nn.Linear(2*self.base_channels, self.output_classes)
        nn.init.xavier_uniform_(self.out_layer.weight)
    
    def forward(self, *args):
        if self.noise_conditional:
            (x, noise) = args
            x = torch.cat([x, noise], dim=1)
            noise = self.noise_embedding(noise.flatten(start_dim=1))
        else:
            (x,) = args
        for block in self.blocks.values():
            if self.noise_conditional:
                x = block(x, noise)
            else:
                x = block(x)
        x = x.mean(dim=-1)
        x = self.out_layer(x)
        return x