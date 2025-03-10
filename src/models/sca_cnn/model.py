from typing import *
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from ..common import *
from .building_blocks import *

class SCA_CNN(Module):
    def __init__(self,
        input_shape: Sequence[int] = (1, 1000),
        output_classes: int = 256,
        base_channels: int = 32,
        head_count: int = 1,
        block_count: int = 3,
        head_kwargs: dict = {},
        block_kwargs: dict = {},
        noise_conditional: bool = False
    ):
        if noise_conditional:
            self.embedded_noise_dim = base_channels*2**block_count
            block_kwargs['noise_shape'] = (self.embedded_noise_dim,)
        super().__init__(**{key: val for key, val in locals().items() if key not in ('self', 'key', 'val')})

    def construct(self):
        if self.noise_conditional:
            self.noise_embedding = nn.Sequential(
                    nn.Linear(np.prod(self.input_shape), self.embedded_noise_dim),
                    nn.ELU()
            )
        trunk_modules = []
        in_channels = self.input_shape[0]
        if self.noise_conditional:
            in_channels *= 2
        out_channels = self.base_channels
        for block_idx in range(self.block_count):
            trunk_modules.append((f'block_{block_idx+1}', Block(in_channels, out_channels, **self.block_kwargs)))
            in_channels = out_channels
            out_channels *= 2
        self.trunk = nn.ModuleDict(OrderedDict(trunk_modules))
        x = torch.randn(1, (2 if self.noise_conditional else 1)*self.input_shape[0], *self.input_shape[1:])
        x_i = x
        for mod in self.trunk.values():
            x = mod(*([x, torch.zeros(1, self.embedded_noise_dim)] if self.noise_conditional else [x]))
        self.heads = nn.ModuleList([
            Head(np.prod(x.shape), self.output_classes, **self.head_kwargs) for _ in range(self.head_count)
        ])
    
    def forward(self, *args):
        if self.noise_conditional:
            (x, noise) = args
            noise = (noise - noise.mean())/(noise.std() + 1e-4)
            embedded_noise = self.noise_embedding(noise.flatten(start_dim=1))
            x = torch.cat([x, noise], dim=1)
        else:
            (x,) = args
        batch_size, *input_shape = x.shape
        for mod in self.trunk.values():
            x = mod(*([x, embedded_noise] if self.noise_conditional else [x]))
        x = x.view(batch_size, -1)
        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits