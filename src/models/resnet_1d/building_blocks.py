# Based on this implementation: https://github.com/okrasolar/pytorch-timeseries/blob/master/src/models/resnet_baseline.py

from typing import *
import numpy as np
import torch
from torch import nn

class NoiseConditionalBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, noise_shape: Sequence[int], **kwargs):
        kwargs['affine'] = False
        super().__init__(**kwargs)
        self.to_weight_and_bias = nn.Linear(np.prod(noise_shape), 2*self.num_features)
        self.to_weight_and_bias.weight.data.zero_()
        self.to_weight_and_bias.bias.data.zero_()
        self.to_weight_and_bias.bias.data[:self.num_features] = 1.

    def forward(self, x, noise):
        x_norm = super().forward(x)
        weight_and_bias = self.to_weight_and_bias(noise.view(noise.size(0), -1))
        weight = weight_and_bias[:, :self.num_features].view(-1, self.num_features, 1)
        bias = weight_and_bias[:, self.num_features:].view(-1, self.num_features, 1)
        out = weight*x_norm + bias
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, noise_shape: Optional[Sequence[int]] = None):
        super().__init__()
        
        self.noise_shape = noise_shape
        self.block = nn.ModuleDict([
            ('conv', nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding='same')),
            ('norm', nn.BatchNorm1d(out_channels) if noise_shape is None else NoiseConditionalBatchNorm1d(noise_shape, num_features=out_channels)),
            ('act', nn.ReLU())
        ])
    
    def forward(self, *args):
        if self.noise_shape is not None:
            (x, noise) = args
        else:
            (x,) = args
        for layer in self.block.values():
            if isinstance(layer, NoiseConditionalBatchNorm1d):
                x = layer(x, noise)
            else:
                x = layer(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, noise_shape: Optional[Sequence[int]] = None):
        super().__init__()
        self.channels = [in_channels, out_channels, out_channels, out_channels]
        self.kernel_sizes = [9, 5, 3]
        self.noise_shape = noise_shape
        
        self.resid_blocks = nn.ModuleDict([
            (f'block_{idx+1}', ConvBlock(self.channels[idx], self.channels[idx+1], self.kernel_sizes[idx], stride=1))
            for idx in range(len(self.kernel_sizes))
        ])
        self.match_channels = in_channels != out_channels
        if self.match_channels:
            self.skip_block = nn.ModuleDict([
                ('conv', nn.Conv1d(in_channels, out_channels, 1)),
                ('norm', nn.BatchNorm1d(out_channels))
            ])
    
    def forward(self, *args):
        if self.noise_shape is not None:
            (x, noise) = args
        else:
            (x,) = args
        x_resid = x
        x_skip = x
        for block in self.resid_blocks.values():
            if self.noise_shape is not None:
                x_resid = block(x_resid, noise)
            else:
                x_resid = block(x_resid)
        if self.match_channels:
            for layer in self.skip_block.values():
                if isinstance(layer, NoiseConditionalBatchNorm1d):
                    x_skip = layer(x_skip, noise)
                else:
                    x_skip = layer(x_skip)
        return x_resid + x_skip