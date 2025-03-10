from typing import *
import numpy as np
import torch
from torch import nn

from ..common import Module
from .building_blocks import Block, Head

class LeNet(Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_dims: int = 10,
        conv_channels: Sequence[int] = [6, 16],
        hidden_neurons: Sequence[int] = [120, 84]
    ):
        super().__init__(input_shape, output_dims, conv_channels, hidden_neurons)
    
    def construct(self):
        self.trunk = nn.Sequential(OrderedDict([
            (f'block_{idx+1}', Block(in_channels, out_channels))
            for idx, (in_channels, out_channels) in enumerate(zip([self.input_shape, *self.conv_channels[:-1]], self.conv_channels))
        ]))
        eg_input = torch.randn(1, *self.input_shape)
        eg_output = self.trunk(eg_input)
        self.head = Head(np.prod(eg_output.shape), self.output_dims, self.hidden_neurons)
    
    def forward(self, x):
        batch_size, *_ = x.shape
        x = self.trunk(x)
        x = self.head(x.view(batch_size, -1))
        return x

class LeNet5(LeNet):
    def __init__(self,
        input_shape: Sequence[int],
        output_dims: int = 10
    ):
        LeNet.__init__(self, input_shape, output_dims=output_dims, conv_channels=[6, 16], hidden_neurons=[120, 84])