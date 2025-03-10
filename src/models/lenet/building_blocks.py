from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from ..common import Module

class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

    def construct(self):
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=5, bias=False)
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class Head(Module):
    def __init__(self, in_features, out_features, hidden_neurons=[120, 84]):
        super().__init__(in_features=in_features, out_features=out_features, hidden_neurons=hidden_neurons)
    
    def construct(self):
        modules = []
        for idx, (n_i, n_o) in enumerate(zip([self.in_features, *self.hidden_neurons[:-1]], self.hidden_neurons)):
            modules.append((f'dense_{idx+1}', nn.Linear(n_i, n_o)))
            modules.append((f'act_{idx+1}', nn.ReLU()))
        modules.append((f'dense_{len(modules)+1}', nn.Linear(self.hidden_neurons[-1], self.out_features)))
        self.head = nn.Sequential(OrderedDict(modules))
    
    def init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(mod.bias, 0)
    
    def forward(self, x):
        return self.head(x)