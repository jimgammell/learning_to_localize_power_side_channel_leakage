from typing import *
import numpy as np
import torch
from torch import nn

class Module(nn.Module):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.construct()
        self.init_weights()
    
    def construct(self):
        raise NotImplementedError
    
    def init_weights(self):
        pass

    def extra_repr(self):
        if not hasattr(self, 'param_count'):
            self.param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        rv = []
        rv.append(f'Parameter count: {self.param_count:.4e}')
        rv = '\n'.join(rv)
        return rv