from typing import *
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from .keras_to_pytorch_utils import FlattenTranspose

class GenericWoutersNet(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256,
        input_pool_size: int = 2,
        block_settings: List[Dict[str, Any]] = [],
        dense_widths: List[int] = [20, 20, 20]
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.input_pool_size = input_pool_size
        self.block_settings = block_settings
        self.dense_widths = dense_widths
        
        self.input_pool = nn.AvgPool1d(self.input_pool_size)
        blocks = []
        in_channels = self.input_shape[0]
        for block_settings in self.block_settings:
            assert all(key in block_settings.keys() for key in ['channels', 'conv_kernel_size', 'pool_size'])
            block = nn.Sequential(OrderedDict([
                ('conv', nn.Conv1d(in_channels, block_settings['channels'], block_settings['conv_kernel_size'], padding='same')),
                ('selu', nn.SELU()),
                ('norm', nn.BatchNorm1d(block_settings['channels'])),
                ('pool', nn.AvgPool1d(block_settings['pool_size']))
            ]))
            blocks.append(block)
            in_channels = block_settings['channels']
        self.conv_stage = nn.Sequential(OrderedDict([(f'block_{block_idx+1}', block) for block_idx, block in enumerate(blocks)]))
        self.flatten = FlattenTranspose()
        eg_input = torch.randn((1, *self.input_shape))
        eg_output = self.flatten(self.conv_stage(self.input_pool(eg_input)))
        in_dims = np.prod(eg_output.shape)
        fc_modules = []
        for dense_idx, dense_width in enumerate(self.dense_widths):
            fc_modules.append((f'dense_{dense_idx+1}', nn.Linear(in_dims, dense_width)))
            fc_modules.append((f'selu_{dense_idx+1}', nn.SELU()))
            in_dims = dense_width
        fc_modules.append(('classifier', nn.Linear(in_dims, self.output_classes)))
        self.fc_stage = nn.Sequential(OrderedDict(fc_modules))
        
        for mod in self.modules():
            if isinstance(mod, nn.Conv1d):
                nn.init.kaiming_uniform_(mod.weight)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.BatchNorm1d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
            else:
                pass
    
    def forward(self, x):
        x = self.input_pool(x)
        x = self.conv_stage(x)
        x = self.flatten(x)
        x = self.fc_stage(x)
        return x

class GenericZaidNet(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256,
        block_settings: List[Dict[str, Any]] = [],
        dense_widths: List[int] = [20, 20, 20],
        omit_first_selu: bool = False # The Wouters implementation of ZaidNet omits the first SELU activation for ASCADv1-fixed-desync=100.
                                      # Likely a typo, but we must replicate it to use their pretrained weights.
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.block_settings = block_settings
        self.dense_widths = dense_widths
        self.omit_first_selu = omit_first_selu
        
        blocks = []
        in_channels = self.input_shape[0]
        for block_idx, block_settings in enumerate(self.block_settings):
            assert all(key in block_settings.keys() for key in ['channels', 'conv_kernel_size', 'pool_size'])
            block = nn.Sequential(OrderedDict([
                ('conv', nn.Conv1d(in_channels, block_settings['channels'], block_settings['conv_kernel_size'], padding='same')),
                ('selu', nn.SELU() if block_idx > 0 or not(self.omit_first_selu) else nn.Identity()),
                ('norm', nn.BatchNorm1d(block_settings['channels'])),
                ('pool', nn.AvgPool1d(block_settings['pool_size']))
            ]))
            blocks.append(block)
            in_channels = block_settings['channels']
        self.conv_stage = nn.Sequential(OrderedDict([(f'block_{block_idx+1}', block) for block_idx, block in enumerate(blocks)]))
        self.flatten = FlattenTranspose()
        eg_input = torch.randn((1, *self.input_shape))
        eg_output = self.flatten(self.conv_stage(eg_input))
        in_dims = np.prod(eg_output.shape)
        fc_modules = []
        for dense_idx, dense_width in enumerate(self.dense_widths):
            fc_modules.append((f'dense_{dense_idx+1}', nn.Linear(in_dims, dense_width)))
            fc_modules.append((f'selu_{dense_idx+1}', nn.SELU()))
            in_dims = dense_width
        fc_modules.append(('classifier', nn.Linear(in_dims, self.output_classes)))
        self.fc_stage = nn.Sequential(OrderedDict(fc_modules))
        
        for mod in self.modules():
            if isinstance(mod, nn.Conv1d):
                nn.init.kaiming_uniform_(mod.weight)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.BatchNorm1d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
            else:
                pass
        nn.init.xavier_uniform_(self.fc_stage.classifier.weight)
    
    def forward(self, x):
        x = self.conv_stage(x)
        x = self.flatten(x)
        x = self.fc_stage(x)
        return x