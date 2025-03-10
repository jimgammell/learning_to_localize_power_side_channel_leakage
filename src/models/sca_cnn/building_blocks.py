import torch
from torch import nn

from ..common import *
from ..soft_xor_layer import SoftXOR

class NoiseConditionalBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, noise_shape, **kwargs):
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

class Head(Module):
    def __init__(self, input_dims, output_dims, hidden_dims=256, activation_constructor=nn.ELU, xor_output=False):
        super().__init__(**{key: val for key, val in locals().items() if key not in ('self', 'key', 'val')})
    
    def construct(self):
        self.dense_1 = nn.Linear(self.input_dims, self.hidden_dims)
        self.act_1 = self.activation_constructor()
        self.dense_2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.act_2 = self.activation_constructor()
        if self.xor_output:
            self.output = SoftXOR(self.hidden_dims, int(np.log(self.output_dims)/np.log(2)), skip=False, xor_copies=8)
        else:
            self.output = nn.Linear(self.hidden_dims, self.output_dims)
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.dense_1(x)
        x = self.act_1(x)
        x = self.dense_2(x)
        x = self.act_2(x)
        x = self.output(x)
        return x

class Block(Module):
    def __init__(self,
        in_channels, out_channels, conv_kernel_size=11, pool_kernel_size=4,
        activation_constructor=nn.ELU, noise_shape=None
    ):
        super().__init__(**{key: val for key, val in locals().items() if key not in ('self', 'key', 'val')})
    
    def construct(self):
        self.conv = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_kernel_size//2)
        self.act = self.activation_constructor()
        if self.noise_shape is None:
            self.norm = nn.BatchNorm1d(self.out_channels)
        else:
            self.norm = NoiseConditionalBatchNorm1d(self.noise_shape, num_features=self.out_channels)
        self.pool = nn.AvgPool1d(self.pool_kernel_size)
    
    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, *args):
        if self.noise_shape is not None:
            (x, noise) = args
        else:
            (x,) = args
        x = self.conv(x)
        x = self.act(x)
        if self.noise_shape is not None:
            x = self.norm(x, noise)
        else:
            x = self.norm(x)
        x = self.pool(x)
        return x