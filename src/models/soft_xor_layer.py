import numpy as np
from numba import jit
import torch
from torch import nn

class SoftXOR(nn.Module):
    def __init__(self, in_dims, out_bits, xor_copies=1, skip=False):
        super().__init__()
        self.in_dims = in_dims
        self.out_bits = out_bits
        self.xor_copies = xor_copies
        self.skip = skip
        self.to_x_y = nn.Linear(self.in_dims, (2*self.xor_copies+(1 if self.skip else 0))*2**self.out_bits)
        self.to_out = nn.Conv1d(self.xor_copies+(1 if self.skip else 0), 1, 1)
        nn.init.xavier_uniform_(self.to_x_y.weight)
        nn.init.constant_(self.to_x_y.bias, 0)
        nn.init.constant_(self.to_out.weight, 1/(2*self.xor_copies+(1 if self.skip else 0)))
        nn.init.constant_(self.to_out.bias, 0)
    
    @torch.compile
    def forward(self, input):
        batch_size = input.size(0)
        out_dims = 2**self.out_bits
        input = input.view(batch_size, -1)
        x_y = self.to_x_y(input)
        x = x_y[:, :self.xor_copies*out_dims]
        y = x_y[:, self.xor_copies*out_dims:2*self.xor_copies*out_dims]
        if self.skip:
            skip = x_y[:, 2*self.xor_copies*out_dims:]
        x = x.reshape(batch_size*self.xor_copies, out_dims)
        y = y.reshape(batch_size*self.xor_copies, out_dims)
        z = soft_xor(x, y)
        z = z.reshape(batch_size, self.xor_copies, out_dims)
        if self.skip:
            skip = skip.reshape(batch_size, 1, out_dims)
            z = torch.cat([z, skip], dim=1)
        out = self.to_out(z).squeeze(1)
        return out

def soft_xor(x, y):
    N = x.shape[-1]
    n = int(np.log2(N))
    device = x.device
    indices = torch.arange(N, device=device).unsqueeze(1) ^ torch.arange(N, device=device).unsqueeze(0)
    z = torch.logsumexp(x.unsqueeze(2) + y[:, indices], dim=1)
    return z