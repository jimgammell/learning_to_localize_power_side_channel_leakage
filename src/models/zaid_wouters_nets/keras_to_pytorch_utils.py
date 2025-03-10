import h5py
from itertools import chain
import numpy as np
import torch
from torch import nn

class FlattenTranspose(nn.Module): # The intermediate conv activations in Keras are transposed relative to those in PyTorch
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=1)
        return x

def unpack_keras_params(pretrained_model_path):
    with h5py.File(pretrained_model_path) as model_state:
        keras_params = {}
        for layer_name in model_state['model_weights'].keys():
            module = model_state['model_weights'][layer_name]
            if not layer_name in module.keys():
                continue
            module = module[layer_name]
            keras_params[layer_name] = {
                key: np.array(val) for key, val in module.items()
            }
    return keras_params

def keras_to_torch_param(keras_param, torch_param, transpose=False, dtype=np.float32):
    param = np.array(keras_param, dtype=dtype)
    if transpose:
        param = param.transpose()
    torch_param.data = torch.from_numpy(param)

def keras_to_torch_mod(keras_mod, torch_mod):
    if isinstance(torch_mod, nn.Conv1d):
        keras_to_torch_param(keras_mod['kernel:0'], torch_mod.weight, transpose=True)
        keras_to_torch_param(keras_mod['bias:0'], torch_mod.bias)
    elif isinstance(torch_mod, nn.Linear):
        keras_to_torch_param(keras_mod['kernel:0'], torch_mod.weight, transpose=True)
        keras_to_torch_param(keras_mod['bias:0'], torch_mod.bias)
    elif isinstance(torch_mod, nn.BatchNorm1d):
        keras_to_torch_param(keras_mod['gamma:0'], torch_mod.weight)
        keras_to_torch_param(keras_mod['beta:0'], torch_mod.bias)
        keras_to_torch_param(keras_mod['moving_mean:0'], torch_mod.running_mean)
        keras_to_torch_param(keras_mod['moving_variance:0'], torch_mod.running_var)
    else:
        assert False

def load_pretrained_keras_weights(model, pretrained_weights_path):
    with h5py.File(pretrained_weights_path) as pretrained_weights:
        for name, param in chain(model.named_parameters(), model.named_buffers()):
            _, layer_name, param_type = name.split('.')
            keras_layer = pretrained_weights['model_weights'][layer_name][layer_name]
            if param_type == 'weight':
                if 'kernel:0' in keras_layer.keys():
                    keras_param = keras_layer['kernel:0']
                elif 'gamma:0' in keras_layer.keys():
                    keras_param = keras_layer['gamma:0']
                else:
                    assert False
                param.data = torch.from_numpy(np.array(keras_param, dtype=np.float32).transpose())
            elif param_type == 'bias':
                if 'bias:0' in keras_layer.keys():
                    keras_param = keras_layer['bias:0']
                elif 'beta:0' in keras_layer.keys():
                    keras_param = keras_layer['beta:0']
                else:
                    assert False
                param.data = torch.from_numpy(np.array(keras_param, dtype=np.float32))
            elif param_type == 'running_mean':
                param.data = torch.from_numpy(np.array(keras_layer['moving_mean:0'], dtype=np.float32))
            elif param_type == 'running_var':
                param.data = torch.from_numpy(np.array(keras_layer['moving_variance:0'], dtype=np.float32))