import numpy as np
import numba
from numba import jit
import torch
from torch.utils.data import Dataset

from ..common import *

@jit(nopython=True)
def apply_ema(trace, ema_coeff):
    avg = np.mean(trace)
    for time_idx in range(trace.shape[-1]):
        trace[time_idx] = ema_coeff*avg + (1-ema_coeff)*trace[time_idx]
        avg = trace[time_idx]
    return trace

@jit('float32(uint32)', nopython=True)
def get_hamming_weight(number):
    hamming_weight = np.uint32(0)
    while number != 0:
        hamming_weight += number & 1
        number >>= 1
    hamming_weight = np.float32(hamming_weight)
    return hamming_weight

@jit(nopython=True)
def generate_trace(data, timestep_count, fixed_noise, random_noise, data_var):
    trace = np.empty((timestep_count,), dtype=np.float32)
    for t_idx in range(timestep_count):
        data_val = data[t_idx]
        data_power = np.sqrt(data_var)*(4 - get_hamming_weight(data_val))/np.sqrt(2)
        trace[t_idx] = data_power + fixed_noise[t_idx] + random_noise[t_idx]
    return trace