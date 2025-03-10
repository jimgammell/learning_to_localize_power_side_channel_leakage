import builtins
import os
import yaml
import shutil
import time
import datetime
from typing import *
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import multiprocessing

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{times} \usepackage{amsmath} \usepackage{amssymb}'

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = 10*gpu_properties.major + gpu_properties.minor
    if arch >= 70:
        torch.set_float32_matmul_precision('high')

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
CONFIG_DIRNAME = 'config'
CONFIG_DIR = os.path.join(PROJ_DIR, CONFIG_DIRNAME)
with open(os.path.join(CONFIG_DIR, 'global_variables.yaml')) as f:
    global_config = yaml.load(f, Loader=yaml.FullLoader)
RESOURCE_DIRNAME = global_config['resource_dirname'] if 'resource_dirname' in global_config else 'resources'
RESOURCE_DIR = os.path.join(PROJ_DIR, RESOURCE_DIRNAME)
OUTPUT_DIRNAME = global_config['output_dirname'] if 'output_dirname' in global_config else 'outputs'
OUTPUT_DIR = os.path.join(PROJ_DIR, OUTPUT_DIRNAME)
numpy_rng = np.random.default_rng()
_trial_name = None
_verbose = None
_seed = None

PLOT_KWARGS = {'rasterized': True}
PLOT_WIDTH = 4
SAVEFIG_KWARGS = {'dpi': 150}

def get_trial_name():
    assert _trial_name is not None
    return _trial_name

def get_trial_dir():
    trial_dir = os.path.join(OUTPUT_DIR, get_trial_name())
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

def get_log_path():
    log_path = os.path.join(get_trial_dir(), 'log.out')
    return log_path

def set_trial_name(name: str):
    global _trial_name
    _trial_name = name

def set_seed(
    seed: Optional[int] = None
) -> int:
    if seed is None:
        seed = time.time_ns() & 0xFFFFFFFF
    global NUMPY_RNG
    NUMPY_RNG = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return seed

def set_verbosity(
    verbose: bool = True
):
    global _verbose
    _verbose = verbose

_print_backlog = []
def print(*args, **kwargs):
    global _print_backlog
    if _trial_name is not None:
        with open(get_log_path(), 'a+') as f:
            for _args, _kwargs in _print_backlog:
                builtins.print(*_args, file=f, **_kwargs)
            _print_backlog = []
            builtins.print(*args, file=f, **kwargs)
    else:
        _print_backlog.append((args, kwargs))
    if _verbose:
        builtins.print(*args, **kwargs)

_seed = set_seed()
set_verbosity()
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESOURCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)