from copy import copy
import numpy as np
from torch.utils.data import Dataset, Subset, ConcatDataset

from utils.chunk_iterator import chunk_iterator

def calculate_dataset_stats(dataset: Dataset, chunk_size: int = 1024, calc_var=True):
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    if isinstance(dataset, ConcatDataset):
        orig_transforms = []
        for base_dataset in dataset.datasets:
            orig_transforms.append(base_dataset.transform)
            base_dataset.transform = None
    else:
        orig_transform = copy(base_dataset.transform)
        base_dataset.transform = None
    
    mean = np.zeros((base_dataset.timesteps_per_trace,), dtype=np.float32)
    for idx, (trace, *_) in enumerate(chunk_iterator(dataset, chunk_size=chunk_size)):
        mean = (idx/(idx+1))*mean + (1/(idx+1))*trace
    if calc_var:
        var = np.zeros((base_dataset.timesteps_per_trace,), dtype=np.float32)
        for idx, (trace, *_) in enumerate(chunk_iterator(dataset, chunk_size=chunk_size)):
            var = (idx/(idx+1))*var + (1/(idx+1))*(trace - mean)**2
    if isinstance(dataset, ConcatDataset):
        for base_dataset, orig_transform in zip(dataset.datasets, orig_transforms):
            base_dataset.transform = orig_transform
    else:
        base_dataset.transform = orig_transform
    return (mean, var) if calc_var else mean