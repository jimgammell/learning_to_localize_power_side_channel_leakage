import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

def chunk_iterator(dataset: Dataset, chunk_size: int = 1024):
    if isinstance(dataset, ConcatDataset):
        return dataset
    else:
        chunk_count = int(np.ceil(len(dataset)/chunk_size))
        for chunk_idx in range(chunk_count):
            min_idx = chunk_idx*chunk_size
            max_idx = min(len(dataset), (chunk_idx+1)*chunk_size)
            indices = np.arange(min_idx, max_idx)
            return_vals = dataset[indices]
            for datapoint_idx in range(len(return_vals[0])):
                yield tuple([
                    {key: val[datapoint_idx, ...] for key, val in x.items()} if isinstance(x, dict) else x[datapoint_idx, ...]
                    for x in return_vals
                ])