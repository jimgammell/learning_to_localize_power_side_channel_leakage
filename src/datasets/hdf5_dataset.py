# Based on comments here: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

from common import *

def collate_fn(batch):
    return [torch.concatenate(samples, dim=0) for samples in batch]

class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_workers=0, shuffle=False):
        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)
        self.length = int(np.ceil(len(self.data_source)/self.batch_size))
        if num_workers != 0:
            assert batch_size // num_workers > 0
        self.subbatch_sizes = [(idx+1)*(batch_size//num_workers) for idx in range(num_workers if num_workers > 0 else 1)]
        self.subbatch_sizes[0] += self.batch_size - sum(self.subbatch_sizes)
        self.subbatch_indices = np.cumsum([0] + self.subbatch_sizes)
    
    def __iter__(self):
        if self.shuffle:
            indices = NUMPY_RNG.choice(len(self.data_source), len(self.data_source), replace=False)
        else:
            indices = np.arange(len(self.data_source))
        for bidx in range(self.length):
            bindices = indices[bidx*self.batch_size : min((bidx+1)*self.batch_size, len(self.data_source))]
            yield [bindices[idx1:idx2] for idx1, idx2 in zip(self.subbatch_indices[:-1], self.subbatch_indices[1:])]
    
    def __len__(self):
        return self.length

class HDF5DataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=0, shuffle=False, **kwargs):
        assert (batch_size % num_workers) == 0
        assert not(any(key in kwargs.keys() for key in ('sampler', 'batch_sampler', 'collate_fn', 'persistent_workers')))
        batch_sampler = BatchSampler(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        super().__init__(
            dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, persistent_workers=True, **kwargs
        )

class ConcatHDF5Dataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.subbatch_indices = np.cumsum([0] + [len(x) for x in self.datasets])
        self.length = sum(len(x) for x in self.datasets)
    
    def __getitem__(self, indices):
        items = None
        for dataset, idx1, idx2 in zip(self.datasets, self.subbatch_indices[:-1], self.subbatch_indices[1:]):
            subindices = np.array([x for x in indices if idx1 <= x < idx2])
            if items is None:
                items = dataset[subindices]
            else:
                _items = dataset[subindices]
                items = [torch.concatenate([x0, x1], dim=0) for x0, x1 in zip(items, _items)]
        return items
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        for dataset in self.datasets:
            del dataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, keys, transforms=None):
        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)
        super().__init__()
        self.hdf5_file = None
        if self.transforms is None:
            self.transforms = len(self.keys)*[None]
        self.length = None
        with open(self.hdf5_path, 'r') as hdf5_file:
            for key in self.keys():
                if self.length is None:
                    self.length = hdf5_file[key].shape[0]
                else:
                    assert self.length == hdf5_file[key].shape[0]
    
    def get_hdf5_file(self): # Dataset will no longer be serializable after this is called.
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self.hdf5_file

    def __getitem__(self, indices):
        hdf5_file = self.get_hdf5_file()
        items = []
        for key, transform in zip(self.keys, self.transforms):
            item = np.array(hdf5_file[key][indices, ...])
            if transform is not None:
                item = transform(item)
            items.append(item)
        return items
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()