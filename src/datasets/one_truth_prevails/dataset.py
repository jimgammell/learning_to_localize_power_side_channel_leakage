import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class OneTruthPrevails(Dataset):
    def __init__(self,
        root=None,
        train=True,
        transform=None,
        target_transform=None,
        profiling_dataset_size=100000, # there are like 64 million datapoints, but I (and they in their paper) can easily get near 100% accuracy with 0.1% of them.
        overwrite_small_p=False
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = False
        self.profiling_dataset_size = profiling_dataset_size
        self.overwrite_small_p = overwrite_small_p
        
        if self.train:
            if self.overwrite_small_p or not os.path.exists(os.path.join(self.root, '1024', 'small_p.npy')): # this may take ~10 mins for profiling_dataset_size=100k
                traces = np.load(os.path.join(self.root, '1024', 'p.npy'), mmap_mode='r')
                labels = np.loadtxt(os.path.join(self.root, '1024', 'p_labels.txt'), dtype=np.uint8)
                pos_indices = np.where(labels == 1)[0]
                neg_indices = np.where(labels == 0)[0]
                assert self.profiling_dataset_size <= min(len(pos_indices), len(neg_indices))
                indices = np.concatenate([np.random.choice(pos_indices, self.profiling_dataset_size//2, replace=False), np.random.choice(neg_indices, self.profiling_dataset_size//2, replace=False)])
                np.random.shuffle(indices)
                traces = np.array(traces[indices, ...], dtype=np.float32)
                labels = np.array(labels[indices], dtype=np.int64)
                np.save(os.path.join(self.root, '1024', 'small_p.npy'), traces)
                np.save(os.path.join(self.root, '1024', 'small_p_labels.npy'), labels)
            self.traces = np.load(os.path.join(self.root, '1024', 'small_p.npy'))
            self.labels = np.load(os.path.join(self.root, '1024', 'small_p_labels.npy'))
        else:
            self.traces = np.load(os.path.join(self.root, '1024', 'a.npy')).astype(np.float32)
            self.labels = np.load(os.path.join(self.root, '1024', 'a_labels.npy')).astype(np.int64)
        self.dataset_length = len(self.traces)
        assert self.dataset_length == len(self.labels)
        self.data_shape = self.traces[0, np.newaxis, :].shape
        self.timesteps_per_trace = np.prod(self.data_shape)
        self.class_count = 2
    
    def __getitem__(self, indices):
        trace = self.traces[indices, np.newaxis, :].astype(np.float32)
        label = self.labels[indices]
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_metadata:
            return trace, label, {'label': label}
        else:
            return trace, label
    
    def __len__(self):
        return self.dataset_length