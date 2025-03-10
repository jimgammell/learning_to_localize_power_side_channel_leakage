import os
import numpy as np
from numba import jit
from torch.utils.data import Dataset

from utils.aes import *

@jit(nopython=True)
def to_key_preds(int_var_preds, plaintext, constants=None):
    return int_var_preds[AES_SBOX[np.arange(256, dtype=np.uint8) ^ plaintext]]

class AES_RD(Dataset):
    def __init__(self,
        root=None,
        train=True,
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = False
        self.construct()
    
    def construct(self):
        if self.train:
            self.traces = np.load(os.path.join(self.root, 'AES_RD_dataset', 'profiling_traces_AES_RD.npy')).astype(np.float32)
            self.targets = np.load(os.path.join(self.root, 'AES_RD_dataset', 'profiling_labels_AES_RD.npy')).astype(np.uint8)
            self.plaintexts = np.load(os.path.join(self.root, 'AES_RD_dataset', 'profiling_plaintext_AES_RD.npy'))[:, 0].astype(np.uint8)
        else:
            self.traces = np.load(os.path.join(self.root, 'AES_RD_dataset', 'attack_traces_AES_RD.npy')).astype(np.float32)
            self.targets = np.load(os.path.join(self.root, 'AES_RD_dataset', 'attack_labels_AES_RD.npy')).astype(np.uint8)
            self.plaintexts = np.load(os.path.join(self.root, 'AES_RD_dataset', 'attack_plaintext_AES_RD.npy'))[:, 0].astype(np.uint8)
        self.key = np.load(os.path.join(self.root, 'AES_RD_dataset', 'key.npy')).astype(np.uint8)
        self.metadata = {
            'subbytes': self.targets,
            'plaintext': self.plaintexts,
            'key': self.key[0] * np.ones_like(self.targets)
        }
        self.dataset_length = len(self.traces)
        self.data_shape = self.traces[0].shape
        self.timesteps_per_trace = self.data_shape[-1]
    
    def __getitem__(self, indices):
        trace = self.traces[indices, np.newaxis, :]
        target = self.targets[indices].squeeze()
        metadata = {key: val[indices].squeeze() for key, val in self.metadata.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self):
        return self.dataset_length