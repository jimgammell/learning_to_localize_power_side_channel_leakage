import os
import numpy as np
from numba import jit
import torch
from torch.utils.data import Dataset

from utils.aes import *

@jit(nopython=True)
def to_key_preds(int_var_preds, args, constants):
    if args.ndim == 1:
        plaintext = args[0]
        offset = args[1]
    elif args.ndim == 2:
        plaintext = args[:, 0]
        offset = args[:, 1]
    else:
        assert False
    mask = constants[0]
    return int_var_preds[AES_SBOX[np.arange(256, dtype=np.uint8) ^ plaintext] ^ mask[(offset+1)%16]]

class DPAv4(Dataset):
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
            self.traces = np.load(os.path.join(self.root, 'DPAv4_dataset', 'profiling_traces_dpav4.npy')).astype(np.float32)
            self.targets = np.load(os.path.join(self.root, 'DPAv4_dataset', 'profiling_labels_dpav4.npy')).astype(np.uint8)
            self.plaintexts = np.load(os.path.join(self.root, 'DPAv4_dataset', 'profiling_plaintext_dpav4.npy')).astype(np.uint8)
        else:
            self.traces = np.load(os.path.join(self.root, 'DPAv4_dataset', 'attack_traces_dpav4.npy')).astype(np.float32)
            self.targets = np.load(os.path.join(self.root, 'DPAv4_dataset', 'attack_labels_dpav4.npy')).astype(np.uint8)
            self.plaintexts = np.load(os.path.join(self.root, 'DPAv4_dataset', 'attack_plaintext_dpav4.npy')).astype(np.uint8)
        self.key = np.load(os.path.join(self.root, 'DPAv4_dataset', 'key.npy')).astype(np.uint8)
        self.mask = np.load(os.path.join(self.root, 'DPAv4_dataset', 'mask.npy')).astype(np.uint8)
        self.metadata = {
            'subbytes': self.targets,
            'plaintext': self.plaintexts[:, 0],
            'key': self.key[0] * np.ones_like(self.targets),
            'label': self.targets
        }
        if not self.train:
            self.offset = np.load(os.path.join(self.root, 'DPAv4_dataset', 'attack_offset_dpav4.npy'))[:, 0].astype(np.uint8)
            self.metadata.update({'offset': self.offset})
        self.dataset_length = len(self.traces)
        self.data_shape = self.traces[0].shape
        self.timesteps_per_trace = self.data_shape[-1]
        self.class_count = 256
    
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