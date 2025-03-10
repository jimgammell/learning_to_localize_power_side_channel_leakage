import os
import numpy as np
from numba import jit
import h5py
import torch
from torch.utils.data import Dataset

from utils.aes import *

# Note: labels are given by AES_INVERSE_SBOX[ciphertexts[:, 11]] ^ ciphertexts[:, 7] -- I think this is backwards from what a few papers state

KEY = np.array([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c], dtype=np.uint8)

@jit(nopython=True)
def to_key_preds(int_var_preds, args, constants=None):
    if args.ndim == 1:
        ciphertext_11 = args[0]
        ciphertext_7 = args[1]
    elif args.ndim == 2:
        ciphertext_11 = args[:, 0]
        ciphertext_7 = args[:, 1]
    return int_var_preds[AES_INVERSE_SBOX[np.arange(256, dtype=np.uint8) ^ ciphertext_11] ^ ciphertext_7]

class AES_HD(Dataset):
    def __init__(self,
        root=None,
        train=True,
        extended_version=False,
        target_values='last_state',
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.extended_version = extended_version
        self.target_values = [target_values] if isinstance(target_values, str) else target_values
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = False
        self.construct()
    
    def construct(self):
        if not self.extended_version:
            if self.train:
                self.traces = np.load(os.path.join(self.root, 'AES_HD_dataset', 'profiling_traces_AES_HD.npy')).astype(np.float32)
                self.targets = np.load(os.path.join(self.root, 'AES_HD_dataset', 'profiling_labels_AES_HD.npy')).astype(np.uint8)
                self.ciphertexts = np.load(os.path.join(self.root, 'AES_HD_dataset', 'profiling_ciphertext_AES_HD.npy')).astype(np.uint8)
            else:
                self.traces = np.load(os.path.join(self.root, 'AES_HD_dataset', 'attack_traces_AES_HD.npy')).astype(np.float32)
                self.targets = np.load(os.path.join(self.root, 'AES_HD_dataset', 'attack_labels_AES_HD.npy')).astype(np.uint8)
                self.ciphertexts = np.load(os.path.join(self.root, 'AES_HD_dataset', 'attack_ciphertext_AES_HD.npy')).astype(np.uint8)
            self.metadata = {
                'label': self.targets,
                'ciphertext': self.ciphertexts,
                'ciphertext_11': self.ciphertexts[:, 11],
                'ciphertext_7': self.ciphertexts[:, 7],
                'last_state': self.targets,
                'key': np.zeros_like(self.targets) # Not the actual key, but the output of the key schedule at this point in the algorithm
            }
        else:
            with h5py.File(os.path.join(self.root, 'aes_hd_ext.h5'), 'r') as f:
                if self.train:
                    dataset = f['Profiling_traces']
                else:
                    dataset = f['Attack_traces']
                self.traces = np.array(dataset['traces'], dtype=np.float32)
                self.plaintexts = np.array(dataset['metadata']['plaintext'], dtype=np.uint8)
                self.ciphertexts = np.array(dataset['metadata']['ciphertext'], dtype=np.uint8)
            self.targets = AES_INVERSE_SBOX[self.ciphertexts[:, 11]] ^ self.ciphertexts[:, 7]
            self.metadata = {
                'label': self.targets,
                'ciphertext': self.ciphertexts,
                'ciphertext_11': self.ciphertexts[:, 11],
                'ciphertext_7': self.ciphertexts[:, 7],
                'last_state': self.targets,
                'key': np.zeros_like(self.targets)
            }
        self.dataset_length = len(self.traces)
        assert self.dataset_length == len(self.targets) == len(self.ciphertexts)
        self.data_shape = self.traces[0].shape
        self.timesteps_per_trace = self.data_shape[-1]
        self.class_count = 256
    
    def __getitem__(self, indices):
        trace = self.traces[indices, np.newaxis, :]
        #trace = trace + np.random.rand(*trace.shape).astype(np.float32) - 0.5
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