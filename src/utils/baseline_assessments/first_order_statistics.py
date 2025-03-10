from typing import *
from copy import copy
import numpy as np
from torch.utils.data import Dataset, Subset

from utils.chunk_iterator import chunk_iterator

def _prepare_dataset(dataset: Dataset):
    dataset = copy(dataset)
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    base_dataset.return_metadata = True
    base_dataset.transform = None
    base_dataset.target_transform = None
    return dataset, base_dataset

def _get_key(target: Union[str, int], byte: Optional[int]):
    if byte is None:
        return target
    else:
        return (target, byte)

def _get_hamming_weight(x):
    return np.unpackbits(x.astype(np.uint8)).astype(np.float64).sum()

class FirstOrderStatistics:
    def __init__(self, dataset, targets: Union[str, Sequence[str]] = 'label', bytes: Optional[int] = None, chunk_size: int = 1024):
        self.dataset, self.base_dataset = _prepare_dataset(dataset)
        if isinstance(targets, str):
            targets = [targets]
        if (bytes is None) or isinstance(bytes, int):
            bytes = [bytes]
        self.targets = targets
        self.bytes = bytes
        self.chunk_size = chunk_size
        self.timesteps_per_trace = self.base_dataset.timesteps_per_trace
        self.class_count = self.base_dataset.class_count
        self.compute_basic_stats()
        self.compute_snr()
        self.compute_sosd()
        self.compute_cpa()
    
    def get_chunk_iterator(self):
        return chunk_iterator(self.dataset, chunk_size=self.chunk_size)

    def get_key_iterator(self):
        for target in self.targets:
            for byte in self.bytes:
                yield target, byte, _get_key(target, byte)

    def compute_basic_stats(self):
        if not(hasattr(self, 'per_target_means')) or not(hasattr(self, 'per_target_counts')) or not(hasattr(self, 'mean_hamming_weights')):
            self.per_target_means = {key: np.zeros((self.class_count, self.timesteps_per_trace), dtype=np.float64) for *_, key in self.get_key_iterator()}
            self.per_target_counts = {key: np.zeros(self.dataset.class_count, dtype=int) for *_, key in self.get_key_iterator()}
            self.mean_hamming_weights = {key: 0. for *_, key in self.get_key_iterator()}
            self.mean_trace = np.zeros(self.timesteps_per_trace, dtype=np.float64)
            for count, (trace, _, metadata) in enumerate(self.get_chunk_iterator()):
                for target, byte, key in self.get_key_iterator():
                    target_val = metadata[target]
                    if (byte is not None) and (target_val.size > 1):
                        target_val = target_val[byte]
                    current_mean = self.per_target_means[key][target_val, :]
                    current_count = self.per_target_counts[key][target_val]
                    self.per_target_means[key][target_val, :] = (current_count/(current_count+1))*current_mean + (1/(current_count+1))*trace
                    self.per_target_counts[key][target_val] += 1
                    self.mean_hamming_weights[key] = (
                        (current_count/(current_count+1))*self.mean_hamming_weights[key]
                        + (1/(current_count+1))*_get_hamming_weight(target_val)
                    )
                    self.mean_trace = (count/(count+1))*self.mean_trace + (1/(count+1))*trace
        if not(hasattr(self, 'noise_variance')) or not(hasattr(self, 'unnormalized_correlation')) or not(hasattr(self, 'hamming_weight_variance')):
            self.noise_variance = {key: np.zeros(self.timesteps_per_trace, dtype=np.float64) for *_, key in self.get_key_iterator()}
            self.trace_variance = np.zeros(self.timesteps_per_trace, dtype=np.float64)
            self.unnormalized_correlation = {key: np.zeros(self.timesteps_per_trace, dtype=np.float64) for *_, key in self.get_key_iterator()}
            self.hamming_weight_variance = {key: 0. for *_, key in self.get_key_iterator()}
            for count, (trace, _, metadata) in enumerate(self.get_chunk_iterator()):
                for target, byte, key in self.get_key_iterator():
                    target_val = metadata[target]
                    if (byte is not None) and (target_val.size > 1):
                        target_val = target_val[byte]
                    key = _get_key(target, byte)
                    current_var = self.noise_variance[key]
                    self.noise_variance[key] = (count/(count+1))*current_var + (1/(count+1))*(trace - self.per_target_means[key][target_val])**2
                    self.unnormalized_correlation[key] = (
                        (count/(count+1))*self.unnormalized_correlation[key]
                        + (1/(count+1))*(trace - self.mean_trace)*(_get_hamming_weight(target_val)-self.mean_hamming_weights[key])
                    )
                    self.hamming_weight_variance[key] = (
                        (count/(count+1))*self.hamming_weight_variance[key]
                        + (1/(count+1))*(_get_hamming_weight(target_val)-self.mean_hamming_weights[key])**2
                    )
                    self.trace_variance = (count/(count+1))*self.trace_variance + (1/(count+1))*(trace - self.mean_trace)**2
    
    def compute_snr(self):
        signal_variance = {key: np.var(self.per_target_means[key], axis=0) for *_, key in self.get_key_iterator()}
        snr_vals = {key: signal_variance[key]/self.noise_variance[key] for *_, key in self.get_key_iterator()}
        self.snr_vals = snr_vals

    def compute_sosd(self):
        sosd_vals = {key: np.zeros(self.timesteps_per_trace, dtype=np.float64) for *_, key in self.get_key_iterator()}
        for *_, key in self.get_key_iterator():
            for i in range(self.class_count):
                for j in range(i+1, self.class_count):
                    sosd_vals[key] += (self.per_target_means[key][i, :] - self.per_target_means[key][j, :])**2
        self.sosd_vals = sosd_vals
    
    def compute_cpa(self):
        cpa_vals = {key: self.unnormalized_correlation[key] / np.sqrt(self.trace_variance*self.hamming_weight_variance[key]) for *_, key in self.get_key_iterator()}
        self.cpa_vals = cpa_vals