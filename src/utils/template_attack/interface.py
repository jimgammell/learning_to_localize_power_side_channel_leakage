from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset

from .functional import *
from utils.metrics.rank import _accumulate_ranks
from utils.aes import *
from common import *

class TemplateAttack:
    def __init__(self, points_of_interest: Sequence[int], target_key: str = 'subbytes', target_byte: Optional[int] = None):
        self.points_of_interest = points_of_interest
        self.target_key = target_key
        self.target_byte = target_byte
    
    def has_profiled(self):
        return hasattr(self, 'means') and hasattr(self, 'Ls') and hasattr(self, 'log_p_y')
    
    def profile(self, profiling_dataset: Dataset):
        traces, metadata = extract_dataset(profiling_dataset, self.points_of_interest, metadata_keys=self.target_key, target_byte=self.target_byte)
        targets = metadata[self.target_key]
        self.class_count = get_class_count(targets)
        self.log_p_y = get_log_p_y(targets, self.class_count)
        self.means = fit_means(traces, targets, self.class_count)
        covs = fit_covs(traces, targets, self.means, self.class_count)
        self.Ls = choldecomp_covs(covs)
    
    def get_predictions(self, attack_dataset: Dataset):
        if n_traces is None:
            n_traces = len(attack_dataset)
        assert self.has_profiled()
        traces, _ = extract_dataset(attack_dataset, self.points_of_interest, metadata_keys=[], target_byte=self.target_byte)
        log_p_x_given_y = get_log_p_x_given_y(traces, self.means, self.Ls, self.class_count)
        predictions = log_p_x_given_y + self.log_p_y
        return predictions
    
    def get_ranks(self, attack_dataset):
        traces, metadata = extract_dataset(attack_dataset, self.points_of_interest, metadata_keys=self.target_key, target_byte=self.target_byte)
        log_p_x_given_y = get_log_p_x_given_y(traces, self.means, self.Ls, self.class_count)
        predictions = log_p_x_given_y + self.log_p_y
        for key, targets in metadata.items():
            sorted_indices = np.argsort(-predictions, axis=1)
            ranks = np.array([np.where(sorted_indices[idx] == targets[idx])[0][0] for idx in range(len(predictions))])
        return ranks
    
    def attack(self, attack_dataset: Dataset, n_repetitions=100, n_traces: Optional[int] = None, arg_keys=[], constants=[], int_var_to_key_fn=subbytes_to_keys):
        if n_traces is None:
            n_traces = len(attack_dataset)
        assert len(attack_dataset) >= n_traces
        assert self.has_profiled()
        traces, metadata = extract_dataset(attack_dataset, self.points_of_interest, metadata_keys=[*arg_keys, 'key'])
        log_p_x_given_y = get_log_p_x_given_y(traces, self.means, self.Ls, self.class_count)
        indices = np.stack([np.random.choice(len(attack_dataset), n_traces, replace=False) for _ in range(n_repetitions)])
        predictions = log_p_x_given_y + self.log_p_y
        args = np.stack([metadata[arg_key] for arg_key in arg_keys], axis=-1)
        rank_over_time = _accumulate_ranks(predictions, metadata['key'], args, constants, indices, int_var_to_key_fn)
        return rank_over_time