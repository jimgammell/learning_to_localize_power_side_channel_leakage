from typing import *
import time
import numpy as np
from numba import jit
from scipy.stats import kendalltau
import torch
from torch.utils.data import Dataset

from ._functional import *

class GMMMutInfCorrelation:
    def __init__(self,
        leakage_assessment: Sequence[float],
        device: Optional[str] = 'cpu'
    ):
        self.leakage_assessment = leakage_assessment
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_dataset(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        traces, labels = extract_dataset(dataset, np.arange(len(self.leakage_assessment)))
        

class GMMPerformanceCorrelation:
    def __init__(self,
        feature_ranking: Sequence[int],
        feature_count: int = 1,
        device: Optional[str] = 'cpu'
    ):
        if feature_count != 1:
            raise NotImplementedError
        if len(feature_ranking) % feature_count != 0:
            resid_len = feature_count - (len(feature_ranking) % feature_count)
            feature_ranking = np.concatenate([feature_ranking, feature_ranking[-resid_len:]])
        self.feature_ranking = feature_ranking
        self.feature_count = feature_count
        self.partitions = self.feature_ranking.reshape(-1, self.feature_count)
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_dataset(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        traces, labels = extract_dataset(dataset, self.partitions)
        traces = torch.tensor(traces, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        return traces, labels
    
    def profile(self, profiling_dataset: Dataset):
        traces, labels = self.load_dataset(profiling_dataset)
        self.class_count = get_class_count(labels)
        self.means, self.vars = get_means_and_vars(traces, labels, self.class_count)
        self.log_p_y = get_log_p_y(labels)
        assert torch.all(torch.isfinite(self.means))
        assert torch.all(torch.isfinite(self.vars))
        assert torch.all(torch.isfinite(self.log_p_y))
    
    def has_profiled(self) -> bool:
        return hasattr(self, 'class_count') and hasattr(self, 'means') and hasattr(self, 'vars')
    
    def get_logits(self, traces: torch.Tensor) -> np.ndarray:
        assert self.has_profiled()
        log_p_x_mid_y = get_log_p_x_mid_y(traces, self.means, self.vars)
        log_p_x_y = log_p_x_mid_y + self.log_p_y.unsqueeze(0).unsqueeze(-1)
        return log_p_x_y.cpu().numpy()
    
    def get_ranks(self, logits, labels):
        logits = logits.transpose(0, 2, 1)
        labels = labels.cpu().numpy()
        @jit(nopython=True)
        def _get_ranks(logits, labels):
            partition_count, datapoint_count, class_count = logits.shape
            ranks = np.full((partition_count, datapoint_count), np.nan, dtype=int)
            for partition_idx in range(partition_count):
                for datapoint_idx in range(datapoint_count):
                    label = labels[datapoint_idx]
                    correct_logit = logits[partition_idx, datapoint_idx, label]
                    rank = np.sum(logits[partition_idx, datapoint_idx] >= correct_logit) - 1
                    ranks[partition_idx, datapoint_idx] = rank
            return ranks
        return _get_ranks(logits, labels)
    
    def get_hard_rank_correlation(self, ranks):
        mean_ranks = ranks.mean(axis=-1)
        return kendalltau(mean_ranks, -np.arange(len(mean_ranks))).statistic
    
    def get_soft_rank_correlation(self, ranks):
        mean_ranks = ranks.mean(axis=-1)
        var_ranks = ranks.var(axis=-1)
        return soft_kendall_tau(ranks, -np.arange(len(mean_ranks)))
    
    def __call__(self, attack_dataset, return_ranks=False):
        traces, labels = self.load_dataset(attack_dataset)
        logits = self.get_logits(traces)
        ranks = self.get_ranks(logits, labels)
        metric = self.get_soft_rank_correlation(ranks)
        if return_ranks:
            return metric, ranks
        else:
            return metric