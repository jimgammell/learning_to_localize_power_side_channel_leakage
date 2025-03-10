from typing import *
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.metrics import get_rank

@torch.no_grad()
def compute_dnn_performance_auc(
    dataloader: DataLoader,
    dnn: nn.Module,
    leakage_assessment: np.ndarray,
    device: Optional[str] = None,
    cluster_count: Optional[int] = 100,
    average: bool = True, # if false, will return the curve itself rather than its mean
    logarithmic_mode: bool = False, # if true, will ablate 1, then 2, then 4, etc. points
    multi_classifiers: bool = False
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dnn = dnn.to(device)
    traces, labels = next(iter(dataloader))
    traces, labels = traces.to(device), labels.to(device)
    if cluster_count is None:
        cluster_count = traces.shape[-1]
    timesteps_per_trace = traces.shape[-1]
    leakage_ranking = leakage_assessment.reshape(-1).argsort()
    if not logarithmic_mode:
        indices = leakage_ranking
        if not len(indices) % cluster_count == 0:
            indices = np.concatenate([
                indices[:cluster_count - (len(indices)%cluster_count)], indices
            ])
        indices = torch.tensor(indices.reshape(cluster_count, -1), dtype=torch.long)
    else:
        indices = [np.array([])]
        idx = 0
        while len(indices[-1]) < timesteps_per_trace:
            indices.append(leakage_ranking[:2**idx])
            idx += 1
    mask = torch.zeros(1, timesteps_per_trace, dtype=torch.float, device=device)
    ranks = []
    for index_cluster in indices:
        mask[:, index_cluster] = 1.
        masked_traces = mask.unsqueeze(0)*traces
        if multi_classifiers:
            logits = dnn(masked_traces, mask.unsqueeze(0).repeat(traces.shape[0], 1, 1))
        else:
            logits = dnn(masked_traces)
        rank = get_rank(logits, labels).mean()
        ranks.append(rank)
    if average:
        reverse_auc = np.mean(ranks)
    else:
        reverse_auc = np.array(ranks)
        
    leakage_ranking = leakage_assessment.reshape(-1).argsort()[::-1].copy()
    if not logarithmic_mode:
        indices = leakage_ranking
        if not len(indices) % cluster_count == 0:
            indices = np.concatenate([
                indices[:cluster_count - (len(indices)%cluster_count)], indices
            ])
        indices = torch.tensor(indices.reshape(cluster_count, -1), dtype=torch.long)
    else:
        indices = [np.array([])]
        idx = 0
        while len(indices[-1]) < timesteps_per_trace:
            indices.append(leakage_ranking[:2**idx])
            idx += 1
    mask = torch.zeros(1, timesteps_per_trace, dtype=torch.float, device=device)
    ranks = []
    for index_cluster in indices:
        mask[:, index_cluster] = 1.
        masked_traces = mask.unsqueeze(0)*traces
        if multi_classifiers:
            logits = dnn(masked_traces, mask.unsqueeze(0).repeat(traces.shape[0], 1, 1))
        else:
            logits = dnn(masked_traces)
        rank = get_rank(logits, labels).mean()
        ranks.append(rank)
    if average:
        forward_auc = np.mean(ranks)
    else:
        forward_auc = np.array(ranks)
    
    return {'forward_dnn_auc': forward_auc, 'reverse_dnn_auc': reverse_auc}