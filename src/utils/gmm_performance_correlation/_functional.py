from typing import *
from copy import copy
import numpy as np
from numba import jit
from scipy.special import erf
import torch
from torch.utils.data import Dataset, Subset

from utils.chunk_iterator import chunk_iterator

@torch.no_grad()
def extract_dataset(dataset: Dataset, partitions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dataset = copy(dataset)
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    base_dataset.transform = base_dataset.target_transform = None
    datapoint_count = len(dataset)
    partition_count, feature_count = partitions.shape
    partitions = partitions.reshape(-1)
    traces = np.full((partition_count, datapoint_count, feature_count), np.nan, dtype=np.float32)
    labels = np.full((datapoint_count,), -1, dtype=np.int64)
    for datapoint_idx, (trace, label) in enumerate(chunk_iterator(dataset)):
        trace = trace.squeeze()
        traces[:, datapoint_idx, :] = trace[partitions].reshape(partition_count, feature_count)
        labels[datapoint_idx] = label
    assert np.all(np.isfinite(traces))
    assert np.all(labels >= 0)
    traces -= traces.mean(axis=1, keepdims=True)
    traces /= traces.std(axis=1, keepdims=True)
    return traces, labels

@torch.no_grad()
def get_class_count(labels: torch.Tensor) -> int:
    return len(labels.unique())

@torch.no_grad()
def get_means_and_vars(traces: torch.Tensor, labels: torch.Tensor, class_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    partition_count, datapoint_count, feature_count = traces.shape
    if feature_count != 1:
        raise NotImplementedError
    assert datapoint_count == len(labels)
    means = torch.full((partition_count, class_count, feature_count), np.nan, dtype=traces.dtype, device=traces.device)
    vars = torch.full((partition_count, class_count, feature_count), np.nan, dtype=traces.dtype, device=traces.device)
    for label in range(class_count):
        means[:, label, :] = traces[:, labels==label, :].mean(dim=1)
        vars[:, label, :] = traces[:, labels==label, :].var(dim=1)
    assert torch.all(torch.isfinite(means))
    assert torch.all(torch.isfinite(vars))
    return means, vars

@torch.no_grad()
def get_log_p_y(labels: torch.Tensor) -> torch.Tensor:
    _, counts = torch.unique(labels, sorted=True, return_counts=True)
    counts = counts.to(torch.float)
    return counts.log() - counts.sum().log()

@torch.no_grad()
def get_log_gaussian_density(x: torch.Tensor, mu: torch.Tensor, sigmasq: torch.Tensor) -> torch.Tensor:
    rv = -(1/(2*sigmasq))*(x-mu)**2 - 0.5*torch.log(2*np.pi*sigmasq)
    rv = rv[..., 0]
    return rv

@torch.no_grad()
def get_log_p_x_mid_y(traces: torch.Tensor, means: torch.Tensor, vars: torch.Tensor):
    _, datapoint_count, _ = traces.shape
    _, class_count, _ = means.shape
    traces = traces.unsqueeze(1).repeat(1, class_count, 1, 1)
    means = means.unsqueeze(2).repeat(1, 1, datapoint_count, 1)
    vars = vars.unsqueeze(2).repeat(1, 1, datapoint_count, 1)
    log_gaussian_densities = get_log_gaussian_density(traces, means, vars)
    return log_gaussian_densities

@jit(nopython=True)
def get_sample_mean(sample):
    count = len(sample)
    sum = np.sum(sample, axis=1)
    return sum / count

@jit(nopython=True)
def get_sample_var(sample, sample_mean):
    count = len(sample)
    sum_squared_error = np.sum((sample - sample_mean)**2, axis=1)
    return sum_squared_error / (count - 1)

@jit(nopython=True)
def get_normal_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

@jit(nopython=True)
def prob_geq(x_mean, x_var, y_mean, y_var):
    diff_mean = x_mean - y_mean
    diff_var = x_var + y_var
    if diff_var == 0:
        return 1. if diff_mean >= 0 else 0.
    else:
        z_score = -diff_mean / np.sqrt(diff_var)
        prob = 0.5 - 0.5*erf(z_score / np.sqrt(2))
    return prob

@jit(nopython=True)
def prob_diff(x_mean, x_var, x_count, y_mean, y_var, y_count):
    z = (x_mean - y_mean) / np.sqrt((x_var/x_count) + (y_var/y_count))
    return 2*get_normal_cdf(np.abs(z)) - 1

@jit(nopython=True)
def _soft_kendall_tau(target, ref, count):
    num = 0.
    denom = 0.
    for i in range(len(ref)):
        for j in range(i+1, len(ref)):
            weight = np.abs(ref[i] - ref[j])
            if ref[i] > ref[j]:
                num += weight*(1 if target[i] > target[j] else -1)
            else:
                num += weight*(1 if target[j] > target[i] else -1)
            denom += weight
    tau = num/denom
    return tau

def soft_kendall_tau(target, ref):
    count = ref.shape[-1]
    return _soft_kendall_tau(target, ref, count)