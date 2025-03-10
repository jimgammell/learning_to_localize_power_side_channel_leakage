import numpy as np
from numba import jit
from scipy.special import erf

@jit(nopython=True)
def mean(x):
    return np.sum(x, axis=0) / x.shape[0]

@jit(nopython=True)
def var(x, x_mean):
    if x.shape[0] == 1:
        return np.zeros(x.shape[1], dtype=np.float32)
    else:
        return np.sum((x - x_mean[np.newaxis, :])**2, axis=0) / (x.shape[0] - 1)

@jit(nopython=True)
def prob_geq(a_mean, a_var, b_mean, b_var):
    mean_diff, var_diff = a_mean - b_mean, a_var + b_var
    if var_diff == 0:
        return 1. if mean_diff >= 0 else 0.
    else:
        z_score = -mean_diff / np.sqrt(var_diff)
        prob = 0.5 - 0.5*erf(z_score / np.sqrt(2))
    return prob

@jit(nopython=True)
def _calculate_tau(x_mean, x_var, y_mean, y_var, count):
    sum_coeff = 0.
    _count = 0
    for i in range(count):
        for j in range(i+1, count):
            p_xigxj = prob_geq(x_mean[i], x_var[i], x_mean[j], x_var[j])
            p_yigyj = prob_geq(y_mean[i], y_var[i], y_mean[j], y_var[j])
            p_conc = p_xigxj*p_yigyj + (1-p_xigxj)*(1-p_yigyj)
            coeff = 2*p_conc - 1
            sum_coeff += coeff
            _count = _count + 1
    tau = sum_coeff / _count
    return tau

def soft_kendall_tau(x, y, x_var=None, y_var=None):
    count = x.shape[-1]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    else:
        assert x.ndim == 2
    if y.ndim == 1:
        y = y[np.newaxis, :]
    else:
        assert y.ndim == 2
    x_mean = mean(x)
    if x_var is None:
        x_var = var(x, x_mean)
    y_mean = mean(y)
    if y_var is None:
        y_var = var(y, y_mean)
    tau = _calculate_tau(x_mean, x_var, y_mean, y_var, count)
    return tau

def partition_timesteps(leakage_measurements, poi_count):
    ranking = leakage_measurements.argsort()
    partition = ranking[-poi_count*(len(ranking)//poi_count):].reshape(-1, poi_count)
    if len(ranking) > poi_count*(len(ranking)//poi_count):
        partition = np.concatenate([ranking[np.newaxis, :poi_count], partition])
    return partition