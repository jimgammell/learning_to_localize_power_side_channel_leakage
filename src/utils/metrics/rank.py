from typing import *
import numpy as np
import numba
import torch
from torch import nn
from torch.utils.data import Subset
from torchmetrics import Metric

from common import *
from utils.aes import *

@numba.jit(nopython=True)
def _get_rank(logits: np.ndarray, targets: np.ndarray):
    batch_size, num_classes = logits.shape
    assert (batch_size == targets.shape[0]) and (len(targets.shape) == 1)
    ranks = np.zeros(batch_size, dtype=np.float32)
    for batch_idx in range(batch_size):
        target = targets[batch_idx]
        correct_logit = logits[batch_idx, target]
        rank = (logits[batch_idx, :] >= correct_logit).sum()
        ranks[batch_idx] = rank
    return ranks

def get_rank(logits: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    else:
        assert isinstance(logits, np.ndarray)
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    else:
        assert isinstance(targets, np.ndarray)
    if len(logits.shape) == 1:
        logits = logits[np.newaxis, :]
    else:
        assert len(logits.shape) == 2
    if len(targets.shape) == 0:
        targets = targets[np.newaxis]
    else:
        assert len(targets.shape) == 1
    rank = _get_rank(logits, targets)
    return rank

@torch.no_grad()
def _process_dataloader_for_rank_accumulation(lightning_module, metadata_keys=['plaintext',], constants=None, skip_forward_passes=False):
    dataloader = lightning_module.trainer.datamodule.test_dataloader()
    if not skip_forward_passes:
        model = lightning_module.model
        device = lightning_module.device
        model.eval()
    dataset = dataloader.dataset
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    batch_size = dataloader.batch_size
    orig_ret_mdata = base_dataset.return_metadata
    base_dataset.return_metadata = True
    
    predictions = np.empty((len(dataset), 256), dtype=np.float32)
    keys = np.empty((len(dataset),), dtype=np.uint8)
    args = [np.empty((len(dataset),), dtype=np.uint8) for _ in metadata_keys]
    for batch_idx, (traces, _, metadata) in enumerate(dataloader):
        start_idx = batch_idx*batch_size
        end_idx = min((batch_idx+1)*batch_size, len(dataset))
        if not skip_forward_passes:
            traces = traces.to(device)
            logits = model(traces).cpu().squeeze(1)
            prediction = nn.functional.log_softmax(logits, dim=-1)
            predictions[start_idx:end_idx, ...] = prediction.numpy()
        _keys = metadata['key']
        keys[start_idx:end_idx] = _keys.numpy()
        for idx, metadata_key in enumerate(metadata_keys):
            args[idx][start_idx:end_idx] = metadata[metadata_key].numpy()
    if constants is not None:
        constants = [getattr(base_dataset, constant_key) for constant_key in constants]
    
    base_dataset.return_metadata = orig_ret_mdata
    return predictions, keys, args, constants

@numba.jit(nopython=True)
def _accumulate_ranks(model_outputs, keys, args, constants, indices, int_var_to_key_fn=subbytes_to_keys):
    attack_count, trace_count = indices.shape
    rank_over_time = np.empty((attack_count, trace_count), dtype=np.int32)
    for attack_idx in range(attack_count):
        predictions = np.zeros((trace_count, 256), dtype=np.float32)
        for res_idx, trace_idx in enumerate(indices[attack_idx, :]):
            key_probs = int_var_to_key_fn(model_outputs[trace_idx], args[trace_idx], constants)
            if res_idx == 0:
                predictions[res_idx] = key_probs
            else:
                predictions[res_idx] = key_probs + predictions[res_idx-1]
        ranks = _get_rank(predictions, keys[indices[attack_idx, :]])
        rank_over_time[attack_idx, :] = ranks
    return rank_over_time

def accumulate_ranks(lightning_module, attack_count=1000, traces_per_attack=1000, int_var_to_key_fn=subbytes_to_keys, args=['plaintext',], constants=None):
    predictions, keys, args, constants = _process_dataloader_for_rank_accumulation(lightning_module, metadata_keys=args, constants=constants)
    args = np.stack(args, axis=1)
    if traces_per_attack > len(predictions):
        traces_per_attack = len(predictions)
    indices = np.stack([NUMPY_RNG.choice(len(predictions), traces_per_attack, replace=False) for _ in range(attack_count)])
    rank_over_time = _accumulate_ranks(predictions, keys, args, constants, indices, int_var_to_key_fn=int_var_to_key_fn)
    return rank_over_time

class Rank(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('rank_sum', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        batch_size, _ = logits.size()
        assert batch_size == targets.size(0)
        correct_logits = logits[torch.arange(batch_size), targets].view(batch_size, 1)
        rank_sum = (logits >= correct_logits).to(torch.long).sum()
        self.rank_sum += rank_sum
        self.total += batch_size
    
    def compute(self):
        return (self.rank_sum / self.total) - 1