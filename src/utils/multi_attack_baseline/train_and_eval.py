from typing import *
import time
from collections import defaultdict
from tqdm.auto import tqdm
from copy import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda, Compose
from lightning import Trainer as LightningTrainer

from .multi_mlp import MultiMLPModule
from .multi_template_attack import MultiTemplateAttackModule
from utils.metrics import get_rank

class TimeSubsampleTransform(nn.Module):
    def __init__(self, time_indices: Sequence[int]):
        super().__init__()
        self.time_indices = time_indices
    def forward(self, x):
        x = x[..., self.time_indices]
        return x

def apply_time_subsample_transform(dataset: Dataset, time_indices: Sequence[int]):
    if dataset.transform is None:
        dataset.transform = TimeSubsampleTransform(time_indices)
    else:
        dataset.transform = Compose([dataset.transform, TimeSubsampleTransform(time_indices)])

class MultiAttackTrainer:
    def __init__(self,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        attack_type: Literal['mlp', 'template'],
        window_size: int = 5,
        max_parallel_timesteps: Optional[int] = None
    ):
        self.generator_seed = time.time_ns()&0xFFFFFFFF
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        trace_mean = self.profiling_dataset.traces.mean(axis=0).reshape(1, -1)
        trace_std = self.profiling_dataset.traces.std(axis=0).reshape(1, -1)
        standardize_transform = Lambda(lambda x: (x - trace_mean) / (trace_std + 1e-4))
        self.profiling_dataset.transform = self.attack_dataset.transform = standardize_transform
        self.attack_type = attack_type
        self.window_size = window_size
        self.max_parallel_timesteps = max_parallel_timesteps
        self.class_count = self.profiling_dataset.class_count
        self.batch_size = int(1e6/self.profiling_dataset.timesteps_per_trace)
        
        if self.attack_type == 'template':
            self.means = [
                np.zeros((torch.zeros(len(timesteps)).unfold(-1, self.window_size, 1).shape[0], self.class_count, self.window_size), dtype=np.float32)
                for timesteps in self.next_timesteps_sequence()
            ]
            self.p_y = np.zeros(self.class_count)
            for trace, label in self.profiling_dataset:
                if isinstance(label, torch.Tensor):
                    label = label.cpu().numpy()
                for idx, timesteps in enumerate(self.next_timesteps_sequence()):
                    _trace = trace[..., timesteps]
                    self.means[idx][:, label, :] += torch.tensor(_trace).reshape(-1).unfold(-1, self.window_size, 1).numpy()
                self.p_y[label] += 1
            for idx in range(len(self.means)):
                self.means[idx] /= self.p_y.reshape(1, -1, 1)
            self.p_y /= self.p_y.sum()
    
    def get_subsampled_datasets(self, timesteps: Sequence[int] = None):
        if timesteps is None:
            timesteps = np.arange(self.profiling_dataset.timesteps_per_trace)
        profiling_dataset = copy(self.profiling_dataset)
        attack_dataset = copy(self.attack_dataset)
        apply_time_subsample_transform(profiling_dataset, timesteps)
        apply_time_subsample_transform(attack_dataset, timesteps)
        return profiling_dataset, attack_dataset
    
    def next_timesteps_sequence(self):
        t0 = 0
        t1 = self.max_parallel_timesteps
        done = False
        while not done:
            if t1 >= self.profiling_dataset.timesteps_per_trace:
                done = True
                t1 = self.profiling_dataset.timesteps_per_trace
            yield torch.arange(t0, t1)
            t0 = t1 - self.window_size + 1
            t1 = t0 + self.max_parallel_timesteps
    
    def template_attack_sequence(self, timesteps, means: Optional[Sequence[float]] = None):
        generator = torch.Generator()
        generator.manual_seed(self.generator_seed)
        training_module = MultiTemplateAttackModule(len(timesteps), self.class_count, window_size=self.window_size, p_y=self.p_y, means=means)
        profiling_dataset, attack_dataset = self.get_subsampled_datasets(timesteps)
        profiling_dataloader = DataLoader(profiling_dataset, shuffle=True, batch_size=self.batch_size, num_workers=5, generator=generator)
        attack_dataloader = DataLoader(attack_dataset, batch_size=len(attack_dataset), num_workers=5)
        trainer = LightningTrainer(max_steps=1000, logger=False, enable_checkpointing=False)
        trainer.fit(training_module, train_dataloaders=profiling_dataloader)
        with torch.no_grad():
            attack_traces, attack_labels = next(iter(attack_dataloader))
            log_p_y_mid_x = training_module.template_attacker.get_log_p_y_mid_x(attack_traces).cpu().numpy()
            batch_size, window_count, class_count = log_p_y_mid_x.shape
            _attack_labels = attack_labels.reshape(batch_size, 1).repeat(1, window_count).cpu().numpy()
            ranks = get_rank(log_p_y_mid_x.reshape(-1, self.class_count), _attack_labels.reshape(-1)).reshape(batch_size, window_count)
            rank_mean, rank_std = ranks.mean(axis=0), ranks.std(axis=0)
            mutinf = training_module.template_attacker.get_pointwise_mutinf(attack_traces).cpu().numpy()
            info = {
                'log_p_y_mid_x': log_p_y_mid_x.mean(axis=0),
                'rank_mean': rank_mean,
                'rank_std': rank_std,
                'mutinf': mutinf
            }
        return info
    
    def mlp_attack_sequence(self, timesteps: int):
        generator = torch.Generator()
        generator.manual_seed(self.generator_seed)
        training_module = MultiMLPModule(len(timesteps), self.class_count, self.window_size)
        profiling_dataset, attack_dataset = self.get_subsampled_datasets(timesteps)
        profiling_dataloader = DataLoader(profiling_dataset, shuffle=True, batch_size=self.batch_size, num_workers=5, generator=generator)
        attack_dataloader = DataLoader(attack_dataset, batch_size=len(attack_dataset), num_workers=5)
        trainer = LightningTrainer(max_steps=1000, logger=False, enable_checkpointing=False)
        trainer.fit(training_module, train_dataloaders=profiling_dataloader)
        with torch.no_grad():
            attack_traces, attack_labels = next(iter(attack_dataloader))
            log_p_y_mid_x = training_module.multi_mlp.get_log_p_y_mid_x(attack_traces)
            batch_size, window_count, class_count = log_p_y_mid_x.shape
            _attack_labels = attack_labels.reshape(batch_size, 1).repeat(1, window_count).cpu().numpy()
            ranks = get_rank(log_p_y_mid_x.reshape(-1, self.class_count), _attack_labels.reshape(-1)).reshape(batch_size, window_count)
            rank_mean, rank_std = ranks.mean(axis=0), ranks.std(axis=0)
            mutinf = training_module.multi_mlp.get_pointwise_mutinf(attack_traces).cpu().numpy()
            info = {
                'log_p_y_mid_x': log_p_y_mid_x.mean(axis=0),
                'rank_mean': rank_mean,
                'rank_std': rank_std,
                'mutinf': mutinf
            }
        return info
    
    def get_info(self):
        info = defaultdict(list)
        if self.attack_type == 'template':
            for timesteps, means in zip(tqdm(self.next_timesteps_sequence()), self.means):
                _info = self.template_attack_sequence(timesteps, means=means)
                for key, val in _info.items():
                    info[key].append(val)
        else:
            for timesteps in tqdm(self.next_timesteps_sequence()):
                _info = self.mlp_attack_sequence(timesteps)
                for key, val in _info.items():
                    info[key].append(val)
        info = {key: np.concatenate(val, axis=0) for key, val in info.items()}
        return info