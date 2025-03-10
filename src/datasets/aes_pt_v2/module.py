import os
from copy import copy
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
import lightning as L

from common import *
from .dataset import AES_PTv2
from utils.calculate_dataset_stats import calculate_dataset_stats

class DataModule(L.LightningDataModule):
    def __init__(self,
        profiling_dataset,
        attack_dataset,
        val_prop: float = 0.1,
        train_batch_size = 256,
        eval_batch_size = 2048,
        dataloader_kwargs: dict = {}
    ):
        self.profiling_dataset = copy(profiling_dataset)
        self.attack_dataset = copy(attack_dataset)
        self.val_prop = val_prop
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataloader_kwargs = dataloader_kwargs
        super().__init__()

    def setup(self, stage: str):
        self.data_mean, self.data_var = calculate_dataset_stats(self.profiling_dataset)
        self.data_mean, self.data_var = map(
            lambda x: torch.tensor(x, dtype=torch.float) if isinstance(x, np.ndarray) else x.to(torch.float), (self.data_mean, self.data_var)
        )
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float)),
            transforms.Lambda(lambda x: (x - self.data_mean)/self.data_var.sqrt())
        ])
        target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        if isinstance(self.profiling_dataset, ConcatDataset):
            for dataset in self.profiling_dataset.datasets:
                dataset.transform = transform
                dataset.target_transform = target_transform
            for dataset in self.attack_dataset.datasets:
                dataset.transform = transform
                dataset.target_transform = target_transform
        else:
            self.profiling_dataset.transform = self.attack_dataset.transform = transform
            self.profiling_dataset.target_transform = self.attack_dataset.target_transform = target_transform
        self.train_indices = np.random.choice(len(self.profiling_dataset), int((1-self.val_prop)*len(self.profiling_dataset)), replace=False)
        self.val_indices = np.array([x for x in np.arange(len(self.profiling_dataset)) if not(x in self.train_indices)])
        self.val_dataset = Subset(self.profiling_dataset, self.val_indices)
        self.train_dataset = Subset(self.profiling_dataset, self.train_indices)
        if not 'num_workers' in self.dataloader_kwargs:
            self.dataloader_kwargs['num_workers'] = max(1, os.cpu_count()//10)

    def train_dataloader(self, override_batch_size=None):
        return DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.train_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )
    
    def val_dataloader(self, override_batch_size=None):
        return DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.eval_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )
    
    def test_dataloader(self, override_batch_size=None):
        return DataLoader(
            self.attack_dataset, shuffle=False, batch_size=self.eval_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )