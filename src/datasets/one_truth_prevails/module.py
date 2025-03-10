import os
from copy import copy
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import lightning as L

from .dataset import OneTruthPrevails
from utils.calculate_dataset_stats import calculate_dataset_stats

class DataModule(L.LightningDataModule):
    def __init__(self,
        root: str,
        train_batch_size=256,
        eval_batch_size=100000,
        dataset_kwargs={},
        dataloader_kwargs={},
        val_prop=0.1
    ):
        self.root = root
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        self.val_prop = val_prop
        super().__init__()
    
    def setup(self, stage):
        self.profiling_dataset = OneTruthPrevails(root=self.root, train=True, transform=None, target_transform=None)
        self.attack_dataset = OneTruthPrevails(root=self.root, train=False, transform=None, target_transform=None)
        self.val_count = int(self.val_prop*len(self.profiling_dataset))
        self.train_count = len(self.profiling_dataset) - self.val_count
        assert self.train_count + self.val_count <= len(self.profiling_dataset)
        indices = np.random.choice(len(self.profiling_dataset), self.train_count+self.val_count, replace=False)
        self.train_indices = indices[:self.train_count]
        self.val_indices = indices[self.train_count:]
        self.train_dataset = Subset(copy(self.profiling_dataset), self.train_indices)
        self.val_dataset = Subset(copy(self.profiling_dataset), self.val_indices)
        self.mean, self.std = calculate_dataset_stats(self.profiling_dataset)
        self.mean, self.std = map(lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x.to(torch.float32), (self.mean, self.std))
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: (x - self.mean)/self.std)
        ])
        self.target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        self.train_dataset.dataset.transform = self.val_dataset.dataset.transform = self.attack_dataset.transform = self.transform
        self.train_dataset.dataset.target_transform = self.val_dataset.dataset.target_transform = self.attack_dataset.target_transform = self.target_transform
        if not 'num_workers' in self.dataloader_kwargs.keys():
            self.dataloader_kwargs['num_workers'] = max(os.cpu_count()//10, 1)
    
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