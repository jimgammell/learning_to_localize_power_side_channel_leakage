from typing import *
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import lightning as L

from common import *
from .dataset import AES_RD
from utils.calculate_dataset_stats import calculate_dataset_stats

class DataModule(L.LightningDataModule):
    def __init__(self,
        root: str,
        train_batch_size: int = 256,
        eval_batch_size: int = 2048,
        data_mean: Optional[Union[float, Sequence[float]]] = None,
        data_var: Optional[Union[float, Sequence[float]]] = None,
        dataset_kwargs: dict = {},
        dataloader_kwargs: dict = {}
    ):
        self.root = root
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_mean = data_mean
        self.data_var = data_var
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        super().__init__()
    
    def setup(self, stage: Any):
        stage = None
        self.train_dataset = AES_RD(root=self.root, train=True)
        if (self.data_mean is None) or (self.data_var is None):
            self.data_mean, self.data_var = calculate_dataset_stats(self.train_dataset)
        self.data_mean, self.data_var = map(
            lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x.to(torch.float32), (self.data_mean, self.data_var)
        )
        basic_transform_mods = [
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: (x - self.data_mean) / self.data_var.sqrt())
        ]
        transform = eval_transform = transforms.Compose(basic_transform_mods)
        target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        self.train_dataset.transform = transform
        self.train_dataset.target_transform = target_transform
        self.train_indices = np.random.choice(len(self.train_dataset), int(0.9*len(self.train_dataset)), replace=False)
        self.val_indices = np.array([x for x in np.arange(len(self.train_dataset)) if not(x in self.train_indices)])
        self.val_dataset = Subset(self.train_dataset, self.val_indices)
        self.train_dataset = Subset(self.train_dataset, self.train_indices)
        self.test_dataset = AES_RD(self.root, train=False, transform=transform, target_transform=target_transform)
        if not 'num_workers' in self.dataloader_kwargs.keys():
            self.dataloader_kwargs['num_workers'] = os.cpu_count()//10
    
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
            self.test_dataset, shuffle=False, batch_size=self.eval_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )