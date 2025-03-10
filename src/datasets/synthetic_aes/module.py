from typing import *
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as L

from common import *
from .dataset import SyntheticAES, SyntheticAESLike
from utils.calculate_dataset_stats import calculate_dataset_stats
from ..augmentation.additive_noise import AdditiveNoise

class DataModule(L.LightningDataModule):
    def __init__(self,
        train_dataset: Optional[Dataset] = None,
        aug: bool = False,
        train_dataset_size: int = 100000,
        val_dataset_size: int = 10000,
        test_dataset_size: int = 10000,
        train_batch_size: int = 256,
        eval_batch_size: int = 2048,
        data_mean: Optional[Union[float, Sequence[float]]] = None,
        data_var: Optional[Union[float, Sequence[float]]] = None,
        dataset_kwargs: dict = {},
        dataloader_kwargs: dict = {}
    ):
        for key, val in locals().items():
            if key not in ('self', 'key', 'val'):
                setattr(self, key, val)
        super().__init__()
    
    def setup(self, stage: str):
        if self.train_dataset is None:
            self.train_dataset = SyntheticAES(self.train_dataset_size, **self.dataset_kwargs)
        if (self.data_mean is None) or (self.data_var is None):
            self.data_mean, self.data_var = calculate_dataset_stats(self.train_dataset)
        self.data_mean, self.data_var = map(
            lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x.to(torch.float32), (self.data_mean, self.data_var)
        )
        basic_transform_mods = [
            transforms.Lambda(lambda x: torch.tensor(x[np.newaxis, :], dtype=torch.float32)),
            transforms.Lambda(lambda x: (x - self.data_mean) / self.data_var.sqrt())
        ]
        aug_transform_mods = [AdditiveNoise(std=0.5)] if self.aug else []
        train_transform = transforms.Compose([*basic_transform_mods, *aug_transform_mods])
        eval_transform = transforms.Compose(basic_transform_mods)
        target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        self.train_dataset.transform = train_transform
        self.train_dataset.target_transform = target_transform
        self.val_dataset = SyntheticAESLike(self.train_dataset, epoch_length=self.val_dataset_size, fixed_key=NUMPY_RNG.integers(256).astype(np.uint32))
        self.val_dataset.transform = eval_transform
        self.val_dataset.target_transform = target_transform
        self.test_dataset = SyntheticAESLike(self.train_dataset, epoch_length=self.test_dataset_size, fixed_key=NUMPY_RNG.integers(256).astype(np.uint32))
        self.test_dataset.transform = eval_transform
        self.test_dataset.target_transform = target_transform
        if not 'num_workers' in self.dataloader_kwargs.keys():
            self.dataloader_kwargs['num_workers'] = os.cpu_count() // 10
    
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