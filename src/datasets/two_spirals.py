# Adapted from https://conx.readthedocs.io/en/latest/Two-Spirals.html

from typing import *
from copy import copy
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import lightning as L

class TwoSpiralsDataModule(L.LightningDataModule):
    def __init__(self,
        train_dataset_size: int = 10000,
        eval_dataset_size: int = 10000,
        train_batch_size: int = 10000,
        eval_batch_size: int = 10000,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        dataset_kwargs: dict = {},
        dataloader_kwargs: dict = {}
    ):
        self.train_dataset_size = train_dataset_size
        self.eval_dataset_size = eval_dataset_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_dataset = copy(train_dataset)
        self.eval_dataset = copy(eval_dataset)
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        super().__init__()
    
    def setup(self, stage: str):
        if self.train_dataset is None:
            self.train_dataset = TwoSpiralsDataset(self.train_dataset_size, **self.dataset_kwargs)
        if self.eval_dataset is None:
            self.eval_dataset = TwoSpiralsDataset(self.eval_dataset_size, **self.dataset_kwargs)
        self.train_dataset.ret_mdata = self.eval_dataset.ret_mdata = False
        self.data_mean = self.train_dataset.x.mean(axis=0)
        self.data_std = self.train_dataset.x.std(axis=0)
        self.data_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - self.data_mean)/self.data_std),
            transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.float))
        ])
        self.target_transform = transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.long))
        self.train_dataset.transform = self.eval_dataset.transform = self.data_transform
        self.train_dataset.target_transform = self.eval_dataset.target_transform = self.target_transform
        if not 'num_workers' in self.dataloader_kwargs.keys():
            self.dataloader_kwargs['num_workers'] = 1
    
    def train_dataloader(self, override_batch_size=None):
        return DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.train_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )
    
    def val_dataloader(self, override_batch_size=None):
        return DataLoader(
            self.eval_dataset, shuffle=False, batch_size=self.eval_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )
    
    def test_dataloader(self, override_batch_size=None):
        raise NotImplementedError

class TwoSpiralsDataset(Dataset):
    def __init__(self,
        datapoint_count=1000,
        xor_hard_feature=False,
        easy_feature_sigma=1.0,
        random_feature_count=7,
        easy_feature_count=1,
        no_hard_feature=False,
        transform=None,
        target_transform=None
    ):
        self.datapoint_count = datapoint_count
        self.xor_hard_feature = xor_hard_feature
        self.easy_feature_sigma = easy_feature_sigma
        self.random_feature_count = random_feature_count
        self.easy_feature_count = easy_feature_count
        self.no_hard_feature = no_hard_feature
        self.transform = transform
        self.target_transform = target_transform
        self.x, self.y = self.sample_datapoint(self.datapoint_count)
        self.ret_mdata = False
        self.timesteps_per_trace = 2*int(not(self.no_hard_feature)) + self.easy_feature_count + self.random_feature_count
        
    def construct_spiral(self, s, y):
        if self.xor_hard_feature:
            r = np.random.randint(2, size=y.shape)
            yr = y ^ r
            x1 = np.random.randn(*s.shape) + (2*r.reshape(-1, 1) - 1)
            x2 = np.random.randn(*s.shape) + (2*yr.reshape(-1, 1) - 1)
        else:
            phi = np.pi*s/16
            r = 6.5*((104 - s)/104)
            x1 = (r*np.cos(phi)*(2*y.reshape(-1, 1) - 1))/13 + 0.5
            x2 = (r*np.sin(phi)*(2*y.reshape(-1, 1) - 1))/13 + 0.5
        x = np.concatenate([x1, x2], axis=-1)
        return x
    
    def sample_datapoint(self, count):
        s = np.random.uniform(0, 96, size=(count, 1))
        y = np.random.randint(2, size=count, dtype=np.uint8)
        if not self.no_hard_feature:
            x12 = self.construct_spiral(s, y)
        x3 = self.easy_feature_sigma*np.random.randn(count, self.easy_feature_count) + (2*y.reshape(-1, 1) - 1)
        xrand = np.random.randn(count, self.random_feature_count)
        if not self.no_hard_feature:
            x = np.concatenate([x12, x3, xrand], axis=-1)
        else:
            x = np.concatenate([x3, xrand], axis=-1)
        return x, y
    
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        if self.ret_mdata:
            return x, y, {'target': y}
        else:
            return x, y
    
    def __len__(self):
        return self.datapoint_count