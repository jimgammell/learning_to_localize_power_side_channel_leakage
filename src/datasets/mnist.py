from typing import *
from copy import copy
import os
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning as L

from common import *

ROOT = os.path.join(RESOURCE_DIR, 'mnist')
os.makedirs(ROOT, exist_ok=True)

def download():
    _ = MNIST(root=ROOT, train=True, download=True)
    _ = MNIST(root=ROOT, train=False, download=True)

class DataModule(L.LightningDataModule):
    def __init__(self,
        aug: bool = False,
        val_prop: Optional[float] = 0.1,
        train_batch_size=256,
        eval_batch_size=2048
    ):
        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)
        super().__init__()
    
    def setup(self, stage: str):
        basic_transform_mods = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            transforms.Lambda(lambda x: x.float())
        ]
        aug_transform_mods = [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1))
        ] if self.aug else []
        train_transform = transforms.Compose([*aug_transform_mods, *basic_transform_mods])
        eval_transform = transforms.Compose(basic_transform_mods)
        target_transform = transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.long))
        if stage == 'fit':
            dataset = MNIST(root=ROOT, train=True, transform=None, target_transform=target_transform)
            if self.val_prop is not None:
                val_len = int(self.val_prop*len(dataset))
                train_len = len(dataset) - val_len
                train_indices = NUMPY_RNG.choice(len(dataset), train_len, replace=False)
                val_indices = np.array([x for x in np.arange(len(dataset)) if not(x in train_indices)])
                train_dataset = Subset(dataset, train_indices)
                val_dataset = Subset(copy(dataset), val_indices)
                train_dataset.dataset.transform = train_transform
                val_dataset.dataset.transform = eval_transform
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset
            else:
                self.train_dataset = dataset
                self.train_dataset.transform = train_transform
        elif stage == 'test':
            self.test_dataset = MNIST(root=ROOT, train=False, transform=eval_transform, target_transform=target_transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size)