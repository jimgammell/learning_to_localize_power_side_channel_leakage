from typing import *
import torch
from torch import nn, optim
import lightning as L

import models
import utils.lr_schedulers
from ..utils import *
from utils.metrics import get_rank

class Module(L.LightningModule):
    def __init__(self,
        classifier_name: str,
        classifier_kwargs: dict = {},
        lr_scheduler_name: str = None,
        lr_scheduler_kwargs: dict = {},
        lr: float = 2e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        noise_scale: Optional[float] = None,
        timesteps_per_trace: Optional[int] = None,
        class_count: int = 256
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        assert self.hparams.timesteps_per_trace is not None
        if self.hparams.lr_scheduler_name is None:
            self.hparams.lr_scheduler_name = 'NoOpLRSched'
        self.classifier = models.load(
            self.hparams.classifier_name, input_shape=(1, self.hparams.timesteps_per_trace),
            output_classes=self.hparams.class_count, **self.hparams.classifier_kwargs
        )
    
    def configure_optimizers(self):
        yes_weight_decay, no_weight_decay = [], []
        for name, param in self.classifier.named_parameters():
            if ('weight' in name) and not('norm' in name):
                yes_weight_decay.append(param)
            else:
                no_weight_decay.append(param)
        param_groups = [{'params': yes_weight_decay, 'weight_decay': self.hparams.weight_decay}, {'params': no_weight_decay, 'weight_decay': 0.0}]
        self.optimizer = optim.AdamW(param_groups, lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2), eps=self.hparams.eps)
        lr_scheduler_constructor = (
            self.hparams.lr_scheduler_name if isinstance(self.hparams.lr_scheduler_name, optim.lr_scheduler.LRScheduler)
            else getattr(utils.lr_schedulers, self.hparams.lr_scheduler_name) if hasattr(utils.lr_schedulers, self.hparams.lr_scheduler_name)
            else getattr(optim.lr_scheduler, self.hparams.lr_scheduler_name)
        )
        if self.trainer.max_epochs != -1:
            self.total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
        elif self.trainer.max_steps != -1:
            self.total_steps = self.trainer.max_steps
        else:
            assert False
        self.lr_scheduler = lr_scheduler_constructor(self.optimizer, total_steps=self.total_steps, **self.hparams.lr_scheduler_kwargs)
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}
    
    def unpack_batch(self, batch):
        trace, label = batch
        assert trace.size(0) == label.size(0)
        if self.hparams.noise_scale is not None:
            trace += self.hparams.noise_scale*torch.randn_like(trace)
        return trace, label

    def step(self, batch, train: bool = False):
        if train:
            optimizer = self.optimizers()
            lr_scheduler = self.lr_schedulers()
            optimizer.zero_grad()
        trace, label = self.unpack_batch(batch)
        batch_size = trace.size(0)
        rv = {}
        logits = self.classifier(trace)
        logits = logits.reshape(-1, logits.size(-1))
        loss = nn.functional.cross_entropy(logits, label)
        rv.update({'loss': loss.detach(), 'rank': get_rank(logits, label).mean()})
        if train:
            self.manual_backward(loss)
            rv.update({'rms_grad': get_rms_grad(self.classifier)})
            optimizer.step()
            lr_scheduler.step()
        assert all(torch.all(torch.isfinite(param)) for param in self.classifier.parameters())
        return rv
    
    def training_step(self, batch):
        rv = self.step(batch, train=True)
        for key, val in rv.items():
            self.log(f'train_{key}', val, on_step=False, on_epoch=True)
    
    def validation_step(self, batch):
        rv = self.step(batch, train=False)
        for key, val in rv.items():
            self.log(f'val_{key}', val, on_step=False, on_epoch=True)