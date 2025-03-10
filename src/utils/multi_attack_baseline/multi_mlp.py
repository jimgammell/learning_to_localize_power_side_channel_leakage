from typing import *
import numpy as np
import torch
from torch import nn, optim
from lightning import LightningModule

class MultiMLPModule(LightningModule):
    def __init__(self,
        timesteps_per_trace: int,
        class_count: int,
        window_size: int = 5,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.multi_mlp = MultiMLP(self.hparams.timesteps_per_trace, self.hparams.class_count, self.hparams.window_size, self.hparams.hidden_dim)
    
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.multi_mlp.parameters(), weight_decay=1e-2)
        return {'optimizer': self.optimizer}
    
    def training_step(self, batch):
        trace, labels = batch
        optimizer = self.optimizers()
        loss = None
        def closure():
            nonlocal loss
            optimizer.zero_grad()
            logits = self.multi_mlp(trace)
            batch_size, window_count, class_count = logits.shape
            _labels = labels.unsqueeze(1).repeat(1, window_count)
            loss = nn.functional.cross_entropy(logits.reshape(-1, self.hparams.class_count), _labels.reshape(-1), reduction='none')
            loss = loss.reshape(batch_size, window_count).sum(dim=-1).mean()
            self.manual_backward(loss)
            return loss
        optimizer.step(closure)
        return loss

class MultiMLP(nn.Module):
    def __init__(self,
        timesteps_per_trace: int,
        class_count: int,
        window_size: int = 5,
        hidden_dim: int = 256,
        log_p_y: Optional[Sequence[float]] = None
    ):
        super().__init__()
        self.timesteps_per_trace = timesteps_per_trace
        self.class_count = class_count
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.window_count = self.timesteps_per_trace - self.window_size + 1
        self.log_p_y = log_p_y
        
        self.dense_1 = nn.Conv1d(
            self.window_count, self.window_count*self.hidden_dim, kernel_size=self.window_size, groups=self.window_count
        )
        self.dense_2 = nn.Conv1d(
            self.window_count*self.hidden_dim, self.window_count*self.class_count, kernel_size=1, groups=self.window_count
        )
    
    def forward(self, x):
        batch_size, _, timestep_count = x.shape
        x = x.reshape(batch_size, timestep_count).unfold(-1, self.window_size, 1)
        x = self.dense_1(x)
        x = nn.functional.relu(x)
        x = self.dense_2(x).squeeze(-1).reshape(batch_size, self.window_count, self.class_count)
        return x
    
    def get_log_p_y_mid_x(self, x):
        return nn.functional.log_softmax(self(x), dim=-1)

    def get_pointwise_mutinf(self, x, y=None):
        log_p_y_mid_x = self.get_log_p_y_mid_x(x)
        if self.log_p_y is not None:
            log_p_y = self.log_p_y.reshape(1, 1, -1)
        else:
            log_p_y = -np.log(self.class_count)*torch.ones((1, 1, self.class_count), dtype=x.dtype, device=x.device)
        if y is None:
            ptwent_y_mid_x = -(log_p_y_mid_x.exp() * log_p_y_mid_x).sum(dim=-1).mean(dim=0)
        else:
            p_y_mid_x = nn.functional.one_hot(y.to(torch.long), num_classes=self.class_count).unsqueeze(1).to(torch.float)
            ptwent_y_mid_x = -(log_p_y_mid_x*p_y_mid_x).sum(dim=-1).mean(dim=0)
        ptwent_y = -(log_p_y.exp() * log_p_y).sum(dim=-1).mean(dim=0)
        ptw_mutinf = ptwent_y - ptwent_y_mid_x
        return ptw_mutinf