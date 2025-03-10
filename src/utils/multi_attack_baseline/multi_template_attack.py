from typing import *
import numpy as np
import torch
from torch import nn, optim
from lightning import LightningModule

class MultiTemplateAttackModule(LightningModule):
    def __init__(self,
        timesteps_per_trace: int,
        class_count: int,
        window_size: int = 5,
        p_y: Optional[Sequence[float]] = None,
        means: Optional[Sequence[float]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.template_attacker = MultiTemplateAttack(
            self.hparams.timesteps_per_trace,
            self.hparams.class_count,
            window_size=self.hparams.window_size,
            p_y=self.hparams.p_y,
            means=self.hparams.means
        )
    
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.template_attacker.parameters(), weight_decay=1e-2 if self.hparams.window_size > 1 else 0.0)
        return {'optimizer': self.optimizer}
    
    def training_step(self, batch):
        trace, labels = batch
        optimizer = self.optimizers()
        loss = None
        def closure():
            nonlocal loss
            optimizer.zero_grad()
            logits = self.template_attacker(trace)
            batch_size, window_count, class_count = logits.shape
            _labels = labels.unsqueeze(1).repeat(1, window_count)
            loss = nn.functional.nll_loss(logits.reshape(-1, self.hparams.class_count), _labels.reshape(-1), reduction='none')
            loss = loss.reshape(batch_size, window_count).sum(dim=-1).mean()
            self.manual_backward(loss)
            return loss
        optimizer.step(closure)
        return loss

class MultiTemplateAttack(nn.Module):
    def __init__(self,
        timesteps_per_trace: int,
        class_count: int,
        window_size: int = 5,
        means: Optional[Sequence[float]] = None,
        p_y: Optional[Sequence[float]] = None
    ):
        super().__init__()
        self.timesteps_per_trace = timesteps_per_trace
        self.class_count = class_count
        self.window_size = window_size
        assert self.window_size > 0
        self.window_count = self.timesteps_per_trace - self.window_size + 1
        if p_y is None:
            p_y = torch.ones(self.class_count, dtype=torch.float) / self.class_count
        else:
            assert len(p_y) == self.class_count
            p_y = torch.tensor(p_y, dtype=torch.float)
        
        self.register_buffer('log_p_y', p_y.log())
        if means is not None:
            self.register_buffer('means', torch.tensor(means, dtype=torch.float))
        else:
            self.means = nn.Parameter(torch.zeros(self.window_count, self.class_count, self.window_size, dtype=torch.float))
        self._cholesky_diag = nn.Parameter(-torch.ones(self.window_count, self.class_count, self.window_size, dtype=torch.float))
        if self.window_size > 1:
            self.cholesky_ltri = nn.Parameter(torch.zeros(self.window_count, self.class_count, (self.window_size**2 - self.window_size)//2, dtype=torch.float))
        
    def get_cholesky_diag(self):
        return nn.functional.softplus(self._cholesky_diag) + 1e-4
        
    def get_precision_cholesky_matrices(self):
        assert self.window_size > 1
        cholesky_matrices = torch.zeros(
            self.window_count, self.class_count, self.window_size, self.window_size,
            dtype=self._cholesky_diag.dtype, device=self._cholesky_diag.device
        )
        cholesky_diag = self.get_cholesky_diag()
        diag_indices = torch.arange(self.window_size)
        cholesky_matrices[:, :, diag_indices, diag_indices] = cholesky_diag
        tril_indices = torch.tril_indices(self.window_size, self.window_size, offset=-1)
        cholesky_matrices[:, :, tril_indices[0], tril_indices[1]] = self.cholesky_ltri
        return cholesky_matrices
    
    def get_precision_matrices(self):
        assert self.window_size > 1
        cholesky_matrices = self.get_precision_cholesky_matrices()
        precision = cholesky_matrices @ cholesky_matrices.mT
        return precision
    
    def get_precision_logdet(self):
        assert self.window_size > 1
        cholesky_diag = self.get_cholesky_diag()
        precision_det = 2*cholesky_diag.log().sum(dim=-1)
        return precision_det
    
    def get_log_p_x_mid_y(self, x):
        batch_size, _, timesteps_per_trace = x.shape
        assert timesteps_per_trace == self.timesteps_per_trace
        x = x.reshape(batch_size, timesteps_per_trace).unfold(-1, self.window_size, 1)
        assert (x.shape[0] == batch_size) and (x.shape[1] == self.window_count) and (x.shape[2] == self.window_size)
        mean_diff = x.unsqueeze(2) - self.means.unsqueeze(0)
        if self.window_size > 1:
            precisions = self.get_precision_matrices().unsqueeze(0)
            prec_logdets = self.get_precision_logdet().unsqueeze(0)
            log_p_x_mid_y = (
                -0.5*torch.einsum('bwcij,bwcjk,bwcik->bwc', mean_diff.unsqueeze(-2), precisions, mean_diff.unsqueeze(-1)).squeeze(-1)
                - 0.5*self.window_size*np.log(2*np.pi) + 0.5*prec_logdets
            )
        else:
            precisions = self.get_cholesky_diag().unsqueeze(0)**2
            log_p_x_mid_y = (-0.5*precisions*mean_diff**2 - 0.5*np.log(2*np.pi) + precisions.log()).squeeze(dim=-1)
        return log_p_x_mid_y
    
    def get_log_p_y(self):
        log_p_y = self.log_p_y.reshape(1, 1, self.class_count)
        return log_p_y
    
    def get_log_p_x_y(self, x):
        return self.get_log_p_x_mid_y(x) + self.get_log_p_y()
    
    def get_log_p_x(self, x, log_p_x_y = None):
        if log_p_x_y is None:
            log_p_x_y = self.get_log_p_x_y(x)
        log_p_x = torch.logsumexp(log_p_x_y, dim=2)
        return log_p_x
    
    def get_log_p_y_mid_x(self, x):
        log_p_x_y = self.get_log_p_x_y(x)
        log_p_x = self.get_log_p_x(x, log_p_x_y).unsqueeze(-1)
        log_p_y_mid_x = log_p_x_y - log_p_x
        return log_p_y_mid_x
    
    def get_pointwise_mutinf(self, x, y=None):
        log_p_y_mid_x = self.get_log_p_y_mid_x(x)
        log_p_y = self.get_log_p_y()
        if y is None:
            ptwent_y_mid_x = -(log_p_y_mid_x.exp() * log_p_y_mid_x).sum(dim=-1).mean(dim=0)
        else:
            p_y_mid_x = nn.functional.one_hot(y.to(torch.long), num_classes=self.class_count).unsqueeze(1).to(torch.float)
            ptwent_y_mid_x = -(log_p_y_mid_x*p_y_mid_x).sum(dim=-1).mean(dim=0)
        ptwent_y = -(log_p_y.exp() * log_p_y).sum(dim=-1).mean(dim=0)
        ptw_mutinf = ptwent_y - ptwent_y_mid_x
        return ptw_mutinf
    
    def forward(self, x):
        return self.get_log_p_y_mid_x(x)