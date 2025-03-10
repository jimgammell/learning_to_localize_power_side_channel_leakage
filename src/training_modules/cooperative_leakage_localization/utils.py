from typing import *
import numpy as np
import torch
from torch import nn

import models

class CondMutInfEstimator(nn.Module):
    def __init__(self,
        classifiers_name: str,
        input_shape: Sequence[int],
        output_classes: int,
        mutinf_estimate_with_labels: bool = True,
        classifiers_kwargs: dict = {}
    ):
        super().__init__()
        self.classifiers_name = classifiers_name
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.mutinf_estimate_with_labels = mutinf_estimate_with_labels
        self.classifiers_kwargs = classifiers_kwargs
        
        self.classifiers = models.load(
            self.classifiers_name,
            input_shape=self.input_shape,
            output_classes=self.output_classes,
            noise_conditional=True,
            **self.classifiers_kwargs
        )
    
    def get_logits(self, input: torch.Tensor, condition_mask: torch.Tensor):
        masked_input = condition_mask*input + (1-condition_mask)*torch.randn_like(input)
        logits = self.classifiers(masked_input, condition_mask)
        logits = logits.reshape(-1, logits.size(-1))
        return logits
    
    def get_mutinf_estimate_from_logits(self, logits: torch.Tensor, labels: Optional[torch.Tensor] = None):
        ent_y = torch.full((logits.size(0),), np.log(self.classifiers.output_classes), dtype=logits.dtype, device=logits.device)
        if self.mutinf_estimate_with_labels:
            assert labels is not None
            ent_y_mid_x = nn.functional.cross_entropy(logits, labels)
        else:
            ent_y_mid_x = -(nn.functional.softmax(logits, dim=-1)*nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
        mutinf = ent_y - ent_y_mid_x
        return mutinf
    
    def get_mutinf_estimate(self, input: torch.Tensor, condition_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        logits = self.get_logits(input, condition_mask, calibrated=self.calibrate_classifiers)
        mutinf = self.get_mutinf_estimate_from_logits(logits, labels)
        return mutinf
    
    def forward(self, *args, **kwargs):
        assert False

class SelectionMechanism(nn.Module):
    def __init__(self, timesteps_per_trace: int, C: Optional[float] = None, beta: Optional[float] = None, average_gamma: bool = False, adversarial_mode: bool = False):
        super().__init__()
        self.timesteps_per_trace = timesteps_per_trace
        if C is None:
            assert beta is not None
            log_C = np.log(self.timesteps_per_trace) + np.log(beta) - np.log(1-beta)
        else:
            assert beta is None
            log_C = np.log(C)
        self.register_buffer('log_C', torch.tensor(log_C, dtype=torch.float))
        self.average_gamma = average_gamma
        self.adversarial_mode = adversarial_mode
        self.etat = nn.Parameter(torch.zeros((1, self.timesteps_per_trace), dtype=torch.float), requires_grad=True)
        self.register_buffer('accumulated_gamma', torch.zeros((1, self.timesteps_per_trace), dtype=torch.float))
        self.register_buffer('update_count', torch.tensor(0))
    
    @torch.no_grad()
    def update_accumulated_gamma(self):
        gamma = self.get_gamma()
        if self.average_gamma:
            self.accumulated_gamma = (self.update_count/(self.update_count+1))*self.accumulated_gamma + (1/(self.update_count+1))*gamma
        else:
            self.accumulated_gamma = gamma
        self.update_count += 1
    
    @torch.no_grad()
    def get_accumulated_gamma(self):
        return self.accumulated_gamma.cpu().numpy()
    
    def get_etat(self):
        return self.etat
    
    def get_eta(self):
        etat = self.get_etat()
        eta = nn.functional.softmax(etat.reshape(-1), dim=-1).reshape(*etat.shape)
        return eta
    
    def get_log_eta(self):
        etat = self.get_etat()
        log_eta = nn.functional.log_softmax(etat.reshape(-1), dim=-1).reshape(*etat.shape)
        return log_eta
    
    def get_gammat(self):
        etat = self.get_etat()
        gammat = etat + self.log_C - torch.logsumexp(etat.squeeze(0), dim=0)
        return gammat
    
    def get_gamma(self):
        gammat = self.get_gammat()
        return nn.functional.sigmoid(gammat)
    
    def get_log_gamma(self):
        gammat = self.get_gammat()
        return nn.functional.logsigmoid(gammat)
    
    def get_log_1mgamma(self):
        gammat = self.get_gammat()
        return nn.functional.logsigmoid(-gammat)
    
    @torch.no_grad()
    def sample(self, batch_size):
        gammat = self.get_gammat()
        gamma = nn.functional.sigmoid(gammat)
        gamma = gamma.unsqueeze(0).repeat(batch_size, 1, 1)
        alpha = gamma.bernoulli_()
        return alpha
    
    def log_pmf(self, alpha):
        gammat = self.get_gammat().unsqueeze(0)
        log_pdf = (alpha*nn.functional.logsigmoid(gammat) + (1-alpha)*nn.functional.logsigmoid(-gammat)).sum(-1).squeeze(-1)
        return log_pdf
    
    def forward(self, *args, **kwargs):
        assert False