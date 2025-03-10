from typing import *
import os
from copy import copy
import numpy as np
import torch
from torch import nn
from captum.attr import InputXGradient, FeatureAblation, Saliency, LRP

from utils.chunk_iterator import chunk_iterator
from training_modules.supervised_deep_sca import SupervisedModule
from models.zaid_wouters_nets import pretrained_models

class ReshapeOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        logits = self.model(x)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

class NeuralNetAttribution:
    def __init__(self, dataloader, model: Union[nn.Module, str, os.PathLike], seed: Optional[int] = None, device: Optional[str] = None):
        self.dataloader = dataloader
        if isinstance(model, (str, os.PathLike)):
            if 'ZaidNet' in model or 'Wouters' in model:
                model_class = getattr(pretrained_models, model)
                assert seed is not None
                model = model_class(pretrained_seed=seed)
            else:
                logging_dir = model
                assert os.path.exists(os.path.join(logging_dir, 'best_checkpoint.ckpt'))
                training_module = SupervisedModule.load_from_checkpoint(os.path.join(logging_dir, 'best_checkpoint.ckpt'))
                model = training_module.classifier
        self.base_model = model
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.base_model = self.base_model.to(self.device)
        self.trace_shape = self.base_model.input_shape
        self.model = ReshapeOutput(self.base_model)
    
    def accumulate_attributions(self, attr_fn: Callable):
        attribution_map = torch.zeros(*self.trace_shape)
        count = 0
        for trace, target in self.dataloader:
            batch_size = trace.size(0)
            trace, target = trace.to(self.device), target.to(self.device)
            batch_attribution_map = attr_fn(trace, target)
            attribution_map = (count/(count+batch_size))*attribution_map + (batch_size/(count+batch_size))*batch_attribution_map
            count += batch_size
        return attribution_map.numpy()
    
    def compute_gradvis(self):
        def attr_fn(trace, target):
            trace.requires_grad = True
            logits = self.model(trace)
            loss = nn.functional.cross_entropy(logits, target)
            self.model.zero_grad()
            loss.backward()
            trace_grad = trace.grad.detach().abs().mean(dim=0).cpu()
            return trace_grad
        return self.accumulate_attributions(attr_fn)
    
    def compute_saliency(self):
        saliency = Saliency(self.model)
        def attr_fn(trace, target):
            trace.requires_grad = True
            return saliency.attribute(trace, target=target.to(torch.long)).detach().abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)
    
    def compute_lrp(self):
        lrp = LRP(self.model)
        def attr_fn(trace, target):
            trace.requires_grad = True
            return lrp.attribute(trace, target=target.to(torch.long)).detach().abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)
    
    @torch.no_grad()
    def compute_occlusion(self):
        ablator = FeatureAblation(self.model)
        def attr_fn(trace, target):
            return ablator.attribute(trace, target=target.to(torch.long), perturbations_per_eval=10).abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)
    
    def compute_inputxgrad(self):
        input_x_grad = InputXGradient(self.model)
        def attr_fn(trace, target):
            trace.requires_grad = True
            return input_x_grad.attribute(trace, target=target.to(torch.long)).detach().abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)