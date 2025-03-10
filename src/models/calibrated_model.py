from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim

class CalibratedModel(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.input_shape = self.classifier.input_shape
        self.output_classes = self.classifier.output_classes
        self.noise_conditional = self.classifier.noise_conditional

        if self.noise_conditional:
            self.to_unsquashed_temperature = nn.Sequential(OrderedDict([
                ('dense_1', nn.Linear(np.prod(self.input_shape), 512)),
                ('relu', nn.ReLU()),
                ('dense_2', nn.Linear(512, 1))
            ]))
            #self.to_unsquashed_temperature = nn.Linear(np.prod(self.input_shape), 1)
            self.to_unsquashed_temperature.requires_grad_(False)
            nn.init.constant_(self.to_unsquashed_temperature.dense_2.weight, 0)
            nn.init.constant_(self.to_unsquashed_temperature.dense_2.bias, np.log(np.exp(1)-1))
        else:
            self.unsquashed_temperature = nn.Parameter(
                torch.tensor(np.log(np.exp(1)-1), dtype=torch.float), requires_grad=False
            )
    
    def forward(self, *args):
        return self.classifier(*args)
    
    def get_temperature(self, *args):
        if self.noise_conditional:
            (noise,) = args
            noise = noise.flatten(start_dim=1)
            temperature = 1+nn.functional.softplus(self.to_unsquashed_temperature(noise))
        else:
            temperature = 1+nn.functional.softplus(self.unsquashed_temperature)
        return temperature.view(-1, 1)
    
    def calibrated_forward(self, *args):
        logits = self.classifier(*args)
        logits = logits.view(-1, logits.size(-1))
        if self.noise_conditional:
            (_, noise) = args
            temperature = self.get_temperature(noise)
        else:
            temperature = self.get_temperature()
        return logits/temperature
    
    def calibrate_temperature(self, *args, **kwargs):
        if self.noise_conditional:
            self._calibrate_temperature_conditional(*args, **kwargs)
        else:
            self._calibrate_temperature_unconditional(*args, **kwargs)
    
    def _calibrate_temperature_unconditional(self, dataloader):
        self.unsquashed_temperature.data.fill_(np.log(np.exp(1)-1))
        logits, targets = [], []
        device = self.unsquashed_temperature.device
        for trace, target in dataloader:
            trace, target = map(lambda x: x.to(device), (trace, target))
            targets.append(target)
            with torch.no_grad():
                logits.append(self.classifier(trace))
        logits, targets = map(lambda x: torch.cat(x, dim=0), (logits, targets))
        self.unsquashed_temperature.requires_grad_(True)
        optimizer = optim.LBFGS([self.unsquashed_temperature], line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                logits/self.get_temperature(), targets
            )
            loss.backward()
            return loss
        optimizer.step(closure)
        self.unsquashed_temperature.requires_grad_(False)
    
    def _calibrate_temperature_conditional(self, dataloader, get_classifier_args, noise_samples=1):
        self.to_unsquashed_temperature.weight.data.zero_()
        self.to_unsquashed_temperature.bias.data.fill_(np.log(np.exp(1)-1))
        logits, targets, noises = [], [], []
        device = self.to_unsquashed_temperature.weight.device
        #self.classifier.eval()
        for trace, target in dataloader:
            trace, target = map(lambda x: x.to(device), (trace, target))
            for _ in range(noise_samples):
                with torch.no_grad():
                    obfuscated_trace, noise = get_classifier_args(trace)
                    logits.append(self.classifier(obfuscated_trace, noise).squeeze(1))
                    if logits[-1].size(0) > trace.size(0):
                        factor = logits[-1].size(0) // trace.size(0)
                        assert logits[-1].size(0) == factor*trace.size(0)
                        target = target.repeat(factor)
                    noises.append(noise)
                    targets.append(target)
        logits, targets, noises = map(lambda x: torch.cat(x, dim=0), (logits, targets, noises))
        self.to_unsquashed_temperature.requires_grad_(True)
        optimizer = optim.LBFGS(self.to_unsquashed_temperature.parameters(), line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                logits/self.get_temperature(noises), targets
            )
            loss.backward()
            return loss
        optimizer.step(closure)
        self.to_unsquashed_temperature.requires_grad_(False)