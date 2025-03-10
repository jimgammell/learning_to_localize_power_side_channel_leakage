import numpy as np
from torch.optim.lr_scheduler import LambdaLR

class CosineDecayLRSched(LambdaLR):
    def __init__(self, optimizer, total_steps, warmup_prop=0.0, const_prop=0.0, final_prop=0.0):
        for key, val in locals().items():
            if key not in ('self', 'key', 'val'):
                setattr(self, key, val)
        self.warmup_steps = int(self.warmup_prop * self.total_steps)
        self.const_steps = int(self.const_prop * self.total_steps)
        self.decay_steps = self.total_steps - self.warmup_steps - self.const_steps
        self.schedule = np.concatenate([
            np.linspace(0, 1, self.warmup_steps),
            np.ones(self.const_steps),
            (1 - self.final_prop)*(0.5*np.cos(np.linspace(0, np.pi, self.decay_steps)) + 0.5) + self.final_prop
        ])
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, current_step):
        assert current_step >= 0
        if current_step >= self.total_steps:
            return 0.
        else:
            return self.schedule[current_step]