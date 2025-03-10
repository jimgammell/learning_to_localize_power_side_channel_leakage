from torch.optim.lr_scheduler import LambdaLR

from .cosine_decay import CosineDecayLRSched

class NoOpLRSched(LambdaLR):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, lr_lambda=lambda _: 1.0)