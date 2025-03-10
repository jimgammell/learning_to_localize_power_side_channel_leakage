import torch
from lightning import Callback

class RecalibrateBatchnorm(Callback):
    def on_train_epoch_end(self, trainer, module):
        model = module.model
        device = module.device
        dataloader = trainer.datamodule.train_dataloader(override_batch_size=trainer.datamodule.eval_batch_size)
        training_mode = model.training
        model.eval()
        for module in model.modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            module.reset_running_stats()
            module.momentum = None
            module.train()
        for x, *_ in dataloader:
            x = x.to(device)
            _ = model(x)
        model.train(training_mode)