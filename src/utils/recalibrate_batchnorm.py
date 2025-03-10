from typing import * 
import torch

@torch.no_grad()
def recalibrate_batchnorm(trainer, module, mode: Literal['supervised', 'ALL'] = 'supervised'):
    if mode == 'supervised':
        model = module.model
    elif mode == 'ALL':
        model = module.classifiers
    else:
        assert False
    device = module.device
    kwargs = {}
    if mode == 'supervised':
        kwargs.update({'override_batch_size': trainer.datamodule.eval_batch_size})
    elif mode == 'ALL':
        kwargs.update({'override_aug_batch_size': trainer.datamodule.eval_batch_size})
    dataloader = trainer.datamodule.train_dataloader(**kwargs)
    training_mode = model.training
    model.eval()
    for mod in model.modules():
        if not isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            continue
        mod.reset_running_stats()
        mod.momentum = None
        mod.train()
    for batch in dataloader:
        if mode == 'ALL':
            [(x, y), _] = batch
        elif mode == 'supervised':
            x, y = batch
        else:
            assert False
        x, y = x.to(device), y.to(device)
        _ = module.step_theta(x, y, train=False)
    model.train(training_mode)