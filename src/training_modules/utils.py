import torch

@torch.no_grad()
def get_rms_grad(model):
    rms_grad, param_count = 0.0, 0
    for param in model.parameters():
        if param.grad is not None:
            rms_grad += (param.grad**2).sum().item()
            param_count += torch.numel(param)
    rms_grad = (rms_grad / param_count)**0.5
    return rms_grad