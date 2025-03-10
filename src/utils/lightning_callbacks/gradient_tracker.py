from lightning import Callback

class GradientTracker(Callback):
    def on_after_backward(self, trainer, module):
        total_norm = 0.
        for name, param in module.named_parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item()**2
                trainer.logger.experiment.add_histogram(f'{name}_grad', param.grad, trainer.current_epoch)
        total_norm **= 0.5
        trainer.logger.log_metrics({'total_gradient_norm': total_norm}, step=trainer.global_step)