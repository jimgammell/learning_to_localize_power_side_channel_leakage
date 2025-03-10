import os
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from lightning import pytorch as pl_lightning

from common import *
from trials.localization_assessment import dnn_ablation

class TrackAndPlotThings(pl_lightning.Callback):
    def __init__(self, logging_dir, epoch_interval=100, classifier=None):
        super().__init__()
        self.logging_dir = logging_dir
        self.epoch_interval = epoch_interval
        self.classifier = classifier
        self.current_epoch = 0
        self.erasure_probs_dir = os.path.join(self.logging_dir, 'erasure_probs')
    
    def on_train_epoch_start(self, trainer, pl_module):
        if (self.current_epoch > 0) and (self.current_epoch % self.epoch_interval == 0):
            self.compute_metrics(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.current_epoch += 1
    
    def compute_metrics(self, trainer, pl_module):
        self.record_metrics(trainer, pl_module)
        self.plot_metrics()
    
    def data_path(self, name):
        return os.path.join(self.logging_dir, f'{name}_log.npy')
    
    def load_data(self, name):
        path = self.data_path(name)
        if os.path.exists(path):
            vals = np.load(path)
        else:
            vals = None
        return vals
    
    def save_metric(self, name, vals):
        path = self.data_path(name)
        mean_vals = np.mean(vals)
        std_vals = np.std(vals)
        existing_vals = self.load_data(name)
        if existing_vals is not None:
            new_vals = np.concatenate([existing_vals, np.array([mean_vals, std_vals])[np.newaxis, :]], axis=0)
        else:
            new_vals = np.array([mean_vals, std_vals])[np.newaxis, :]
        np.save(path, new_vals)
    
    def save_erasure_probs(self, unsquashed_probs):
        existing_probs = self.load_data('erasure_probs')
        current_probs = nn.functional.sigmoid(unsquashed_probs).detach().cpu().numpy().squeeze()
        if existing_probs is not None:
            new_probs = np.concatenate([existing_probs, current_probs[np.newaxis, :]], axis=0)
        else:
            new_probs = current_probs[np.newaxis, :]
        np.save(self.data_path('erasure_probs'), new_probs)
        return current_probs
    
    def save_dnn_ablation(self, erasure_probs, attack_dataloader):
        ranks = dnn_ablation(self.classifier, attack_dataloader, erasure_probs)
        existing_ablation = self.load_data('dnn_ablation')
        if existing_ablation is not None:
            new_ablation = np.concatenate([existing_ablation, ranks[np.newaxis, :]], axis=0)
        else:
            new_ablation = ranks[np.newaxis, :]
        np.save(self.data_path('dnn_ablation'), new_ablation)
    
    def record_metrics(self, trainer, pl_module):
        self.save_metric('log_likelihood', pl_module.log_likelihood_log)
        self.save_metric('l2_norm_penalty', pl_module.l2_norm_penalty_log)
        self.save_metric('log_likelihood_grad', pl_module.log_likelihood_grad_log)
        self.save_metric('l2_norm_penalty_grad', pl_module.l2_norm_penalty_grad_log)
        pl_module.log_likelihood_log = []
        pl_module.l2_norm_penalty_log = []
        pl_module.log_likelihood_grad_log = []
        pl_module.l2_norm_penalty_grad_log = []
        erasure_probs = self.save_erasure_probs(pl_module.unsquashed_obfuscation_weights)
        self.save_dnn_ablation(erasure_probs, trainer.datamodule.val_dataloader())
    
    def _plot_metric(self, metric, ax, **plot_kwargs):
        mean, std = metric[:, 0], metric[:, 1]
        xx = self.epoch_interval*np.arange(1, metric.shape[0]+1)
        ax.fill_between(xx, mean-std, mean+std, alpha=0.25, **plot_kwargs)
        ax.plot(xx, mean, **plot_kwargs)
    
    def plot_metrics(self):
        fig, axes = plt.subplots(1, 6, figsize=(4*6, 4*1))
        self._plot_metric(self.load_data('log_likelihood'), axes[0], color='blue')
        self._plot_metric(self.load_data('l2_norm_penalty'), axes[1], color='blue')
        self._plot_metric(self.load_data('log_likelihood_grad'), axes[2], color='blue')
        self._plot_metric(self.load_data('l2_norm_penalty_grad'), axes[3], color='blue')
        dnn_ablation = self.load_data('dnn_ablation')
        xx = self.epoch_interval*np.arange(1, dnn_ablation.shape[0]+1)
        axes[4].plot(self.load_data('dnn_ablation').mean(axis=-1), color='blue')
        erasure_probs = self.load_data('erasure_probs')
        axes[5].plot(xx, erasure_probs.min(axis=-1), color='red', label='min')
        axes[5].plot(xx, np.percentile(erasure_probs, 25, axis=-1), color='orange', label='25\%ile')
        axes[5].plot(xx, np.median(erasure_probs, axis=-1), color='blue', label='median')
        axes[5].plot(xx, np.percentile(erasure_probs, 75, axis=-1), color='purple', label='75\%ile')
        axes[5].plot(xx, erasure_probs.max(axis=-1), color='green', label='max')
        for ax in axes:
            ax.set_xlabel('Training step')
        axes[0].set_ylabel('log-likelihood')
        axes[1].set_ylabel('l2 norm penalty')
        axes[2].set_ylabel('log-likelihood gradient')
        axes[3].set_ylabel('l2 norm penalty gradient')
        axes[4].set_ylabel('DNN ablation AUC')
        axes[5].set_ylabel('erasure probs')
        axes[5].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'results.png'))
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(erasure_probs[-1, :], marker='.', linestyle='none', markersize=1, color='blue')
        ax.set_xlabel('Timestep $t$')
        ax.set_ylabel('Estimated leakage of $X_t$')
        ax.set_ylim(0, 1)
        os.makedirs(self.erasure_probs_dir, exist_ok=True)
        fig.savefig(os.path.join(self.erasure_probs_dir, f'epoch_{self.current_epoch}.png'))