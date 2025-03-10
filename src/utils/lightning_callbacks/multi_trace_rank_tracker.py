import numpy as np
from matplotlib import pyplot as plt
from lightning import Callback

from utils.metrics import accumulate_ranks

class MultiTraceRankTracker(Callback):
    def on_validation_epoch_end(self, trainer, module):
        ranks_over_time = accumulate_ranks(module).astype(np.float32)
        module.log('mean-rank', ranks_over_time.mean(), on_step=False, on_epoch=True, prog_bar=True)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_ylim(0, 256)
        mean_rank_over_time = ranks_over_time.mean(axis=0)
        std_rank_over_time = ranks_over_time.std(axis=0)
        ax.plot(mean_rank_over_time, color='blue', linestyle='--')
        ax.fill_between(range(std_rank_over_time.shape[0]), std_rank_over_time, color='blue', alpha=0.5)
        ax.set_xlabel('Traces seen')
        ax.set_ylabel('Rank of correct trace')
        ax.set_yscale('symlog', linthresh=1)
        fig.tight_layout()
        module.logger.experiment.add_figure('rank_over_time', fig, module.current_epoch)