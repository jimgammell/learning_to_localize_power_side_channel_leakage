from typing import *
import os
from numbers import Number
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from common import *
from trials.utils import *
from utils.baseline_assessments import FirstOrderStatistics
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer

def _plot_leakage_assessments(dest, leakage_assessments, leaking_instruction_timesteps=None, title=None, to_label=None):
    print(leakage_assessments)
    assert False
    keys = list(leakage_assessments.keys())
    if isinstance(keys[0], Number):
        assert all(isinstance(key, Number) for key in keys)
        keys.sort()
        leakage_assessments = {key: leakage_assessments[key] for key in keys}
    row_count = max(len(x) for x in leakage_assessments.values())
    fig, axes = plt.subplots(row_count, len(leakage_assessments), figsize=(PLOT_WIDTH*len(leakage_assessments), PLOT_WIDTH*row_count))
    if isinstance(leaking_instruction_timesteps, int):
        for ax in axes.flatten():
            ax.axvline(leaking_instruction_timesteps, linestyle=':', color='black', label='leaking instruction')
    elif hasattr(leaking_instruction_timesteps, '__iter__'):
        for x, axes_col in zip(leaking_instruction_timesteps, axes.transpose()):
            for ax in axes_col:
                if (x is None) or (x.dtype == type(None)):
                    continue
                else:
                    for xx in x:
                        ax.axvline(xx, color='black', linestyle='--', linewidth=0.5)
    for col_idx, (setting, _leakage_assessments) in enumerate(leakage_assessments.items()):
        for row_idx, (budget, leakage_assessment) in enumerate(_leakage_assessments.items()):
            ax = axes[row_idx, col_idx]
            ax.plot(leakage_assessment, color='blue', marker='.', markersize=3, linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'Est. lkg. of $X_t$' + f' (budget={budget})')
            ax.set_ylim(0, 1)
            if to_label is not None:
                ax.set_title(to_label(setting))
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None,
        override_run_kwargs: dict = {},
        override_leakage_localization_kwargs: dict = {},
        batch_size: int = 1000,
        timestep_count: int = 101,
        trial_count: int = 8,
        seed_count: int = 1,
        pretrain_classifiers_only: bool = False
    ):
        self.logging_dir = logging_dir
        self.run_kwargs = {'max_steps': 10000, 'anim_gammas': False}
        self.leakage_localization_kwargs = {'classifiers_name': 'mlp-1d', 'theta_lr': 1e-3, 'theta_weight_decay': 1e-2, 'etat_lr': 1e-3, 'calibrate_classifiers': False, 'ent_penalty': 0.0, 'starting_prob': 0.5}
        self.run_kwargs.update(override_run_kwargs)
        self.leakage_localization_kwargs.update(override_leakage_localization_kwargs)
        self.batch_size = batch_size
        self.timestep_count = timestep_count
        self.trial_count = trial_count
        self.seed_count = seed_count
        self.pretrain_classifiers_only = pretrain_classifiers_only
        self.betas = np.array([1 - 0.5**n for n in range(self.trial_count)][::-1])
        self.leaky_pt_counts = np.array([0] + [1 + 2*x for x in range(self.trial_count-1)])
        self.no_op_counts = np.array([0] + [1 + 4*x for x in range(self.trial_count-1)])
        self.shuffle_loc_counts = np.array([1 + 2*x for x in range(self.trial_count)])
    
    def construct_datasets(self,
        leaky_1o_count: int = 1,
        leaky_2o_count: int = 0,
        data_var: float = 1.0,
        shuffle_locs: int = 1,
        max_no_ops: int = 0,
        lpf_beta: float = 0.5   
    ):
        leaky_count = shuffle_locs*(leaky_1o_count + 2*leaky_2o_count)
        if leaky_count > 0:
            leaky_pts = np.linspace(0, self.timestep_count-1, leaky_count+2)[1:-1].astype(int)
            leaky_1o_pts = leaky_pts[:shuffle_locs*leaky_1o_count] if leaky_1o_count > 0 else None
            leaky_2o_pts = leaky_pts[shuffle_locs*leaky_1o_count:].reshape(2, -1) if leaky_2o_count > 0 else None
        else:
            leaky_pts = leaky_1o_pts = leaky_2o_pts = None
        profiling_dataset = SyntheticAES(
            infinite_dataset=True,
            timesteps_per_trace=self.timestep_count,
            leaking_timestep_count_1o=0,
            leaking_timestep_count_2o=0,
            leaky_1o_pts=leaky_1o_pts,
            leaky_2o_pts=leaky_2o_pts,
            data_var=data_var,
            shuffle_locs=shuffle_locs,
            max_no_ops=max_no_ops,
            lpf_beta=lpf_beta
        )
        attack_dataset = SyntheticAESLike(profiling_dataset, fixed_key=0)
        return profiling_dataset, attack_dataset, leaky_1o_pts, leaky_2o_pts
    
    def construct_trainer(self, profiling_dataset, attack_dataset):
        trainer = LeakageLocalizationTrainer(
            profiling_dataset, attack_dataset,
            default_data_module_kwargs={'train_batch_size': self.batch_size},
            default_training_module_kwargs={**self.leakage_localization_kwargs}
        )
        return trainer
    
    def run_experiment(self, logging_dir, kwargs):
        leakage_assessments = {}
        os.makedirs(logging_dir, exist_ok=True)
        profiling_dataset, attack_dataset, locs_1o, locs_2o = self.construct_datasets(**kwargs)
        #if not os.path.exists(os.path.join(logging_dir, 'classifiers_pretrain', 'best_checkpoint.ckpt')):
        #    trainer = self.construct_trainer(profiling_dataset, attack_dataset) # classifier pretraining is independent of budget
        #    trainer.pretrain_classifiers(os.path.join(logging_dir, 'classifiers_pretrain'), max_steps=self.run_kwargs['max_steps'])
        for starting_prob in [0.05, 0.1, 0.5, 0.9, 0.95]:
            if not os.path.exists(os.path.join(logging_dir, f'starting_prob={starting_prob}', 'leakage_assessments.npz')):
                self.leakage_localization_kwargs['starting_prob'] = starting_prob
                trainer = self.construct_trainer(profiling_dataset, attack_dataset)
                leakage_assessment = trainer.run(
                    os.path.join(logging_dir, f'starting_prob={starting_prob}'),
                    pretrained_classifiers_logging_dir=None, #os.path.join(logging_dir, 'classifiers_pretrain'),
                    **self.run_kwargs
                )
                np.savez(os.path.join(logging_dir, f'starting_prob={starting_prob}', 'leakage_assessments.npz'), leakage_assessment=leakage_assessment, locs_1o=locs_1o, locs_2o=locs_2o)
            else:
                data = np.load(os.path.join(logging_dir, f'starting_prob={starting_prob}', 'leakage_assessments.npz'), allow_pickle=True)
                leakage_assessment = data['leakage_assessment']
                locs_1o = data['locs_1o']
                locs_2o = data['locs_2o']
        return leakage_assessment, locs_1o, locs_2o
    
    def plot_leakage_assessments(self, *args, **kwargs):
        if not self.pretrain_classifiers_only:
            _plot_leakage_assessments(*args, **kwargs)
    
    def run_1o_beta_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_beta_sweep')
        leakage_assessments = {}
        for seed in range(self.seed_count):
            for beta in self.betas:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'beta={beta}')
                leakage_assessments[1-beta], *_ = self.run_experiment(subdir, {'lpf_beta': beta})
    
    def plot_1o_beta_sweep(self, axes=None, subsample=None):
        full_plot = axes is None
        betas = self.betas[::-1] if subsample is None else self.betas[subsample][::-1]
        if full_plot:
            fig, axes = plt.subplots(5, self.trial_count, figsize=(0.75*self.trial_count*PLOT_WIDTH, 0.75*5*PLOT_WIDTH))
        else:
            axes = axes.reshape(1, len(betas))
        exp_dir = os.path.join(self.logging_dir, '1o_beta_sweep')
        for beta_idx, beta in enumerate(betas):
            for starting_prob_idx, starting_prob in enumerate([0.05, 0.1, 0.5, 0.9, 0.95] if full_plot else [0.5]):
                leakage_assessments = []
                ax = axes[starting_prob_idx, beta_idx]
                for seed in range(self.seed_count):
                    subdir = os.path.join(exp_dir, f'seed={seed}', f'beta={beta}', f'starting_prob={starting_prob}')
                    assert os.path.exists(os.path.join(subdir, 'leakage_assessments.npz'))
                    data = np.load(os.path.join(subdir, 'leakage_assessments.npz'), allow_pickle=True)
                    leakage_assessment = data['leakage_assessment'].reshape(-1)
                    loc_1o = data['locs_1o'][0]
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                ax.axvline(loc_1o, linestyle='--', color='black')
                ax.fill_between(np.arange(self.timestep_count), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
                ax.plot(np.arange(self.timestep_count), np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=5, linestyle='-', linewidth=0.1, **PLOT_KWARGS)
                if full_plot:
                    ax.set_xlabel(r'Timestep $t$')
                    ax.set_ylabel(r'Estimated leakage of $X_t$')
                ax.set_title(r'LPF $\beta$: $'+f'{beta}'+r'$', fontsize=18)
                ax.set_xlim(0, self.timestep_count-1)
                ax.set_ylim(0.0, 1.0)
        if full_plot:
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, 'beta_sweep.png'))
    
    def run_1o_data_var_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_data_var_sweep')
        leakage_assessments = {}
        for seed in range(self.seed_count):
            for var in [1.0] + [0.5**(-2*n) for n in range(1, self.trial_count//2)] + [0.5**(2*n) for n in range(1, self.trial_count//2)] + [0.0]:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'var={var}')
                leakage_assessments[var], *_ = self.run_experiment(subdir, {'data_var': var})
    
    def run_1o_leaky_pt_count_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_leaky_pt_sweep')
        leakage_assessments = {}
        locss = []
        for seed in range(self.seed_count):
            for count in [0] + [1 + 2*x for x in range(self.trial_count-1)]:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'count={count}')
                leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'leaky_1o_count': count})
                locss.append(locs)
                
    def plot_1o_leaky_pt_count_sweep(self, axes=None, subsample=None):
        full_plot = axes is None
        leaky_pt_counts = self.leaky_pt_counts if subsample is None else self.leaky_pt_counts[subsample]
        if full_plot:
            fig, axes = plt.subplots(5, self.trial_count, figsize=(0.75*self.trial_count*PLOT_WIDTH, 0.75*5*PLOT_WIDTH))
        else:
            axes = axes.reshape(1, len(leaky_pt_counts))
        exp_dir = os.path.join(self.logging_dir, '1o_leaky_pt_sweep')
        for leaky_pt_count_idx, leaky_pt_count in enumerate(leaky_pt_counts):
            for starting_prob_idx, starting_prob in enumerate([0.05, 0.1, 0.5, 0.9, 0.95] if full_plot else [0.5]):
                leakage_assessments = []
                ax = axes[starting_prob_idx, leaky_pt_count_idx]
                for seed in range(self.seed_count):
                    subdir = os.path.join(exp_dir, f'seed={seed}', f'count={leaky_pt_count}', f'starting_prob={starting_prob}')
                    assert os.path.exists(os.path.join(subdir, 'leakage_assessments.npz'))
                    data = np.load(os.path.join(subdir, 'leakage_assessments.npz'), allow_pickle=True)
                    leakage_assessment = data['leakage_assessment'].reshape(-1)
                    locs_1o = data['locs_1o']
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                try:
                    for loc_1o in locs_1o:
                        ax.axvline(loc_1o, linestyle='--', color='black')
                except TypeError:
                    pass
                ax.fill_between(np.arange(self.timestep_count), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
                ax.plot(np.arange(self.timestep_count), np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=5, linestyle='-', linewidth=0.1, **PLOT_KWARGS)
                if full_plot:
                    ax.set_xlabel(r'Timestep $t$')
                    ax.set_ylabel(r'Estimated leakage of $X_t$')
                ax.set_title(f'Leaky pt. cnt.: {leaky_pt_count}', fontsize=18)
                ax.set_xlim(0, self.timestep_count-1)
                ax.set_ylim(0.0, 1.0)
        if full_plot:
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, 'leaky_pt_count_sweep.png'), **SAVEFIG_KWARGS)
    
    def run_1o_no_op_count_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_no_op_sweep')
        leakage_assessments = {}
        locss = []
        for seed in range(self.seed_count):
            for count in self.no_op_counts:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'count={count}')
                leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'max_no_ops': count})
                locss.append(locs)

    def plot_1o_no_op_count_sweep(self, axes=None, subsample=None):
        full_plot = axes is None
        no_op_counts = self.no_op_counts if subsample is None else self.no_op_counts[subsample]
        if full_plot:
            fig, axes = plt.subplots(5, self.trial_count, figsize=(0.75*self.trial_count*PLOT_WIDTH, 0.75*5*PLOT_WIDTH))
        else:
            axes = axes.reshape(1, len(no_op_counts))
        exp_dir = os.path.join(self.logging_dir, '1o_no_op_sweep')
        for no_op_count_idx, no_op_count in enumerate(no_op_counts):
            for starting_prob_idx, starting_prob in enumerate([0.05, 0.1, 0.5, 0.9, 0.95] if full_plot else [0.5]):
                leakage_assessments = []
                ax = axes[starting_prob_idx, no_op_count_idx]
                for seed in range(self.seed_count):
                    subdir = os.path.join(exp_dir, f'seed={seed}', f'count={no_op_count}', f'starting_prob={starting_prob}')
                    assert os.path.exists(os.path.join(subdir, 'leakage_assessments.npz'))
                    data = np.load(os.path.join(subdir, 'leakage_assessments.npz'), allow_pickle=True)
                    leakage_assessment = data['leakage_assessment'].reshape(-1)
                    locs_1o = data['locs_1o']
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                try:
                    for loc_1o in locs_1o:
                        ax.axvline(loc_1o, linestyle='--', color='black')
                except TypeError:
                    pass
                ax.fill_between(np.arange(self.timestep_count), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
                ax.plot(np.arange(self.timestep_count), np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=5, linestyle='-', linewidth=0.1, **PLOT_KWARGS)
                if full_plot:
                    ax.set_xlabel(r'Timestep $t$')
                    ax.set_ylabel(r'Estimated leakage of $X_t$')
                ax.set_title(f'Max no-op cnt.: {no_op_count}', fontsize=18)
                ax.set_xlim(0, self.timestep_count-1)
                ax.set_ylim(0.0, 1.0)
        if full_plot:
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, 'no_op_count_sweep.png'), **SAVEFIG_KWARGS)
    
    def run_1o_shuffle_loc_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_shuffle_sweep')
        leakage_assessments = {}
        locss = []
        for seed in range(self.seed_count):
            for count in self.shuffle_loc_counts:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'count={count}')
                leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'shuffle_locs': count})
                locss.append(locs)

    def plot_1o_shuffle_loc_sweep(self, axes=None, subsample=None):
        full_plot = axes is None
        shuffle_loc_counts = self.shuffle_loc_counts if subsample is None else self.shuffle_loc_counts[subsample]
        if full_plot:
            fig, axes = plt.subplots(5, self.trial_count, figsize=(0.75*self.trial_count*PLOT_WIDTH, 0.75*5*PLOT_WIDTH))
        else:
            axes = axes.reshape(1, len(shuffle_loc_counts))
        exp_dir = os.path.join(self.logging_dir, '1o_shuffle_sweep')
        for shuffle_loc_count_idx, shuffle_loc_count in enumerate(shuffle_loc_counts):
            for starting_prob_idx, starting_prob in enumerate([0.05, 0.1, 0.5, 0.9, 0.95] if full_plot else [0.5]):
                leakage_assessments = []
                ax = axes[starting_prob_idx, shuffle_loc_count_idx]
                for seed in range(self.seed_count):
                    subdir = os.path.join(exp_dir, f'seed={seed}', f'count={shuffle_loc_count}', f'starting_prob={starting_prob}')
                    assert os.path.exists(os.path.join(subdir, 'leakage_assessments.npz'))
                    data = np.load(os.path.join(subdir, 'leakage_assessments.npz'), allow_pickle=True)
                    leakage_assessment = data['leakage_assessment'].reshape(-1)
                    locs_1o = data['locs_1o']
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                try:
                    for loc_1o in locs_1o:
                        ax.axvline(loc_1o, linestyle='--', color='black')
                except TypeError:
                    pass
                ax.fill_between(np.arange(self.timestep_count), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
                ax.plot(np.arange(self.timestep_count), np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=5, linestyle='-', linewidth=0.1, **PLOT_KWARGS)
                if full_plot:
                    ax.set_xlabel(r'Timestep $t$')
                    ax.set_ylabel(r'Estimated leakage of $X_t$')
                ax.set_title(f'Shuffle loc. cnt.: {shuffle_loc_count}', fontsize=18)
                ax.set_xlim(0, self.timestep_count-1)
                ax.set_ylim(0.0, 1.0)
        if full_plot:
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, 'shuffle_loc_count_sweep.png'), **SAVEFIG_KWARGS)
    
    def plot_main_paper_sweeps(self):
        subsample = np.linspace(0, self.trial_count-1, 4).astype(int)
        #subsample = np.arange(self.trial_count)
        fig, axes = plt.subplots(4, len(subsample), figsize=(0.75*len(subsample)*PLOT_WIDTH, 0.75*4*PLOT_WIDTH))
        self.plot_1o_beta_sweep(axes[0, :], subsample)
        self.plot_1o_leaky_pt_count_sweep(axes[1, :], subsample)
        self.plot_1o_no_op_count_sweep(axes[2, :], subsample)
        self.plot_1o_shuffle_loc_sweep(axes[3, :], subsample)
        for ax in axes[:, 0]:
            ax.set_ylabel(r'Estimated leakage of $X_t$', fontsize=14)
        for ax in axes[-1, :]:
            ax.set_xlabel(r'Timestep $t$', fontsize=14)
        for ax in axes.flatten():
            ax.set_xlim(0, self.timestep_count-1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 1.0])
            ax.set_yticklabels(['0.', '1.'])
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'main_paper_sweep.pdf'), **SAVEFIG_KWARGS)
    
    def run_2o_trial(self):
        exp_dir = os.path.join(self.logging_dir, '2o_trial')
        leakage_assessment, _, locs = self.run_experiment(exp_dir, {'leaky_1o_count': 0, 'leaky_2o_count': 1})
        self.plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            {None: leakage_assessment},
            locs,
            title=r'Second-order leakage'
        )
    
    def __call__(self):
        self.run_1o_beta_sweep()
        self.plot_1o_beta_sweep()
        self.run_1o_leaky_pt_count_sweep()
        self.plot_1o_leaky_pt_count_sweep()
        self.run_1o_no_op_count_sweep()
        self.plot_1o_no_op_count_sweep()
        self.run_1o_shuffle_loc_sweep()
        self.plot_1o_shuffle_loc_sweep()
        self.plot_main_paper_sweeps()