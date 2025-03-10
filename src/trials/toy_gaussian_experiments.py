from typing import *
import os
from collections import defaultdict
from copy import copy
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, LogLocator
import numpy as np
from torch.utils.data import DataLoader

from common import *
from datasets.simple_gaussian import SimpleGaussianDataset
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer
from training_modules.supervised_deep_sca import SupervisedTrainer
from utils.baseline_assessments import NeuralNetAttribution, FirstOrderStatistics
from trials.utils import *

to_names = {
    'snr': 'SNR',
    'sosd': 'SoSD',
    'cpa': 'CPA',
    'gradvis': 'GradVis',
    'lrp': 'LRP',
    'saliency': 'saliency',
    'occlusion': 'occlusion',
    'inputxgrad': 'inpXgrad',
    'leakage_localization': 'ALL (Ours)'
}

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None,
        seed_count: int = 1,
        trial_count: int = 11,
        run_baselines: bool = True
    ):
        self.logging_dir = logging_dir
        self.seed_count = seed_count
        self.trial_count = trial_count
        self.run_kwargs = {'max_steps': 10000, 'anim_gammas': False}
        self.supervised_kwargs = {'classifier_name': 'mlp-1d', 'classifier_kwargs': {'layer_count': 1}, 'lr': 1e-3}
        self.leakage_localization_kwargs = {
            'classifiers_name': 'mlp-1d', 'classifiers_kwargs': {'layer_count': 1}, 'theta_lr': 1e-3, 'etat_lr': 1e-3,
            'adversarial_mode': True, 'ent_penalty': 0., 'starting_prob': 0.5,
        }
        self.run_baselines = run_baselines
    
    def run_experiments(self, logging_dir, dataset_kwargss, run_baselines: bool = True):
        leakage_assessments = {}
        for trial_name, dataset_kwargs in dataset_kwargss:
            leakage_assessments[trial_name] = {}
            profiling_dataset = SimpleGaussianDataset(**dataset_kwargs)
            attack_dataset = SimpleGaussianDataset(**dataset_kwargs)
            if run_baselines and self.run_baselines:
                first_order_stats = FirstOrderStatistics(profiling_dataset)
                leakage_assessments[trial_name]['snr'] = first_order_stats.snr_vals['label'].reshape(-1)
                leakage_assessments[trial_name]['sosd'] = first_order_stats.sosd_vals['label'].reshape(-1)
                leakage_assessments[trial_name]['cpa'] = first_order_stats.cpa_vals['label'].reshape(-1)
                sup_trainer = SupervisedTrainer(
                    profiling_dataset, attack_dataset,
                    default_data_module_kwargs={'train_batch_size': len(profiling_dataset)//10},
                    default_training_module_kwargs=self.supervised_kwargs
                )
                sup_trainer.run(
                    os.path.join(logging_dir, trial_name, 'supervised'), max_steps=self.run_kwargs['max_steps']
                )
                neural_net_attributor = NeuralNetAttribution(
                    DataLoader(profiling_dataset, batch_size=len(profiling_dataset)), model=os.path.join(logging_dir, trial_name, 'supervised')
                )
                leakage_assessments[trial_name]['gradvis'] = neural_net_attributor.compute_gradvis()
                leakage_assessments[trial_name]['saliency'] = neural_net_attributor.compute_saliency()
                leakage_assessments[trial_name]['lrp'] = neural_net_attributor.compute_lrp()
                leakage_assessments[trial_name]['occlusion'] = neural_net_attributor.compute_occlusion()
                leakage_assessments[trial_name]['inputxgrad'] = neural_net_attributor.compute_inputxgrad()
            for assessment_name, assessment in leakage_assessments[trial_name].items():
                plot_leakage_assessment(assessment.reshape(-1), os.path.join(logging_dir, trial_name, '{}.png'.format(assessment_name.replace('_', r'\_'))))
            #ll_trainer = LeakageLocalizationTrainer(
            #    profiling_dataset, attack_dataset,
            #    default_data_module_kwargs={'train_batch_size': len(profiling_dataset)//10},
            #    default_training_module_kwargs={**self.leakage_localization_kwargs}
            #)
            #ll_trainer.pretrain_classifiers(os.path.join(logging_dir, trial_name, 'pretrain_classifiers'), max_steps=self.run_kwargs['max_steps']//2)
            ll_trainer = LeakageLocalizationTrainer(
                profiling_dataset, attack_dataset,
                default_data_module_kwargs={'train_batch_size': len(profiling_dataset)//10},
                default_training_module_kwargs={**self.leakage_localization_kwargs}
            )
            ll_leakage_assessment = ll_trainer.run(
                os.path.join(logging_dir, trial_name, 'leakage_localization'),
                #pretrained_classifiers_logging_dir=os.path.join(logging_dir, trial_name, 'pretrain_classifiers'),
                max_steps=self.run_kwargs['max_steps'], #//2,
                anim_gammas=self.run_kwargs['anim_gammas']
            )
            leakage_assessments[trial_name]['leakage_localization'] = ll_leakage_assessment
        return leakage_assessments
    
    def tune_1o_count_sweep(self):
        dataset_kwargss = [('none', {'no_hard_feature': True, 'easy_feature_count': 101})]
        for starting_prob in [0.001, 0.01, 0.1, 0.5]:
            for ent_penalty in [0.0, 1e-4, 1e-2, 1e0]:
                out = self.run_experiments(
                    os.path.join(self.logging_dir, '1o_count_tune', f'starting_prob={starting_prob}__ent_penalty={ent_penalty}'),
                    dataset_kwargss=dataset_kwargss, run_baselines=False
                )
                assessment = out['none']['leakage_localization'].reshape(-1)
                assessment -= np.min(assessment)
                assessment /= np.max(assessment)
                print(f'prob={starting_prob}, ent_penalty={ent_penalty}, diff={np.min(assessment[1:]) - assessment[0]}')
    
    def tune_xor_var_sweep(self):
        dataset_kwargss = [('none', {'easy_feature_snrs': 2.0})]
        while True:
            starting_prob = 10**np.random.uniform(-2, 0)
            ent_penalty = 10**np.random.uniform(-6, 2)
            out = self.run_experiments(
                os.path.join(self.logging_dir, 'xor_var_tuning', f'starting_prob={starting_prob}__ent_penalty={ent_penalty}'),
                dataset_kwargss=dataset_kwargss, run_baselines=False
            )
            assessment = out['none']['leakage_localization'].reshape(-1)
            assessment -= np.min(assessment)
            assessment /= np.max(assessment)
            print(f'prob={starting_prob}, ent_penalty={ent_penalty}, assessment={assessment}')
    
    def run_1o_count_sweep(self):
        dataset_kwargss = [
            (f'count={x}', {'no_hard_feature': True, 'easy_feature_count': x}) for x in [2**x for x in range(14)][::-1]
        ]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, '1o_count_sweep', f'seed={seed}')
            if not os.path.exists(os.path.join(logging_dir, 'leakage_assessments.npz')):
                leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss)
                np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
        
    def plot_1o_count_sweep(self):
        counts = [2**x for x in range(14)]
        traces = defaultdict(list)
        for count in counts:
            _traces = defaultdict(list)
            for seed in range(self.seed_count):
                leakage_assessments = np.load(os.path.join(self.logging_dir, '1o_count_sweep', f'seed={seed}', 'leakage_assessments.npz'), allow_pickle=True)['leakage_assessments'].item()
                for key, val in leakage_assessments[f'count={count}'].items():
                    _traces[key].append(val.reshape(-1))
            _traces = {key: np.stack(val) for key, val in _traces.items()}
            for key, val in _traces.items():
                traces[key].append(np.abs(val))
        col_count = 4
        row_count = int(np.ceil(len(traces)/col_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(PLOT_WIDTH*col_count, PLOT_WIDTH*row_count))
        for ax in axes.flatten():
            ax.set_rasterization_zorder(-10)
        for (trace_name, trace), ax in zip(traces.items(), axes.flatten()):
            ax.set_title(to_names[trace_name])
            ax.set_xlabel('Number of leaky points')
            ax.set_ylabel('Estimated leakage of measurement')
            for seed, marker in zip(range(self.seed_count), ['.', 'v', '^', '1', '2']):
                for idx, (count, assessment) in enumerate(zip(counts, trace)):
                    ax.plot(count*[count], assessment[seed, 1:], color='blue', marker=marker, linestyle='none', label='leaking' if idx == seed == 0 else None, **PLOT_KWARGS)
            for seed, marker in zip(range(self.seed_count), ['.', 'v', '^', '1', '2']):
                for idx, (count, assessment) in enumerate(zip(counts, trace)):
                    ax.plot([count], [assessment[seed, 0]], color='red', marker=marker, linestyle='none', label='non-leaking' if idx == seed == 0 else None, **PLOT_KWARGS)
            ax.set_xscale('log')
            ax.set_yscale('log')
        for ax in axes.flatten()[len(traces):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, '1o_count_sweep', 'sweep.png'), **SAVEFIG_KWARGS)
    
    def plot_xor_var_sweep(self):
        vars = [0.5**n for n in range(1, self.trial_count//2+1)][::-1] + [1.0] + [2.0**n for n in range(1, self.trial_count//2+1)]
        traces = defaultdict(list)
        for var in vars:
            _traces = defaultdict(list)
            for seed in range(self.seed_count):
                leakage_assessments = np.load(os.path.join(self.logging_dir, 'xor_var_sweep', f'seed={seed}', 'leakage_assessments.npz'), allow_pickle=True)['leakage_assessments'].item()
                for key, val in leakage_assessments[f'var={var}'].items():
                    _traces[key].append(val.reshape(-1))
            _traces = {key: np.stack(val) for key, val in _traces.items()}
            for key, val in _traces.items():
                traces[key].append(np.abs(val))
        traces = {key: np.stack(val) for key, val in traces.items()}
        col_count = 4
        row_count = int(np.ceil(len(traces)/col_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(PLOT_WIDTH*col_count, PLOT_WIDTH*row_count))
        for ax in axes.flatten():
            ax.set_rasterization_zorder(-10)
        for (trace_name, trace), ax in zip(traces.items(), axes.flatten()):
            ax.set_title(to_names[trace_name])
            ax.set_xlabel('SNR of 1st-order measurement')
            ax.set_ylabel('Estimated leakage of measurement')
            ax.plot(vars, np.median(trace[:, :, 0], axis=-1), color='red', label='Random', **PLOT_KWARGS)
            ax.fill_between(vars, np.min(trace[:, :, 0], axis=-1), np.max(trace[:, :, 0], axis=-1), color='red', alpha=0.25, **PLOT_KWARGS)
            ax.plot(vars, np.median(trace[:, :, 1], axis=-1), color='blue', label='1st-order', **PLOT_KWARGS)
            ax.fill_between(vars, np.min(trace[:, :, 1], axis=-1), np.max(trace[:, :, 1], axis=-1), color='blue', alpha=0.25, **PLOT_KWARGS)
            ax.plot(vars, np.median(trace[:, :, 2], axis=-1), color='green', label='2nd-order (share 1)', **PLOT_KWARGS)
            ax.fill_between(vars, np.min(trace[:, :, 2], axis=-1), np.max(trace[:, :, 2], axis=-1), color='green', alpha=0.25, **PLOT_KWARGS)
            ax.plot(vars, np.median(trace[:, :, 3], axis=-1), color='purple', label='2nd-order (share 2)', **PLOT_KWARGS)
            ax.fill_between(vars, np.min(trace[:, :, 3], axis=-1), np.max(trace[:, :, 3], axis=-1), color='purple', alpha=0.25, **PLOT_KWARGS)
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
        for ax in axes.flatten()[len(traces):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'xor_var_sweep', 'sweep.png'), **SAVEFIG_KWARGS)
    
    def create_main_paper_plot(self):
        fig, axes = plt.subplots(2, 3, figsize=(3*0.75*PLOT_WIDTH, 2*0.75*PLOT_WIDTH))
        for ax in axes.flatten():
            ax.set_rasterization_zorder(-10)
        axes[0, 0].set_title('SNR', fontsize=18)
        axes[0, 1].set_title('1-occlusion', fontsize=18)
        axes[0, 2].set_title(r'\textbf{ALL (Ours)}', fontsize=18)
        for ax in axes.flatten():
            ax.set_xscale('log')
            ax.set_yscale('log')
        for ax in axes[0, :]:
            ax.set_xlabel(r'SNR of $X_{\mathrm{1o}}$', fontsize=14)
        for ax in axes[1, :]:
            ax.set_xlabel(r'Leaking point count $n$', fontsize=14)
        for ax in axes[:, 0]:
            ax.set_ylabel('Estimated leakage', fontsize=14)
        #for ax in axes.flatten():
        #    ax.xaxis.set_major_locator(LogLocator(base=2, subs=None, numticks=3))
        #    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=3))
        #    ax.xaxis.set_minor_locator(LogLocator(base=2, subs='auto', numticks=3))
        #    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=3))
        traces = defaultdict(list)
        xor_vars = [0.5**n for n in range(1, self.trial_count//2+1)][::-1] + [1.0] + [2.0**n for n in range(1, self.trial_count//2+1)]
        for xor_var in xor_vars:
            _traces = defaultdict(list)
            for seed in range(self.seed_count):
                leakage_assessments = np.load(os.path.join(self.logging_dir, 'xor_var_sweep', f'seed={seed}', 'leakage_assessments.npz'), allow_pickle=True)['leakage_assessments'].item()
                for key, val in leakage_assessments[f'var={xor_var}'].items():
                    if not key in ['snr', 'occlusion', 'leakage_localization']:
                        continue
                    _traces[key].append(val.reshape(-1))
            _traces = {key: np.stack(val) for key, val in _traces.items()}
            for key, val in _traces.items():
                traces[key].append(val)
        traces = {key: np.stack(val) for key, val in traces.items()}
        for key, ax in zip(['snr', 'occlusion', 'leakage_localization'], axes[0, :]):
            trace = traces[key]
            ax.plot(xor_vars, np.median(trace[:, :, 0], axis=-1), color='red', label=r'$X_{\mathrm{rand}}$ (not leaky)', **PLOT_KWARGS)
            ax.fill_between(xor_vars, np.min(trace[:, :, 0], axis=-1), np.max(trace[:, :, 0], axis=-1), color='red', alpha=0.25, **PLOT_KWARGS)
            ax.plot(xor_vars, np.median(trace[:, :, 1], axis=-1), color='blue', label=r'$X_{\mathrm{1o}}$ (leaky)', **PLOT_KWARGS)
            ax.fill_between(xor_vars, np.min(trace[:, :, 1], axis=-1), np.max(trace[:, :, 1], axis=-1), color='blue', alpha=0.25, **PLOT_KWARGS)
            ax.plot(xor_vars, np.median(trace[:, :, 2], axis=-1), color='green', label=r'$X_{\mathrm{2o, 1}}$ (leaky)', **PLOT_KWARGS)
            ax.fill_between(xor_vars, np.min(trace[:, :, 2], axis=-1), np.max(trace[:, :, 2], axis=-1), color='green', alpha=0.25, **PLOT_KWARGS)
            ax.plot(xor_vars, np.median(trace[:, :, 3], axis=-1), color='purple', label=r'$X_{\mathrm{2o, 2}}$ (leaky)', **PLOT_KWARGS)
            ax.fill_between(xor_vars, np.min(trace[:, :, 3], axis=-1), np.max(trace[:, :, 3], axis=-1), color='purple', alpha=0.25, **PLOT_KWARGS)
        traces = defaultdict(list)
        counts = [2**x for x in range(14)]
        for count in counts:
            _traces = defaultdict(list)
            for seed in range(self.seed_count):
                leakage_assessments = np.load(os.path.join(self.logging_dir, '1o_count_sweep', f'seed={seed}', 'leakage_assessments.npz'), allow_pickle=True)['leakage_assessments'].item()
                for key, val in leakage_assessments[f'count={count}'].items():
                    if not key in ['snr', 'occlusion', 'leakage_localization']:
                        continue
                    _traces[key].append(val.reshape(-1))
            _traces = {key: np.stack(val) for key, val in _traces.items()}
            for key, val in _traces.items():
                traces[key].append(val)
        for key, ax in zip(['snr', 'occlusion', 'leakage_localization'], axes[1, :]):
            trace = traces[key]
            for seed, marker in zip(range(self.seed_count), ['.', 'v', '^', '1', '2']):
                for idx, (count, assessment) in enumerate(zip(counts, trace)):
                    ax.plot(count*[count], assessment[seed, 1:], color='blue', marker=marker, linestyle='none', label=r'$X_i,\;i>0$ (leaky)' if idx == seed == 0 else None, **PLOT_KWARGS)
            for seed, marker in zip(range(self.seed_count), ['.', 'v', '^', '1', '2']):
                for idx, (count, assessment) in enumerate(zip(counts, trace)):
                    ax.plot([count], [assessment[seed, 0]], color='red', marker=marker, linestyle='none', label=r'$X_0$ (not leaky)' if idx == seed == 0 else None, **PLOT_KWARGS)
        #axes[0, 0].legend(ncol=2, loc='lower center', fontsize=8, handletextpad=0.5, labelspacing=0.3)
        #axes[1, 0].legend(loc='lower center', fontsize=8, handletextpad=0.5, labelspacing=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'main_paper_plot.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def run_1o_var_sweep(self, budgets: Union[float, Sequence[float]] = 1.0):
        if not hasattr(budgets, '__len__'):
            budgets = self.trial_count*[budgets]
        dataset_kwargss = [
            (f'var={x}', {'no_hard_feature': True, 'random_feature_count': 0, 'easy_feature_count': 2, 'easy_feature_snrs': [1.0, x]})
            for x in [0.5**n for n in range(1, self.trial_count//2+1)][::-1] + [1.0] + [2.0**n for n in range(1, self.trial_count//2+1)]
        ]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, '1o_var_sweep', f'seed={seed}')
            if not os.path.exists(os.path.join(self.logging_dir, 'leakage_assessments.npz')):
                leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss)
                np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
    
    def run_xor_var_sweep(self, budgets: Union[float, Sequence[float]] = 1.0):
        if not hasattr(budgets, '__len__'):
            budgets = self.trial_count*[budgets]
        dataset_kwargss = [
            (f'var={x}', {'easy_feature_snrs': x})
            for x in [0.5**n for n in range(1, self.trial_count//2+1)][::-1] + [1.0] + [2.0**n for n in range(1, self.trial_count//2+1)]
        ]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, 'xor_var_sweep', f'seed={seed}')
            if not os.path.exists(os.path.join(logging_dir, 'leakage_assessments.npz')):
                leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss)
                np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
    
    def __call__(self):
        self.leakage_localization_kwargs['starting_prob'] = 0.5
        self.run_xor_var_sweep()
        self.plot_xor_var_sweep()
        self.leakage_localization_kwargs['starting_prob'] = 0.9
        self.run_1o_count_sweep()
        self.plot_1o_count_sweep()
        self.create_main_paper_plot()