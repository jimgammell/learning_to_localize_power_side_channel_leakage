from typing import *
from copy import copy
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib import gridspec
import torch
from torch.utils.data import DataLoader

from common import *
from .utils import *
from datasets.data_module import DataModule
from datasets.dpav4 import DPAv4
from datasets.ascadv1 import ASCADv1
from datasets.aes_hd import AES_HD
from datasets.ed25519_wolfssl import ED25519
from datasets.one_truth_prevails import OneTruthPrevails
from utils.baseline_assessments import FirstOrderStatistics, NeuralNetAttribution
from training_modules import SupervisedTrainer, SupervisedModule, LeakageLocalizationTrainer, LeakageLocalizationModule
from training_modules.supervised_deep_sca.plot_things import plot_hparam_sweep
from training_modules.cooperative_leakage_localization.plot_things import plot_ll_hparam_sweep
from utils.aes_multi_trace_eval import AESMultiTraceEvaluator
from utils.multi_attack_baseline import MultiAttackTrainer, soft_kendall_tau
from utils.template_attack import TemplateAttack
from utils.dnn_performance_auc import compute_dnn_performance_auc

def get_assessment_name(key):
    return {
        'random': 'Random',
        'ground_truth': 'Ground Truth',
        'snr': 'SNR',
        'sosd': 'SOSD',
        'cpa': 'CPA',
        'gradvis': 'GradVis',
        'lrp': 'LRP',
        'occlusion': '1-Occlusion',
        'saliency': 'Saliency',
        'inputxgrad': r'Input $*$ Grad',
        'zaid_gradvis': 'GradVis (ZaidNet)',
        'zaid_occlusion': '1-Occlusion (ZaidNet)',
        'zaid_saliency': 'Saliency (ZaidNet)',
        'zaid_inputxgrad': r'Input $*$ Grad (ZaidNet)',
        'wouters_gradvis': 'GradVis (WoutersNet)',
        'wouters_occlusion': '1-Occlusion (WoutersNet)',
        'wouters_saliency': 'Saliency (WoutersNet)',
        'wouters_inputxgrad': r'Input $*$ Grad (WoutersNet)',
        'leakage_localization': 'Adversarial Leakage Localization (Ours)'
    }[key]

def get_sensitive_variable_label_and_color(key, dataset_name):
    cmap = cm.get_cmap('tab10', 7)
    colors = [cmap(i) for i in range(7)]
    if 'ascad' in dataset_name:
        labels = {
            'subbytes': r'$\operatorname{Sbox}(k_3 \oplus w_3)$',
            'r_out': r'$r_{\mathrm{out}}$',
            'subbytes__r_out': r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r_{\mathrm{out}}$',
            'r_in': r'$r_{\mathrm{in}}$',
            'subbytes__r_in': r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r_{\mathrm{in}}$',
            'r': r'$r$',
            'subbytes__r': r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r$'
        }
        label = labels[key]
        color = colors[list(labels.keys()).index(key)]
    elif dataset_name == 'dpav4':
        label = r'$\operatorname{Sbox}(k_0 \oplus w_0) \oplus m_0$'
        color = colors[0]
    elif dataset_name == 'aes_hd':
        label = r'$\operatorname{Sbox}^{-1}(k_{11}^* \oplus c_{11}) \oplus c_7$'
        color = colors[0]
    elif dataset_name == 'otiait':
        label = 'Ephemeral key nibble'
        color = colors[0]
    elif dataset_name == 'otp':
        label = 'Dummy load?'
        color = colors[0]
    else:
        assert False
    return (label, color)

class Trial:
    def __init__(self,
        dataset_name: Literal['dpav4', 'ascadv1_fixed', 'ascadv1_variable', 'otiait', 'otp', 'aes_hd'],
        trial_config: dict,
        seed_count: int = 1,
        logging_dir: Optional[Union[str, os.PathLike]] = None
    ):
        self.dataset_name = dataset_name
        self.trial_config = trial_config
        self.seed_count = seed_count
        self.logging_dir = logging_dir if logging_dir is not None else dataset_name
        os.makedirs(self.logging_dir, exist_ok=True)
        self.stats_dir = os.path.join(self.logging_dir, 'first_order_stats')
        os.makedirs(self.stats_dir, exist_ok=True)
        self.supervised_model_dir = os.path.join(self.logging_dir, 'supervised_model')
        os.makedirs(self.supervised_model_dir, exist_ok=True)
        self.nn_attr_dir = os.path.join(self.logging_dir, 'nn_attr_assessments')
        os.makedirs(self.nn_attr_dir, exist_ok=True)
        self.ll_classifiers_pretrain_dir = os.path.join(self.logging_dir, 'll_classifiers_pretrain')
        os.makedirs(self.ll_classifiers_pretrain_dir, exist_ok=True)
        self.leakage_localization_dir = os.path.join(self.logging_dir, 'leakage_localization')
        os.makedirs(self.leakage_localization_dir, exist_ok=True)
        self.supervised_hparam_sweep_dir = os.path.join(self.logging_dir, 'supervised_hparam_sweep')
        os.makedirs(self.supervised_hparam_sweep_dir, exist_ok=True)
        self.ll_classifiers_hparam_sweep_dir = os.path.join(self.logging_dir, 'll_classifiers_hparam_sweep')
        os.makedirs(self.ll_classifiers_hparam_sweep_dir, exist_ok=True)
        self.ll_hparam_sweep_dir = os.path.join(self.logging_dir, 'll_hparam_sweep')
        os.makedirs(self.ll_hparam_sweep_dir, exist_ok=True)
        self.ground_truth_dir = os.path.join(self.logging_dir, 'ground_truth_assessments')
        os.makedirs(self.ground_truth_dir, exist_ok=True)
        self.dnn_auc_dir = os.path.join(self.logging_dir, 'dnn_auc')
        os.makedirs(self.dnn_auc_dir, exist_ok=True)
        
        print('Constructing datasets...')
        if self.dataset_name == 'dpav4':
            self.profiling_dataset = DPAv4(root=trial_config['data_dir'], train=True)
            self.attack_dataset = DPAv4(root=trial_config['data_dir'], train=False)
        elif self.dataset_name == 'ascadv1_fixed':
            self.profiling_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=True)
            self.attack_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=False)
        elif self.dataset_name == 'ascadv1_variable':
            self.profiling_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=True, train=True)
            self.attack_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=True, train=False)
        elif self.dataset_name == 'aes_hd':
            self.profiling_dataset = AES_HD(root=trial_config['data_dir'], train=True)
            self.attack_dataset = AES_HD(root=trial_config['data_dir'], train=False)
        elif dataset_name == 'otiait':
            self.profiling_dataset = ED25519(root=trial_config['data_dir'], train=True)
            self.attack_dataset = ED25519(root=trial_config['data_dir'], train=False)
        elif dataset_name == 'otp':
            self.profiling_dataset = OneTruthPrevails(root=trial_config['data_dir'], train=True)
            self.attack_dataset = OneTruthPrevails(root=trial_config['data_dir'], train=False)
        else:
            assert False
        print('\tDone.')
        
    def compute_ground_truth_assessments(self):
        for seed in range(self.seed_count):
            for attack_type in ['template']: #['mlp', 'template']:
                assessments = {}
                for window_size in [5]: #[1, 3, 5]:
                    name = f'{attack_type}__seed={seed}__window_size={window_size}'
                    assessments[window_size] = {}
                    if 'ascadv1' in self.dataset_name:
                        targets = ['subbytes', 'r_out', 'subbytes__r_out', 'r_in', 'subbytes__r_in', 'r', 'subbytes__r', 'k__p__r_in']
                    else:
                        targets = ['subbytes']
                    fig, axes = plt.subplots(len(targets), 3, figsize=(PLOT_WIDTH*3, PLOT_WIDTH*len(targets)))
                    if len(targets) == 1:
                        axes = axes.reshape(1, 3)
                    if not os.path.exists(os.path.join(self.ground_truth_dir, f'{name}.npz')):
                        for target_idx, target in enumerate(targets):
                            self.profiling_dataset.target_values = [target]
                            self.attack_dataset.target_values = [target]
                            trainer = MultiAttackTrainer(self.profiling_dataset, self.attack_dataset, attack_type=attack_type, window_size=window_size, max_parallel_timesteps=100)
                            info = trainer.get_info()
                            assessments[window_size][target] = info
                        np.savez(os.path.join(self.ground_truth_dir, f'{name}.npz'), assessments=assessments[window_size])
                    else:
                        assessments[window_size] = np.load(os.path.join(self.ground_truth_dir, f'{name}.npz'), allow_pickle=True)['assessments'].item()
                    for target_idx, target in enumerate(targets):
                        for key_idx, key in enumerate(['log_p_y_mid_x', 'mutinf']):
                            assessment = assessments[window_size][target][key]
                            axes[target_idx, key_idx].plot(assessment, linestyle='none', marker='.', markersize=1, color='blue')
                            axes[target_idx, key_idx].set_xlabel(r'Timesteps $t$')
                            axes[target_idx, key_idx].set_ylabel(r'Estimated leakage of $X_t$')
                            axes[target_idx, key_idx].set_title('Method: ' + key.replace('_', r'\_'))
                        #rank_mean, rank_std = assessments[window_size][target]['rank_mean'], assessments[window_size][target]['rank_std']
                        #axes[target_idx, 2].fill_between(np.arange(len(assessments[window_size][target]['rank_mean'])), rank_mean-rank_std, rank_mean+rank_std, color='blue', alpha=0.25)
                        #axes[target_idx, 2].plot(rank_mean, linestyle='-', color='blue')
                    fig.tight_layout()
                    fig.savefig(os.path.join(self.ground_truth_dir, f'{name}.png'))
                    plt.close(fig)
    
    def get_ground_truth_assessments(self):
        rv = {}
        for attack_type in ['template']: #['mlp', 'template']:
            for window_size in [5]: #[1, 3, 5]:
                assessment_name = f'{attack_type}__window_size={window_size}'
                if 'ascadv1' in self.dataset_name:
                    targets = ['r_out', 'subbytes__r_out', 'r_in', 'subbytes__r_in', 'r', 'subbytes__r', 'subbytes']
                else:
                    targets = ['subbytes']
                mutinfs = defaultdict(list)
                for seed in range(self.seed_count):
                    for target in targets:
                        name = f'{attack_type}__seed={seed}__window_size={window_size}'
                        assessment = np.load(os.path.join(self.ground_truth_dir, f'{name}.npz'), allow_pickle=True)['assessments'].item()[target]
                        mutinfs[target].append(assessment['mutinf'])
                mutinfs = {key: np.stack(val) for key, val in mutinfs.items()}
                rv[assessment_name] = mutinfs
        return rv
    
    def run_template_attacks(self):
        leakage_assessments = self.get_leakage_assessments()
        for name, assessments in leakage_assessments.items():
            if assessments.ndim == 1:
                assessments = assessments.reshape(1, -1)
            for seed, assessment in assessments:
                for poi_count in [1, 5, 9, 13]:
                    points_of_interest = assessments.argsort()[-poi_count:]
                    template_attacker = TemplateAttack(points_of_interest, target_key='label')
                    template_attacker.profile(self.profiling_dataset)
                    template_attacker.attack(self.attack_dataset)
                    
    def compute_random_assessment(self):
        self.random_assessment = {'random': np.random.randn(self.seed_count, self.profiling_dataset.timesteps_per_trace)}
        
    def compute_ascad_first_order_stats(self):
        for target in ['subbytes', 'subbytes__r_in', 'subbytes__r', 'subbytes__r_out', 'r_in', 'r_out', 'r']:
            first_order_stats = FirstOrderStatistics(self.profiling_dataset, targets=target)
            snr = first_order_stats.snr_vals[target].reshape(-1)
            plot_leakage_assessment(snr, os.path.join(self.stats_dir, f'snr_target={target}.png'))
        
    def compute_first_order_stats(self):
        if not os.path.exists(os.path.join(self.stats_dir, 'stats.npy')):
            print('Computing first-order statistical assessments...')
            first_order_stats = FirstOrderStatistics(self.profiling_dataset)
            snr = first_order_stats.snr_vals['label'].reshape(-1)
            sosd = first_order_stats.sosd_vals['label'].reshape(-1)
            cpa = first_order_stats.cpa_vals['label'].reshape(-1)
            np.save(os.path.join(self.stats_dir, 'stats.npy'), np.stack([snr, sosd, cpa]))
            print('\tDone.')
        else:
            rv = np.load(os.path.join(self.stats_dir, 'stats.npy'))
            snr = rv[0, :]
            sosd = rv[1, :]
            cpa = rv[2, :]
            print('Found precomputed first-order statistical assessments.')
        plot_leakage_assessment(snr, os.path.join(self.stats_dir, 'snr.png'))
        plot_leakage_assessment(sosd, os.path.join(self.stats_dir, 'sosd.png'))
        plot_leakage_assessment(cpa, os.path.join(self.stats_dir, 'cpa.png'))
        self.first_order_stats = {
            'snr': np.abs(snr), 'sosd': np.abs(sosd), 'cpa': np.abs(cpa)
        }
    
    def run_supervised_hparam_sweep(self):
        if not os.path.exists(os.path.join(self.supervised_hparam_sweep_dir, 'results.pickle')):
            print('Running supervised hparam sweep...')
            supervised_trainer = SupervisedTrainer(
                self.profiling_dataset, self.attack_dataset,
                default_training_module_kwargs=self.trial_config['supervised_training_kwargs'],
                default_data_module_kwargs={'gaussian_noise_std': 0.5 if self.dataset_name == 'aes_hd' else 0.0}
            )
            supervised_trainer.hparam_tune(logging_dir=self.supervised_hparam_sweep_dir, max_steps=self.trial_config['max_classifiers_pretrain_steps'])
            print('\tDone.')
        else:
            print('Found existing supervised hparam sweep.')
        self.optimal_hparams = plot_hparam_sweep(self.supervised_hparam_sweep_dir)
        print(f'Optimal hyperparameters on {self.dataset_name}: {self.optimal_hparams}')
    
    def run_ll_classifiers_hparam_sweep(self):
        if not os.path.exists(os.path.join(self.ll_classifiers_hparam_sweep_dir, 'results.pickle')):
            print('Running LL classifiers hparam sweep...')
            kwargs = copy(self.trial_config['default_kwargs'])
            kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
            ll_trainer = LeakageLocalizationTrainer(
                self.profiling_dataset, self.attack_dataset,
                default_training_module_kwargs=kwargs,
                default_data_module_kwargs={'gaussian_noise_std': 0.5 if self.dataset_name == 'aes_hd' else 0.0}
            )
            ll_trainer.htune_pretrain_classifiers(logging_dir=self.ll_classifiers_hparam_sweep_dir, max_steps=self.trial_config['max_classifiers_pretrain_steps'])
            print('\tDone.')
        else:
            print('Found existing LL classifier pretraining sweep.')
        self.optimal_ll_pretrain_hparams = plot_hparam_sweep(self.ll_classifiers_hparam_sweep_dir)
        print(f'Optimal hyperparameters on {self.dataset_name}: {self.optimal_ll_pretrain_hparams}')
    
    def compute_dnn_auc_vals_on_baselines(self):
        leakage_assessments = self.get_leakage_assessments()
        forward_auc_vals, reverse_auc_vals = defaultdict(list), defaultdict(list)
        for seed in range(self.seed_count):
            training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, f'seed={(seed+1)%self.seed_count}', 'best_checkpoint.ckpt'))
            supervised_dnn = training_module.classifier
            for assessment_name, _leakage_assessments in leakage_assessments.items():
                if _leakage_assessments.ndim > 1:
                    leakage_assessment = _leakage_assessments[seed, :]
                else:
                    assert _leakage_assessments.ndim == 1
                    leakage_assessment = _leakage_assessments
                auc_vals = compute_dnn_performance_auc(
                    DataLoader(self.attack_dataset, batch_size=len(self.attack_dataset), num_workers=max(1, os.cpu_count()//4)),
                    supervised_dnn, leakage_assessment, device='cuda', cluster_count=10
                )
                forward_auc_vals[assessment_name].append(auc_vals['forward_dnn_auc'])
                reverse_auc_vals[assessment_name].append(auc_vals['reverse_dnn_auc'])
        forward_auc_vals, reverse_auc_vals = {key: np.stack(val) for key, val in forward_auc_vals.items()}, {key: np.stack(val) for key, val in reverse_auc_vals.items()}
        for key in leakage_assessments.keys():
            print(f'Assessment: {key}')
            print(f'\tForward AUC: {forward_auc_vals[key].mean()} +/- {forward_auc_vals[key].std()}')
            print(f'\tReverse AUC: {reverse_auc_vals[key].mean()} +/- {reverse_auc_vals[key].std()}')
    
    def run_ll_hparam_sweep(self):
        training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, 'll_eval', 'best_checkpoint.ckpt'))
        supervised_dnn = training_module.classifier
        use_pretrained_classifiers = self.dataset_name not in ['otp', 'otiait', 'dpav4']
        if not os.path.exists(os.path.join(self.ll_hparam_sweep_dir, 'results.pickle')):
            print('Running LL hparam sweep...')
            kwargs = copy(self.trial_config['default_kwargs'])
            kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
            kwargs.update(self.trial_config['leakage_localization_kwargs'])
            if use_pretrained_classifiers:
                kwargs.update(self.optimal_ll_pretrain_hparams)
            ll_trainer = LeakageLocalizationTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=kwargs)
            ll_trainer.htune_leakage_localization(
                self.ll_hparam_sweep_dir,
                pretrained_classifiers_logging_dir=os.path.join(self.ll_classifiers_pretrain_dir, f'seed=0') if use_pretrained_classifiers else None,
                trial_count=25 if use_pretrained_classifiers else 50,
                max_steps=self.trial_config['max_leakage_localization_steps'],
                supervised_dnn=supervised_dnn,
                references={key: val.mean(axis=0) for key, val in self.get_ground_truth_assessments().items()}
            )
        else:
            print('Found existing LL hparam sweep.')
        self.ll_optimal_hparams = plot_ll_hparam_sweep(self.ll_hparam_sweep_dir)
        print(f'Optimal LL hyperparameters on {self.dataset_name}: {self.ll_optimal_hparams}')
        
    def run_leakage_localization(self):
        training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, 'll_eval', 'best_checkpoint.ckpt'))
        supervised_dnn = training_module.classifier
        use_pretrained_classifiers = self.dataset_name not in ['otp', 'otiait', 'dpav4']
        assessments = []
        for seed in range(self.seed_count):
            subdir = os.path.join(self.leakage_localization_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            if not os.path.exists(os.path.join(subdir, 'best_checkpoint.ckpt')):
                print('Running leakage localization...')
                trainer = LeakageLocalizationTrainer(
                    self.profiling_dataset, self.attack_dataset,
                    default_training_module_kwargs=self.trial_config['default_kwargs']
                )
                leakage_localization_kwargs = copy(self.trial_config['default_kwargs'])
                leakage_localization_kwargs.update(self.trial_config['leakage_localization_kwargs'])
                leakage_localization_kwargs.update({'supervised_dnn': supervised_dnn})
                leakage_localization_kwargs.update(self.ll_optimal_hparams)
                leakage_assessment = trainer.run(
                    logging_dir=subdir,
                    pretrained_classifiers_logging_dir=os.path.join(self.ll_classifiers_pretrain_dir, f'seed={seed}') if use_pretrained_classifiers else None,
                    max_steps=self.trial_config['max_leakage_localization_steps'],
                    override_kwargs=leakage_localization_kwargs,
                    anim_gammas=False
                )
            else:
                assert os.path.exists(os.path.join(subdir, 'leakage_assessment.npy'))
                leakage_assessment = np.load(os.path.join(subdir, 'leakage_assessment.npy'))
            assessments.append(leakage_assessment)
        self.leakage_localization_assessments = {
            'leakage_localization': np.stack(assessments)
        }
        print('\tDone.')
    
    def train_supervised_model(self):
        for subdir in ['ll_eval', *[f'seed={seed}' for seed in range(self.seed_count)]]:
            os.makedirs(os.path.join(self.supervised_model_dir, subdir), exist_ok=True)
            if not os.path.exists(os.path.join(self.supervised_model_dir, subdir, 'final_checkpoint.ckpt')):
                print('Training supervised model...')
                training_module_kwargs = copy(self.trial_config['supervised_training_kwargs'])
                training_module_kwargs.update(self.optimal_hparams)
                supervised_trainer = SupervisedTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=training_module_kwargs)
                supervised_trainer.run(logging_dir=os.path.join(self.supervised_model_dir, subdir), max_steps=self.trial_config['max_classifiers_pretrain_steps'])
                print('\tDone.')
            else:
                print('Found pretrained supervised model.')
    
    def plot_supervised_training_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_WIDTH))
        colormap = plt.cm.get_cmap('tab10', self.seed_count)
        min_ranks, min_losses = [], []
        for seed in range(self.seed_count):
            color = colormap(seed)
            subdir = os.path.join(self.supervised_model_dir, f'seed={seed}')
            assert os.path.exists(os.path.join(subdir, 'final_checkpoint.ckpt'))
            training_curves = load_training_curves(subdir)
            axes[0].plot(*training_curves['train_rank'], linestyle='--', color=color, **PLOT_KWARGS)
            axes[0].plot(*training_curves['val_rank'], linestyle='-', color=color, **PLOT_KWARGS)
            axes[1].plot(*training_curves['train_loss'], linestyle='--', color=color)
            axes[1].plot(*training_curves['val_loss'], linestyle='-', color=color)
            optimal_idx = np.argmin(training_curves['val_rank'][-1])
            min_ranks.append(training_curves['val_rank'][-1][optimal_idx])
            min_losses.append(training_curves['val_loss'][-1][optimal_idx])
        axes[0].set_xlabel('Training step')
        axes[1].set_xlabel('Training step')
        axes[0].set_ylabel('Rank')
        axes[1].set_ylabel('Loss')
        class_count = self.profiling_dataset.class_count
        axes[0].axhline((class_count+1)/2, color='black', linestyle=':')
        axes[1].axhline(np.log(class_count), color='black', linestyle=':')
        legend_elems = [
            Line2D([0], [0], color='black', linestyle='--', label='train'),
            Line2D([0], [0], color='black', linestyle='-', label='val'),
            Line2D([0], [0], color='black', linestyle=':', label='random guessing')
        ]
        axes[0].legend(handles=legend_elems, loc='lower left')
        axes[1].legend(handles=legend_elems, loc='lower left')
        axes[1].set_yscale('symlog')
        dset_name = self.dataset_name.replace(r'_', r'\_')
        fig.suptitle(f'Dataset: {dset_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(self.supervised_model_dir, 'training_curves.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
        print(f'Supervised ranks: {np.mean(min_ranks)} +/- {np.std(min_ranks)}')
        print(f'Supervised losses: {np.mean(min_losses)} +/- {np.std(min_losses)}')
    
    def compute_supervised_ranks_over_time(self, wouters_zaid_model=None):
        data_module = DataModule(self.profiling_dataset, self.attack_dataset)
        attack_dataloader = data_module.test_dataloader()
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
        colormap = plt.cm.get_cmap('tab10', self.seed_count)
        for seed in range(self.seed_count):
            subdir = os.path.join(self.supervised_model_dir, f'seed={seed}')
            to_name = lambda x: x if wouters_zaid_model is None else f'zaid_{x}' if 'Zaid' in wouters_zaid_model else f'wouters_{x}' if 'Wouters' in wouters_zaid_model else None
            assert to_name('') is not None
            assert os.path.exists(subdir)
            if not os.path.exists(os.path.join(subdir, to_name('rank_over_time.npy'))):
                evaluator = AESMultiTraceEvaluator(attack_dataloader, subdir if wouters_zaid_model is None else wouters_zaid_model, seed=seed, dataset_name=self.dataset_name)
                rank_over_time = evaluator()
                np.save(os.path.join(subdir, to_name('rank_over_time.npy')), rank_over_time)
            else:
                rank_over_time = np.load(os.path.join(subdir, to_name('rank_over_time.npy')))
            color = colormap(seed)
            ax.plot(np.arange(1, len(rank_over_time)+1), rank_over_time, color=color)
        ax.axhline(0.5*(self.profiling_dataset.class_count+1), linestyle=':', color='black', label='random guessing')
        ax.legend(loc='upper right')
        ax.set_xscale('log')
        ax.set_xlabel('Traces seen')
        ax.set_ylabel('Rank')
        fig.tight_layout()
        fig.savefig(os.path.join(self.supervised_model_dir, to_name('rank_over_time.png')), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def create_paper_rot_plot(self):
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
        colormap = plt.cm.get_cmap('tab10', self.seed_count)
        for seed in range(self.seed_count):
            subdir = os.path.join(self.supervised_model_dir, f'seed={seed}')
            rank_over_time = np.load(os.path.join(subdir, 'rank_over_time.npy'))
            color = colormap(seed)
            ax.plot(np.arange(1, len(rank_over_time)+1), rank_over_time, color=color)
            if os.path.exists(os.path.join(subdir, 'wouters_rank_over_time.npy')):
                wouters_rank_over_time = np.load(os.path.join(subdir, 'wouters_rank_over_time.npy'))
                ax.plot(np.arange(1, len(wouters_rank_over_time)+1), wouters_rank_over_time, color=color, linestyle='--')
            else:
                wouters_rank_over_time = None
            if os.path.exists(os.path.join(subdir, 'zaid_rank_over_time.npy')):
                zaid_rank_over_time = np.load(os.path.join(subdir, 'zaid_rank_over_time.npy'))
                ax.plot(np.arange(1, len(zaid_rank_over_time)+1), zaid_rank_over_time, color=color, linestyle='-.')
            else:
                zaid_rank_over_time = None
        ax.axhline(0.5*(self.profiling_dataset.class_count+1), linestyle=':', color='black', label='random guessing')
        legend_handles = [
            Line2D([0], [0], color='black', linestyle=':', label='random guessing'),
            Line2D([0], [0], color='gray', linestyle='-', label='ours')
        ]
        if wouters_rank_over_time is not None:
            legend_handles.append(Line2D([0], [0], color='gray', linestyle='--', label='Wouters et al.'))
        if zaid_rank_over_time is not None:
            legend_handles.append(Line2D([0], [0], color='gray', linestyle='-.', label='Zaid et al.'))
        ax.legend(loc='upper right', handles=legend_handles)
        ax.set_xscale('log')
        ax.set_xlabel('Traces seen')
        ax.set_ylabel('Rank')
        ax.set_title('Dataset:' + self.dataset_name.replace('_', r'\_'))
        fig.tight_layout()
        fig.savefig(os.path.join(self.supervised_model_dir, 'paper_rank_over_time.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
        
    def compute_neural_net_attributions(self, wouters_zaid_model=None):
        data_module = DataModule(self.profiling_dataset, self.attack_dataset, val_prop=0.0)
        profiling_dataloader = data_module.train_dataloader()
        gradviss, saliencies, occlusions, inputxgrads, lrps = [], [], [], [], []
        for seed in range(self.seed_count):
            subdir = os.path.join(self.nn_attr_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            nn_attributor = NeuralNetAttribution(profiling_dataloader, os.path.join(self.supervised_model_dir, f'seed={seed}') if wouters_zaid_model is None else wouters_zaid_model, seed=seed)
            to_name = lambda x: x if wouters_zaid_model is None else f'zaid_{x}' if 'Zaid' in wouters_zaid_model else f'wouters_{x}' if 'Wouters' in wouters_zaid_model else None
            assert to_name('') is not None
            if wouters_zaid_model is None and not os.path.exists(os.path.join(subdir, to_name('lrp.npy'))):
                print('Computing LRP...')
                lrp = nn_attributor.compute_lrp().reshape(-1)
                np.save(os.path.join(subdir, to_name('lrp.npy')), lrp)
                print('\tDone.')
            elif wouters_zaid_model is None:
                lrp = np.load(os.path.join(subdir, to_name('lrp.npy')))
                print('Found precomputed LRP.')
            if not os.path.exists(os.path.join(subdir, to_name('gradvis.npy'))):
                print('Computing GradVis...')
                gradvis = nn_attributor.compute_gradvis().reshape(-1)
                np.save(os.path.join(subdir, to_name('gradvis.npy')), gradvis)
                print('\tDone.')
            else:
                gradvis = np.load(os.path.join(subdir, to_name('gradvis.npy')))
                print('Found precomputed GradVis.')
            if not os.path.exists(os.path.join(subdir, to_name('saliency.npy'))):
                print('Computing saliency...')
                saliency = nn_attributor.compute_saliency().reshape(-1)
                np.save(os.path.join(subdir, to_name('saliency.npy')), saliency)
                print('\tDone.')
            else:
                saliency = np.load(os.path.join(subdir, to_name('saliency.npy')))
                print('Found precomputed saliency.')
            if not os.path.exists(os.path.join(subdir, to_name('occlusion.npy'))):
                print('Computing occlusion...')
                occlusion = nn_attributor.compute_occlusion().reshape(-1)
                np.save(os.path.join(subdir, to_name('occlusion.npy')), occlusion)
                print('\tDone.')
            else:
                occlusion = np.load(os.path.join(subdir, to_name('occlusion.npy')))
                print('Found precomputed occlusion.')
            if not os.path.exists(os.path.join(subdir, to_name('inputxgrad.npy'))):
                print('Computing inputxgrad...')
                inputxgrad = nn_attributor.compute_inputxgrad().reshape(-1)
                np.save(os.path.join(subdir, to_name('inputxgrad.npy')), inputxgrad)
                print('\tDone.')
            else:
                inputxgrad = np.load(os.path.join(subdir, to_name('inputxgrad.npy')))
                print('Found precomputed inputxgrad.')
            plot_leakage_assessment(gradvis, os.path.join(subdir, to_name('gradvis.png')))
            plot_leakage_assessment(saliency, os.path.join(subdir, to_name('saliency.png')))
            plot_leakage_assessment(occlusion, os.path.join(subdir, to_name('occlusion.png')))
            plot_leakage_assessment(inputxgrad, os.path.join(subdir, to_name('inputxgrad.png')))
            if wouters_zaid_model is None:
                plot_leakage_assessment(lrp, os.path.join(subdir, to_name('lrp.png')))
                lrps.append(lrp)
            gradviss.append(gradvis)
            saliencies.append(saliency)
            occlusions.append(occlusion)
            inputxgrads.append(inputxgrad)
        setattr(self, to_name('nn_attr_assessments'), {
            to_name('gradvis'): np.stack(gradviss), to_name('saliency'): np.stack(saliencies), to_name('inputxgrad'): np.stack(inputxgrads), to_name('occlusion'): np.stack(occlusions),
            **({to_name('lrp'): np.stack(lrps)} if wouters_zaid_model is None else {})
        })
    
    def pretrain_leakage_localization_classifiers(self):
        for seed in range(self.seed_count):
            subdir = os.path.join(self.ll_classifiers_pretrain_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            if not os.path.exists(os.path.join(subdir, 'final_checkpoint.ckpt')):
                print('Pretraining leakage localization classifiers...')
                trainer = LeakageLocalizationTrainer(
                    self.profiling_dataset, self.attack_dataset,
                    default_training_module_kwargs=self.trial_config['default_kwargs']
                )
                classifiers_pretrain_kwargs = copy(self.trial_config['default_kwargs'])
                classifiers_pretrain_kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
                classifiers_pretrain_kwargs.update(self.optimal_ll_pretrain_hparams)
                trainer.pretrain_classifiers(
                    logging_dir=subdir,
                    max_steps=self.trial_config['max_classifiers_pretrain_steps'],
                    override_kwargs=classifiers_pretrain_kwargs
                )
                print('\tDone.')
            else:
                print('Found pretrained leakage localization classifiers.')
    
    def get_leakage_assessments(self):
        leakage_assessments = {}
        leakage_assessments.update(self.random_assessment)
        if hasattr(self, 'first_order_stats'):
            leakage_assessments.update(self.first_order_stats)
        if hasattr(self, 'nn_attr_assessments'):
            leakage_assessments.update(self.nn_attr_assessments)
        if hasattr(self, 'zaid_nn_attr_assessments'):
            leakage_assessments.update(self.zaid_nn_attr_assessments)
        if hasattr(self, 'wouters_nn_attr_assessments'):
            leakage_assessments.update(self.wouters_nn_attr_assessments)
        if hasattr(self, 'leakage_localization_assessments'):
            leakage_assessments.update(self.leakage_localization_assessments)
        return leakage_assessments
    
    def eval_leakage_assessments(self): # should print out valid code for a Latex booktabs table
        leakage_assessments = self.get_leakage_assessments()
        ground_truth_assessments = self.get_ground_truth_assessments()
        kendalltau_evaluations = {}
        for leakage_assessment_name, leakage_assessment in leakage_assessments.items():
            kendalltau_evaluations[leakage_assessment_name] = {}
            print(leakage_assessment_name)
            for ground_truth_assessment_name, ground_truth_assessment in ground_truth_assessments.items():
                print(f'\t{ground_truth_assessment_name}')
                ground_truth_assessment = np.mean(np.stack(list(ground_truth_assessment.values())), axis=0)
                kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name] = []
                for seed in range(self.seed_count):
                    if leakage_assessment.ndim == 1:
                        _leakage_assessment = leakage_assessment
                    else:
                        _leakage_assessment = leakage_assessment[seed, :]
                    window_size = int(ground_truth_assessment_name.split('=')[-1])
                    _leakage_assessment = torch.tensor(_leakage_assessment).unfold(0, window_size, 1).numpy().mean(axis=-1)
                    #mutinf, rank_mean, rank_std = ground_truth_assessment[seed, 0, :], ground_truth_assessment[seed, 1, :], ground_truth_assessment[seed, 2, :]
                    mutinf = ground_truth_assessment[0, :]
                    kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name].append(spearmanr(_leakage_assessment, mutinf).statistic)  #soft_kendall_tau(_leakage_assessment, mutinf))
                kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name] = np.stack(kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name])
                print(f'\t\tKendall tau: {kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name].mean()} +/- {kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name].std()}')
        return kendalltau_evaluations
    
    def create_main_paper_dnn_auc_plots(self):
        forward_results = defaultdict(list)
        reverse_results = defaultdict(list)
        datamodule = DataModule(self.profiling_dataset, self.attack_dataset, eval_batch_size=len(self.attack_dataset))
        attack_dataloader = datamodule.test_dataloader()
        progress_bar = tqdm(total=self.seed_count*len(self.get_leakage_assessments()))
        for seed in range(self.seed_count):
            training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, f'seed={(seed+1)%5}', 'best_checkpoint.ckpt'))
            supervised_dnn = training_module.classifier
            for leakage_assessment_name, leakage_assessment in self.get_leakage_assessments().items():
                forward_result_name = f'{leakage_assessment_name}_forward_seed={seed}.npy'
                reverse_result_name = f'{leakage_assessment_name}_reverse_seed={seed}.npy'
                if not(os.path.exists(os.path.join(self.dnn_auc_dir, forward_result_name)) and os.path.exists(os.path.join(self.dnn_auc_dir, reverse_result_name))):
                    if leakage_assessment.ndim > 1:
                        assert leakage_assessment.shape[-1] >= self.seed_count
                        _leakage_assessment = leakage_assessment[seed, :]
                    else:
                        _leakage_assessment = leakage_assessment
                    out = compute_dnn_performance_auc(attack_dataloader, supervised_dnn, _leakage_assessment, 'cuda', average=False, cluster_count=None)
                    np.save(os.path.join(self.dnn_auc_dir, forward_result_name), out['forward_dnn_auc'])
                    np.save(os.path.join(self.dnn_auc_dir, reverse_result_name), out['reverse_dnn_auc'])
                forward_result = np.load(os.path.join(self.dnn_auc_dir, forward_result_name))
                reverse_result = np.load(os.path.join(self.dnn_auc_dir, reverse_result_name))
                forward_results[leakage_assessment_name].append(forward_result)
                reverse_results[leakage_assessment_name].append(reverse_result)
                progress_bar.update(1)
        forward_results = {key: np.stack(val) for key, val in forward_results.items()}
        reverse_results = {key: np.stack(val) for key, val in reverse_results.items()}
        
        print('Forward ablation test:')
        for key, val in forward_results.items():
            print(f'\t{key}: {np.mean(val)} +/- {np.std(np.mean(val, axis=-1))}')
        print('Reverse ablation test:')
        for key, val in reverse_results.items():
            print(f'\t{key}: {np.mean(val)} +/- {np.std(np.mean(val, axis=-1))}')
        
        if 'ascad' in self.dataset_name: # Main paper plot
            fig, axes = plt.subplots(2, 1, figsize=(PLOT_WIDTH, 2*PLOT_WIDTH))
            if 'fixed' in self.dataset_name:
                forward_stat_baseline = forward_results['cpa'].reshape(self.seed_count, -1)
                forward_nn_baseline = forward_results['inputxgrad'].reshape(self.seed_count, -1)
                reverse_stat_baseline = reverse_results['cpa'].reshape(self.seed_count, -1)
                reverse_nn_baseline = reverse_results['inputxgrad'].reshape(self.seed_count, -1)
            elif 'variable' in self.dataset_name:
                forward_stat_baseline = forward_results['sosd'].reshape(self.seed_count, -1)
                forward_nn_baseline = forward_results['gradvis'].reshape(self.seed_count, -1)
                reverse_stat_baseline = reverse_results['sosd'].reshape(self.seed_count, -1)
                reverse_nn_baseline = reverse_results['gradvis'].reshape(self.seed_count, -1)
            else:
                assert False
            xx = np.arange(forward_results['random'].shape[-1])
            axes[0].fill_between(xx, np.min(forward_results['random'], axis=0), np.max(forward_results['random'], axis=0), color='red', alpha=0.25, **PLOT_KWARGS)
            axes[0].fill_between(xx, np.min(forward_stat_baseline, axis=0), np.max(forward_stat_baseline, axis=0), color='green', alpha=0.25, **PLOT_KWARGS)
            axes[0].fill_between(xx, np.min(forward_nn_baseline, axis=0), np.max(forward_nn_baseline, axis=0), color='purple', alpha=0.25, **PLOT_KWARGS)
            axes[0].fill_between(xx, np.min(forward_results['leakage_localization'], axis=0), np.max(forward_results['leakage_localization'], axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            axes[0].plot(xx, np.median(forward_results['random'], axis=0), color='red', **PLOT_KWARGS)
            axes[0].plot(xx, np.median(forward_stat_baseline, axis=0), color='green', **PLOT_KWARGS)
            axes[0].plot(xx, np.median(forward_nn_baseline, axis=0), color='purple', **PLOT_KWARGS)
            axes[0].plot(xx, np.median(forward_results['leakage_localization'], axis=0), color='blue', **PLOT_KWARGS)
            axes[0].set_xlabel('Number of un-occluded inputs')
            axes[0].set_ylabel('Mean rank on test dataset')
            axes[0].set_title('Forward DNN occlusion test')
            axes[1].fill_between(xx, np.min(reverse_results['random'], axis=0), np.max(reverse_results['random'], axis=0), color='red', alpha=0.25, **PLOT_KWARGS)
            axes[1].fill_between(xx, np.min(reverse_stat_baseline, axis=0), np.max(reverse_stat_baseline, axis=0), color='green', alpha=0.25, **PLOT_KWARGS)
            axes[1].fill_between(xx, np.min(reverse_nn_baseline, axis=0), np.max(reverse_nn_baseline, axis=0), color='purple', alpha=0.25, **PLOT_KWARGS)
            axes[1].fill_between(xx, np.min(reverse_results['leakage_localization'], axis=0), np.max(reverse_results['leakage_localization'], axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            axes[1].plot(xx, np.median(reverse_results['random'], axis=0), color='red', **PLOT_KWARGS)
            axes[1].plot(xx, np.median(reverse_stat_baseline, axis=0), color='green', **PLOT_KWARGS)
            axes[1].plot(xx, np.median(reverse_nn_baseline, axis=0), color='purple', **PLOT_KWARGS)
            axes[1].plot(xx, np.median(reverse_results['leakage_localization'], axis=0), color='blue', **PLOT_KWARGS)
            axes[1].set_xlabel('Number of un-occluded inputs')
            axes[1].set_ylabel('Mean rank on test dataset')
            axes[1].set_title('Reverse DNN occlusion test')
            fig.tight_layout()
            fig.savefig(os.path.join(self.dnn_auc_dir, 'main_paper_dnn_occlusion.pdf'), **SAVEFIG_KWARGS)
        
        # appendix plot
        col_count = 6
        row_count = 2*int(np.ceil(len(forward_results)/col_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(0.75*col_count*PLOT_WIDTH, 0.75*row_count*PLOT_WIDTH))
        forward_axes = axes[:row_count//2, :].flatten()
        reverse_axes = axes[row_count//2:, :].flatten()
        for idx, assessment_name in enumerate(forward_results.keys()):
            forward_ax = forward_axes[idx]
            reverse_ax = reverse_axes[idx]
            forward_result = forward_results[assessment_name].reshape(self.seed_count, -1)
            reverse_result = reverse_results[assessment_name].reshape(self.seed_count, -1)
            forward_ax.fill_between(range(forward_result.shape[-1]), np.min(forward_result, axis=0), np.max(forward_result, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            forward_ax.plot(np.median(forward_result, axis=0), color='blue', **PLOT_KWARGS)
            reverse_ax.fill_between(range(reverse_result.shape[-1]), np.min(reverse_result, axis=0), np.max(reverse_result, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            reverse_ax.plot(np.median(reverse_result, axis=0), color='blue', **PLOT_KWARGS)
            forward_ax.set_xlabel('Number of un-occluded inputs')
            forward_ax.set_ylabel('Mean rank on test dataset')
            reverse_ax.set_xlabel('Number of un-occluded inputs')
            reverse_ax.set_ylabel('Mean rank on test dataset')
            forward_ax.set_title(f'Forward DNN occlusion test:\n{get_assessment_name(assessment_name)}')
            reverse_ax.set_title(f'Reverse DNN occlusion test:\n{get_assessment_name(assessment_name)}')
        for ax in forward_axes[len(forward_results):]:
            ax.axis('off')
        for ax in reverse_axes[len(reverse_results):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.dnn_auc_dir, 'full_dnn_occlusion.png'), **SAVEFIG_KWARGS)
        
        r"""leakage_assessments = self.get_leakage_assessments()['leakage_localization'].reshape(self.seed_count, -1)
        random_assessments = self.get_leakage_assessments()['random'].reshape(self.seed_count, -1)
        all_dnn_forward_curves, all_dnn_backward_curves, random_dnn_forward_curves, random_dnn_backward_curves = [], [], [], []
        for seed in range(self.seed_count):
            training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, f'seed={(seed+1)%5}', 'best_checkpoint.ckpt'))
            supervised_dnn = training_module.classifier
            datamodule = DataModule(self.profiling_dataset, self.attack_dataset, eval_batch_size=len(self.attack_dataset))
            attack_dataloader = datamodule.test_dataloader()
            all_dnn_curve = compute_dnn_performance_auc(
                attack_dataloader,
                supervised_dnn, leakage_assessments[seed, :], 'cuda', average=False, cluster_count=None
            )
            random_dnn_curve = compute_dnn_performance_auc(
                attack_dataloader,
                supervised_dnn, random_assessments[seed, :], 'cuda', average=False, cluster_count=None
            )
            all_dnn_forward_curves.append(all_dnn_curve['forward_dnn_auc'])
            all_dnn_backward_curves.append(all_dnn_curve['reverse_dnn_auc'])
            random_dnn_forward_curves.append(random_dnn_curve['forward_dnn_auc'])
            random_dnn_backward_curves.append(random_dnn_curve['reverse_dnn_auc'])
        all_dnn_forward_curves, all_dnn_backward_curves, random_dnn_forward_curves, random_dnn_backward_curves = map(
            lambda x: np.stack(x), (all_dnn_forward_curves, all_dnn_backward_curves, random_dnn_forward_curves, random_dnn_backward_curves)
        )
        fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
        mask_sizes = np.arange(all_dnn_forward_curves.shape[-1])
        axes[0].plot(mask_sizes, np.median(all_dnn_backward_curves, axis=0), color='blue')
        axes[0].fill_between(mask_sizes, np.min(all_dnn_backward_curves, axis=0), np.max(all_dnn_backward_curves, axis=0), color='blue', alpha=0.25)
        axes[0].plot(mask_sizes, np.median(random_dnn_backward_curves, axis=0), color='red')
        axes[0].fill_between(mask_sizes, np.min(random_dnn_backward_curves, axis=0), np.max(random_dnn_backward_curves, axis=0), color='red', alpha=0.25)
        axes[1].plot(mask_sizes, np.median(all_dnn_forward_curves, axis=0), color='blue')
        axes[1].fill_between(mask_sizes, np.min(all_dnn_forward_curves, axis=0), np.max(all_dnn_forward_curves, axis=0), color='blue', alpha=0.25)
        axes[1].plot(mask_sizes, np.median(random_dnn_forward_curves, axis=0), color='red')
        axes[1].fill_between(mask_sizes, np.min(random_dnn_forward_curves, axis=0), np.max(random_dnn_forward_curves, axis=0), color='red', alpha=0.25)
        axes[0].set_title('Reverse ablation test')
        axes[1].set_title('Forward ablation test')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'dnn_auc_plots.png'))"""
    
    def create_main_paper_leakage_assessment_plots(self):
        leakage_assessments = self.get_leakage_assessments()['leakage_localization'].reshape(self.seed_count, -1)
        random_assessment = self.get_leakage_assessments()['random'].reshape(self.seed_count, -1)
        if self.dataset_name == 'ascadv1_fixed':
            stat_baseline_assessment = self.get_leakage_assessments()['cpa'].reshape(1, -1)
            nn_attr_baseline_assessment = self.get_leakage_assessments()['occlusion'].reshape(self.seed_count, -1)
            stat_baseline_name = 'CPA'
            nn_attr_baseline_name = '1-occlusion'
        else:
            stat_baseline_assessment = self.get_leakage_assessments()['sosd'].reshape(1, -1)
            nn_attr_baseline_assessment = self.get_leakage_assessments()['gradvis'].reshape(self.seed_count, -1)
            stat_baseline_name = 'SoSD'
            nn_attr_baseline_name = 'GradVis'
        ground_truth_assessments = self.get_ground_truth_assessments()['template__window_size=5']
        mean_gta = np.mean(np.stack(list(ground_truth_assessments.values())), axis=0)
        #fig = plt.figure(figsize=(PLOT_WIDTH, 3*PLOT_WIDTH))
        #gs = gridspec.GridSpec(3, 1)
        #axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
        #comp_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2, 0], wspace=0.3, hspace=0.3)
        #comp_axes = [fig.add_subplot(comp_gs[0, 0]), fig.add_subplot(comp_gs[0, 1]), fig.add_subplot(comp_gs[1, 0]), fig.add_subplot(comp_gs[1, 1])]
        fig, axes = plt.subplots(2, 1, figsize=(PLOT_WIDTH, 2*PLOT_WIDTH))
        cmap = cm.get_cmap('tab10', len(ground_truth_assessments))
        for key, _ground_truth_assessment in ground_truth_assessments.items():
            name, color = get_sensitive_variable_label_and_color(key, self.dataset_name)
            axes[0].fill_between(range(_ground_truth_assessment.shape[-1]), np.min(_ground_truth_assessment, axis=0), np.max(_ground_truth_assessment, axis=0), color=color, alpha=0.25, **PLOT_KWARGS)
            axes[0].plot(np.median(_ground_truth_assessment, axis=0), color=color, marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
        legend_handles = []
        for key in ground_truth_assessments.keys():
            name, color = get_sensitive_variable_label_and_color(key, self.dataset_name)
            legend_handles.append(Line2D([], [], color=color, label=name, marker='.', linestyle='none'))
        axes[0].legend(handles=legend_handles, loc='upper center', ncol=2, fontsize=8, handletextpad=0.5, labelspacing=0.3, framealpha=0.25)
        axes[0].set_xlabel(r'Timestep $t$')
        axes[0].set_ylabel(r'Estimated leakage of $X_t$')
        axes[0].set_title('Estimated leakage by\n`omniscient\' Gaussian mixture model', pad=5)
        axes[0].set_yscale('log')
        
        axes[1].fill_between(range(leakage_assessments.shape[-1]), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
        axes[1].plot(np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
        axes[1].set_xlabel(r'Timestep $t$')
        axes[1].set_ylabel(r'Estimated leakage of $X_t$')
        axes[1].set_title('Adversarial leakage localization (ours)')
        
        r"""markers = ['o', 'v', '^', 'p', '*']
        def to_ranks(assessment):
            if assessment.shape[-1] > mean_gta.shape[-1]:
                averaged_assessment = torch.tensor(assessment).unfold(0, 5, 1).numpy().mean(axis=-1)
            else:
                averaged_assessment = assessment
            averaged_assessment = np.abs(averaged_assessment)
            #averaged_assessment -= np.min(averaged_assessment)
            #averaged_assessment /= np.max(averaged_assessment)
            return averaged_assessment
        #print(mean_gta.shape, random_assessment.shape, stat_baseline_assessment.shape, nn_attr_baseline_assessment.shape, leakage_assessments.shape)
        for seed in range(self.seed_count):
            comp_axes[0].plot(mean_gta[seed, :], to_ranks(random_assessment[seed, :]), color='red', marker=markers[seed], markersize=1, linestyle='none', label='Random', alpha=0.5)
            if seed == 0:
                comp_axes[1].plot(mean_gta[seed, :], to_ranks(stat_baseline_assessment[seed, :]), color='green', marker=markers[seed], markersize=1, linestyle='none', label=stat_baseline_name, alpha=0.5)
            comp_axes[2].plot(mean_gta[seed, :], to_ranks(nn_attr_baseline_assessment[seed, :]), color='purple', marker=markers[seed], markersize=1, linestyle='none', label=nn_attr_baseline_name, alpha=0.5)
            comp_axes[3].plot(mean_gta[seed, :], to_ranks(leakage_assessments[seed, :]), color='blue', marker=markers[seed], markersize=1, linestyle='none', label='ALL (Ours)', alpha=0.5)
        comp_axes[0].set_title('Random')
        comp_axes[1].set_title(stat_baseline_name)
        comp_axes[2].set_title(nn_attr_baseline_name)
        comp_axes[3].set_title('ALL (Ours)')
        fig.text(0.02, 1/6, r'Estimated leakage of $X_t$', va='center', rotation='vertical', fontsize=10)
        fig.text(0.55, 0.005, 'Estimated leakage by `omniscient\' GMM', ha='center', fontsize=10)
        for ax in comp_axes:
            ax.set_xscale('log')
            ax.tick_params(axis='y', labelrotation=45)"""
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'main_paper_leakage_assessments.pdf'), **SAVEFIG_KWARGS)
    
    def create_appendix_leakage_assessment_plots(self):
        leakage_assessments = self.get_leakage_assessments()
        ground_truth_assessment = self.get_ground_truth_assessments()['template__window_size=5']
        ground_truth_assessment = np.mean(np.stack(list(ground_truth_assessment.values())), axis=0)[0, :]
        col_count = 6
        row_count = int(2*np.ceil((len(leakage_assessments))/col_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(0.75*col_count*PLOT_WIDTH, 0.75*row_count*PLOT_WIDTH))
        comparison_axes = axes[row_count//2:, :]
        assessment_axes = axes[:row_count//2, :]
        for (assessment_name, assessment), comparison_ax, assessment_ax in zip(leakage_assessments.items(), comparison_axes.flatten(), assessment_axes.flatten()):
            assessment = np.abs(assessment.reshape(-1, assessment.shape[-1]))
            if assessment_name != 'ground_truth':
                averaged_assessment = np.stack([
                    torch.tensor(_assessment).unfold(0, 5, 1).numpy().mean(axis=-1)
                    for _assessment in assessment
                ])
            else:
                assessment = assessment.reshape(1, -1)
                averaged_assessment = assessment
            comparison_ax.errorbar(
                ground_truth_assessment, np.median(averaged_assessment, axis=0),
                yerr=(np.median(averaged_assessment, axis=0)-np.min(averaged_assessment, axis=0), np.max(averaged_assessment, axis=0)-np.median(averaged_assessment, axis=0)),
                color='blue', fmt='.', markersize=3, elinewidth=0.5, capsize=2, linestyle='none', **PLOT_KWARGS
            )
            assessment_ax.fill_between(np.arange(assessment.shape[-1]), np.min(assessment, axis=0), np.max(assessment, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            assessment_ax.plot(np.median(assessment, axis=0), color='blue', linewidth=0.25, **PLOT_KWARGS)
            comparison_ax.set_xscale('log')
            comparison_ax.set_yscale('log')
            assessment_ax.set_yscale('log')
            comparison_ax.set_xlabel(r'Ground truth-like leakage of $X_t$')
            comparison_ax.set_ylabel(r'Estimated leakage of $X_t$')
            assessment_ax.set_xlabel(r'Timestep $t$')
            assessment_ax.set_ylabel(r'Estimated leakage of $X_t$')
            assessment_ax.set_title(f'{get_assessment_name(assessment_name)}')
            comparison_ax.set_title(f'{get_assessment_name(assessment_name)}')
        for ax in comparison_axes.flatten()[len(leakage_assessments):]:
            ax.axis('off')
        for ax in assessment_axes.flatten()[len(leakage_assessments):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'appendix_leakage_assessment_plots.png'), **SAVEFIG_KWARGS)
    
    def plot_leakage_assessments(self):
        leakage_assessments = self.get_leakage_assessments()
        row_count = 2
        col_count = int(np.ceil(len(leakage_assessments)/row_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(col_count*PLOT_WIDTH, row_count*PLOT_WIDTH))
        for ax, (la_name, la) in zip(axes.flatten(), leakage_assessments.items()):
            if la.ndim == 1:
                ax.plot(la, color='blue', marker='.', linestyle='none', markersize=1, **PLOT_KWARGS)
            elif la.ndim == 2:
                median = np.median(la, axis=0)
                min = np.min(la, axis=0)
                max = np.max(la, axis=0)
                #ax.errorbar(range(len(median)), median, yerr=[median-min, max-median], fmt='none', ecolor='red', label='min--max')
                ax.plot(median, color='blue', marker='.', linestyle='none', markersize=1, label='median', **PLOT_KWARGS)
                ax.legend()
            else:
                assert False
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'Estimated leakage of $X_t$')
            la_name = la_name.replace('_', r'\_')
            ax.set_title(f'Technique: {la_name}')
        for ax in axes.flatten()[len(leakage_assessments):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'leakage_assessments.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def __call__(self):
        self.compute_random_assessment()
        if 'ascad' in self.dataset_name:
            self.compute_ascad_first_order_stats()
        #if ('compute_ground_truth_assessments' in self.trial_config) and self.trial_config['compute_ground_truth_assessments']:
        #    self.compute_ground_truth_assessments()
        if ('compute_first_order_stats' in self.trial_config) and self.trial_config['compute_first_order_stats']:
            self.compute_first_order_stats()
        if ('run_supervised_hparam_sweep' in self.trial_config) and self.trial_config['run_supervised_hparam_sweep']:
            self.run_supervised_hparam_sweep()
        if ('train_supervised_model' in self.trial_config) and self.trial_config['train_supervised_model']:
            self.train_supervised_model()
            self.plot_supervised_training_curves()
        if ('compute_nn_attributions' in self.trial_config) and self.trial_config['compute_nn_attributions']:
            self.compute_neural_net_attributions()
            if self.dataset_name == 'dpav4':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__DPAv4')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__DPAv4')
                self.compute_supervised_ranks_over_time()
                self.compute_supervised_ranks_over_time(wouters_zaid_model='ZaidNet__DPAv4')
                self.compute_supervised_ranks_over_time(wouters_zaid_model='WoutersNet__DPAv4')
                self.create_paper_rot_plot()
            elif self.dataset_name == 'ascadv1_fixed':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__ASCADv1f')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__ASCADv1f')
                self.compute_supervised_ranks_over_time()
                self.compute_supervised_ranks_over_time(wouters_zaid_model='ZaidNet__ASCADv1f')
                self.compute_supervised_ranks_over_time(wouters_zaid_model='WoutersNet__ASCADv1f')
                self.create_paper_rot_plot()
            elif self.dataset_name == 'ascadv1_variable':
                self.compute_supervised_ranks_over_time()
                self.create_paper_rot_plot()
            elif self.dataset_name == 'aes_hd':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__AES_HD')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__AES_HD')
                self.compute_supervised_ranks_over_time()
                self.compute_supervised_ranks_over_time(wouters_zaid_model='ZaidNet__AES_HD')
                self.compute_supervised_ranks_over_time(wouters_zaid_model='WoutersNet__AES_HD')
                self.create_paper_rot_plot()
        if self.dataset_name not in ['otiait', 'otp', 'dpav4']:
            if ('run_ll_classifiers_hparam_sweep' in self.trial_config) and self.trial_config['run_ll_classifiers_hparam_sweep']:
                self.run_ll_classifiers_hparam_sweep()
            if ('pretrain_classifiers' in self.trial_config) and self.trial_config['pretrain_classifiers']:
                self.pretrain_leakage_localization_classifiers()
        if ('run_ll_hparam_sweep' in self.trial_config) and self.trial_config['run_ll_hparam_sweep']:
            self.run_ll_hparam_sweep()
        if ('run_leakage_localization' in self.trial_config) and self.trial_config['run_leakage_localization']:
            self.run_leakage_localization()
        self.create_main_paper_dnn_auc_plots()
        self.eval_leakage_assessments()
        self.plot_leakage_assessments()
        self.create_main_paper_leakage_assessment_plots()
        self.create_appendix_leakage_assessment_plots()
        #self.compute_dnn_auc_vals_on_baselines()