from typing import *
import os
from copy import copy
import numpy as np

from datasets.aes_pt_v2 import AES_PTv2, VALID_DEVICES as _VALID_DEVICES
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer
VALID_DEVICES = [x for x in _VALID_DEVICES if x != 'Pinata']

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike],
        countermeasure: Literal['MS1', 'MS2', 'Unprotected'] = 'Unprotected',
        trial_config: dict = {},
        seed_count: int = 1,
        batch_size: int = 1000,
        budgets: Union[float, Sequence[float]] = np.logspace(0, 4, 9),
        override_run_kwargs: dict = {},
        override_leakage_localization_kwargs: dict = {},
        pretrain_classifiers_only: bool = False
    ):
        self.logging_dir = logging_dir
        self.countermeasure = countermeasure
        self.trial_config = trial_config
        self.seed_count = seed_count
        if self.seed_count != 1:
            raise NotImplementedError
        self.batch_size = batch_size
        self.budgets = budgets if hasattr(budgets, '__len__') else [budgets]
        self.run_kwargs = {'max_steps': 10000, 'anim_gammas': False}
        self.leakage_localization_kwargs = {'classifiers_name': 'mlp-1d', 'theta_lr': 1e-3, 'etat_lr': 1e-2, 'etat_lr_scheduler_name': 'CosineDecayLRSched'}
        self.run_kwargs.update(override_run_kwargs)
        self.leakage_localization_kwargs.update(override_leakage_localization_kwargs)
        self.pretrain_classifiers_only = pretrain_classifiers_only
    
    def get_leave_3_out_datasets(self):
        out = {}
        for device in VALID_DEVICES:
            profiling_dataset = AES_PTv2(
                root=self.trial_config['data_dir'], train=True, devices=device
            )
            attack_dataset = AES_PTv2(
                root=self.trial_config['data_dir'], train=False, devices=[d for d in VALID_DEVICES if d != device]
            )
            out[device] = (profiling_dataset, attack_dataset)
        return out
    
    def get_leave_1_out_datasets(self):
        out = {}
        for device in VALID_DEVICES:
            profiling_dataset = AES_PTv2(
                root=self.trial_config['data_dir'], train=True, devices=[d for d in VALID_DEVICES if d != device]
            )
            attack_dataset = AES_PTv2(
                root=self.trial_config['data_dir'], train=False, devices=device
            )
            out[device] = (profiling_dataset, attack_dataset)
        return out
    
    def get_leave_0_out_dataset(self):
        profiling_dataset = AES_PTv2(
            root=self.trial_config['data_dir'], train=True, devices=VALID_DEVICES
        )
        attack_dataset = AES_PTv2(
            root=self.trial_config['data_dir'], train=False, devices=VALID_DEVICES
        )
        return (profiling_dataset, attack_dataset)
    
    def get_datasets(self):
        return {
            **{key: val for key, val in self.get_leave_3_out_datasets().items()},
            **{f'-{key}': val for key, val in self.get_leave_1_out_datasets().items()},
            'all': self.get_leave_0_out_dataset()
        }
        
    def get_trainer(self, profiling_dataset, attack_dataset, budget):
        trainer = LeakageLocalizationTrainer(
            profiling_dataset, attack_dataset,
            default_data_module_kwargs={'train_batch_size': self.batch_size},
            default_training_module_kwargs={'budget': budget, **self.leakage_localization_kwargs}
        )
        return trainer
    
    def run_hparam_sweep(self):
        base_dir = os.path.join(self.logging_dir, 'hparam_sweeps')
        for dataset_name, dataset in self.get_datasets().items():
            if not os.path.exists(os.path.join(self.ll_hparam_sweep_dir, 'results.pickle')):
                print('Running LL hparam sweep...')
                kwargs = copy(self.trial_config['default_kwargs'])
                kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
                kwargs.update(self.trial_config['leakage_localization_kwargs'])
                ll_trainer = LeakageLocalizationTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=kwargs)
                ll_trainer.htune_leakage_localization(
                    self.ll_hparam_sweep_dir,
                    pretrained_classifiers_logging_dir=None,
                    trial_count=50,
                    max_steps=10000,
                    supervised_dnn=None,
                    references=None, #{key: val.mean(axis=0) for key, val in self.get_ground_truth_assessments().items()}
                )
            else:
                print('Found existing LL hparam sweep.')
        #self.ll_optimal_hparams = plot_ll_hparam_sweep(self.ll_hparam_sweep_dir)
        #print(f'Optimal LL hyperparameters on {self.dataset_name}: {self.ll_optimal_hparams}')
    
    def __call__(self):
        self.run_hparam_sweep()