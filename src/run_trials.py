import os
from copy import copy
import yaml
import argparse
import time
from torch.utils.data import DataLoader

from common import *
from utils.flatten_dict import flatten_dict, unflatten_dict
from training_modules.supervised_deep_sca import SupervisedTrainer
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer
from utils.baseline_assessments import FirstOrderStatistics, NeuralNetAttribution
from trials.utils import *
from utils.gmm_performance_correlation import GMMPerformanceCorrelation
from trials.real_dataset_baseline_comparison import Trial as RealBaselineComparisonTrial
from trials.synthetic_data_experiments import Trial as SyntheticTrial
from trials.portability_experiments import Trial as PortabilityTrial
from trials.toy_gaussian_experiments import Trial as ToyGaussianTrial

AVAILABLE_DATASETS = (
    [x.split('.')[0] for x in os.listdir(CONFIG_DIR) if x.endswith('.yaml') and not(x in ['default_config.yaml', 'global_variables.yaml'])]
     + ['synthetic', 'portability', 'toy_gaussian']
)
with open(os.path.join(CONFIG_DIR, 'default_config.yaml'), 'r') as f:
    DEFAULT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

GAUSSIAN_LEAKAGE_TYPES = ['1o-sweep']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=AVAILABLE_DATASETS)
    parser.add_argument('--seed-count', type=int, default=1, action='store')
    parser.add_argument('--lr-count', type=int, default=20, action='store')
    parser.add_argument('--lambda-count', type=int, default=20, action='store')
    parser.add_argument('--trial-dir', default=None, action='store')
    parser.add_argument('--override-theta-pretrain-path', default=None, action='store')
    clargs = parser.parse_args()
    dataset = clargs.dataset
    seed_count = clargs.seed_count
    trial_dir = os.path.join(OUTPUT_DIR, dataset if clargs.trial_dir is None else clargs.trial_dir)
    assert seed_count > 0
    
    if dataset == 'synthetic':
        trial = SyntheticTrial(
            logging_dir=trial_dir,
            seed_count=seed_count
        )
        trial()
    elif dataset == 'toy_gaussian':
        trial = ToyGaussianTrial(
            logging_dir=trial_dir,
            seed_count=seed_count
        )
        trial()
    elif dataset == 'portability':
        with open(os.path.join(CONFIG_DIR, 'portability.yaml'), 'r') as f:
            trial_config = yaml.load(f, Loader=yaml.FullLoader)
        trial = PortabilityTrial(
            logging_dir=trial_dir,
            trial_config=trial_config
        )
        trial()
    else:
        with open(os.path.join(CONFIG_DIR, f'{dataset}.yaml'), 'r') as f:
            trial_config = yaml.load(f, Loader=yaml.FullLoader)
        trial = RealBaselineComparisonTrial(
            dataset_name=dataset,
            trial_config=trial_config,
            seed_count=seed_count,
            logging_dir=trial_dir
        )
        trial()

if __name__ == '__main__':
    main()