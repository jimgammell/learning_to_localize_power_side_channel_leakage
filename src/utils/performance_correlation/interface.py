from typing import *
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from multiprocessing import pool
from scipy.stats import kendalltau

from .functional import *
from common import *
#from utils.template_attack import TemplateAttack

class MeasurePerformanceCorrelation:
    def __init__(self,
        leakage_measurements,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        attack_type: Literal['template-attack'] = 'template-attack',
        target_keys: Union[str, Sequence[str]] = 'subbytes',
        target_bytes: Optional[Union[int, Sequence[int]]] = None
    ):
        self.leakage_measurements = leakage_measurements
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.attack_type = attack_type
        self.target_keys = [target_keys] if isinstance(target_keys, str) else target_keys
        self.target_bytes = [target_bytes] if (isinstance(target_bytes, int) or (target_bytes is None)) else target_bytes
    
    def measure_performance(self,
            poi_count: int = 10, seed_count: int = 1, fast: bool = False
    ):
        ranking = self.leakage_measurements.squeeze().argsort()
        if (len(ranking)%poi_count) != 0:
            ranking = np.concatenate([ranking[:poi_count - (len(ranking)%poi_count)], ranking])
            assert (len(ranking)%poi_count) == 0
        poi_sets = ranking.reshape(-1, poi_count)
        if fast:
            poi_sets = poi_sets[np.linspace(0, len(poi_sets)-1, 5).astype(int)] #poi_sets[[0, 1, 2, -3, -2, -1], :]
        performance_metrics = np.full((len(self.attack_dataset), poi_sets.shape[0]), np.nan, dtype=np.float32)
        for idx, poi_set in enumerate(tqdm(poi_sets)):
            _performance_metrics = []
            for target_key in self.target_keys:
                template_attack = TemplateAttack(poi_set, target_key=target_key)
                template_attack.profile(self.profiling_dataset)
                ranks = template_attack.get_ranks(self.attack_dataset)
                _performance_metrics.append(ranks)
            performance_metrics[:, idx] = _performance_metrics[np.argmin([np.mean(x) for x in _performance_metrics])]
        assert np.all(np.isfinite(performance_metrics))
        correlation = soft_kendall_tau(performance_metrics, -np.arange(performance_metrics.shape[-1]))
        means = np.mean(performance_metrics, axis=0)
        stds = np.std(performance_metrics, axis=0)
        return correlation, means, stds