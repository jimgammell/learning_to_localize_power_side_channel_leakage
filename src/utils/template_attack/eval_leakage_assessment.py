from typing import *
from collections import defaultdict
from torch.utils.data import Dataset

from .interface import TemplateAttack
from models.soft_xor_layer import soft_xor
from utils.aes import *
from metrics.rank import get_rank

class TemplateAttackEvaluator:
    def __init__(self,
        leakage_assessment: Sequence[float],
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        dataset_name: Literal['dpav4', 'ascadv1_fixed', 'ascadv1_variable', 'otiait', 'otp', 'aes_hd'],
        poi_counts: Sequence[int] = [1, 5, 9, 13]
    ):
        self.leakage_assessment = leakage_assessment
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.dataset_name = dataset_name
        self.poi_counts = poi_counts
        self.targets = ['r_out', 'subbytes__r_out'] if dataset_name.split('_')[0] == 'ascadv1' else ['label']
        
    def compute_key_predictions(self):
        self.predictions = {}
        self.label_predictions = {}
        for poi_count in self.poi_counts:
            predictions = {}
            for target in self.targets:
                points_of_interest = self.leakage_assessment.argsort()[-poi_count:]
                template_attacker = TemplateAttack(points_of_interest, target_key=target)
                template_attacker.profile(self.profiling_dataset)
                predictions[target] = template_attacker.get_predictions(self.attack_dataset)
            if self.dataset_name.split('_')[0] == 'ascadv1':
                subbytes_predictions = soft_xor(predictions['r_out'], predictions['subbytes__r_out'])
                plaintexts = self.attack_dataset.metadata['plaintext']
                key_predictions = subbytes_predictions[:, AES_SBOX[np.arange(256, dtype=np.uint8) ^ plaintexts]]
            elif self.dataset_name == 'dpav4':
                masked_subbytes_predictions = predictions['label']
                plaintexts = self.attack_dataset.metadata['plaintext']
                offsets = self.attack_dataset.metadata['offset']
                mask = self.attack_dataset.metadata['mask']
                key_predictions = masked_subbytes_predictions[:, AES_SBOX[np.arange(256, dtype=np.uint8) ^ plaintexts] ^ mask[(offsets+1)%16]]
            elif self.dataset_name == 'aes_hd':
                state_predictions = predictions['label']
                ciphertexts_7 = self.attack_dataset.metadata['ciphertext_7']
                ciphertexts_11 = self.attack_dataset.metadata['ciphertext_11']
                key_predictions = state_predictions[:, AES_SBOX[np.arange(256, dtype=np.uint8) ^ ciphertexts_11] ^ ciphertexts_7]
            elif self.dataset_name == 'otiait': # I don't know how to get from sensitive variable to key for these algorithms and don't want to bother.
                                                # Will just evaluate accuracy of the sensitive variable predictions.
                key_predictions = None
            elif self.dataset_name == 'otp':
                key_predictions = None
            else:
                raise NotImplementedError
            self.predictions[poi_count] = key_predictions
            self.label_predictions[poi_count] = predictions['label']
    
    def compute_mean_rank(self):
        self.mean_ranks = {}
        for poi_count in self.poi_counts:
            label_predictions = self.label_predictions[poi_count]
            true_labels = self.attack_dataset.metadata['label']
            self.mean_ranks[poi_count] = get_rank(label_predictions, true_labels).mean()
    
    def compute_mean_accumulated_rank(self):
        assert hasattr(self, 'predictions')
        assert not self.dataset_name in ['otp', 'otiait']
        self.rank_over_times = defaultdict(list)
        for poi_count in self.poi_counts:
            predictions = self.predictions[poi_count]
            true_keys = self.attack_dataset.metadata['key']
            assert all(key == true_keys[0] for key in true_keys)
            true_key = true_keys[0]
            log_probs = None
            for prediction in predictions:
                if log_probs is None:
                    log_probs = prediction
                else:
                    log_probs += prediction
                rank = get_rank(log_probs, true_key)
                self.rank_over_times[poi_count].append(rank)