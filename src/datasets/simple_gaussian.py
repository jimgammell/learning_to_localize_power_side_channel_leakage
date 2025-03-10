from typing import *
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

class SimpleGaussianDataset(Dataset):
    def __init__(self,
        buffer_size: int = 10000,
        random_feature_count: int = 1,
        easy_feature_count: int = 1,
        easy_feature_snrs: Union[float, Sequence[float]] = 1.0,
        no_hard_feature: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.random_feature_count = random_feature_count
        self.easy_feature_count = easy_feature_count
        self.easy_feature_snrs = easy_feature_snrs if hasattr(easy_feature_snrs, '__len__') else self.easy_feature_count*[easy_feature_snrs]
        assert self.easy_feature_count == len(self.easy_feature_snrs)
        self.easy_feature_snrs = np.array(self.easy_feature_snrs)
        self.no_hard_feature = no_hard_feature
        self.transform = transform
        self.target_transform = target_transform
        self.timesteps_per_trace = self.random_feature_count + self.easy_feature_count + (0 if self.no_hard_feature else 2)
        self.class_count = 2
        self.return_metadata = False
        self.item_iterator = self.get_item_iterator()
    
    def generate_data(self):
        labels = np.random.randint(2, size=(self.buffer_size,))
        random_features = np.random.randn(self.buffer_size, 1, self.random_feature_count)
        easy_feature_noise_std = np.sqrt(1/(1+self.easy_feature_snrs))
        easy_feature_signal_std = np.sqrt(self.easy_feature_snrs/(1+self.easy_feature_snrs))
        easy_features = (
            easy_feature_noise_std.reshape(1, 1, -1)*np.random.randn(self.buffer_size, 1, self.easy_feature_count)
            + 2*easy_feature_signal_std.reshape(1, 1, -1)*labels.reshape(-1, 1, 1).astype(float)
        )
        if not self.no_hard_feature:
            hard_snr = 1e0
            masks = np.random.randint(2, size=(self.buffer_size,))
            masked_labels = masks ^ labels
            masks_feature = (
                np.sqrt(1/(1+hard_snr))*np.random.randn(self.buffer_size, 1, 1)
                + 2*np.sqrt(hard_snr)*masks.reshape(-1, 1, 1).astype(float)
            )
            masked_labels_feature = (
                np.sqrt(1/(1+hard_snr))*np.random.randn(self.buffer_size, 1, 1)
                + 2*np.sqrt(hard_snr)*masked_labels.reshape(-1, 1, 1).astype(float)
            )
            datapoints = np.concatenate([random_features, easy_features, masks_feature, masked_labels_feature], axis=-1)
        else:
            datapoints = np.concatenate([random_features, easy_features], axis=-1)
        return datapoints, labels
    
    def get_item_iterator(self):
        while True:
            datapoints, labels = self.generate_data()
            for datapoint, label in zip(datapoints, labels):
                if self.transform is not None:
                    datapoint = self.transform(datapoint)
                if self.target_transform is not None:
                    label = self.target_transform(label)
                yield datapoint, label
    
    def __getitem__(self, idx):
        if hasattr(idx, '__len__'):
            datapoints, labels = [], []
            for _ in range(len(idx)):
                datapoint, label = next(self.item_iterator)
                datapoints.append(datapoint)
                labels.append(label)
            datapoints = np.stack(datapoints)
            labels = np.stack(labels)
        else:
            datapoints, labels = next(self.item_iterator)
        datapoints = datapoints.astype(np.float32)
        labels = labels.astype(np.int64)
        if self.return_metadata:
            return datapoints, labels, {'label': labels}
        else:
            return datapoints, labels
    
    def __len__(self):
        return self.buffer_size