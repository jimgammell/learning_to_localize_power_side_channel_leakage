import os
import h5py
import numpy as np
from torch.utils.data import Dataset

class ED25519(Dataset):
    def __init__(self,
        root=None,
        train=True,
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.path = os.path.join(self.root, 'databaseEdDSA.h5')
        assert os.path.exists(self.path)
        with h5py.File(self.path) as database_file:
            if self.train:
                database_file = database_file['Profiling_traces']
            else:
                database_file = database_file['Attack_traces']
            self.traces = np.array(database_file['traces'], dtype=np.float32)[:, np.newaxis, :]
            self.labels = np.array(database_file['label'], dtype=np.uint8)
        self.dataset_length = len(self.traces)
        assert self.dataset_length == len(self.labels)
        self.data_shape = self.traces[0].shape
        self.timesteps_per_trace = np.prod(self.data_shape)
        self.return_metadata = False
        self.class_count = 16

    def __getitem__(self, indices):
        trace = self.traces[indices, ...]
        target = self.labels[indices]
        metadata = {'label': target}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self):
        return self.dataset_length