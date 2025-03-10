from typing import *
import numpy as np
from torch.utils.data import Dataset

from .data_generation import *
from common import *
from utils.aes import *

# To do:
#  - Set up the fixed noise profile to reflect no-ops

LPF_BURN_IN_CYCLES = 100
NO_OP_INSTRUCTION = 0

class SyntheticAES(Dataset):
    def __init__(self,
        epoch_length: int = int(1e5), # number of datapoints per 'full traversal' of the dataset
        infinite_dataset: bool = False, # whether or not to generate a new random datapoint for every __getitem__ call, or to compute epoch_length datapoints ahead of time
        timesteps_per_trace: int = 1000, # number of power measurements per trace
        bit_count: int = 8, # number of bits
        operation_count: int = 32, # number of operations
        leaking_timestep_count_1o: Optional[int] = 1, # number of timesteps with 1st-order leakage
        leaking_timestep_count_2o: Optional[int] = 0, # number of pairs of timesteps with 2nd-order leakage
        leaky_1o_pts: Optional[Sequence[int]] = None, # specify the timesteps at which we should have first-order leakage
        leaky_2o_pts: Optional[Tuple[Sequence[int], Sequence[int]]] = None, # specify the timesteps at which we should have first-order leakage of the (mask, masked_subbytes)
        data_var: float = 1.0, # variance due to data-dependent power consumption
        operation_var: float = 1.0, # variance of the Gaussian distribution from which per-operation biases are drawn
        residual_var: float = 1.0, # variance due to things other than data and operations
        shuffle_locs: int = 1, # number of possible positions for each leaking timestep, to simulate shuffling countermeasure
        max_no_ops: int = 0, # max no-ops which may be inserted before each leaking timestep, to simulate random delay countermeasure
        lpf_beta: float = 0.0, # Traces will be discrete low-pass filtered according to x_lpf[t+1] = lpf_beta*x_lpf[t] + (1-lpf_beta)*x[t+1]
        target_values: Union[str, Sequence[str]] = 'subbytes', # The sensitive variable to be targeted. Options: ['subbytes', 'mask', 'masked_subbytes']
        fixed_key: Optional[np.uint32] = None, # Instead of randomly sampling keys, fix the key to this value.
        fixed_mask: Optional[np.uint32] = None, # Instead of randomly sampling masks, fix the mask to this value.
        fixed_plaintext: Optional[np.uint32] = None, # Instead of randomly sampling plaintexts, fix the plaintext to this value.
        transform: Optional[Callable] = None, # Traces will be transformed by this function before being returned.
        target_transform: Optional[Callable] = None, # Target values will be transformed by this function before being returned.
        should_generate_data: bool = True, # can use this to disable data generation -- useful if we want to copy another dataset
        return_metadata: bool = False # whether to return metadata -- key, plaintext, subbytes, no-op, etc. regardless of whether they are used as the target
    ):
        self.settings = {}
        for key, val in locals().items():
            if key not in ('self', 'key', 'val', 'settings'):
                setattr(self, key, val)
                self.settings[key] = val
        assert self.bit_count <= 32
        super().__init__()
        if isinstance(self.target_values, str):
            self.target_values = [self.target_values]
        if self.should_generate_data:
            self.generate_data()
        self.class_count = 2**self.bit_count
    
    def generate_data(self):
        leaking_cycles = NUMPY_RNG.choice(
            self.timesteps_per_trace-self.max_no_ops, self.shuffle_locs*(self.leaking_timestep_count_1o + 2*self.leaking_timestep_count_2o), replace=False
        )
        self.leaking_subbytes_cycles = leaking_cycles[:self.shuffle_locs*self.leaking_timestep_count_1o]
        self.leaking_mask_cycles = leaking_cycles[self.shuffle_locs*self.leaking_timestep_count_1o:-self.shuffle_locs*self.leaking_timestep_count_2o]
        self.leaking_masked_subbytes_cycles = leaking_cycles[-self.shuffle_locs*self.leaking_timestep_count_2o:]
        if self.leaky_1o_pts is not None:
            self.leaking_subbytes_cycles = np.concatenate([self.leaking_subbytes_cycles, self.leaky_1o_pts])
            self.leaking_timestep_count_1o = len(self.leaking_subbytes_cycles)//self.shuffle_locs
        if self.leaky_2o_pts is not None:
            self.leaking_mask_cycles = np.concatenate([self.leaking_mask_cycles, self.leaky_2o_pts[0, :]])
            self.leaking_masked_subbytes_cycles = np.concatenate([self.leaking_masked_subbytes_cycles, self.leaky_2o_pts[1, :]])
            self.leaking_timestep_count_2o = (len(self.leaking_mask_cycles) + len(self.leaking_masked_subbytes_cycles))//(2*self.shuffle_locs)
        self.per_operation_power_consumption = np.sqrt(self.operation_var)*NUMPY_RNG.standard_normal(size=self.operation_count, dtype=np.float32)
        self.operations = NUMPY_RNG.choice(self.operation_count, self.timesteps_per_trace+LPF_BURN_IN_CYCLES, replace=True)
        self.fixed_noise_profile = self.per_operation_power_consumption[self.operations]
        if not self.infinite_dataset:
            self.traces, self.metadata = self.generate_datapoints(self.epoch_length)
    
    def generate_datapoints(self, count):
        if self.infinite_dataset:
            numpy_rng = np.random.default_rng() # multiprocessing in dataloaders screws up randomness when we have only one RNG. I'm just going to do this as a temporary fix.
        else:
            numpy_rng = NUMPY_RNG
        if self.fixed_key is not None:
            keys = np.full((count,), self.fixed_key, dtype=np.uint32)
        else:
            keys = numpy_rng.choice(2**self.bit_count, count, replace=True)
        if self.fixed_plaintext is not None:
            plaintexts = np.full((count,), self.fixed_plaintext, dtype=np.uint32)
        else:
            plaintexts = numpy_rng.choice(2**self.bit_count, count, replace=True)
        if self.fixed_mask is not None:
            masks = np.full((count,), self.fixed_mask, dtype=np.uint32)
        else:
            masks = numpy_rng.choice(2**self.bit_count, count, replace=True)
        subbytes = AES_SBOX[keys ^ plaintexts]
        masked_subbytes = masks ^ subbytes
        traces = np.empty((count, 1, self.timesteps_per_trace), dtype=np.float32)
        for idx in range(count):
            data = numpy_rng.choice(2**self.bit_count, self.timesteps_per_trace+LPF_BURN_IN_CYCLES, replace=True).astype(np.uint32)
            locs = np.array([], dtype=np.uint32)
            vals = np.array([], dtype=np.uint32)
            if self.leaking_timestep_count_1o > 0:
                locs = np.concatenate([locs, numpy_rng.choice(self.leaking_subbytes_cycles, self.leaking_timestep_count_1o, replace=False)], axis=0)
                vals = np.concatenate([vals, np.full(self.leaking_timestep_count_1o, subbytes[idx], dtype=np.uint32)], axis=0)
            if self.leaking_timestep_count_2o > 0:
                locs = np.concatenate([locs, numpy_rng.choice(self.leaking_mask_cycles, self.leaking_timestep_count_2o, replace=False)], axis=0)
                vals = np.concatenate([vals, np.full(self.leaking_timestep_count_2o, masks[idx], dtype=np.uint32)], axis=0)
                locs = np.concatenate([locs, numpy_rng.choice(self.leaking_masked_subbytes_cycles, self.leaking_timestep_count_2o, replace=False)], axis=0)
                vals = np.concatenate([vals, np.full(self.leaking_timestep_count_2o, masked_subbytes[idx], dtype=np.uint32)], axis=0)
            locs += LPF_BURN_IN_CYCLES
            fixed_noise_profile = self.fixed_noise_profile.copy()
            if self.max_no_ops > 0:
                for noop_idx in range(len(locs)):
                    loc = locs[noop_idx]
                    no_ops = numpy_rng.integers(self.max_no_ops+1)
                    if no_ops > 0:
                        #fixed_noise_profile[loc+no_ops:] = fixed_noise_profile[loc:-no_ops]
                        #fixed_noise_profile[loc:loc+no_ops] = self.per_operation_power_consumption[NO_OP_INSTRUCTION]
                        locs[noop_idx] += no_ops
            data[locs] = vals
            random_noise = np.sqrt(self.residual_var)*numpy_rng.standard_normal(size=self.timesteps_per_trace+LPF_BURN_IN_CYCLES, dtype=np.float32)
            trace = generate_trace(data, self.timesteps_per_trace+LPF_BURN_IN_CYCLES, fixed_noise_profile, random_noise, self.data_var)
            if self.lpf_beta > 0:
                trace = apply_ema(trace, self.lpf_beta)
            trace = trace[LPF_BURN_IN_CYCLES:]
            traces[idx, 0, ...] = trace
        metadata = {'key': keys, 'plaintext': plaintexts, 'subbytes': subbytes, 'mask': masks, 'masked_subbytes': masked_subbytes}
        return traces, metadata
    
    def __getitem__(self, idx):
        if self.infinite_dataset:
            trace, metadata = self.generate_datapoints(1 if not hasattr(idx, '__len__') else len(idx))
            if not hasattr(idx, '__len__'):
                trace = trace[0, ...]
                metadata = {key: val[0] for key, val in metadata.items()}
            target = np.stack([metadata[key] for key in self.target_values]).squeeze()
        else:
            trace = self.traces[idx, ...]
            target = np.stack([self.metadata[key][idx] for key in self.target_values]).squeeze()
            metadata = {key: val[idx] for key, val in self.metadata.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            metadata.update({'label': target})
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self):
        return self.epoch_length
    
    def __repr__(self):
        return '\n\t'.join((
            f'{self.__class__.__name__}(',
            *[f'{key}={val}' for key, val in self.settings.items()]
        )) + '\n)'

class SyntheticAESLike(SyntheticAES):
    def __init__(self,
        base_dataset: SyntheticAES,
        epoch_length: int = 10000,
        infinite_dataset: bool = False,
        fixed_key: Optional[np.uint8] = None,
        fixed_plaintext: Optional[np.uint8] = None,
        fixed_mask: Optional[np.uint8] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        super().__init__(
            epoch_length=epoch_length,
            infinite_dataset=infinite_dataset,
            timesteps_per_trace=base_dataset.timesteps_per_trace,
            bit_count=base_dataset.bit_count,
            operation_count=base_dataset.operation_count,
            leaking_timestep_count_1o=base_dataset.leaking_timestep_count_1o,
            leaking_timestep_count_2o=base_dataset.leaking_timestep_count_2o,
            data_var=base_dataset.data_var,
            operation_var=base_dataset.operation_var,
            residual_var=base_dataset.residual_var,
            shuffle_locs=base_dataset.shuffle_locs,
            max_no_ops=base_dataset.max_no_ops,
            lpf_beta=base_dataset.lpf_beta,
            target_values=base_dataset.target_values,
            fixed_key=fixed_key,
            fixed_mask=fixed_mask,
            fixed_plaintext=fixed_plaintext,
            transform=transform,
            target_transform=target_transform,
            return_metadata=return_metadata,
            should_generate_data=False
        )
        self.leaking_subbytes_cycles = base_dataset.leaking_subbytes_cycles
        self.leaking_mask_cycles = base_dataset.leaking_mask_cycles
        self.leaking_masked_subbytes_cycles = base_dataset.leaking_masked_subbytes_cycles
        self.per_operation_power_consumption = base_dataset.per_operation_power_consumption
        self.operations = base_dataset.operations
        self.fixed_noise_profile = base_dataset.fixed_noise_profile
        if not self.infinite_dataset:
            self.traces, self.metadata = self.generate_datapoints(self.epoch_length)