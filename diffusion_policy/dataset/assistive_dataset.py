from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask, create_indices, get_create_indices_fn_whole_sequence, PaddingType)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class AssistiveLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            whole_sequence_sampling: bool = False,
            gap_horizon: int = None,
            out_obs_horizon: int = None
    ):
        """
        whole_sequence_sampling - flag.
            If on, full episodes embedded into the horizon-sized buffer of zeros will be sampled.
            If off, classical chunk sampling.
        """
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        
        if whole_sequence_sampling:
            self.create_indices_fn = get_create_indices_fn_whole_sequence(gap_horizon, out_obs_horizon)
            self.padding = PaddingType.ZERO_ACT_REPLICATE_STATE
        else:
            self.create_indices_fn = create_indices
            self.padding = PaddingType.REPLICATE

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            create_indices_fn=self.create_indices_fn,
            padding=self.padding,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            create_indices_fn=self.create_indices_fn,
            padding=self.padding,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer, no_len=True)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample, no_len=False):
        data = {
            'obs': sample['state'].astype(np.float32), # T, S
            'action': sample['action'].astype(np.float32) # T, A
        }
        if not no_len:
            data.update({"length": sample["length"].astype(np.int_)})
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    from pathlib import Path
    ROOT_DIR = Path(__file__).absolute().parent.parent.parent
    zarr_path = ROOT_DIR / "teleop_datasets" / "FeedingJaco-v1.zarr"
    dataset = AssistiveLowdimDataset(str(zarr_path), horizon=50)
    dataset_full = AssistiveLowdimDataset(str(zarr_path), horizon=200, whole_sequence_sampling=True)
    print(dataset_full[0]["length"])
    import ipdb
    ipdb.set_trace()

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == "__main__":
    test()