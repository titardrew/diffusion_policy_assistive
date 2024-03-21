from pathlib import Path

from einops import rearrange, reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

from torch.utils.data import DataLoader

from diffusion_policy.dataset.assistive_dataset import AssistiveLowdimDataset

def normalize(normalizer, x):
    if normalizer is None:
        return x
    else:
        return normalizer.normalize(x)


def unnormalize(normalizer, x):
    if normalizer is None:
        return x
    else:
        return normalizer.unnormalize(x)


class StatePredictor(nn.Module):
    def __init__(self, observation_size, action_size, horizon, hidden_sizes=(256, 256)):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        net = ()

        in_size = (observation_size + action_size) * (horizon - 1)
        out_size = observation_size

        sizes = (in_size, *hidden_sizes, out_size)
        for i in range(len(sizes) - 1):
            net += (
                nn.Linear(in_features=sizes[i], out_features=sizes[i + 1]),
                nn.SELU(),
            )

        self.net = nn.Sequential(*net[:-1])
        self.normalizer = None
 
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def forward(self, batch):
        if self.normalizer:
            batch = self.normalizer.normalize(batch)

        act_obs = torch.cat([batch['obs'], batch['action']], axis=-1)
        next_obs = self.net(act_obs) #+ batch['obs']

        if self.normalizer:
            next_obs = self.normalizer['obs'].unnormalize(next_obs)
        return next_obs

    def compute_loss(self, batch):
        this_batch_size = batch['action'].shape[0]
        input_act = batch['action'][:, :-1, :].reshape(this_batch_size, -1)
        input_obs = batch['obs'][:, :-1, :].reshape(this_batch_size, -1)
        output_obs = batch['obs'][:, -1, :]
        pred = self.forward({'action': input_act, 'obs': input_obs})
        return F.mse_loss(output_obs, pred).mean()
    

def train(zarr_path: Path):
    num_epochs = 50
    n_obs_steps = 1
    n_action_steps = 2
    horizon = 5
    batch_size = 256
    observation_size = 25
    action_size = 7
    lr = 3e-4
    hidden_sizes = (256, 256)
    val_ratio = 0.1

    dataset = AssistiveLowdimDataset(
        str(zarr_path),
        horizon=horizon,
        pad_before=n_obs_steps - 1,
        pad_after=n_action_steps - 1,
        val_ratio=val_ratio,
        max_train_episodes=None
    )
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #normalizer = dataset.get_normalizer()

    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    state_predictor = StatePredictor(
        observation_size=observation_size,
        action_size=action_size,
        horizon=horizon,
        hidden_sizes=hidden_sizes
    )
    #state_predictor.set_normalizer(normalizer)

    optimizer = torch.optim.Adam(state_predictor.parameters(), lr=lr)
        
    def _val_epoch():
        with torch.no_grad():
            val_losses = []
            for batch in val_dataloader:
                loss = state_predictor.compute_loss(batch)
                val_losses.append(loss.cpu().numpy())
            return np.mean(val_losses)

    val_loss = _val_epoch()
    print(f"Initial validation loss: {val_loss}")
    for i_epoch in range(num_epochs):
        losses = []
        state_predictor.train()
        for i, batch in enumerate(train_dataloader):
            loss = state_predictor.compute_loss(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            #print(f"Loss {np.mean(losses)}")
        val_loss = _val_epoch()
        print(f"Epoch: {i_epoch}, Loss {np.mean(losses):.6f}. Val loss {val_loss:.6f}")

train("teleop_dataset/FeedingJaco-v1.zarr")

