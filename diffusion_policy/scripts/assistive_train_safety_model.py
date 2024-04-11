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

class SafetyModel(nn.Module):
    def check_safety_runner(self, batch):
        """
        batch: Not multibatch, just a {'obs': shape=(Batch, To, Obs), 'action': shape=(Batch, Ta, Action)}
        Returns:
            * is_safe: Bool tensor with True for entries of batch which define a "safe state/plan", i.e.
                when a robot doesn't have to recalculate the plan, retrace back to a safer state,
                or to halt the episode.
        """
        raise NotImplementedError()
    
    def reset(self):
        pass

    def forward(self, batch):
        raise NotImplementedError()
    
    def compute_loss(self, batch):
        raise NotImplementedError()

    def override_params(self, params):
        raise NotImplementedError()


"""
# State prediction safety model
  ## One-step
    pred_sT+1 = Pred(s0, s1, ..., sT, a0, a1, ..., aT)
    Error = mse(pred_sT+1, sT+1)
    If Error >= Threshold -> Resample / Back off / Halt
  ## Multi-step
    pred_sT+1, ..., pred_sT+N = Pred(s0, s1, ..., sT, a0, a1, ..., aT)
    Error = 1/N Sum_i=1^N [mse(pred_sT+i, sT+i)]
    If Error >= Threshold -> Resample / Back off / Halt
  ## K-Ensemble (One-step)
    pred_sT+1_1, ..., pred_sT+1_K = Pred(s0, s1, ..., sT, a0, a1, ..., aT)
    Uncertainty = STD(pred_sT+1_1, ..., pred_sT+1_K)
    If Uncertainty >= Threshold -> Resample / Back off / Halt
# Q-value safety model
  Use CQL offline RL evaluation procedure to fit Q(s, a) function.
  Reward is 1, if Success
  Reward is 0, if Failure
  If Q-value <= Thresh(t) -> Resample / Back off / Halt
"""


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    def __init__(self, in_size, out_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False, device='cpu'):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(in_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = out_size
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)

        total_loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            total_loss += self.get_decay_loss()

        mse_loss = torch.sum(torch.pow(mean - labels, 2), dim=2)
        return total_loss, mse_loss

"""
Safety Model Via Offline Q-value Network Ensembles
::Q value ensemble::
Q_1(s, a), ..., Q_N(s, a) -> mean, var
if mean is abnormaly low (lower than before) -> resample/halt
if var is abnormaly high                     -> resample/halt
"""
class EnsembleStatePredictionSM(SafetyModel):
    def __init__(
        self,
        observation_size,
        action_size,
        in_horizon,
        out_horizon=1,
        predict_delta=False,
        device="cpu",
    ):
        super().__init__()
        self.in_horizon = in_horizon
        self.out_horizon = out_horizon
        self.action_size = action_size
        self.observation_size = observation_size
        self.predict_delta = predict_delta
        self.ensemble_size = 5
        self.device = device

        self.net = EnsembleModel(
            in_size=(observation_size + action_size) * in_horizon,
            #in_size=observation_size * in_horizon,
            out_size=observation_size * out_horizon,
            ensemble_size=self.ensemble_size,
            hidden_size=200,
            use_decay=False,
            device=device,
        )

        self.normalizer = None

        self.params = {
            "check_mse": False,
            "mse_threshold": 5.0,
            "check_ensemble": True,
            "std_threshold": 0.1,
        }
 
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def forward(self, multibatch):
        if self.normalizer:
            multibatch = self.normalizer.normalize(multibatch)

        # [Ensemble, Batch, Obs] + [Ensemble, Batch, Act] => [Ensemble, Batch, ObsAct]
        obs_act = torch.cat([multibatch['obs'], multibatch['action']], axis=-1)
        #obs_act = torch.cat([multibatch['obs']], axis=-1)
        
        net_out_mean, net_out_logvar = self.net(obs_act, ret_log_var=True)
        if self.predict_delta:
            latest_obs = multibatch['obs'][..., -self.observation_size:]
            next_obs = net_out_mean + latest_obs.repeat(1, self.out_horizon)
            next_obs_logvar = net_out_logvar
        else:
            next_obs = net_out_mean
            next_obs_logvar = net_out_logvar

        if self.normalizer:
            next_obs = self.normalizer['obs'].unnormalize(next_obs)
        return next_obs, next_obs_logvar

    def compute_loss(self, multibatch, return_mse=False):
        this_ensemble_size = multibatch['action'].shape[0]
        this_batch_size = multibatch['action'].shape[1]
        input_act = multibatch['action'][..., :-self.out_horizon, :].reshape(this_ensemble_size, this_batch_size, -1)
        input_obs = multibatch['obs'][..., :-self.out_horizon, :].reshape(this_ensemble_size, this_batch_size, -1)
        output_obs = multibatch['obs'][..., -self.out_horizon:, :].reshape(this_ensemble_size, this_batch_size, -1)
        pred_means, pred_logvars = self.forward({
            'action': input_act,  # [Ensemble, Batch, Act]
            'obs': input_obs      # [Ensemble, Batch, Obs]
        })  # -> [Ensemble, Batch, Obs], [Ensemble, Batch, Obs]
        
        loss, mse = self.net.loss(pred_means, pred_logvars, output_obs, inc_var_loss=True)
        if return_mse:
            return loss, mse
        else:
            return loss
    
    def compute_ensemble_mse(self, multibatch):
        assert torch.allclose(multibatch['action'].mean(dim=0), multibatch['action'][0]), "Currently multibatch must be just a broadcasted tensor for ensembling."
        this_ensemble_size = multibatch['action'].shape[0]
        this_batch_size = multibatch['action'].shape[1]
        input_act = multibatch['action'][..., :-self.out_horizon, :].reshape(this_ensemble_size, this_batch_size, -1)
        input_obs = multibatch['obs'][..., :-self.out_horizon, :].reshape(this_ensemble_size, this_batch_size, -1)
        output_obs = multibatch['obs'][..., -self.out_horizon:, :].reshape(this_ensemble_size, this_batch_size, -1)
        pred_means, pred_logvars = self.forward({
            'action': input_act,  # [Ensemble, Batch, Act]
            'obs': input_obs      # [Ensemble, Batch, Obs]
        })  # -> [Ensemble, Batch, Obs], [Ensemble, Batch, Obs]
        
        return torch.sum((pred_means.mean(dim=0) - output_obs[0])**2, dim=-1).mean()
    
    def check_safety_mse(self, batch, mse_threshold=1.0, rr_log=False):

        with torch.no_grad():
            _, mse_batch = self.compute_loss(batch, return_mse=True)
            mse_batch = mse_batch.cpu().numpy()
            is_safe = mse_batch.mean(axis=0) < mse_threshold

            if rr_log:
                import rerun as rr
                rr.log("safety_model/ensemble_state_predictor_mse", rr.Scalar(mse_batch.mean(axis=0)[0]))
                rr.log("safety_model/ensemble_state_predictor_mse_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'mse': mse_batch} 

    def check_safety_ensemble(self, batch, std_threshold=1.0, rr_log=False):

        with torch.no_grad():
            this_ensemble_size = batch['action'].shape[0]
            this_batch_size = batch['action'].shape[1]
            input_act = batch['action'][..., :-self.out_horizon, :].reshape(this_ensemble_size, this_batch_size, -1)
            input_obs = batch['obs'][..., :-self.out_horizon, :].reshape(this_ensemble_size, this_batch_size, -1)
            pred_means, pred_logvars = self.forward({'action': input_act, 'obs': input_obs})
            # TODO(aty): maybe check pred_logvars? We can detect invalid ensemble members using it (e.g. it's too large).
            mean_std = torch.std(pred_means, dim=0).cpu().mean(dim=1).numpy()
            is_safe = mean_std < std_threshold
            if rr_log:
                import rerun as rr
                rr.log("safety_model/ensemble_state_predictor_std", rr.Scalar(mean_std[0]))
                rr.log("safety_model/ensemble_state_predictor_std_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'std': mean_std} 
    
    def reset(self):
        pass
    
    def override_params(self, params):
        if params is not None:
            self.params.update(params)
    
    def check_safety_runner(self, batch, rr_log=False):
        with torch.no_grad():
            horizon = self.out_horizon + self.in_horizon
            multibatch = multi_repeat({
                'obs': batch['obs'][:, :horizon],
                'action': batch['action'][:, :horizon],
            }, self.ensemble_size)

            check_mse = self.params['check_mse']
            mse_threshold = self.params['mse_threshold']
            check_ensemble = self.params['check_ensemble']
            std_threshold = self.params['std_threshold']

            if check_mse:
                is_safe, stats = self.check_safety_mse(multibatch, mse_threshold, rr_log)
            elif check_ensemble:
                is_safe, stats = self.check_safety_ensemble(multibatch, std_threshold, rr_log)
            
            return is_safe
            

class MultibatchDataLoader:
    def __init__(self, ensemble_size, *dataloader_args, **dataloader_kwargs):
        self.dataloaders = [DataLoader(*dataloader_args, **dataloader_kwargs) for _ in range(ensemble_size)]
        self.iters = None

    def __iter__(self):
        self.iters = [iter(dataloader) for dataloader in self.dataloaders]
        return self
    
    def __next__(self):
        if self.iters is None:
            raise StopIteration
        vals = []
        for data_iter in self.iters:
            vals.append(next(data_iter))

        if isinstance(vals[0], dict):
            # Loaders that we use return dicts of batches.
            # So we have to stack those batches for each key.
            multibatch = dict()
            for key in vals[0].keys():
                key_vals = []
                for val in vals:
                    key_vals.append(val[key])
                multibatch[key] = torch.stack(key_vals)
        else:
            # A normal batch
            multibatch = torch.stack(vals)

        return multibatch

def multi_repeat(x, times):
    # x - [Batch, Horizon, Dim]
    def _repeat(x):
        # => [Times, Batch, Horizon, Dim]
        return x.repeat([times, 1, 1, 1])  #.transpose(2, 0, 1)
    if isinstance(x, dict):
        return {key: _repeat(val) for key, val in x.items()}
    else:
        return _repeat(x)


import click
@click.command()
@click.option("--zarr_path", default="teleop_datasets/FeedingJaco-v1.zarr", type=Path, help="Path to a datset.")
@click.option("--num_epochs", default=50, help="Number of epochs to train.")
@click.option("--n_obs_steps", default=2, help="To")
@click.option("--n_action_steps", default=2, help="Ta")
@click.option("--in_horizon", default=1, help="Number of obs as input to the safety model.")
@click.option("--out_horizon", default=1, help="Number of obs as out from the safety model.")
@click.option("--batch_size", default=256)
@click.option("--observation_size", default=25, help="Default - FeedingJaco-v1 (25)")
@click.option("--action_size", default=7, help="Default - FeedingJaco-v1 (7)")
@click.option("--lr", default=3e-4, help="Learing rate.")
@click.option("--val_ratio", default=0.1, help="Fraction of data used for validation.")
@click.option("--save_path", default="ensemble_state_prediction_sm3.pth", help="Output checkpoint path.")
@click.option("--ensemble_size", default=5, help="Number of members of ensemble.")
def train(
    zarr_path: Path,
    num_epochs=50,
    n_obs_steps=2,
    n_action_steps=2,
    in_horizon=1,
    out_horizon=1,
    batch_size=256,
    observation_size=25,
    action_size=7,
    lr=3e-4,
    val_ratio=0.1,
    save_path="ensemble_state_prediction_sm3.pth",
    ensemble_size=5,
):

    dataset = AssistiveLowdimDataset(
        str(zarr_path),
        horizon=in_horizon+out_horizon,
        pad_before=n_obs_steps - 1,
        pad_after=n_action_steps - 1,
        val_ratio=val_ratio,
        max_train_episodes=None
    )
    #print(f"T_o: {n_obs_steps}\nT_a: {n_action_steps}\nH: {in_horizon+out_horizon}")
    #val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #for batch in val_dataloader:
    #    for key, val in batch.items():
    #        print(f"{key}: {val.shape}")
    #    break

    train_dataloader = MultibatchDataLoader(ensemble_size, dataset, batch_size=batch_size, shuffle=True)
    #normalizer = dataset.get_normalizer()

    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    state_predictor = EnsembleStatePredictionSM(
        observation_size=observation_size,
        action_size=action_size,
        in_horizon=in_horizon,
        out_horizon=out_horizon,
    )
    #state_predictor.set_normalizer(normalizer)

    optimizer = torch.optim.Adam(state_predictor.parameters(), lr=lr)
        
    def _val_epoch():
        with torch.no_grad():
            val_losses = []
            val_ens_mses = []
            val_mses = []
            for batch in val_dataloader:
                multibatch = multi_repeat(batch, times=ensemble_size)
                loss, mse = state_predictor.compute_loss(multibatch, return_mse=True)
                ensemble_mse = state_predictor.compute_ensemble_mse(multibatch)
                val_losses.append(loss.item())
                val_mses.append(mse.mean(dim=0).mean().item())
                val_ens_mses.append(ensemble_mse.item())
            return np.mean(val_losses), np.mean(val_mses), np.mean(val_ens_mses)

    val_loss, val_mse, val_ens_mse = _val_epoch()
    print(f"Initial validation loss: {val_loss:.6f}, [mse: {val_mse:.6f} enseble mse: {val_ens_mse:.6f}]")
    for i_epoch in range(num_epochs):
        losses = []
        state_predictor.train()
        for i, multibatch in enumerate(train_dataloader):
            loss = state_predictor.compute_loss(multibatch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            #print(f"Loss {np.mean(losses)}")
        val_loss, val_mse, val_ens_mse = _val_epoch()
        print(f"Epoch: {i_epoch}, Loss {np.mean(losses):.6f}. Val loss {val_loss:.6f}. Val mse {val_mse:.6f} Val ensemble mse {val_ens_mse:.6f}")
    
    torch.save(state_predictor, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()

