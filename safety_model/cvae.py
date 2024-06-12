import numpy as np
import torch
from torch import nn 
import torch.nn.functional as F

from safety_model.detr.models.detr_vae import DETRVAE, TransformerEncoder, TransformerEncoderLayer
from safety_model.detr.models.transformer import Transformer
from safety_model import SafetyModel
from safety_model.nets.ensemble_vae import EnsembleVAE
from safety_model.utils import multi_repeat

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ConditionalVariationalAutoEncoderSM(SafetyModel):
    def __init__(
        self,
        observation_size,
        action_size,
        in_horizon,
        gap_horizon,
        out_obs_horizon,
        max_horizon,
        device="cpu",
        kl_weight=5,
        use_times=False,
    ):
        super().__init__()
        self.action_size = action_size
        self.observation_size = observation_size

        self.in_horizon = in_horizon
        self.gap_horizon = gap_horizon
        self.out_obs_horizon = out_obs_horizon
        self.max_horizon = max_horizon
        self.kl_weight = kl_weight
        self.ensemble_size = 1  # HACK
        self.use_times = use_times

        def build_encoder(hidden_dim, dropout, dim_feedforward, normalize_before, nheads, num_encoder_layers):
            d_model = hidden_dim
            activation = "relu"
            encoder_layer = TransformerEncoderLayer(d_model, nheads, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            return TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._encoder = build_encoder(
            hidden_dim=256,  # 512
            dropout=0.1,
            dim_feedforward=2048,  # 3200
            normalize_before=False,
            nheads=8,
            num_encoder_layers=4,
        )

        self._transformer = Transformer(
            d_model=256,
            dropout=0.1,
            nhead=8,
            dim_feedforward=2048,
            num_encoder_layers=4,
            num_decoder_layers=7,
            normalize_before=False,
            return_intermediate_dec=True,
        )

        self.net = DETRVAE(
            backbones=None,
            transformer=self._transformer,
            encoder=self._encoder,
            state_dim=observation_size,
            in_horizon=in_horizon if in_horizon > 0 else max_horizon,
            out_horizon=out_obs_horizon,
            camera_names=[],
            vq=None, vq_class=None, vq_dim=None,
            action_dim=action_size,
        )

        self.normalizer = None

        self.params = {
            "metric": "recon",
            "reconstruction_threshold": 4.0,
        }
        self.device = device
        self.to(device)
 
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def forward(self, input_obs, input_act, input_is_pad, input_lengths, output_obs, output_is_pad, normalize=True):
        if normalize and self.normalizer:
            input_obs = self.normalizer['obs'].normalize(input_obs)
            input_act = self.normalizer['action'].normalize(input_act)
            output_obs = self.normalizer['obs'].normalize(output_obs)

        input_times = (input_lengths.unsqueeze(1) - self.gap_horizon - self.out_obs_horizon) / 200
        output_times = (input_lengths.unsqueeze(1)) / 200  # HARDCODED
        if not self.use_times:
            input_times *= 0.0
            output_times *= 0.0

        # [Batch, Time, Obs](out), [Batch, Time, Obs](in), [Batch, Time, Act](in) => [Batch, Time, Obs](out)

        # Interleave masks for states and actions m_s, m_a, m_s, m_a...
        stacked = torch.stack([input_is_pad, input_is_pad], dim=2)
        input_is_pad = torch.flatten(stacked, start_dim=1, end_dim=2)

        output_obs_recon, is_pad_recon, [z_mu, z_logvar], _, _ = self.net(input_obs, input_act, input_is_pad, in_times=input_times, out_states=output_obs, out_is_pad=output_is_pad, out_times=output_times)

        if normalize and self.normalizer:
            output_obs_recon = self.normalizer['obs'].unnormalize(output_obs_recon)

        return output_obs_recon, is_pad_recon, z_mu, z_logvar
    
    def preprocess(self, multibatch, no_output=False):
        batch = {k: v[0] for k, v in multibatch.items()}  # HACK: no ensembling here
        input_length = batch['length'][:, 0].to(self.device)
        full_episode = self.in_horizon == 0

        B, T, D = batch['obs'].shape
        if full_episode:
            in_horizon = input_length - self.gap_horizon - self.out_obs_horizon

            input_act = torch.zeros_like(batch['action'], device=self.device)
            input_obs = torch.zeros_like(batch['obs'], device=self.device)
            # input_time = self.time_encoding((B, T), self.time_size, 0, T, device=self.device)
            input_is_pad = torch.full((B, self.max_horizon), fill_value=False, dtype=torch.bool, device=self.device)
            # HACK(aty): couldn't find a better way to batch the slicing.
            # zeroing out everything after the input horizon.
            for i_seq, i_seq_horizon in enumerate(in_horizon):
                #input_act[i_seq, :i_seq_horizon] = batch['action'][i_seq, :i_seq_horizon]
                #input_obs[i_seq, :i_seq_horizon] = batch['obs'][i_seq, :i_seq_horizon]
                input_is_pad[i_seq, i_seq_horizon:] = True  # everything else is padding
        else:
            in_horizon = self.in_horizon
            input_act = batch['action'][:, :in_horizon, :]
            input_obs = batch['obs'][:, :in_horizon, :]
            # input_time = self.time_encoding((B, in_horizon), self.time_size, input_length - in_horizon, 200, device=self.device)
            input_is_pad = torch.full((B, in_horizon), fill_value=False, dtype=torch.bool, device=self.device) 

        if no_output:
            return input_act, input_obs, input_length, input_is_pad
        else:
            if full_episode:
                B, T, D = batch['obs'].shape
                out_start = (in_horizon + self.gap_horizon).reshape(-1)
                
                # HACK(aty): couldn't find a better way to batch the slicing.
                obs = batch["obs"]
                obs_list = [
                    obs[i: i+1, out_start[i]: out_start[i] + self.out_obs_horizon, :]
                    for i in range(B)
                ]
                output_obs = torch.cat(obs_list, dim=0)
                assert output_obs.shape == (B, self.out_obs_horizon, D), (output_obs, (B, self.out_obs_horizon, D))
                output_is_pad = torch.full((B, self.out_obs_horizon), fill_value=False, dtype=torch.bool, device=self.device)

            else:
                out_start = in_horizon + self.gap_horizon
                output_obs = batch["obs"][:, out_start: out_start + self.out_obs_horizon, :]
                output_is_pad = torch.full((B, self.out_obs_horizon), fill_value=False, dtype=torch.bool, device=self.device)

            return input_act, input_obs, input_length, input_is_pad, output_obs, output_is_pad

    def compute_loss(self, multibatch):
        input_act, input_obs, input_length, input_is_pad, output_obs, output_is_pad = self.preprocess(multibatch)

        if self.normalizer:
            input_obs = self.normalizer['obs'].normalize(input_obs)
            input_act = self.normalizer['action'].normalize(input_act)
            output_obs = self.normalizer['obs'].normalize(output_obs)

        output_obs_recon, output_is_pad_recon, z_mu, z_logvar = self.forward(input_obs, input_act, input_is_pad, input_length, output_obs, output_is_pad, normalize=False)

        output_obs_recon = output_obs_recon[:, :self.out_obs_horizon]
        output_is_pad_recon = output_is_pad_recon[:, :self.out_obs_horizon]

        losses_info = dict()
        if self.net.encoder is None:
            total_kld = [torch.tensor(0.0)]
        else:
            total_kld, dim_wise_kld, mean_kld = kl_divergence(z_mu, z_logvar)

        all_l1 = F.l1_loss(output_obs, output_obs_recon, reduction='none')
        l1 = (all_l1 * ~output_is_pad.unsqueeze(-1)).mean()
        losses_info['l1'] = l1
        losses_info['kl'] = total_kld[0]
        losses_info['loss'] = losses_info['l1'] + losses_info['kl'] * self.kl_weight
        loss = losses_info['loss']

        return loss, losses_info
    
    def compute_validation_loss(self, multibatch):
        with torch.no_grad():
            _, loss_dict = self.compute_loss(multibatch)

            for key, val in loss_dict.items():
                loss_dict[key] = val.cpu().numpy()

            return loss_dict

    def check_safety_recon(self, batch, recon_thresh=1.0, rr_log=False):

        with torch.no_grad():
            _, losses_info = self.compute_loss(batch)
            recon_loss_batch = losses_info["l1"].cpu().numpy()
            is_safe = recon_loss_batch < recon_thresh
            if rr_log:
                import rerun as rr
                rr.log("safety_model/cvae_recon", rr.Scalar(recon_loss_batch[0]))

            return is_safe, {'recon': recon_loss_batch} 
    
    def reset(self):
        pass
    
    def override_params(self, params):
        if params is not None:
            self.params.update(params)
    
    def check_safety_runner(self, batch, rr_log=False, return_stats=False, return_score=False):
        with torch.no_grad():
            multibatch = multi_repeat(batch, 1)

            metric = self.params['metric']

            if metric == "recon":
                recon_thresh = self.params['reconstruction_threshold']
                is_safe, stats = self.check_safety_recon(
                    multibatch,
                    recon_thresh=recon_thresh,
                    rr_log=rr_log)
                score = stats['recon']
            else:
                raise NotImplementedError()

            if return_stats or return_score:
                return (
                    is_safe,
                    *((stats,) if return_stats else ()),
                    *((score,) if return_score else ())
                )
            else:
                return is_safe
