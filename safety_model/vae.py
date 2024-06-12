import numpy as np
import torch

from safety_model import SafetyModel
from safety_model.nets.ensemble_vae import EnsembleVAE
from safety_model.utils import multi_repeat


def mean_pairwise_kl_divergence(mu_tensor, logvar_tensor):
    def gaussian_kl_divergence(mu1, logvar1, mu2, logvar2):
        kl = 0.5*(logvar2 - logvar1 + (logvar1.exp() + (mu1 - mu2)**2) / logvar2.exp() - 1.0)
        return kl.sum(dim=-1)

    N = mu_tensor.shape[0]
    total_kl_divergence = 0
    num_pairs = 0
    
    for i in range(N):
        for j in range(i+1, N):
            mu1, logvar1 = mu_tensor[i], logvar_tensor[i]
            mu2, logvar2 = mu_tensor[j], logvar_tensor[j]
            kl_div_ij = gaussian_kl_divergence(mu1, logvar1, mu2, logvar2)
            total_kl_divergence += kl_div_ij
            num_pairs += 1
    
    return total_kl_divergence / num_pairs


class VariationalAutoEncoderSM(SafetyModel):
    def __init__(
        self,
        observation_size,
        action_size,
        in_horizon,
        horizon,
        device="cpu",
        ensemble_size=5,
        hidden_size=200,
        embedding_size=32,
    ):
        """
            I/O comprehensive example:
                in_horizon = 4

                [ s0   s1   s2  s3 ]
                [ a0   a1   a2  a3 ]

                [] - input/output
        """

        super().__init__()
        self.action_size = action_size
        self.observation_size = observation_size

        self.in_horizon = in_horizon
        self.horizon = horizon

        self.ensemble_size = ensemble_size
        self.device = device
        self.forbid_mse = False
        self.embedding_size = embedding_size

        in_size = observation_size + action_size
        self.net = EnsembleVAE(
            in_size=in_size,
            in_horizon=horizon,
            embedding_size=self.embedding_size,
            ensemble_size=self.ensemble_size,
            hidden_size=hidden_size,
            use_decay=True,
            device=device,
        )

        self.normalizer = None

        self.params = {
            "metric": "pw_kl",

            "reconstruction_threshold": 4.0,
            "pw_kl_threshold": 0.005,
        }
 
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def forward(self, multibatch, ensemble_sampling=False, normalize=True):
        if normalize and self.normalizer:
            multibatch['obs'] = self.normalizer['obs'].normalize(multibatch['obs'])
            multibatch['action'] = self.normalizer['action'].normalize(multibatch['action'])

        # [Ensemble, Batch, Time, Obs] + [Ensemble, Batch, Time, Act] => [Ensemble, Batch, Time, ObsAct]
        obs_act = torch.cat([multibatch['obs'], multibatch['action']], axis=-1)
        #obs_act = torch.cat([multibatch['obs']], axis=-1)
        
        obs_act_recon, z_mu, z_logvar, z, z_all = self.net(obs_act)#, ensemble_sampling=ensemble_sampling)

        if normalize and self.normalizer:
            obs_recon = self.normalizer['obs'].unnormalize(obs_act_recon[..., :self.observation_size])
            act_recon = self.normalizer['action'].unnormalize(obs_act_recon[..., self.observation_size:])
            obs_act_recon = torch.cat([obs_recon, act_recon], axis=-1)

        return obs_act_recon, z_mu, z_logvar, z, z_all
    
    def preprocess(self, multibatch):
        # HACK(aty): multibatch get's incorrectly broadcasted, thus squeeze (E, 1, B, 1) -> (E, B)
        # While it should be (E, B, 1, 1). FIXME
        input_length = multibatch['length'][..., 0].to(self.device)
        full_episode = self.in_horizon == 0

        E, B, T, D = multibatch['obs'].shape
        if full_episode:
            in_horizon = input_length

            input_act = torch.zeros_like(multibatch['action'], device=self.device)
            input_obs = torch.zeros_like(multibatch['obs'], device=self.device)
            # input_time = self.time_encoding((E, B, T), self.time_size, 0, T, device=self.device)
            # HACK(aty): couldn't find a better way to batch the slicing.
            # zeroing out everything after the input horizon.
            for i_ens, i_ens_horizons in enumerate(in_horizon):
                for i_seq, i_seq_horizon in enumerate(i_ens_horizons):
                    input_act[i_ens, i_seq, :i_seq_horizon] = multibatch['action'][i_ens, i_seq, :i_seq_horizon]
                    input_obs[i_ens, i_seq, :i_seq_horizon] = multibatch['obs'][i_ens, i_seq, :i_seq_horizon]
        else:
            in_horizon = self.in_horizon
            input_act = multibatch['action'][..., :in_horizon, :]
            input_obs = multibatch['obs'][..., :in_horizon, :]
            # input_time = self.time_encoding((E, B, in_horizon), self.time_size, input_length - in_horizon, 200, device=self.device)
        
        return input_act, input_obs, input_length  #, input_time
    

    def compute_loss(self, multibatch, inference=False):
        input_act, input_obs, input_length = self.preprocess(multibatch)

        if self.normalizer:
            input_obs = self.normalizer['obs'].normalize(input_obs)
            input_act = self.normalizer['action'].normalize(input_act)

        obs_act = {
            'action': input_act,     # [Ensemble, Batch, Time, ActDim]
            'obs': input_obs,        # [Ensemble, Batch, Time, ObsDim]
            'length': input_length,  # [Ensemble, Batch]
        }

        obs_act_recon, z_mu, z_logvar, z, z_all = self.forward(obs_act, ensemble_sampling=inference, normalize=False)
        obs_act = torch.cat([obs_act['obs'], obs_act['action']], axis=-1)
        loss, losses_info = self.net.loss(obs_act, obs_act_recon, z_mu, z_logvar, keep_batch=inference)

        # [Ensemble_size, Batch, EmbedDim] -> [Batch]
        z_std = z.std(dim=0).mean(dim=-1)
        pw_kl = mean_pairwise_kl_divergence(z_mu, z_logvar)
        if not inference:
            z_std = torch.sum(z_std, dim=0)
            pw_kl = torch.sum(pw_kl, dim=0)

        losses_info['metric_z_std'] = z_std
        losses_info['metric_pw_kl'] = pw_kl

        return loss, losses_info
    
    def compute_validation_loss(self, multibatch):
        with torch.no_grad():
            _, loss_dict = self.compute_loss(multibatch)
            # [Ensemble x Batch x Dims]
            _, ens_loss_dict = self.compute_loss(multibatch, inference=True)

            for key, val in ens_loss_dict.items():
                loss_dict[f"ens_{key}"] = val.mean()

            for key, val in loss_dict.items():
                loss_dict[key] = val.cpu().numpy()
             
            return loss_dict

    def check_safety_ensemble(self, batch, use_recon=False, recon_thresh=1.0, use_pw_kl=False, pw_kl_thresh=0.0005, rr_log=False):
        assert use_recon or use_pw_kl

        with torch.no_grad():
            #input_act, input_obs, input_length = self.preprocess(batch)
            #multibatch = {'action': input_act, 'obs': input_obs, 'length': input_length}

            _, losses_info = self.compute_loss(batch, inference=True)
            recon_loss_batch = losses_info["recon_loss"].cpu().numpy()
            z_std_batch = losses_info["metric_z_std"].cpu().numpy()
            pw_kl_batch = losses_info["metric_pw_kl"].cpu().numpy()
            is_safe = np.ones_like(recon_loss_batch, dtype=np.bool_)
            if use_recon:
                is_safe &= recon_loss_batch < recon_thresh
            if use_pw_kl:
                is_safe &= pw_kl_batch < pw_kl_thresh
            if rr_log:
                import rerun as rr
                rr.log("safety_model/ensemble_vae_recon", rr.Scalar(recon_loss_batch[0]))
                rr.log("safety_model/ensemble_vae_z_std", rr.Scalar(z_std_batch[0]))
                rr.log("safety_model/ensemble_vae_pw_kl", rr.Scalar(pw_kl_batch[0]))
                rr.log("safety_model/ensemble_vae_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'recon': recon_loss_batch, 'pwkl': pw_kl_batch} 
    
    def reset(self):
        pass
    
    def override_params(self, params):
        if params is not None:
            self.params.update(params)
    
    def check_safety_runner(self, batch, rr_log=False, return_stats=False, return_score=False):
        with torch.no_grad():
            multibatch = multi_repeat(batch, self.ensemble_size)

            metric = self.params['metric']

            if metric == "recon":
                recon_thresh = self.params['reconstruction_threshold']
                is_safe, stats = self.check_safety_ensemble(
                    multibatch,
                    use_recon=True,
                    recon_thresh=recon_thresh,
                    rr_log=rr_log)
                score = stats['recon']
            elif metric == "pw_kl":
                pw_kl_thresh = self.params['pw_kl_threshold']
                is_safe, stats = self.check_safety_ensemble(
                    multibatch,
                    use_pw_kl=True,
                    pw_kl_thresh=pw_kl_thresh,
                    rr_log=rr_log)
                score = stats['pwkl']
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
