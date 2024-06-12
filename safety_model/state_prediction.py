import torch
from safety_model import SafetyModel
from safety_model.utils import multi_repeat
from safety_model.nets.ensemble_state_predictor import EnsembleModel


class EnsembleStatePredictionSM(SafetyModel):
    def __init__(
        self,
        observation_size,
        action_size,
        in_horizon,
        gap_horizon,
        out_obs_horizon,
        horizon,
        predict_delta=False,
        device="cpu",
        ensemble_size=5,
        use_times=False,
    ):
        """
            I/O comprehensive examples

                [] - input
                {} - output

                ----------------------------------------

                in_horizon = 4
                out_obs_horizon = 2
                horizon_gap = 1

                [ s0   s1 ] s2 { s3   s4 }
                [ a0   a1 ] a2   a3   a4

                ----------------------------------------

                in_horizon = 0  => full episode mode
                out_obs_horizon = 2
                horizon_gap = 1

                [ s0 ] s1 { s2   s3 }
                [ a0 ] a1 { a2   a3 }

                [ s0   s1 ] s2 { s3   s4 }
                [ a0   a1 ] a2   a3   a4  

                [ s0   s1   s2 ] s3 { s4   s5 }
                [ a0   a1   a2 ] a3 { a4   a5 }

                ----------------------------------------
        """

        super().__init__()
        self.action_size = action_size
        self.observation_size = observation_size

        assert horizon >= in_horizon + gap_horizon + out_obs_horizon, \
            f"{horizon} < {in_horizon}) + {out_obs_horizon}"

        self.in_horizon = in_horizon
        self.out_obs_horizon = out_obs_horizon
        self.gap_horizon = gap_horizon
        self.horizon = horizon

        self.predict_delta = predict_delta
        self.ensemble_size = ensemble_size
        self.device = device

        # Add time embedding of 4 freqs to input observations.
        self.use_times = use_times
        self.time_size = 4


        time_size = self.time_size if self.use_times else 0
        self.model = EnsembleModel(
            in_size=observation_size + action_size + time_size,
            in_horizon=self.in_horizon if in_horizon > 0 else horizon,
            out_size=observation_size,
            out_horizon=self.out_obs_horizon,
            ensemble_size=self.ensemble_size,
            hidden_size=64,
            use_decay=False,
            device=device,
        )

        self.normalizer = None

        self.params = {
            "metric": "ensemble",

            "mse_threshold": 5.0,

            "nll_threshold": -0.5,

            "std_threshold": 0.1,

            "gate_n_sigmas": 2.0,
            "gate_min_sigma": 0.1,

            "max_variance_threshold": 1.0,
        }
 
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def time_encoding(self, shape, num_freqs: int, starts: torch.Tensor, period: int, device):
        E, B, T = shape
        times = torch.zeros((E, B, T), device=device)
        times[:, :] = torch.arange(0, end=T, step=1.0, device=device)
        times = (times + starts[..., None]) / period
        enc_times = torch.zeros((E, B, T, num_freqs), device=device)
        for i_freq in range(num_freqs):
            enc_times[..., i_freq] = torch.sin(times * torch.pi / period * i_freq**2)
        return enc_times
    
    def preprocess(self, multibatch, no_output=False):
        # HACK(aty): multibatch get's incorrectly broadcasted, thus squeeze (E, 1, B, 1) -> (E, B)
        # While it should be (E, B, 1, 1). FIXME
        input_length = multibatch['length'][..., 0].to(self.device)
        full_episode = self.in_horizon == 0

        E, B, T, D = multibatch['obs'].shape
        if full_episode:
            in_horizon = input_length - self.gap_horizon - self.out_obs_horizon

            input_act = torch.zeros_like(multibatch['action'], device=self.device)
            input_obs = torch.zeros_like(multibatch['obs'], device=self.device)
            input_time = self.time_encoding((E, B, T), self.time_size, 0, T, device=self.device)
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
            input_time = self.time_encoding((E, B, in_horizon), self.time_size, input_length - in_horizon, 200, device=self.device)
        

        if no_output:
            return input_act, input_obs, input_length, input_time
        else:
            if full_episode:
                E, B, T, D = multibatch['obs'].shape
                out_start = (in_horizon + self.gap_horizon).reshape(-1)
                
                # HACK(aty): couldn't find a better way to batch the slicing.
                obs = multibatch["obs"].reshape(E*B, T, D)
                obs_list = [
                    obs[i: i+1, out_start[i]: out_start[i] + self.out_obs_horizon, :]
                    for i in range(E*B)
                ]
                output_obs = torch.cat(obs_list, dim=0).reshape(E, B, self.out_obs_horizon, D)  # cating back to ens/batch

            else:
                out_start = in_horizon + self.gap_horizon
                output_obs = multibatch["obs"][:, :, out_start: out_start + self.out_obs_horizon, :]

            return input_act, input_obs, input_length, input_time, output_obs
    
    def forward(self, multibatch):
        if self.normalizer:
            multibatch['obs'] = self.normalizer['obs'].normalize(multibatch['obs'])
            multibatch['action'] = self.normalizer['action'].normalize(multibatch['action'])

        # (Ensemble, Batch, T, Obs) + (Ensemble, Batch, T, Act) => (Ensemble, Batch, T, ObsAct)
        input_data = [multibatch['obs'], multibatch['action']]
        if self.use_times:
            input_data += [multibatch['time']]
        obs_act = torch.cat(input_data, axis=-1)
        #obs_act = torch.cat([multibatch['obs']], axis=-1)
        
        net_out_mean, net_out_logvar = self.model(obs_act, ret_log_var=True)
        # -> (Ensemble, Batch, T, ObsAct)
        if self.predict_delta:
            latest_obs = multibatch['obs'][..., -1:, :]
            next_obs = net_out_mean + latest_obs
            next_obs_logvar = net_out_logvar
        else:
            next_obs = net_out_mean
            next_obs_logvar = net_out_logvar

        if self.normalizer:
            next_obs = self.normalizer['obs'].unnormalize(next_obs)
        return next_obs, next_obs_logvar

    def compute_loss(self, multibatch):
        input_act, input_obs, input_length, input_time, output_obs = self.preprocess(multibatch)
        pred_means, pred_logvars = self.forward({
            'action': input_act,     # [Ensemble, Batch, Time, ActDim]
            'obs': input_obs,        # [Ensemble, Batch, Time, ObsDim]
            'length': input_length,  # [Ensemble, Batch]
            'time': input_time,      # [Ensemble, Batch, Time, TimeDim]
        })  # -> [Ensemble, Batch, Time, Obs], [Ensemble, Batch, Time, Obs]
        
        loss, mse, nll = self.model.loss(pred_means, pred_logvars, output_obs, inc_var_loss=True)

        return loss, {"mse": mse, "nll": nll}
    
    def compute_validation_loss(self, multibatch):
        with torch.no_grad():
            loss, stats = self.compute_loss(multibatch)
            mse = stats['mse']
            nll = stats['nll']
            ensemble_mse, ensemble_nll = self.compute_ensemble_mse(multibatch)
            return {"total_loss": loss.item(), "nll": nll.mean(dim=0).mean().item(), "mse": mse.mean(dim=0).mean().item(), "ensemble_mse": ensemble_mse.mean(dim=0).item(), "ensemble_nll": ensemble_nll.mean(dim=0).item()}
    
    def compute_ensemble_mse(self, multibatch):
        assert torch.allclose(multibatch['action'].mean(dim=0), multibatch['action'][0]), "Currently multibatch must be just a broadcasted tensor for ensembling."
        input_act, input_obs, input_length, input_time, output_obs = self.preprocess(multibatch)
        pred_means, pred_logvars = self.forward({
            'action': input_act,     # [Ensemble, Batch, Time, ActDim]
            'obs': input_obs,        # [Ensemble, Batch, Time, ObsDim]
            "length": input_length,  # [Ensemble, Batch]
            "time": input_time,      # [Ensemble, Batch, Time, TimeDim]
        })  # -> [Ensemble, Batch, Time, ObsDim], [Ensemble, Batch, Time, ObsDim]
        
        ens_pred_means = pred_means.mean(dim=0)
        ens_pred_logvars = pred_logvars.mean(dim=0)
        error = (ens_pred_means - output_obs[0])**2
        mse = torch.sum(error, dim=(-1, -2))
        nll = torch.sum(error * torch.exp(-ens_pred_logvars), dim=(-1, -2))
        return mse, nll
    
    def check_safety_mse(self, batch, mse_threshold=1.0, rr_log=False):
        with torch.no_grad():
            mse, nll = self.compute_ensemble_mse(batch)
            mse_batch = mse.cpu().numpy()
            is_safe = mse_batch < mse_threshold

            if rr_log:
                import rerun as rr
                rr.log("safety_model/ensemble_state_predictor_mse", rr.Scalar(mse_batch[0]))
                rr.log("safety_model/ensemble_state_predictor_mse_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'mse': mse_batch} 

    def check_safety_nll(self, batch, nll_threshold=1.0, rr_log=False):
        with torch.no_grad():
            mse, nll = self.compute_ensemble_mse(batch)
            nll_batch = nll.cpu().numpy()
            is_safe = nll_batch < nll_threshold

            if rr_log:
                import rerun as rr
                rr.log("safety_model/ensemble_state_predictor_nll", rr.Scalar(nll_batch[0]))
                rr.log("safety_model/ensemble_state_predictor_mse_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'nll': nll_batch} 

    def check_safety_ensemble(self, batch, std_threshold=1.0, rr_log=False):

        with torch.no_grad():
            input_act, input_obs, input_length, input_time = self.preprocess(batch, no_output=True)
            pred_means, pred_logvars = self.forward({'action': input_act, 'obs': input_obs, 'length': input_length, "time": input_time})
            # TODO(aty): maybe check pred_logvars? We can detect invalid ensemble members using it (e.g. it's too large).
            # [Ensemble, Batch, T, Dim] -> [Batch, T, Dim]
            mean_std = torch.std(pred_means, dim=0).cpu().mean(dim=(-1, -2)).numpy()
            is_safe = mean_std < std_threshold
            if rr_log:
                import rerun as rr
                rr.log("safety_model/ensemble_state_predictor_std", rr.Scalar(mean_std[0]))
                rr.log("safety_model/ensemble_state_predictor_std_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'std': mean_std} 
    
    def check_safety_max_variance(self, batch, max_variance_threshold=1.0, rr_log=False):

        with torch.no_grad():
            input_act, input_obs, input_length, input_time = self.preprocess(batch, no_output=True)
            pred_means, pred_logvars = self.forward({'action': input_act, 'obs': input_obs, 'length': input_length, 'time': input_time})
            # [Ensemble, Batch, T, Dim] -> [Ensemble, Batch, T]
            variance_norms = torch.norm(pred_logvars.exp(), dim=-1)
            # -> [Ensemble, Batch]
            variance_norm_max_per_time = variance_norms.max(dim=-1)[0]
            # -> [Batch]
            max_variance = variance_norm_max_per_time.max(dim=0)[0].cpu().numpy()
            is_safe = max_variance < max_variance_threshold
            if rr_log:
                import rerun as rr
                rr.log("safety_model/ensemble_state_predictor_maxvar", rr.Scalar(max_variance[0]))
                rr.log("safety_model/ensemble_state_predictor_maxvar_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'max_variance': max_variance} 

    def check_safety_gating(self, multibatch, num_sigmas=1.0, min_sigma=0.1, rr_log=False):
        with torch.no_grad():
            input_act, input_obs, input_length, input_time, output_obs = self.preprocess(multibatch)
            pred_means, pred_logvars = self.forward({
                'action': input_act,    # (Ensemble, Batch, T, Act)
                'obs': input_obs,       # (Ensemble, Batch, T, Obs)
                'length': input_length, # (Ensemble, Batch)
                'time': input_time,
            })  # -> (Ensemble, Batch, T, Dim), (Ensemble, Batch, T, Dim)
            
            gate_n_sigmas_batch = self.model.compute_gate_size(pred_means, pred_logvars, output_obs, min_sigma=min_sigma)
            # (Ensemble, Batch, T, Dim) -> (Batch, T, Dim) -> (Batch,)
            gate_n_sigmas_batch = gate_n_sigmas_batch.mean(dim=0).cpu().numpy().max(axis=(-1, -2))
            is_safe = gate_n_sigmas_batch < num_sigmas

            if rr_log:
                import rerun as rr
                rr.log("safety_model/ensemble_state_predictor_gating", rr.Scalar(gate_n_sigmas_batch[0]))
                rr.log("safety_model/ensemble_state_predictor_gating_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'gate_n_sigmas': gate_n_sigmas_batch} 
    
    def reset(self):
        pass
    
    def override_params(self, params):
        if params is not None:
            self.params.update(params)
    
    def check_safety_runner(self, batch, rr_log=False, return_stats=False, return_score=False):
        with torch.no_grad():
            multibatch = multi_repeat({
                'obs': batch['obs'],
                'action': batch['action'],
                'length': batch['length'],
            }, self.ensemble_size)

            metric = self.params['metric']

            if metric == "mse":
                mse_threshold = self.params['mse_threshold']
                is_safe, stats = self.check_safety_mse(multibatch, mse_threshold, rr_log)
                score = stats['mse']

            if metric == "nll":
                nll_threshold = self.params['nll_threshold']
                is_safe, stats = self.check_safety_nll(multibatch, nll_threshold, rr_log)
                score = stats['nll']

            elif metric == "ensemble":
                std_threshold = self.params['std_threshold']
                is_safe, stats = self.check_safety_ensemble(multibatch, std_threshold, rr_log)
                score = stats['std']

            elif metric == "max_variance":
                max_variance_threshold = self.params['max_variance_threshold']
                is_safe, stats = self.check_safety_max_variance(multibatch, max_variance_threshold, rr_log)
                score = stats['max_variance']

            elif metric == "gate":
                gating_n_sigmas = self.params['gating_n_sigmas']
                gating_min_sigma = self.params['gating_min_sigma']
                is_safe, stats = self.check_safety_gating(multibatch, gating_n_sigmas, gating_min_sigma, rr_log)
                score = stats['gate_n_sigmas']
            else:
                raise NotImplementedError(f"Wrong metric {metric}")
            
            if return_stats or return_score:
                return (
                    is_safe,
                    *((stats,) if return_stats else ()),
                    *((score,) if return_score else ())
                )
            else:
                return is_safe

