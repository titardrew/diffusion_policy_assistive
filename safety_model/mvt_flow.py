import numpy as np
import torch

from voraus_ad_dataset.configuration import Configuration
from voraus_ad_dataset.normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample

from safety_model import SafetyModel
from safety_model.utils import multi_repeat


class MVTFlowSM(SafetyModel):
    def __init__(
        self,
        observation_size,
        action_size,
        max_horizon,
        device="cpu",
    ):
        super().__init__()
        self.action_size = action_size
        self.observation_size = observation_size

        self.in_horizon = 0
        self.max_horizon = max_horizon

        self.ensemble_size = 1  # HACK
        self.device = device

        configuration = Configuration(
            columns="machine",
            epochs=70,
            frequencyDivider=1,
            trainGain=1.0,
            seed=177,
            batchsize=32,
            nCouplingBlocks=4,
            clamp=1.2,
            learningRate=8e-4,
            normalize=False,
            pad=False,
            nHiddenLayers=0,
            scale=2,
            kernelSize1=13,
            dilation1=2,
            kernelSize2=1,
            dilation2=1,
            kernelSize3=1,
            dilation3=1,
            milestones=[11, 61],
            gamma=0.1,
        )
        n_signals = observation_size + action_size
        n_times = max_horizon
        # Initialize the model, optimizer and scheduler.
        self.net = NormalizingFlow((n_signals, n_times), configuration).float().to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
        #scheduler = optim.lr_scheduler.MultiStepLR(
        #    optimizer, milestones=configuration.milestones, gamma=configuration.gamma
        #)

        self.normalizer = None

        self.params = {
            "metric": "loss",

            "loss_threshold": 1.0,
        }
 
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def forward(self, multibatch):
        if self.normalizer:
            multibatch['obs'] = self.normalizer['obs'].normalize(multibatch['obs'])
            multibatch['action'] = self.normalizer['action'].normalize(multibatch['action'])

        # [Batch, Time, Obs] + [Batch, Time, Act] => [Batch, Time, ObsAct]
        obs_act = torch.cat([multibatch['obs'], multibatch['action']], axis=-1)
        #obs_act = torch.cat([multibatch['obs']], axis=-1)
        
        latent_z, jacobian = self.net(obs_act.transpose(2, 1))
        jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))

        return latent_z, jacobian
    
    def preprocess(self, multibatch):
        batch = {k: v[0] for k, v in multibatch.items()}  # HACK: no ensembling here
        input_length = batch['length'][..., 0].to(self.device)
        full_episode = self.in_horizon == 0
        assert full_episode

        B, T, D = batch['obs'].shape
        in_horizon = input_length

        input_act = torch.zeros_like(batch['action'], device=self.device)
        input_obs = torch.zeros_like(batch['obs'], device=self.device)
        # input_time = self.time_encoding((E, B, T), self.time_size, 0, T, device=self.device)
        # HACK(aty): couldn't find a better way to batch the slicing.
        # zeroing out everything after the input horizon.
        for i_seq, i_seq_horizon in enumerate(in_horizon):
            input_act[i_seq, :i_seq_horizon] = batch['action'][i_seq, :i_seq_horizon]
            input_obs[i_seq, :i_seq_horizon] = batch['obs'][i_seq, :i_seq_horizon]

        return input_act, input_obs, input_length
    

    def compute_loss(self, multibatch, per_sample=False):
        input_act, input_obs, input_length = self.preprocess(multibatch)

        obs_act = {
            'action': input_act,     # [Batch, Time, ActDim]
            'obs': input_obs,        # [Batch, Time, ObsDim]
            'length': input_length,  # [Batch]
        }

        latent_z, jacobian = self.forward(obs_act)
        if per_sample:
            loss = get_loss_per_sample(latent_z, jacobian)
        else:
            loss = get_loss(latent_z, jacobian)
        losses_info = {"loss": loss}
        return loss, losses_info
    
    def compute_validation_loss(self, multibatch):
        with torch.no_grad():
            _, loss_dict = self.compute_loss(multibatch)

            for key, val in loss_dict.items():
                loss_dict[key] = val.cpu().numpy()

            return loss_dict

    def check_safety(self, batch, loss_thresh=1.0, rr_log=False):
        with torch.no_grad():

            _, losses_info = self.compute_loss(batch, per_sample=True)
            loss_batch = losses_info["loss"].cpu().numpy()
            is_safe = loss_batch < loss_thresh
            if rr_log:
                import rerun as rr
                rr.log("safety_model/mvt_flow_loss", rr.Scalar(loss_batch[0]))
                rr.log("safety_model/mvt_flow_safe", rr.Scalar(float(is_safe[0])))

            return is_safe, {'loss': loss_batch} 
    
    def reset(self):
        pass
    
    def override_params(self, params):
        if params is not None:
            self.params.update(params)
    
    def check_safety_runner(self, batch, rr_log=False, return_stats=False, return_score=False):
        with torch.no_grad():
            multibatch = multi_repeat(batch, self.ensemble_size)

            metric = self.params['metric']

            if metric == "loss":
                loss_thresh = self.params['loss_threshold']
                is_safe, stats = self.check_safety(
                    multibatch,
                    loss_thresh=loss_thresh,
                    rr_log=rr_log)
                score = stats['loss']
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
