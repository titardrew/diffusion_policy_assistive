import torch.nn as nn

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

    def compute_validation_loss(self, batch):
        raise NotImplementedError()

    def override_params(self, params):
        raise NotImplementedError()
