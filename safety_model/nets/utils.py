import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std).to(t.device)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape, device=t.device), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True, shared_weights: bool = False) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.shared_weights = shared_weights
        self.ens_dim_size = 1 if shared_weights else ensemble_size
        self.weight_decay = weight_decay
        self.weight = nn.Parameter(torch.Tensor(self.ens_dim_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.ens_dim_size, out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        if hasattr(self, "shared_weights") and self.shared_weights:
            weight = weight.repeat([self.ensemble_size, 1, 1])
            if bias is not None:
                bias = bias.repeat([self.ensemble_size, 1])

        w_times_x = torch.bmm(input, weight)
        return torch.add(w_times_x, bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )