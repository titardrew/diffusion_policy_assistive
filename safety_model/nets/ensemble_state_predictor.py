import torch
import torch.nn as nn
import torch.nn.functional as F

from safety_model.nets.utils import EnsembleFC, Swish, init_weights


class EnsembleFCBackbone(nn.Module):
    def __init__(self, in_size, in_horizon, out_size, out_horizon, ensemble_size, hidden_size):
        super(EnsembleFCBackbone, self).__init__()

        self.in_horizon = in_horizon
        self.in_size = in_size
        self.out_horizon = out_horizon
        self.out_size = out_size

        self.net = nn.Sequential(
            EnsembleFC(in_size * in_horizon, hidden_size, ensemble_size, weight_decay=0.000025),
            Swish(),
            EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005),
            Swish(),
            EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075),
            Swish(),
            EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075),
            Swish(),
            EnsembleFC(hidden_size, out_size * out_horizon, ensemble_size, weight_decay=0.0001),
        )
    
    def forward(self, x):
        """
        x: (ensemble, batch, in_horizon, in_size)
        out: (ensemble, batch, out_horizon, out_size)
        """
        x_flat = x.reshape(*x.shape[:2], -1)  # -> (ensemble, batch, in_horizon * in_size)
        out_flat = self.net(x_flat)
        out = out_flat.reshape(*out_flat.shape[:2], self.out_horizon, self.out_size)
        return out


class Conv1DBackbone(nn.Module):
    def __init__(self, in_size, in_horizon, out_size, out_horizon, hidden_num_channels):
        super(Conv1DBackbone, self).__init__()

        self.in_horizon = in_horizon
        self.in_size = in_size
        self.out_horizon = out_horizon
        self.out_size = out_size

        self.conv_net = nn.Sequential(
            nn.Conv1d(
                in_size,
                in_size,
                kernel_size=3,
                dilation=1,
                padding="same",
                padding_mode="replicate",
            ),
            nn.LayerNorm((in_size, in_horizon)),
            nn.GELU(),
            nn.Conv1d(
                in_size,
                hidden_num_channels,
                kernel_size=3,
                dilation=1,
                padding="same",
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Dropout(),
        )
        in_features = nn.Flatten()(self.conv_net(torch.zeros((1, in_size, in_horizon)))).shape[1]
        print("In features for mlp: ", in_features)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=out_size*out_horizon)
        )
    
    def forward(self, x):
        """
        x: (batch, in_horizon, in_size)
        out: (batch, out_horizon, out_size)
        """
        x_enc = self.conv_net(x.transpose(-2, -1))  # (b, T, s) -> (b, s, T)
        out_flat = self.mlp(x_enc)
        out = out_flat.reshape(out_flat.shape[0], self.out_horizon, self.out_size)
        return out

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock1D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3,
        n_groups=8
    ):
        super().__init__()

        self.block_1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        self.block_2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x in_channels x horizon ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.block_1(x)
        out = self.block_2(out)
        out = out + self.residual_conv(x)
        return out

class EnsembleWrapper(nn.Module):
    def __init__(self, module_fn, ensemble_size):
        super().__init__()
        self.module_list = nn.ModuleList([
            module_fn() for _ in range(ensemble_size)
        ])
        self.ensemble_size = ensemble_size
    
    def forward(self, x):
        # [Ensemble, Batch, Time, Dim] -> [Ensemble, Batch, Time, Dim]
        outs = []
        for i in range(self.ensemble_size):
            out = self.module_list[i](x[i])
            outs.append(out)
        out_ens = torch.stack(outs)  # -> (e, b, T, s)
        return out_ens

import einops

class ResidualConv1DBackbone(nn.Module):
    def __init__(
        self,
        in_size,
        in_horizon,
        out_size,
        out_horizon,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        unet=True,
    ):
        super(ResidualConv1DBackbone, self).__init__()

        self.in_horizon = in_horizon
        self.in_size = in_size
        self.out_horizon = out_horizon
        self.out_size = out_size
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.unet = unet

        all_dims = [in_size] + list(down_dims)
        start_dim = down_dims[0]

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ResidualBlock1D(
                mid_dim, mid_dim,
                kernel_size=kernel_size, n_groups=n_groups,
            ),
            ResidualBlock1D(
                mid_dim, mid_dim,
                kernel_size=kernel_size, n_groups=n_groups,
            ),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ResidualBlock1D(
                    dim_in, dim_out,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                ResidualBlock1D(
                    dim_out, dim_out,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        if unet:
            self.up_modules = nn.ModuleList([])
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                is_last = ind >= (len(in_out) - 1)
                self.up_modules.append(nn.ModuleList([
                    ResidualBlock1D(
                        dim_out * 2,
                        dim_in,
                        kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    ResidualBlock1D(
                        dim_in, dim_in,
                        kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))

            self.final_conv = nn.Sequential(
                Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
                nn.Conv1d(start_dim, out_size, 1),
            )
        else:
            self.final_pooling = nn.Sequential(
                nn.MaxPool1d(kernel_size=5, stride=5),
                nn.Conv1d(down_dims[-1], out_size, 1),
                nn.Flatten(),
            )
            probe_down, _ = self._forward_down(torch.zeros((1, in_size, in_horizon)))
            probe_out = self.final_pooling(probe_down)
            _, in_features = probe_out.shape
            print("In features for final layer: ", in_features)
            self.final_layer = nn.Linear(in_features=in_features, out_features=out_size*out_horizon)
    
    def _forward_down(self, x):
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x)
            x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x)
        return x, h
    
    def forward(self, x):
        """
        x: (batch, in_horizon, in_size)
        out: (batch, out_horizon, out_size)
        """
        x = einops.rearrange(x, 'b t h -> b h t')

        x, h = self._forward_down(x)

        if self.unet:
            for resnet, resnet2, upsample in self.up_modules:
                x = torch.cat((x, h.pop()), dim=1)
                x = resnet(x)
                x = resnet2(x)
                x = upsample(x)
                x = self.final_conv(x)
        else:
            x = self.final_pooling(x)
            x = self.final_layer(x)
            x = x.reshape((-1, self.out_size, self.out_horizon))

        x = einops.rearrange(x, 'b h t -> b t h')
        return x


class TransformerBackbone(nn.Module):
    def __init__(self, in_size, out_size, ensemble_size, hidden_size):
        pass

    def forward(self, x):
        """
        x: (batch, in_size)
        out: (batch, out_size)
        """
        return x


class EnsembleModel(nn.Module):
    def __init__(
        self,
        in_size,
        in_horizon,
        out_size,
        out_horizon,
        ensemble_size,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
        device='cpu',
        backbone_type="mlp",
    ):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.in_horzon = in_horizon
        self.use_decay = use_decay
        self.out_size = out_size
        self.out_horizon = out_horizon
        self.output_dim = out_size * out_horizon
        
        if backbone_type == "mlp":
            # Add variance output
            self.backbone = EnsembleFCBackbone(
                in_size, in_horizon,
                out_size*2, out_horizon,
                ensemble_size, hidden_size)
            self.use_decay = True

        elif backbone_type == "cnn":
            backbone_fn = lambda: Conv1DBackbone(
                in_size, in_horizon,
                out_size*2, out_horizon,
                hidden_num_channels=15,
            )
            backbone_fn_2 = lambda: ResidualConv1DBackbone(
                in_size, in_horizon,
                out_size*2, out_horizon,
                down_dims=[64, 128, 256],
                unet=False,
            )
            self.backbone = EnsembleWrapper(backbone_fn, ensemble_size)
        else:
            raise NotImplementedError(f"Backbone '{backbone_type}' is not supported")

        self.max_logvar = nn.Parameter((torch.ones((1, self.out_horizon, out_size)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.out_horizon, out_size)).float() * 5).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.to(device)

    def forward(self, x, ret_log_var=False):
        # (Ensemble, Batch, T, Dim) -> (Ensemble, Batch, T, Dim)

        output = self.backbone(x)
        mean = output[..., :self.out_size]
        logvar = self.max_logvar - F.softplus(self.max_logvar - output[..., self.out_size:])
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
        mean, logvar: (Ensemble, Batch, T, dim)
        labels: (Ensemble, Batch, T, dim)
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 4
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=(-1, -2)), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=(-1, -2)), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2, 3))
            total_loss = torch.sum(mse_loss)

        total_loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            total_loss += self.get_decay_loss()

        mse_loss = torch.sum(torch.pow(mean - labels, 2), dim=(-1, -2))
        nll = torch.sum(torch.pow(mean - labels, 2) * inv_var, dim=(-1, -2))
        return total_loss, mse_loss, nll
    
    def compute_gate_size(self, mean, logvar, labels, min_sigma=0.1):
        """
        mean, logvar: (Ensemble, Batch, T, dim)
        labels: (Ensemble, Batch, T, dim)

        returns gate_size_in_sigmas
        (Ensemble, Batch, T, dim)
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 4
        sigma = torch.exp(logvar * 0.5)
        gate_size_in_sigmas = torch.abs(labels - mean) / torch.maximum(sigma, min_sigma)
        return gate_size_in_sigmas