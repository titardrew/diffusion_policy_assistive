import einops
import torch
import torch.nn as nn

from safety_model.nets.utils import EnsembleFC, Swish, init_weights

def reparam(mean, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std).to(std.device)
    return mean + std*epsilon

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

class ResidualConv1DBackbone(nn.Module):
    def __init__(
        self,
        in_size,
        in_horizon,
        down_dims=[256, 32],
        kernel_size=3,
        n_groups=8,
    ):
        super(ResidualConv1DBackbone, self).__init__()

        self.in_size = in_size
        self.in_horizon = in_horizon
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups

        all_dims = [in_size] + list(down_dims)
        start_dim = down_dims[0]

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        in_features = in_horizon // (len(down_dims) - 1)**2 * mid_dim
        self.mid_shape = (mid_dim, in_horizon // (len(down_dims) - 1)**2)
        self.mid_modules = nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=mid_dim),
            nn.Linear(in_features=in_features, out_features=mid_dim),
            nn.Linear(in_features=mid_dim, out_features=in_features),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ResidualBlock1D(
                    dim_in, dim_out,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                nn.Dropout(p=0.1),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ResidualBlock1D(
                    dim_out,
                    dim_in,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                nn.Dropout(p=0.1),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, in_size, 1),
        )
    
    def encode(self, x):
        """
        x: (batch, in_horizon, in_size)
        z_mu, z_logvar: (batch, Tz, z_dim)
        """
        x = einops.rearrange(x, 'b t h -> b h t')

        for resnet, drop, downsample in self.down_modules:
            x = resnet(x)
            x = drop(x)
            x = downsample(x)

        x = torch.flatten(x, start_dim=1)
        z_mu = self.mid_modules[0](x)
        z_logvar = self.mid_modules[1](x)
        return z_mu, z_logvar
    
    def decode(self, z):
        """
        z: (batch, in_horizon*in_size)
        """
        x = self.mid_modules[2](z)
        x = x.reshape(-1, *self.mid_shape)
        for resnet, drop, upsample in self.up_modules:
            x = resnet(x)
            x = drop(x)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b h t -> b t h')
        return x


class EnsembleWrapper(nn.Module):
    def __init__(self, module_fn, ensemble_size):
        super().__init__()
        self.module_list = nn.ModuleList([
            module_fn() for _ in range(ensemble_size)
        ])
        self.ensemble_size = ensemble_size
    
    def encode(self, x):
        # [Ensemble, Batch, Time, Dim] -> [Ensemble, Batch, H, Z]
        z_mus, z_logvars = [], []
        for i in range(self.ensemble_size):
            z_mu, z_logvar = self.module_list[i].encode(x[i])
            z_mus.append(z_mu)
            z_logvars.append(z_logvar)
        z_mu_ens = torch.stack(z_mus)  # -> (e, b, H, Z)
        z_logvar_ens = torch.stack(z_logvars)  # -> (e, b, H, Z)
        return z_mu_ens, z_logvar_ens

    def decode(self, z):
        # [Ensemble, Batch, H, Z] -> [Ensemble, Batch, Time, Dim]
        outs = []
        for i in range(self.ensemble_size):
            out = self.module_list[i].decode(z[i])
            outs.append(out)
        out_ens = torch.stack(outs)  # -> (e, b, T, s)
        return out_ens   


class EnsembleFCBackbone(nn.Module):
    def __init__(self, in_size, in_horizon, embedding_size, ensemble_size, hidden_size):
        super(EnsembleFCBackbone, self).__init__()
        self.encoder = nn.Sequential(
            EnsembleFC(in_size * in_horizon, hidden_size, ensemble_size, shared_weights=False),
            Swish(),
            EnsembleFC(hidden_size, hidden_size, ensemble_size, shared_weights=False),
            Swish(),
            EnsembleFC(hidden_size, embedding_size * 2, ensemble_size, shared_weights=False),
        )
        self.decoder = nn.Sequential(
            EnsembleFC(embedding_size, hidden_size, ensemble_size, shared_weights=False),
            Swish(),
            EnsembleFC(hidden_size, hidden_size, ensemble_size, shared_weights=False),
            Swish(),
            EnsembleFC(hidden_size, in_size * in_horizon, ensemble_size, shared_weights=False),
        )
        self.in_size = in_size
        self.in_horizon = in_horizon
        self.embedding_size = embedding_size
        self.ensemble_size = ensemble_size
        self.hidden_size = hidden_size
    
    def encode(self, x):
        """
        x: (ensemble, batch, in_horizon, in_size)
        mean, logvar: (ensemble, batch, embedding_size)
        """
        x = x.reshape(*x.shape[:2], -1)  # -> (ensemble, batch, in_size * in_horizon)
        x_enc = self.encoder(x)
        mean, logvar = x_enc[:, :, :self.embedding_size], x_enc[:, :, self.embedding_size:]
        return mean, logvar
    
    def decode(self, z):
        """
        z: (ensemble, batch, embedding_size)
        out: (ensemble, batch, in_horizon, in_size)
        """
        x_dec = self.decoder(z)
        x_dec = x_dec.reshape(*x_dec.shape[:2], self.in_horizon, self.in_size)
        return x_dec


class EnsembleVAE(nn.Module):
    def __init__(self, in_size, in_horizon, embedding_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False, device='cpu'):
        super(EnsembleVAE, self).__init__()
        self.in_size = in_size
        self.in_horizon = in_horizon
        self.hidden_size = hidden_size
        self.kl_weight = 1.0
        self.device = device
        self.embedding_size = embedding_size
        self.ensemble_size = ensemble_size

        self.backbone = EnsembleFCBackbone(in_size, in_horizon, embedding_size, ensemble_size, hidden_size).to(device)
        backbone_fn = lambda: ResidualConv1DBackbone(in_size, in_horizon, down_dims=[128, 64, 32], kernel_size=3, n_groups=4)
        self.backbone = EnsembleWrapper(backbone_fn, ensemble_size)

        self.use_decay = use_decay

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
    
    def forward(self, x, ensemble_sampling=False):
        mean, logvar = self.backbone.encode(x)
        z = reparam(mean, logvar)
        z_all = z

        if self.ensemble_size > 1 and ensemble_sampling:
            z[:] = z.mean(dim=0, keepdim=True)

        x_recon = self.backbone.decode(z)
        return x_recon, mean, logvar, z, z_all

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss
    
    def loss(self, x, x_recon, mean, logvar, keep_batch=False):
        """
        x, x_recon: Ensemble x Batch x Time x Dim
        mean, logvar: Ensemble x Batch x ?Time x Dim
        """
        if len(mean.shape) == 3:
            reduce_dims = (-1,)
        elif len(mean.shape) == 4:
            reduce_dims = (-1, -2)
        else:
            raise NotImplementedError()
        # recon == mse for cont spaces
        if keep_batch:
            # -> Ensemble_size x Batch -> Batch
            recon_loss = torch.mean(torch.pow(x_recon - x, 2), dim=(-1, -2)).sum(dim=0)
            kl_loss = torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()), dim=reduce_dims).sum(dim=0)
        else:
            # -> Ensemble_size -> 1
            recon_loss = torch.mean(torch.pow(x_recon - x, 2))
            kl_loss = torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))

        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
