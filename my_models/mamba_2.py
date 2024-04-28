import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from Mamba.mamba import Mamba, MambaConfig
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from Mamba.pscan import pscan
from multiscan_v3 import snake_flatten, to_uturn_sequence
from tokenlearner_pytorch import TokenLearner, TokenFuser


# helpers

@dataclass
class MambaConfig:
    d_model: int  # Dimensionality of the model
    n_layers: int  # Number of layers in the Mamba model
    dt_rank: Union[int, str] = 'auto'  # Rank for delta in S4 layer, 'auto' calculates it based on d_model
    d_state: int = 16  # State dimensionality in S4 layer
    expand_factor: int = 2  # Expansion factor for inner dimension
    d_conv: int = 4  # Convolution kernel size
    seq_length: int = 9
    dt_min: float = 0.001  # Minimum delta value
    dt_max: float = 0.1  # Maximum delta value
    dt_init: str = "random"  # Initialization mode for delta, "random" or "constant"
    dt_scale: float = 1.0  # Scaling factor for delta initialization
    dt_init_floor: float = 1e-4  # Minimum value for delta initialization

    bias: bool = True  # Whether to use bias in linear layers
    conv_bias: bool = True  # Whether to use bias in convolutional layers

    pscan: bool = True  # Use parallel scan mode or sequential mode when training
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.dt_min >= self.dt_max:
            raise ValueError("dt_min must be less than dt_max")

        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # Iterate over each layer in the Mamba model
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # Apply normalization, pass through the MambaBlock, and add a residual connection
        normalized_x = self.norm(x)
        output = self.mixer(normalized_x) + x
        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_proj_x = nn.Linear(config.d_model, config.d_inner, bias=config.bias)
        self.in_proj_z = nn.Linear(config.d_model, config.d_inner, bias=config.bias)
        self.conv1d_f = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                  kernel_size=config.d_conv, bias=config.conv_bias,
                                  groups=config.d_inner, padding=config.d_conv - 1)
        self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                  kernel_size=config.d_conv, bias=config.conv_bias,
                                  groups=config.d_inner, padding=config.d_conv - 1)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=True)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.initialize_dt(config)
        self.initialize_parameters(config)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.tokenizer_dropout = nn.Dropout(config.dropout)
        self.pro_to = nn.Linear(2 * config.d_inner, config.d_inner)

    def forward(self, x_):
        _, L, _ = x_.shape
        x = self.in_proj_x(x_)
        z = self.in_proj_z(x_)

        # seq1, seq2 = self.split_sequence_auto(x)

        x_f = x
        x_b = torch.flip(x, dims=[1])

        x_f = self.conv1d_f(x_f.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_b = self.conv1d_b(x_b.transpose(1, 2))[:, :, :L].transpose(1, 2)

        x_f = F.silu(x_f)
        x_b = F.silu(x_b)

        y_f = self.ssm(x_f)
        y_b = self.ssm(x_b)

        # y_f = self.tokenizer(y_f)
        # y_b = self.tokenizer(y_b)

        y_cat = torch.cat((y_f, y_b), dim=-1)
        # # print('y_cat',y_cat.shape)
        y = self.pro_to(y_cat)

        y = self.tokenizer(y)

        z = F.silu(z)

        output = self.out_proj(y * z)

        return output

    def split_sequence_auto(self, sequence):
        """
        Splits a sequence into two subsequences automatically. The first subsequence is in
        the natural order starting from the beginning, and the second subsequence is in
        reverse order starting from the end, with an overlap of one element in the middle.

        Args:
            sequence (torch.Tensor): The input sequence tensor with shape (batch, length, dimension).

        Returns:
            torch.Tensor: The first subsequence.
            torch.Tensor: The second subsequence.
        """
        _, length, _ = sequence.shape
        mid_point = length // 2 + 1

        # Extract the subsequences
        subsequence1 = sequence[:, :mid_point, :]
        subsequence2 = torch.flip(sequence[:, mid_point - 1:, :], dims=[1])

        return subsequence1, subsequence2

    def tokenizer(self, x):
        wa = self.token_wA.transpose(1, 2)
        A = torch.einsum('bij,bjk->bik', x, wa).transpose(1, 2).softmax(dim=-1)
        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = self.tokenizer_dropout(torch.einsum('bij,bjk->bik', A, VV))
        return T

    def ssm(self, x):
        A = -torch.exp(self.A_log)
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)
        hs = pscan(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3) + D * x
        return y

    def initialize_dt(self, config):
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError("Unsupported dt_init value: {}".format(config.dt_init))
        dt = torch.exp(torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
            config.dt_min)).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def initialize_parameters(self, config):
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)),requires_grad=True)
        self.D = nn.Parameter(torch.ones(config.d_inner),requires_grad=True)
        self.token_wA = nn.Parameter(
            torch.empty(1, config.seq_length, config.d_inner, dtype=torch.float32),requires_grad=True)
        self.token_wV = nn.Parameter(
            torch.empty(1, config.d_inner, config.d_inner,
                        dtype=torch.float32),requires_grad=True)
        nn.init.xavier_normal_(self.token_wA)
        nn.init.xavier_normal_(self.token_wV)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Mamaba_2(nn.Module):
    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.):
        super(Mamaba_2, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            # nn.Conv2d(channels, dim, 3, 1, 1),
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.mamba_config = MambaConfig(d_model=dim, n_layers=depth,
                                        seq_length=num_patches, dropout=emb_dropout, bias=True, expand_factor=2)
        self.mamba = Mamba(self.mamba_config)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, num_classes)
        )
        self.regression = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, channels)
        )
        self.aux_loss_weight = 1

    def forward(self, img):
        img = img.squeeze(1)

        img = snake_flatten(img)

        x = self.to_patch_embedding(img)
        b, n, d = x.shape
        c = int(n + 1) // 2

        x += self.pos_embedding[:, :(n)]

        x = self.dropout(x)

        x = self.mamba(x)

        x_mean = torch.mean(x, dim=1)
        x_last = x[:, -1]
        x_center = x[:, c]
        x_adapool = F.adaptive_avg_pool2d(x, (1, self.dim)).view(b, -1)

        x = self.to_latent(x_mean)

        return self.mlp_head(x), self.regression(x)
