import math
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Mamba.pscan import pscan
from multiscan_v3 import snake_flatten
from einops import rearrange


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

    bias: bool = False  # Whether to use bias in linear layers
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
        self.adapool = nn.AdaptiveAvgPool1d(config.seq_length)
        self.pro = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        # Apply normalization, pass through the MambaBlock, and add a residual connection
        normalized_x = self.norm(x)
        x = self.adapool(rearrange(x, 'b n d -> b d n'))
        x = rearrange(x, 'b d n -> b n d')
        output = self.mixer(normalized_x) + self.pro(x)  # original paper use residual
        # output = self.mixer(normalized_x)  # I don't use it because the result becomes worse here
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
    def __init__(self, config: MambaConfig):
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

        # Forward direction SSM parameters
        self.x_proj_f = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=True)
        self.dt_proj_f = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.A_log_f = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)))
        self.D_f = nn.Parameter(torch.ones(config.d_inner), requires_grad=config.bias)

        # Backward direction SSM parameters
        self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=True)
        self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.A_log_b = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)))
        self.D_b = nn.Parameter(torch.ones(config.d_inner))

        self.initialize_dt(config, 'f')
        self.initialize_dt(config, 'b')

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.token_wA_f = nn.Parameter(torch.randn(1, config.seq_length, config.d_inner, dtype=torch.float32),
                                       requires_grad=True)
        self.token_wA_b = nn.Parameter(torch.randn(1, config.seq_length, config.d_inner, dtype=torch.float32),
                                       requires_grad=True)
        self.token_wV_f = nn.Parameter(torch.randn(1, config.d_inner, config.d_inner, dtype=torch.float32),
                                       requires_grad=True)
        self.token_wV_b = nn.Parameter(torch.randn(1, config.d_inner, config.d_inner, dtype=torch.float32),
                                       requires_grad=True)

        nn.init.kaiming_normal_(self.token_wA_f)
        nn.init.kaiming_normal_(self.token_wV_f)
        nn.init.kaiming_normal_(self.token_wA_b)
        nn.init.kaiming_normal_(self.token_wV_b)
        self.tokenizer_dropout = nn.Dropout(config.dropout)
        self.pro_to = nn.Linear(2 * config.d_inner, config.d_inner)
        self.seq_length = config.seq_length
        self.adapool = nn.AdaptiveAvgPool1d(config.seq_length)
        self.pro = nn.Linear(config.d_inner, config.d_inner)
        self.layernorm = nn.LayerNorm(config.d_inner)

    def forward(self, x_):
        N, L, D = x_.shape
        x = self.in_proj_x(x_)
        z = self.in_proj_z(x_)

        x_sub1, x_sub2 = self.split_sequence_auto(x)
        z_sub1, z_sub2 = self.split_sequence_auto(z)

        # x_f = x
        # x_b = torch.flip(x, dims=[1])
        x_f = x_sub1
        x_b = x_sub2

        # x_f = self.conv1d_f(x_f.transpose(1, 2))[:, :, :L].transpose(1, 2)
        # x_b = self.conv1d_b(x_b.transpose(1, 2))[:, :, :L].transpose(1, 2)

        x_f = F.silu(x_f)
        x_b = F.silu(x_b)

        y_f = self.ssm(x_f, 'f')
        y_b = self.ssm(x_b, 'b')

        y_f_mask = self.gaussian_mask(y_f)
        y_b_mask = self.gaussian_mask(y_b)

        # y_merge = self.submerge(y_f * y_f_mask.unsqueeze(-1), y_b * y_b_mask.unsqueeze(-1))

        y_cat = torch.cat((y_f * y_f_mask.unsqueeze(-1), y_b * y_b_mask.unsqueeze(-1).flip(dims=[1])), dim=-1)
        y = self.pro_to(y_cat)

        y = y * self.gaussian_mask(y).unsqueeze(-1)

        y = self.tokenizer(y, 'f')

        # y_sub1 = self.tokenizer(y_f * y_f_mask.unsqueeze(-1), 'f')
        # y_sub2 = self.tokenizer(y_b * y_b_mask.unsqueeze(-1), 'b')

        # y_mask = self.gaussian_mask(y)

        z = F.silu(self.adapool(rearrange(z, 'b n d -> b d n')))
        z = rearrange(z, 'b d n -> b n d')
        # z_sub1 = F.silu(z_sub1)
        # z_sub2 = F.silu(z_sub2)

        output = self.out_proj(y * z)
        # output = self.out_proj(y_sub2 * z + y_sub1 * z)

        return output

    def submerge(self, sub1, sub2):
        # Slice the sequences to separate the last step
        seq1_initial = sub1[:, :-1, :]  # shape will be (100, 24, 64)
        seq2_initial = sub2[:, :-1, :]  # shape will be (100, 24, 64)
        seq1_last = sub1[:, -1, :]  # shape will be (100, 64)
        seq2_last = sub2[:, -1, :]  # shape will be (100, 64)

        # Sum the last steps
        last_step_summed = seq1_last + seq2_last  # shape will be (100, 64)

        # Reverse the order of the initial part of the second sequence
        seq2_initial_reversed = seq2_initial.flip(dims=[1])  # Reverse along the sequence length dimension

        # Concatenate everything into a new sequence
        new_sequence = torch.cat(
            (seq1_initial, last_step_summed.unsqueeze(1), seq2_initial_reversed),
            dim=1  # Concatenate along the sequence length dimension
        )  # shape will be (100, 49, 64)
        new_sequence = self.pro(new_sequence)
        new_sequence = self.layernorm(new_sequence)

        return new_sequence

    def tokenizer(self, x, direction):
        if direction == 'f':
            wa = self.token_wA_f.transpose(1, 2)
            A = torch.einsum('bij,bjk->bik', x, wa).transpose(1, 2).softmax(dim=-1)
            VV = torch.einsum('bij,bjk->bik', x, self.token_wV_f)
            T = self.tokenizer_dropout(torch.einsum('bij,bjk->bik', A, VV))
        elif direction == 'b':
            wa = self.token_wA_b.transpose(1, 2)
            A = torch.einsum('bij,bjk->bik', x, wa).transpose(1, 2).softmax(dim=-1)
            VV = torch.einsum('bij,bjk->bik', x, self.token_wV_b)
            T = self.tokenizer_dropout(torch.einsum('bij,bjk->bik', A, VV))
        else:
            raise ValueError("Invalid direction")
        return T

    def gaussian_mask(self, sequence):
        length = sequence.shape[1]
        center_index = (length + 1) // 2  # Center index, automatically determined
        last_index = length
        # Create a tensor of index positions [0, 1, 2, ..., length-1]
        indices = torch.arange(0, length, dtype=torch.float32, device=sequence.device)

        # Calculate the Gaussian weights using the distance from the center index
        # Calculate the average index distance to the center
        sigma = torch.abs(indices - center_index).mean()
        weights = torch.exp(-0.5 * ((indices - center_index) ** 2) / (sigma ** 2))

        # Normalize the weights to make them sum to 1
        weights /= weights.sum()

        # Repeat weights across the batch dimension
        weights = weights.repeat(sequence.size(0), 1)  # Shape (batch, length)

        return weights

    @staticmethod
    def split_sequence_auto(sequence):
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

    def ssm(self, x, direction):
        if direction == 'f':
            A = -torch.exp(self.A_log_f)
            D = self.D_f
            deltaBC = self.x_proj_f(x)
            delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
            delta = F.softplus(self.dt_proj_f(delta))
            y = self.selective_scan(x, delta, A, B, C, D)
        elif direction == 'b':
            A = -torch.exp(self.A_log_b)
            D = self.D_b
            deltaBC = self.x_proj_b(x)
            delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
            delta = F.softplus(self.dt_proj_b(delta))
            # print('A', A.shape, 'B',B.shape, 'C', C.shape, 'D', D.shape )
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            raise ValueError("Invalid direction")
        return y

    @staticmethod
    def selective_scan(x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)
        hs = pscan(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3) + D * x
        return y

    def initialize_dt(self, config, direction):
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            init_fn = nn.init.constant_
        elif config.dt_init == "random":
            init_fn = nn.init.uniform_
        else:
            raise NotImplementedError

        if direction == 'f':
            init_fn(self.dt_proj_f.weight, dt_init_std)
        elif direction == 'b':
            init_fn(self.dt_proj_b.weight, dt_init_std)
        else:
            raise ValueError("Invalid direction")

        dt = torch.exp(torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
            config.dt_min)).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            if direction == 'f':
                self.dt_proj_f.bias.copy_(inv_dt)
            elif direction == 'b':
                self.dt_proj_b.bias.copy_(inv_dt)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Mamaba_2(nn.Module):
    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.):
        super(Mamaba_2, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.mamba_config = MambaConfig(d_model=dim, n_layers=depth, seq_length=self.num_patches,
                                        dropout=emb_dropout, d_state=16, bias=True)
        self.mamba = Mamba(self.mamba_config)
        self.layernorm = nn.LayerNorm(dim)
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

    def forward(self, img):
        # img = img.squeeze(1)

        img = snake_flatten(img)

        x = self.to_patch_embedding(img)

        x += self.pos_embedding[:, :self.num_patches]

        x = self.dropout(x)

        x = self.mamba(x)
        b, n, _ = x.shape
        c = int(n + 1) // 2

        x = self.layernorm(x)

        x_mean = torch.mean(x, dim=1)
        x_last = x[:, -1]
        x_center = x[:, c]
        x_adapool = F.adaptive_avg_pool2d(x, (1, self.dim)).view(b, -1)

        x = self.to_latent(x_mean)

        return self.mlp_head(x), self.regression(x)


class MiM(nn.Module):
    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.):
        super(MiM, self).__init__()

        self.T_mamba_1 = Mamaba_2(channels, num_classes, image_size, patch_size=patch_size, dim=dim, depth=depth,
                                  emb_dropout=emb_dropout)
        self.T_mamba_2 = Mamaba_2(channels, num_classes, image_size, patch_size=patch_size, dim=dim, depth=depth,
                                  emb_dropout=emb_dropout)
        self.drop = nn.Dropout(emb_dropout)
        self.aux_loss_weight = 1

    def forward(self, x):
        x = x.squeeze(1)

        out_1, reg_1 = self.T_mamba_1(x)
        # out_2, reg_2 = self.T_mamba_2(rearrange(x, 'b c w h -> b c h w'))

        out = out_1
        reg = reg_1

        return out, reg
