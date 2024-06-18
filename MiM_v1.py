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
from vision_mamba.model import VisionEncoderMambaBlock as vim

torch.autograd.set_detect_anomaly(True)
import numpy as np


# helpers

@dataclass
class MambaConfig:
    d_model: int  # Dimensionality of the model
    n_layers: int  # Number of layers in the Mamba model
    dt_rank: Union[int, str] = 'auto'  # Rank for delta in S4 layer, 'auto' calculates it based on d_model
    d_state: int = 16  # State dimensionality in S4 layer
    expand_factor: int = 2  # Expansion factor for inner dimension
    d_conv: int = 1  # Convolution kernel size

    seq_length: int = 49
    num_tokens: int = 25

    dt_min: float = 0.001  # Minimum delta value
    dt_max: float = 0.1  # Maximum delta value
    dt_init: str = "random"  # Initialization mode for delta, "random" or "constant"
    dt_scale: float = 1.0  # Scaling factor for delta initialization
    dt_init_floor: float = 1e-4  # Minimum value for delta initialization
    rms_norm_eps: float = 1e-5
    bias: bool = True  # Whether to use bias in linear layers
    conv_bias: bool = True  # Whether to use bias in convolutional layers
    inner_layernorms: bool = True  # apply layernorms to internal activations

    pscan: bool = True  # Use parallel scan mode or sequential mode when training
    use_cuda: bool = True  # use official CUDA implementation when training (not compatible with (b)float16)
    dropout: float = 0.1

    def __post_init__(self):
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
        self.mamba = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.adapool = nn.AdaptiveAvgPool1d(config.num_tokens)
        self.pro = nn.Linear(config.d_model, config.d_model)
        self.layernorm = nn.LayerNorm(config.d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Apply normalization, pass through the MambaBlock, and add a residual connection
        normalized_x = self.norm(x)
        x = self.adapool(rearrange(x, 'b n d -> b d n'))
        x = rearrange(x, 'b d n -> b n d')
        output = self.mamba(normalized_x) + self.pro(x)  # original paper use residual
        output = self.tanh(output)
        output = self.layernorm(output)
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


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=1, stride=1),  # Use Conv1d for sequence data
            nn.BatchNorm1d(1),
            nn.ReLU()
        )

    def forward(self, x):
        B, L, C = x.shape
        x = rearrange(x, 'b l c -> b c l')
        B, C, L = x.shape
        max_features = torch.max(x, 1)[0].unsqueeze(1)
        avg_features = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([max_features, avg_features], dim=1)  # Now [B, 2, L]
        # print('x', combined.shape)

        fmap = self.conv1d(combined)  # Apply convolution
        weight_map = torch.sigmoid(fmap)  # Sigmoid to get weights [B, 1, L]

        x_weighted = x * weight_map  # Apply weights
        output = torch.mean(x_weighted, dim=2)  # Mean over sequence

        return output, x_weighted


class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])

    def forward(self, x):
        B, L, C = x.shape
        # print('x1', x.shape)
        Z = torch.zeros(B, self.S, C, dtype=x.dtype, device=x.device)  # Ensure Z has the right type and device
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x)  # [B, C]
            Z[:, i, :] = Ai
        return Z


class TokenFuser(nn.Module):
    def __init__(self, C, S) -> None:
        super().__init__()
        self.projection = nn.Linear(S, S, bias=True)
        self.Bi = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S

    def forward(self, y, x):
        B, S, C = y.shape
        B, L, C = x.shape

        Y = self.projection(y.view(B, C, S)).view(B, S, C)

        Bw = torch.sigmoid(self.Bi(x)).view(B, L, S)  # [B, L, S]

        BwY = torch.matmul(Bw, Y)  # [B, L, C]

        _, xj = self.spatial_attn(x)
        xj = xj.reshape(B, L, C)

        out = (BwY + xj).view(B, L, C)

        return out


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.in_proj_x = nn.Linear(config.d_model, config.d_inner, bias=config.bias)
        self.in_proj_z = nn.Linear(config.d_model, config.d_inner, bias=config.bias)

        # Forward direction SSM parameters
        self.conv1d_f = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                  kernel_size=config.d_conv, bias=config.conv_bias,
                                  groups=config.d_inner,
                                  padding=config.d_conv - 1)
        self.x_proj_f = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=True)
        self.dt_proj_f = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.A_log_f = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)))
        self.D_f = nn.Parameter(torch.ones(config.d_inner), requires_grad=config.bias)

        # Backward direction SSM parameters
        self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                  kernel_size=config.d_conv, bias=config.conv_bias,
                                  groups=config.d_inner, padding=config.d_conv - 1)
        self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=True)
        self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.A_log_b = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)))
        self.D_b = nn.Parameter(torch.ones(config.d_inner))

        self.initialize_dt(config, 'f')
        self.initialize_dt(config, 'b')

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # Forward direction SSM parameters
        self.token_wA_f = nn.Parameter(torch.randn(1, config.num_tokens, config.d_inner, dtype=torch.float32),
                                       requires_grad=True)
        self.token_wV_f = nn.Parameter(torch.randn(1, config.d_inner, config.d_inner, dtype=torch.float32),
                                       requires_grad=True)

        # Backward direction SSM parameters
        self.token_wA_b = nn.Parameter(torch.randn(1, config.num_tokens, config.d_inner, dtype=torch.float32),
                                       requires_grad=True)
        self.token_wV_b = nn.Parameter(torch.randn(1, config.d_inner, config.d_inner, dtype=torch.float32),
                                       requires_grad=True)

        nn.init.kaiming_normal_(self.token_wA_f)
        nn.init.kaiming_normal_(self.token_wV_f)
        nn.init.kaiming_normal_(self.token_wA_b)
        nn.init.kaiming_normal_(self.token_wV_b)

        self.tokenizer_dropout = nn.Dropout(config.dropout)
        self.pro_to_cat = nn.Linear(2 * config.d_inner, config.d_inner)
        self.pro_to_merge = nn.Linear(config.d_inner, config.d_inner)

        self.seq_length = config.seq_length
        self.adapool = nn.AdaptiveAvgPool1d(config.num_tokens)
        self.pro = nn.Linear(config.d_inner, config.d_inner)
        self.layernorm = nn.LayerNorm(config.d_inner)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.tokenlearner = TokenLearner(config.num_tokens)
        self.spatial_attn = SpatialAttention()
        self.tokenfuser = TokenFuser(C=config.d_inner, S=config.num_tokens)
        self.rnn1 = nn.LSTM(config.d_inner, config.d_inner, config.n_layers, batch_first=True)
        self.rnn2 = nn.LSTM(config.d_inner, config.d_inner, config.n_layers, batch_first=True)

    def forward(self, x_):
        # N, L, D = x_.shape
        x = self.in_proj_x(x_)
        z = self.in_proj_z(x_)
        _, L, _ = x.shape

        # x_sub1, x_sub2 = self.split_sequence_auto(x)
        # print('sub0', x_sub1.shape, x_sub2.shape)
        # z_sub1, z_sub2 = self.split_sequence_auto(z)

        x_sub1 = x
        x_sub2 = torch.flip(x, dims=[1])
        x_f = x_sub1
        x_b = x_sub2

        # print('sub0', x_sub1.shape, x_sub2.shape)

        x_f = x_f.transpose(1, 2)  #  (B, ED, L)
        x_f = self.conv1d_f(x_f)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x_f = x_f.transpose(1, 2)  #  (B, L, ED)

        x_b = x_b.transpose(1, 2)  #  (B, ED, L)
        x_b = self.conv1d_b(x_b)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x_b = x_b.transpose(1, 2)  #  (B, L, ED)

        # print('sub1', x_f.shape, x_b.shape)

        x_f = F.silu(x_f)
        x_b = F.silu(x_b)

        y_f = self.ssm(x_f, 'f')
        y_b = self.ssm(x_b, 'b')

        y_f = self.tanh(y_f)
        y_b = self.tanh(y_b)
        # print('sub2', y_f.shape, y_b.shape)

        y_f_mask_1 = self.gaussian_mask_index(y_f, 'center')
        y_f_mask_2 = self.gaussian_mask_vector(y_f, 'center')

        y_b_mask_1 = self.gaussian_mask_index(y_b, 'center')
        y_b_mask_2 = self.gaussian_mask_vector(y_b, 'center')

        y_f_mask = F.softmax(y_f_mask_1 * y_f_mask_2, dim=1)
        y_b_mask = F.softmax(y_b_mask_1 * y_b_mask_2, dim=1)

        y_cat = torch.cat((y_f * y_f_mask.unsqueeze(-1), y_b * y_b_mask.unsqueeze(-1).flip(dims=[1])), dim=-1)
        y = self.pro_to_cat(y_cat)
        y_mask = F.softmax(self.gaussian_mask_index(y_cat, 'center') * self.gaussian_mask_vector(y_cat, 'center'),
                           dim=1)
        y = y * y_mask.unsqueeze(-1)

        # y_merge = self.cross_scan_merge(y_f * y_f_mask.unsqueeze(-1), y_b * y_b_mask.unsqueeze(-1))
        # y = self.pro_to_merge(y_merge)
        # y_merge_mask = F.normalize(self.gaussian_mask_index(y_merge, 'center') * self.gaussian_mask_vector(y_merge, 'center'), p=2)
        # y = y_merge * y_merge_mask.unsqueeze(-1)
        # print('y, z', y.shape, z.shape)

        _, y = self.spatial_attn(y)
        # print('y, z', y.shape, z.shape)

        u = self.STL(rearrange(y, 'b c l -> b l c'), 'f')

        # y_sub1 = self.tokenizer(y_f * y_f_mask.unsqueeze(-1), 'f')
        # y_sub2 = self.tokenizer(y_b * y_b_mask.unsqueeze(-1), 'b')

        z = F.silu(self.adapool(rearrange(z, 'b n d -> b d n')))
        z = rearrange(z, 'b d n -> b n d')

        # print('y, z', y.shape, z.shape)

        output = self.tokenfuser(u, z)

        output = self.out_proj(output)

        output = self.tanh(output)
        # output = self.out_proj(y_sub2 * z + y_sub1 * z)

        return output

    def cross_scan_merge(self, sub1, sub2):
        # Slice the sequences to separate the last step
        seq1_initial = sub1[:, :-1, :]  # shape will be (100, 24, 64)
        seq2_initial = sub2[:, :-1, :]  # shape will be (100, 24, 64)
        seq1_last = sub1[:, -1, :]  # shape will be (100, 64)
        seq2_last = sub2[:, -1, :]  # shape will be (100, 64)

        # Sum the last steps
        last_step_summed = (seq1_last + seq2_last) / 2  # shape will be (100, 64)

        # Reverse the order of the initial part of the second sequence
        seq2_initial = seq2_initial.flip(dims=[1])  # Reverse along the sequence length dimension

        # Concatenate everything into a new sequence
        new_sequence = torch.cat(
            (seq1_initial, last_step_summed.unsqueeze(1), seq2_initial),
            dim=1  # Concatenate along the sequence length dimension
        )  # shape will be (100, 49, 64)
        new_sequence = self.pro(new_sequence)
        new_sequence = self.tanh(new_sequence)
        new_sequence = self.layernorm(new_sequence)

        return new_sequence

    def STL(self, x, direction):
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

    @staticmethod
    def gaussian_mask_index(sequence, type='center'):
        length = sequence.shape[1]
        # print('length', length)
        center_index = (length + 1) // 2  # Center index, automatically determined
        last_index = length - 1  # Corrected to be within bounds

        # Create a tensor of index positions [0, 1, 2, ..., length-1]
        indices = torch.arange(length, dtype=torch.float32, device=sequence.device)
        # print('index', indices.shape)
        # Select the reference index and calculate sigma based on the selected type
        if type == 'center':
            ref_index = center_index
        elif type == 'last':
            ref_index = last_index
        else:
            raise ValueError("Invalid type specified. Choose 'center' or 'last' as the Gaussian decay mask type.")

        # Calculate the average index distance to the reference index
        sigma = torch.abs(indices - ref_index).mean()

        # Calculate the Gaussian weights using the distance from the reference index
        weights = torch.exp(-0.5 * ((indices - ref_index) ** 2) / (sigma ** 2))
        # print('weights', weights.shape)
        # Normalize the weights to make them sum to 1
        weights /= weights.sum()
        # print('weights', weights.shape)
        # Repeat weights across the batch dimension
        weights = weights.repeat(sequence.size(0), 1)  # Shape (batch, length)
        # print('weights', weights.shape)
        return weights

    @staticmethod
    def gaussian_mask_vector(sequence, type='center'):
        length = sequence.shape[1]
        center_index = (length + 1) // 2  # Automatically determined
        last_index = length - 1  # Last index in sequence

        # Select reference vector
        ref_vector = sequence[:, center_index, :] if type == 'center' else sequence[:, last_index, :]

        # Calculate Euclidean distances
        distances = torch.norm(sequence - ref_vector.unsqueeze(1), dim=2)

        # Calculate sigma
        sigma = distances.mean(dim=1, keepdim=True)

        # Compute Gaussian weights
        weights = torch.exp(-0.5 * (distances / sigma) ** 2)

        # Normalize the weights
        weights = weights / weights.sum(dim=1, keepdim=True)  # Ensure this is not in-place

        return weights

    @staticmethod
    def split_sequence_auto(sequence):
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


class T_Mamaba(nn.Module):
    def __init__(self, channels, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.
                 , seq_length=49, num_tokens=25):
        super(T_Mamaba, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        self.num_patches = seq_length
        self.patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.mamba_config = MambaConfig(d_model=dim, n_layers=depth, seq_length=self.num_patches,
                                        dropout=emb_dropout, bias=True, num_tokens=num_tokens)
        self.mamba = Mamba(self.mamba_config)
        # self.vim = vim(dim=dim, dt_rank=dim, dim_inner= dim, d_state=dim)
        # self.rnn = nn.RNN(dim, dim, depth, batch_first=True)
        # self.gru = nn.GRU(dim, dim, depth, batch_first=True)
        # self.lstm = nn.LSTM(dim, dim, depth, batch_first=True)
        # self.gru = nn.GRU(dim, dim, depth, batch_first=True)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
        self.layers = nn.ModuleList()

        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                self.mamba
            )
        self.layernorm = nn.LayerNorm(dim)
        self.pro = nn.Linear(dim * 2, dim)
        self.tanh = nn.Tanh()

    def forward(self, img):
        if img.dim() == 4:
            img = snake_flatten(img)
        else:
            img = img

        if img.size(2) == self.patch_dim:
            x = self.to_patch_embedding(img)
        else:
            x = img

        x += self.pos_embedding[:, :self.num_patches]  # (100, seq_length, 64)

        for layer in self.layers:
            x = layer(x)

        x = self.layernorm(x)

        x = self.tanh(x)

        return x


class MiM_block(nn.Module):
    def __init__(self, channels, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.,
                 seq_length=49, num_tokens=25):
        super(MiM_block, self).__init__()

        # Initialize Mamba_2 models
        self.T_mamba = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                                emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)

    def forward(self, x1, x2, x3, x4):
        x_1 = x1
        x_2 = x2
        x_3 = x3
        x_4 = x4

        # Get outputs and regressions from each transformed input
        out_1 = self.T_mamba(x_1)
        out_4 = self.T_mamba(x_4)
        out_2 = self.T_mamba(x_2)
        out_3 = self.T_mamba(x_3)

        return out_1, out_2, out_3, out_4


class MiM_v1(nn.Module):
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.Parameter, nn.BatchNorm1d, MiM_block,
                          MiM_v1, T_Mamaba, MambaBlock, Mamba, TokenLearner, TokenFuser, SpatialAttention,
                          ResidualBlock)):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=64, depth=1, emb_dropout=0.1,
                 num_tokens=49):
        super(MiM_v1, self).__init__()
        self.mim_1 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=image_size ** 2, num_tokens=7 ** 2)

        self.mim_2 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=7 ** 2, num_tokens=5 ** 2)
        #
        self.mim_3 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=5 ** 2, num_tokens=3 ** 2)
        #
        self.mim_4 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=3 ** 2, num_tokens=1 ** 2)

        self.aux_loss_weight = 1

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, num_classes)
        )
        self.regression = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, channels)
        )

        self.to_latent = nn.Identity()
        # Initialize weights as learnable parameters
        self.k_weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)  # Start with equal weights
        # self.apply(self.weight_init)

    def WTF(self, *o):
        # Normalize weights to ensure they sum to 1
        k_weights = torch.softmax(self.k_weights, dim=0)

        # Check if the number of outputs matches the number of weights
        assert len(o) == len(k_weights), "The number of outputs and weights must match."

        # Weighted sum of outputs
        O = sum(w * out for w, out in zip(k_weights, o))

        # Average over features
        if O.dim() == 3:
            O_mean = torch.mean(O, dim=1)
        else:
            O_mean = O
        return O_mean

    def forward(self, x):
        x_1 = x.squeeze(1)
        x_4 = torch.rot90(x_1, k=-1, dims=(2, 3))
        x_2 = torch.flip(x_4, dims=[3])
        x_3 = torch.rot90(x_2, k=-1, dims=(2, 3))

        x_1 = snake_flatten(x_1)
        x_2 = snake_flatten(x_2)
        x_3 = snake_flatten(x_3)
        x_4 = snake_flatten(x_4)

        tm1_1, tm2_1, tm3_1, tm4_1 = self.mim_1(x_1, x_2, x_3, x_4)
        # print('tm',tm1_1.shape)

        O_1 = self.WTF(tm1_1, tm2_1, tm3_1, tm4_1)
        # print('tm', O_1.shape)

        # tm1_2, tm2_2, tm3_2, tm4_2 = self.mim_2(tm1_1, tm2_1, tm3_1, tm4_1)

        # O_2 = self.WTF(tm1_2, tm2_2, tm3_2, tm4_2)

        # tm1_3, tm2_3, tm3_3, tm4_3 = self.mim_3(tm1_2, tm2_2, tm3_2, tm4_2)

        # O_3 = self.WTF(tm1_3, tm2_3, tm3_3, tm4_3)
        #
        # tm1_4, tm2_4, tm3_4, tm4_4 = self.mim_4(tm1_3, tm2_3, tm3_3, tm4_3)
        #
        # O_4 = self.WTF(tm1_4, tm2_4, tm3_4, tm4_4)

        return self.mlp_head(self.to_latent(O_1))
