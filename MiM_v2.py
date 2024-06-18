import math
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Mamba.pscan import pscan
from multiscan_v3 import snake_flatten, snake_unflatten
from einops import rearrange
import seaborn as sns
torch.autograd.set_detect_anomaly(True)
import numpy as np
from vision_mamba.model import VisionEncoderMambaBlock as vim


# helpers

@dataclass
class MambaConfig:
    d_model: int  # Dimensionality of the model
    n_layers: int  # Number of layers in the Mamba model
    dt_rank: Union[int, str] = 'auto'  # Rank for delta in S4 layer, 'auto' calculates it based on d_model
    d_state: int = 16  # State dimensionality in S4 layer
    expand_factor: int = 2  # Expansion factor for inner dimension
    d_conv: int = 3  # Convolution kernel size

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
        # sns.heatmap(max_features[0].cpu().detach().numpy())
        # plt.title('max feature')
        # plt.show()
        # sns.heatmap(avg_features[0].cpu().detach().numpy())
        # plt.title('avg feature')
        # plt.show()
        # print('max mean', max_features.shape, avg_features.shape)
        combined = torch.cat([max_features, avg_features], dim=1)  # Now [B, 2, L]
        # print('x', combined.shape)

        fmap = self.conv1d(combined)  # Apply convolution
        m = torch.sigmoid(fmap)  # Sigmoid to get weights [B, 1, L]
        # print('m', m.shape)
        # sns.heatmap(m[0].cpu().detach().numpy())
        # plt.title('m')
        # plt.show()
        m_hat = x * m  # Apply weights
        output = torch.mean(m_hat, dim=2)  # Mean over sequence

        return output, m_hat


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


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        # Forward direction SSM parameters
        #  projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        #  projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from Mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

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
        self.rnn = nn.LSTM(config.d_inner, config.d_inner, config.n_layers, batch_first=True)

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        #  x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        x = self.in_proj(x)  # (B, L, 2*ED)
        # print('x1', x.shape)
        x = x.transpose(1, 2)  #  (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  #  (B, L, ED)
        # print('x2', x.shape)
        x = F.silu(x)
        # print('x2', x.shape)
        # y = self.rnn(x)
        y = self.ssm(x)
        # print('y', y.shape)
        output = self.out_proj(y)  #  (B, L, D)

        return output

    def ssm(self, x, z=None):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  #  (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                         delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    @staticmethod
    def selective_scan(x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        #  y : (B, D)
        #  cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  #  (B, ED), (B, ED)

        #  x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  #  (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        #  x : (B, ED)
        #  h : (B, ED, N)

        #  y : (B, ED)
        #  h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x)  #  (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class T_Mamba(nn.Module):
    def __init__(self, channels, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.
                 , seq_length=49, num_tokens=25):
        super(T_Mamba, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        self.num_patches = seq_length
        self.patch_dim = channels * patch_height * patch_width
        self.in_proj_x = nn.Linear(dim, dim, bias=True)
        self.in_proj_z = nn.Linear(dim, dim, bias=True)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.tokenizer_dropout = nn.Dropout(emb_dropout)
        self.pro_to_cat = nn.Linear(2 * dim, dim)
        self.pro_to_merge = nn.Linear(dim * 2, dim)
        self.seq_length = seq_length
        self.adapool = nn.AdaptiveAvgPool1d(num_tokens)
        self.pro = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.tokenlearner = TokenLearner(num_tokens)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.mamba_config = MambaConfig(d_model=dim, n_layers=depth, seq_length=self.num_patches,
                                        dropout=emb_dropout, d_state=dim // 4, bias=True, num_tokens=num_tokens)
        # self.mamba1 = MambaBlock(self.mamba_config)
        self.vim1 = vim(dim=dim, dt_rank=32, dim_inner=dim, d_state=16)
        # self.mamba2 = MambaBlock(self.mamba_config)
        self.vim2 = vim(dim=dim, dt_rank=32, dim_inner=dim, d_state=16)
        # self.rnn1 = nn.LSTM(dim, dim, depth, batch_first=True)
        # self.rnn2 = nn.LSTM(dim, dim, depth, batch_first=True)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4)
        # self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
        # self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
        self.layernorm = nn.LayerNorm(dim)
        self.pro = nn.Linear(dim, dim)
        self.tanh = nn.Tanh()

        self.U1 = nn.Parameter(torch.randn(1, num_tokens, dim, dtype=torch.float32),
                                       requires_grad=True)
        self.U2 = nn.Parameter(torch.randn(1, dim, dim, dtype=torch.float32),
                                       requires_grad=True)

        # nn.init.kaiming_normal_(self.U1)
        # nn.init.kaiming_normal_(self.U2)
        torch.nn.init.xavier_normal_(self.U1)
        torch.nn.init.xavier_normal_(self.U2)

        self.spatial_attn = SpatialAttention()
        self.tokenfuser = TokenFuser(C=dim, S=num_tokens)

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

    @staticmethod
    def split_sequence_auto(sequence):
        _, length, _ = sequence.shape
        mid_point = length // 2 + 1

        # Extract the subsequences
        subsequence1 = sequence[:, :mid_point, :]
        subsequence2 = torch.flip(sequence[:, mid_point - 1:, :], dims=[1])

        return subsequence1, subsequence2

    def forward(self, img):
        if img.dim() == 4:
            img = snake_flatten(img)
        else:
            img = img

        if img.size(2) == self.patch_dim:
            x = self.to_patch_embedding(img)
        else:
            x = img

        x = self.in_proj_x(x)
        z = self.in_proj_z(x)  # (B, L, 2*ED)
        # sns.heatmap(z[0].cpu().detach().numpy())
        # plt.title('z1')
        # plt.show()

        x = x + self.pos_embedding[:, :self.num_patches]  # (100, seq_length, 64)

        x_sub1 = x
        x_sub2 = torch.flip(x, dims=[1])
        # x_sub1, x_sub2 = self.split_sequence_auto(x)

        x_sub1 = self.dropout(x_sub1)
        x_sub2 = self.dropout(x_sub2)

        # y_f = self.mamba1(x_sub1)  # (100, num_tokens, 64)
        y_f = self.vim1(x_sub1)  # (100, num_tokens, 64)
        # y_b = self.mamba2(x_sub2)  # (100, num_tokens, 64)
        y_b = self.vim2(x_sub1)  # (100, num_tokens, 64)

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
        s_hat = y * y_mask.unsqueeze(-1)

        # y_merge = self.cross_scan_merge(y_f * y_f_mask.unsqueeze(-1), y_b * y_b_mask.unsqueeze(-1))
        # y_merge = torch.cat([y_merge, y_merge.flip(dims=[1])], dim=-1)
        # y_merge = self.pro_to_merge(y_merge)
        # y_merge_mask = F.normalize(self.gaussian_mask_index(y_merge, 'center') * self.gaussian_mask_vector(y_merge, 'center'), p=2)
        # s_hat = y_merge * y_merge_mask.unsqueeze(-1)

        # sns.heatmap(s_hat[0].cpu().detach().numpy())
        # plt.title('s hat')
        # plt.show()

        # print('y', y.shape)
        _, m_hat = self.spatial_attn(s_hat)
        # sns.heatmap(m_hat[0].cpu().detach().numpy())
        # plt.title('m hat')
        # plt.show()
        # print('y, z', y.shape, z.shape)

        u = self.STL(rearrange(m_hat, 'b c l -> b l c'))
        # sns.heatmap(u[0].cpu().detach().numpy())
        # plt.title('small u')
        # plt.show()

        # y_sub1 = self.tokenizer(y_f * y_f_mask.unsqueeze(-1), 'f')
        # y_sub2 = self.tokenizer(y_b * y_b_mask.unsqueeze(-1), 'b')

        z = F.silu(self.adapool(rearrange(z, 'b n d -> b d n')))
        z_line = rearrange(z, 'b d n -> b n d')

        # sns.heatmap(z_line[0].cpu().detach().numpy())
        # plt.title('z2')
        # plt.show()

        # print('y, z', y.shape, z.shape)

        output = self.tokenfuser(u, z_line)

        output = self.out_proj(output)

        output = self.tanh(output)

        return output

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

    def STL(self, m_hat):
        U1 = self.U1.transpose(1, 2)
        U2 = self.U2
        # sns.heatmap(U1[0].cpu().detach().numpy())
        # plt.title('U1')
        # plt.show()
        A = torch.einsum('bij,bjk->bik', m_hat, U1).transpose(1, 2).softmax(dim=-1)
        # sns.heatmap(A[0].cpu().detach().numpy())
        # plt.title('A')
        # plt.show()
        V = torch.einsum('bij,bjk->bik', m_hat, U2)
        # sns.heatmap(V[0].cpu().detach().numpy())
        # plt.title('V')
        # plt.show()
        # sns.heatmap(U2[0].cpu().detach().numpy())
        # plt.title('U2')
        # plt.show()
        u = self.tokenizer_dropout(torch.einsum('bij,bjk->bik', A, V))
        return u


class MiM_block(nn.Module):
    def __init__(self, channels, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.,
                 seq_length=49, num_token=25):
        super(MiM_block, self).__init__()
        self.T_mamba = T_Mamba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_token)
        # Incorporate the components from ResidualBlock
        self.norm = RMSNorm(dim)
        self.adapool = nn.AdaptiveAvgPool1d(num_token)
        self.pro = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3, x4):
        inputs = [x1, x2, x3, x4]
        results = []

        for x in inputs:
            normalized_x = self.norm(x)
            x = self.adapool(rearrange(x, 'b n d -> b d n'))
            x = rearrange(x, 'b d n -> b n d')
            output = self.T_mamba(normalized_x) + self.pro(x)  # original paper use residual
            output = self.tanh(output)
            output = self.layernorm(output)

            results.append(output)

        return results  # Return the processed results for all inputs


class TokenFuser(nn.Module):
    def __init__(self, C, S) -> None:
        super().__init__()
        self.projection = nn.Linear(S, S, bias=True)
        self.Z = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S

    def forward(self, u, z_line):
        B, S, C = u.shape
        B, S, C = z_line.shape

        u = self.projection(u.view(B, C, S)).view(B, S, C)

        z_hat = torch.sigmoid(self.Z(z_line)).view(B, S, S)  # [B, S, S]
        # sns.heatmap(z_hat[0].cpu().detach().numpy())
        # plt.title('z  hat')
        # plt.show()

        u_line = torch.matmul(z_hat, u)  # [B, S, C]
        # sns.heatmap(u_line[0].cpu().detach().numpy())
        # plt.title('u line')
        # plt.show()

        _, m_line = self.spatial_attn(z_line)
        m_line = m_line.reshape(B, S, C)
        # sns.heatmap(m_line[0].cpu().detach().numpy())
        # plt.title('m line')
        # plt.show()

        u_hat = (u_line + m_line).view(B, S, C)
        # sns.heatmap(u_hat[0].cpu().detach().numpy())
        # plt.title('u  hat')
        # plt.show()

        return u_hat


class MiM_v2(nn.Module):
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.Parameter, nn.BatchNorm1d, MiM_block,
                          MiM_v2, T_Mamba, MambaBlock, TokenLearner, TokenFuser, SpatialAttention,
                          )):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.,
                 num_tokens=49):
        super(MiM_v2, self).__init__()

        self.mim_1 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=image_size**2, num_token=image_size ** 2)

        self.mim_2 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=image_size**2, num_token=5**2)

        self.mim_3 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=5**2, num_token=3**2)

        self.mim_4 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=3**2, num_token=1 ** 2)

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

        # torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        # torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)

        self.to_latent = nn.Identity()
        # Initialize weights as learnable parameters
        self.k_weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)  # Start with equal weights
        # self.apply(self.weight_init)
        self.embed = nn.Linear(channels, dim, bias=True)

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

        x_1 = self.embed(x_1)
        x_2 = self.embed(x_2)
        x_3 = self.embed(x_3)
        x_4 = self.embed(x_4)

        tm1_1, tm2_1, tm3_1, tm4_1 = self.mim_1(x_1, x_2, x_3, x_4)

        O_1 = self.WTF(tm1_1, tm2_1, tm3_1, tm4_1)

        tm1_2, tm2_2, tm3_2, tm4_2 = self.mim_2(tm1_1, tm2_1, tm3_1, tm4_1)
        #
        O_2 = self.WTF(tm1_2, tm2_2, tm3_2, tm4_2)
        #
        tm1_3, tm2_3, tm3_3, tm4_3 = self.mim_3(tm1_2, tm2_2, tm3_2, tm4_2)
        #
        O_3 = self.WTF(tm1_3, tm2_3, tm3_3, tm4_3)

        tm1_4, tm2_4, tm3_4, tm4_4 = self.mim_4(tm1_3, tm2_3, tm3_3, tm4_3)

        O_4 = self.WTF(tm1_4, tm2_4, tm3_4, tm4_4)

        return self.mlp_head(self.to_latent(O_1 + O_2 + O_3 + O_4))
