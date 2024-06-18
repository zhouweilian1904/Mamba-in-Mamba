import torch
from multiscan_v3 import snake_flatten
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
# import numpy as np


#
#
# class SpatialAttention(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(2, 1, kernel_size=1, stride=1),  # Use Conv1d for sequence data
#             nn.BatchNorm1d(1),
#             nn.SiLU()
#         )
#
#     def forward(self, x):
#         x = rearrange(x, 'b l c -> b c l')
#         max_features = torch.max(x, 1)[0].unsqueeze(1)
#         avg_features = torch.mean(x, 1).unsqueeze(1)
#         combined = torch.cat([max_features, avg_features], dim=1)  # Now [B, 2, L]
#         fmap = self.conv1d(combined)  # Apply convolution
#         m = torch.sigmoid(fmap)  # Sigmoid to get weights [B, 1, L]
#         m_hat = x * m  # Apply weights
#         output = torch.mean(m_hat, dim=2)  # Mean over sequence
#         return output, m_hat
#
#
# class TokenLearner(nn.Module):
#     def __init__(self, S) -> None:
#         super().__init__()
#         self.S = S
#         self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])
#
#     def forward(self, x):
#         B, L, C = x.shape
#         # print('x1', x.shape)
#         Z = torch.zeros(B, self.S, C, dtype=x.dtype, device=x.device)  # Ensure Z has the right type and device
#         for i in range(self.S):
#             Ai, _ = self.tokenizers[i](x)  # [B, C]
#             Z[:, i, :] = Ai
#         return Z
#
#
# class TokenFuser(nn.Module):
#     def __init__(self, C, S) -> None:
#         super().__init__()
#         self.projection = nn.Linear(S, S, bias=True)
#         self.Z = nn.Linear(C, S)
#         self.spatial_attn = SpatialAttention()
#         self.S = S
#
#     def forward(self, u, z_line):
#         B, S, C = z_line.shape
#         u = self.projection(u.view(B, C, S)).view(B, S, C)
#         z_hat = torch.sigmoid(self.Z(z_line)).view(B, S, S)  # [B, S, S]
#         u_line = torch.matmul(z_hat, u)  # [B, S, C]
#         _, m_line = self.spatial_attn(z_line)
#         m_line = m_line.reshape(B, S, C)
#         u_hat = (u_line + m_line).view(B, S, C)
#         return u_hat


class VisionEncoderMambaBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            dt_rank: int,
            dim_inner: int,
            d_state: int,
            num_tokens: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_tokens = num_tokens

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.forward_ssm = SSM(dim, dt_rank, dim_inner, d_state)
        self.backward_ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj3 = nn.Linear(2 * dim, dim)

        # Softplus
        self.softplus = nn.Softplus()
        # pooling sets
        self.adapool = nn.AdaptiveAvgPool1d(num_tokens)

        # # STL sets
        # self.U1 = nn.Parameter(torch.randn(1, num_tokens, dim),
        #                        requires_grad=True)
        # self.U2 = nn.Parameter(torch.randn(1, dim, dim),
        #                        requires_grad=True)
        # torch.nn.init.kaiming_normal_(self.U1)
        # torch.nn.init.kaiming_normal_(self.U2)
        # self.spatial_attn = SpatialAttention()
        # self.tokenfuser = TokenFuser(C=dim, S=num_tokens)

    # def forward(self, x: torch.Tensor):
    #     b, s, d = x.shape
    #
    #     # Skip connection
    #     skip = x
    #     skip = self.silu(self.adapool(rearrange(skip, 'b n d -> b d n')))
    #     skip = rearrange(skip, 'b d n -> b n d')
    #
    #     # Normalization
    #     x = self.norm(x)
    #
    #     # Split x into x1 and x2 with linears
    #     z = self.proj1(x)
    #     x = self.proj2(x)
    #
    #     # forward conv1d
    #     x1 = self.process_direction(
    #         x,
    #         self.forward_conv1d,
    #         self.forward_ssm,
    #     )
    #
    #     # backward conv1d
    #     x2 = self.process_direction(
    #         torch.flip(x, dims=[1]),
    #         self.backward_conv1d,
    #         self.backward_ssm,
    #     )
    #     x2 = torch.flip(x2, dims=[1])
    #
    #     _, x1 = self.spatial_attn(x1)
    #     _, x2 = self.spatial_attn(x2)
    #
    #     x1 = self.STL(rearrange(x1, 'b c l -> b l c'))
    #     x1 = self.silu(x1)
    #     x2 = self.STL(rearrange(x2, 'b c l -> b l c'))
    #     x2 = self.silu(x2)
    #
    #     # Activation
    #     z = self.adapool(rearrange(z, 'b n d -> b d n'))
    #     z = rearrange(z, 'b d n -> b n d')
    #     z = self.silu(z)
    #
    #     # Matmul
    #     x1 = self.tokenfuser(x1, z)
    #     x2 = self.tokenfuser(x2, z)
    #
    #     # Residual connection
    #     return x1 + x2 + skip

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x
        skip = self.adapool(rearrange(skip, 'b n d -> b d n'))
        skip = rearrange(skip, 'b d n -> b n d')

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z = self.proj1(x)
        x = self.proj2(x)

        # forward conv1d
        x1 = self.process_direction(
            x,
            self.forward_conv1d,
            self.forward_ssm,
        )
        x1 = self.adapool(rearrange(x1, 'b n d -> b d n'))
        x1 = rearrange(x1, 'b d n -> b n d')
        x1 = x1 * self.gaussian_decay_mask(x1, 'center', 'index').unsqueeze(-1)
        x1 = self.silu(x1)

        # backward conv1d
        x2 = self.process_direction(
            torch.flip(x, dims=[1]),
            self.backward_conv1d,
            self.backward_ssm,
        )
        x2 = torch.flip(x2, dims=[1])
        x2 = self.adapool(rearrange(x2, 'b n d -> b d n'))
        x2 = rearrange(x2, 'b d n -> b n d')
        x2 = x2 * self.gaussian_decay_mask(x2, 'center', 'index').unsqueeze(-1)
        x2 = self.silu(x2)

        # Activation
        z = self.adapool(rearrange(z, 'b n d -> b d n'))
        z = rearrange(z, 'b d n -> b n d')
        z = self.silu(z)

        # Matmul
        # x = torch.cat([x1, x2], dim=-1)
        # x = self.proj3(x)
        # x = x * self.gaussian_decay_mask(x, 'center', 'index').unsqueeze(-1)
        # x = self.silu(x)

        x1 = z * x1
        x2 = z * x2
        # x = z * x

        # Residual connection
        return x1 + x2 + skip

    # def STL(self, m_hat):
    #     U1 = self.U1.transpose(1, 2)
    #     U2 = self.U2
    #     A = torch.einsum('bij,bjk->bik', m_hat, U1).transpose(1, 2).softmax(dim=-1)
    #     V = torch.einsum('bij,bjk->bik', m_hat, U2)
    #     u = torch.einsum('bij,bjk->bik', A, V)
    #     return u

    @staticmethod
    def gaussian_decay_mask(sequence, type='center', method='index'):
        length = sequence.shape[1]
        # Automatically determine center and last index
        center_index = (length + 1) // 2
        last_index = length - 1

        # Select the reference based on the type
        if type == 'center':
            ref_index_or_vector = center_index if method == 'index' else sequence[:, center_index, :]
        elif type == 'last':
            ref_index_or_vector = last_index if method == 'index' else sequence[:, last_index, :]
        else:
            raise ValueError("Invalid type specified. Choose 'center' or 'last'.")

        if method == 'index':
            # Index-based Gaussian mask
            indices = torch.arange(length, dtype=torch.float32, device=sequence.device)
            sigma = torch.abs(indices - ref_index_or_vector).mean()  # Sigma calculation
            weights = torch.exp(-0.5 * ((indices - ref_index_or_vector) ** 2) / (sigma ** 2))
            weights /= weights.sum()
            weights = weights.repeat(sequence.size(0), 1)  # Repeat weights for batch dimension
        elif method == 'vector':
            # Vector-based Gaussian mask
            distances = torch.norm(sequence - ref_index_or_vector.unsqueeze(1), dim=2)
            sigma = distances.mean(dim=1, keepdim=True)
            weights = torch.exp(-0.5 * (distances / sigma) ** 2)
            weights = weights / weights.sum(dim=1, keepdim=True)
        else:
            raise ValueError("Invalid method specified. Choose 'index' or 'vector'.")

        return weights

    def process_direction(
            self,
            x: Tensor,
            conv1d: nn.Conv1d,
            ssm: SSM,
    ):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class T_Mamaba(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, emb_dropout
                 , seq_length, num_tokens):
        super(T_Mamaba, self).__init__()
        self.num_patches = seq_length
        self.patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.vim = VisionEncoderMambaBlock(dim=dim, dt_rank=dim, dim_inner=dim, d_state=dim, num_tokens=num_tokens)
        self.layers = nn.ModuleList()
        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                self.vim
            )
        self.norm = nn.LayerNorm(dim)
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

        x = x + self.pos_embedding[:, :self.num_patches]  # (100, seq_length, 64)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        x = self.tanh(x)

        return x


class MiM_block(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, emb_dropout,
                 seq_length, num_tokens):
        super(MiM_block, self).__init__()

        # Initialize Mamba_2 models
        self.T_mamba1 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                                 emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)
        # self.T_mamba2 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                          emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)
        # self.T_mamba3 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                          emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)
        # self.T_mamba4 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                          emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)

    def forward(self, x1, x2, x3, x4):
        # Get outputs and regressions from each transformed input
        out_1 = self.T_mamba1(x1)
        out_4 = self.T_mamba1(x4)
        out_2 = self.T_mamba1(x2)
        out_3 = self.T_mamba1(x3)

        return out_1, out_2, out_3, out_4


class MiM_v3(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Parameter, nn.BatchNorm1d, MiM_block,
                          MiM_v3, T_Mamaba, VisionEncoderMambaBlock)):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=64, depth=2, emb_dropout=0.):
        super(MiM_v3, self).__init__()
        self.mim_1 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                               emb_dropout=emb_dropout, seq_length=(image_size // patch_size) ** 2,
                               num_tokens=(image_size // patch_size) ** 2)
        # self.mim_2 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                        emb_dropout=emb_dropout, seq_length=7 ** 2, num_tokens=5 ** 2)
        # self.mim_3 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                        emb_dropout=emb_dropout, seq_length=5 ** 2, num_tokens=3 ** 2)
        # self.mim_4 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                        emb_dropout=emb_dropout, seq_length=3 ** 2, num_tokens=1 ** 2)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, channels),
            nn.LayerNorm(channels),
            Rearrange("b h w d -> b d h w"),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, num_classes)
        )

        self.to_latent = nn.Identity()
        self.k_weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)  # Start with equal weights

    def WMF(self, *o):
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
        x = x.squeeze(1)
        x = self.to_patch_embedding(x)
        x_1 = x
        x_4 = torch.rot90(x_1, k=-1, dims=(2, 3))
        x_2 = torch.flip(x_4, dims=[3])
        x_3 = torch.rot90(x_2, k=-1, dims=(2, 3))

        x_1 = snake_flatten(x_1)
        x_2 = snake_flatten(x_2)
        x_3 = snake_flatten(x_3)
        x_4 = snake_flatten(x_4)

        tm1_1, tm2_1, tm3_1, tm4_1 = self.mim_1(x_1, x_2, x_3, x_4)
        # print('tm',tm1_1.shape)

        O_1 = self.WMF(tm1_1, tm2_1, tm3_1, tm4_1)
        # O_1 = self.tanh(O_1)
        # print('tm', O_1.shape)

        # tm1_2, tm2_2, tm3_2, tm4_2 = self.mim_2(tm1_1, tm2_1, tm3_1, tm4_1)

        # O_2 = self.WMF(tm1_2, tm2_2, tm3_2, tm4_2)
        # O_2 = self.tanh(O_2)

        # tm1_3, tm2_3, tm3_3, tm4_3 = self.mim_3(tm1_2, tm2_2, tm3_2, tm4_2)
        #
        # O_3 = self.WMF(tm1_3, tm2_3, tm3_3, tm4_3)
        # O_3 = self.tanh(O_3)
        #
        # tm1_4, tm2_4, tm3_4, tm4_4 = self.mim_4(tm1_3, tm2_3, tm3_3, tm4_3)
        #
        # O_4 = self.WMF(tm1_4, tm2_4, tm3_4, tm4_4)
        # O_4 = self.tanh(O_4)

        return self.mlp_head(self.to_latent(O_1))
