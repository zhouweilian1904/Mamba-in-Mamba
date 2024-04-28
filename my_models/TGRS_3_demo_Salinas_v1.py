import math

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.nn.utils import weight_norm
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt2
import torch.nn.functional as F
import seaborn as sns
# import optuna
import visdom

viz = visdom.Visdom(env='plot', port=8097)
from torch.autograd import Variable
from torch.nn import init


# from datasets import add_random_mask, add_gaussian_noise

# Transformer classes---------------------------------------------------------------------------------------------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Recur_self_attn(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        # Projection layers
        self.cur_to_qkv = nn.Linear(dim, dim * heads * 3)
        self.pre_to_qkv = nn.Linear(dim, dim * heads * 3)
        self.proj = nn.Linear(dim * heads, dim)

        # Batch normalization layers
        self.bn1 = nn.LayerNorm(dim * heads)
        self.bn2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

        # Parameter initialization
        self.init_alpha()

    def init_alpha(self):
        # Initializing alpha values between 0 and 1
        self.alpha = nn.Parameter(torch.rand(3, self.dim * self.heads, requires_grad=True))
        nn.init.uniform_(self.alpha, 0, 1)

    def forward(self, cur_input_t, pre_hidden):
        b, n, _, h = *cur_input_t.shape, self.heads

        Q_cur, K_cur, V_cur = self.split_qkv(self.cur_to_qkv(cur_input_t), h)
        Q_pre, K_pre, V_pre = self.split_qkv(self.pre_to_qkv(pre_hidden), h)

        # Weighted sum of current and previous Q, K, V with sigmoid gating
        Q = torch.sigmoid(self.alpha[0]) * Q_cur + (1 - torch.sigmoid(self.alpha[0])) * Q_pre
        K = torch.sigmoid(self.alpha[1]) * K_cur + (1 - torch.sigmoid(self.alpha[1])) * K_pre
        V = torch.sigmoid(self.alpha[2]) * V_cur + (1 - torch.sigmoid(self.alpha[2])) * V_pre

        # Self attention calculation
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # sns.heatmap(attn[0,0,:,:].cpu().detach().numpy())
        # plt2.show()

        A_cur_t = attn @ V
        A_cur_t = self.bn1(A_cur_t)

        A_cur_t = rearrange(A_cur_t, 'b h n d -> b n (h d)')
        A_cur_t = self.proj(A_cur_t)
        A_cur_t = self.bn2(A_cur_t)
        A_cur_t = self.dropout(A_cur_t)

        # Residual connection
        A_cur_t += self.relu(cur_input_t)

        return A_cur_t, attn

    def split_qkv(self, x, h):
        """Splits Q, K, V from a single tensor."""
        qkv = rearrange(x, 'b n (qkv h d) -> qkv b h n d', h=h, qkv=3)
        return qkv[0], qkv[1], qkv[2]


class RTCell(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super(RTCell, self).__init__()
        self.attention = Recur_self_attn(dim, heads=heads)

        # Layer Normalizations
        self.layer_norm_cur = nn.LayerNorm(dim)
        self.layer_norm_pre = nn.LayerNorm(dim)
        self.layer_norm_o = nn.LayerNorm(dim)

        # Feed-forward network
        self.ffn = FeedForward(dim, ffn_dim, dropout=0.1)
        self.elu = nn.ELU()

    def forward(self, cur_input_t, pre_hidden=None):
        if pre_hidden is None:
            pre_hidden = torch.zeros_like(cur_input_t, device=cur_input_t.device)

        cur_input_t = self.layer_norm_cur(cur_input_t)
        pre_hidden = self.layer_norm_pre(pre_hidden)

        A_cur_t = self.attention(cur_input_t, pre_hidden)[0]

        cur_hidden = pre_hidden + A_cur_t

        O_hat = self.layer_norm_o(cur_input_t + A_cur_t)

        O_cur_t = self.ffn(O_hat) + self.elu(O_hat)

        return O_cur_t, cur_hidden


class RecursiveTransformer(nn.Module):
    def __init__(self, dim, ffn_dim, depth, heads):
        super(RecursiveTransformer, self).__init__()
        self.cells = nn.ModuleList([RTCell(dim, ffn_dim, heads) for _ in range(depth)])

    def forward(self, sequence, H_pre=None):
        batch_size, n, _, d = sequence.size()

        # If hidden state not provided, initialize
        if H_pre is None:
            H_pre = [nn.Parameter(torch.zeros(batch_size, sequence.size(2), sequence.size(3), device=sequence.device),
                                  requires_grad=True) for _ in range(len(self.cells))]

        outputs = []

        for t in range(n):
            O_t = sequence[:, t, :, :]
            for l, cell in enumerate(self.cells):
                O_t, H_pre[l] = cell(O_t, H_pre[l])
            outputs.append(O_t)

        out = torch.stack(outputs, dim=1)
        return out


class Sub_band_with_group_cnn(nn.Module):
    def __init__(self, dim=10, dropout=0.1):
        super().__init__()

        def create_cnn_block(in_channels, out_channels=dim):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=(1, 1), bias=True),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=(1, 1), bias=True),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(),
                # nn.Dropout(dropout)
            )

        self.group_ranges = [
            (0, 39, 40),
            (40, 79, 40),
            (80, 104, 25),
            (105, 144, 40),
            (145, 199, 55)
        ]

        self.cnns = nn.ModuleList([create_cnn_block(in_channels=size) for _, _, size in self.group_ranges])

        self.depth_cnn = nn.Conv3d(in_channels=1, out_channels=dim, kernel_size=(dim, 3, 3), padding=(0, 1, 1),
                                   stride=(dim, 1, 1))

    def forward(self, image_tensor):
        results = []

        for (start_band, end_band, _), cnn in zip(self.group_ranges, self.cnns):
            sub_tensor = image_tensor[:, start_band:end_band + 1, :, :]
            results.append(cnn(sub_tensor))

        x = torch.cat(results, dim=1)  # Concatenate along channel dimension

        # x_show = x.cpu().detach().numpy()  # Assuming x is a PyTorch tensor
        # y_values = x_show[0, :, 4, 4]  # Extract the values you want to plot
        # x_values = np.arange(y_values.size)  # Create an array of indices with the same size
        # plt2.scatter(x=x_values, y=y_values, c="red", edgecolors="red")  # Plot the values
        # plt2.plot(y_values)
        # # Calculate the polynomial fit (degree 3 for a smoother curve)
        # z = np.polyfit(x_values, y_values, 5)
        # p = np.poly1d(z)
        # # Plot the smooth curve
        # plt2.plot(x_values, p(x_values), "g--", linewidth=3)  # 'b--' indicates a blue dashed line
        # plt2.show()

        # x = self.cnn(x)
        x = x.unsqueeze(1)
        # print('x1', x.shape)
        x = self.depth_cnn(x)
        # print('x2', x.shape)
        x = rearrange(x, 'b c d h w -> b (c d) h w')

        # viz.line(y_values, x_values)

        return x


# class Sub_band_with_group_cnn(nn.Module):
#     def __init__(self, dim=64, dropout = 0.1):
#         super().__init__()
#         self.dim = dim
#         self.depth_cnn = nn.Conv2d(in_channels=dim*5, out_channels=dim*5, kernel_size=3, padding=(1,1), stride=1, groups= 5)
#
#         # Define the band groups with 0-based indexing
#         self.cnn_1 = nn.Sequential(
#             nn.Conv2d(40, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         self.cnn_2 = nn.Sequential(
#             nn.Conv2d(40, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         self.cnn_3 = nn.Sequential(
#             nn.Conv2d(25, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         self.cnn_4 = nn.Sequential(
#             nn.Conv2d(40, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         self.cnn_5 = nn.Sequential(
#             nn.Conv2d(59, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=(1, 1), bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         self.group_ranges = {
#             "g1": (0, 39),
#             "g2": (40, 79),
#             "g3": (80, 104),
#             "g4": (105, 144),
#             "g5": (145, 203)
#         }
#
#     def forward(self, image_tensor):
#         grouped_subtensors = {}
#
#         for group, (start_band, end_band) in self.group_ranges.items():
#             sub_tensor = image_tensor[:, start_band:end_band + 1, :, :]
#             grouped_subtensors[group] = sub_tensor
#
#         result_1 = self.cnn_1(grouped_subtensors["g1"])
#         result_2 = self.cnn_2(grouped_subtensors["g2"])
#         result_3 = self.cnn_3(grouped_subtensors["g3"])
#         result_4 = self.cnn_4(grouped_subtensors["g4"])
#         result_5 = self.cnn_5(grouped_subtensors["g5"])
#
#         x = torch.cat([result_1, result_2, result_3, result_4, result_5], dim=1) #(batch, dim*5, height, width)
#         x = self.depth_cnn(x)
#
#         return x


class Short_term_RT(nn.Module):
    def __init__(self, num=9, dim=16, depth=4, heads=4):
        super(Short_term_RT, self).__init__()

        self.num = num
        # Create a list (or ModuleList) of RecursiveTransformer instances
        self.short_term_RTs = nn.ModuleList(
            [RecursiveTransformer(dim=dim, ffn_dim=dim * 2, depth=depth, heads=heads) for _ in range(num)])

    def process_subsequence(self, sub_seq, idx):
        """Process a single subsequence using its corresponding RecursiveTransformer and return its last output."""
        output = self.short_term_RTs[idx](sub_seq)
        return output[:, -1, :, :]

    def forward(self, x):  # x shape: (100, 180, 25, 16)
        sub_sequences = torch.chunk(x, chunks=self.num, dim=1)  # Each sub_seq shape: (100, 20, 25, 16)
        reshaped_sub_images = [self.process_subsequence(sub_seq, idx) for idx, sub_seq in enumerate(sub_sequences)]

        x_out = torch.stack(reshaped_sub_images, dim=1)  # x_out shape: (100, 9, 25, 16)
        return x_out


class Long_term_RT(nn.Module):
    def __init__(self, dim=25, depth=4, heads=4):
        super(Long_term_RT, self).__init__()

        self.long_term_RT = RecursiveTransformer(dim=dim, ffn_dim=dim * 4, depth=depth, heads=heads)

    def forward(self, x_from_short_term):  # Expected input shape: (100, 9, 25, 16)
        # Rearrange the tensor dimensions. Assuming:
        # b = batch size, n = number of sequences,
        # p1 = original feature dimension, p2 = original sequence length
        # Transforming it to: b = batch size, n = number of sequences,
        # p2 = new feature dimension, p1 = new sequence length
        x_from_short_term = rearrange(x_from_short_term, 'b n p1 p2 -> b n p2 p1')

        x_out = self.long_term_RT(x_from_short_term)
        return x_out


# Houston_2013 = (1905, 349, 144)
# Houston_2018 = (601, 2384, 48)
# Indian_Pine = (145, 145, 200)
# Pavia_U = (610, 340, 100)
Salinas = (512, 217, 200)

subimage_patch_size = (5, 5, 32)
stride = (5, 5, 32)
padding = (0, 0, 0)
frame_patch_size = subimage_patch_size[2]  # HU:12


class RVIT_Salinas(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.Conv2d, nn.LSTM, nn.Parameter,
                          nn.BatchNorm1d, nn.LayerNorm, nn.Dropout, RecursiveTransformer)):
            init.kaiming_normal_(m.weight.data)
            init.zeros_(m.bias.data)

    def __init__(self, n_bands, n_classes, patch_size, batch_size=100, sub_band_groups=5,
                 subimage_patch_size=subimage_patch_size,
                 padding=padding, stride=stride, dim=64, depth=1, heads=1, dropout=0.1):
        super(RVIT_Salinas, self).__init__()
        self.patch_size = patch_size
        self.kernel_size = subimage_patch_size
        self.padding = padding
        self.batch_szie = batch_size
        self.stride = stride
        image_height, image_width = pair(self.patch_size)
        image_frame = dim * sub_band_groups
        self.num_h = int((image_height - self.kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
        self.num_w = int((image_width - self.kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
        self.num_f = int((image_frame - self.kernel_size[2] + 2 * padding[2]) / stride[2]) + 1
        self.num_spatial = self.num_h * self.num_w  ## (spatial aspect)
        self.num_patches = self.num_h * self.num_w * self.num_f  # spatial * spectral aspect
        self.patch_dim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        print('Model: RVIT_Salinas_demo')
        print('num_w:', self.num_h, 'num_w:', self.num_w, 'num_f:',
              self.num_f, 'num_spa:', self.num_spatial, 'num_patches:', self.num_patches, 'sub_patch_dim:',
              self.patch_dim)
        self.layernorm_s = nn.LayerNorm(
            [self.num_spatial, self.kernel_size[0] * self.kernel_size[1], self.kernel_size[2]])
        self.layernorm_f = nn.LayerNorm(
            [self.num_spatial, self.kernel_size[2], self.kernel_size[0] * self.kernel_size[1]])
        self.layernorm_r = nn.LayerNorm(
            [self.num_spatial, self.kernel_size[2], self.kernel_size[0] * self.kernel_size[1]])
        self.sub_band_with_group_cnn = Sub_band_with_group_cnn(dim=dim, dropout=dropout)
        self.short_term_RT = Short_term_RT(num=self.num_spatial, dim=self.kernel_size[2], depth=depth, heads=heads)
        self.long_term_RT_f = Long_term_RT(dim=self.kernel_size[0] * self.kernel_size[1], depth=depth, heads=heads)
        self.long_term_RT_r = Long_term_RT(dim=self.kernel_size[0] * self.kernel_size[1], depth=depth, heads=heads)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(self.kernel_size[0] * self.kernel_size[1] * 2 * self.kernel_size[2]),
            nn.Linear(self.kernel_size[0] * self.kernel_size[1] * 2 * self.kernel_size[2],
                      self.kernel_size[0] * self.kernel_size[1] * 4),
            nn.Dropout(dropout),
            nn.Linear(self.kernel_size[0] * self.kernel_size[1] * 4, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim, n_classes)
        )
        self.softmax = nn.Softmax()

        self.aux_loss_weight = 1

        self.reg = nn.Sequential(
            nn.BatchNorm1d(self.kernel_size[0] * self.kernel_size[1] * 2 * self.kernel_size[2]),
            nn.Linear(self.kernel_size[0] * self.kernel_size[1] * 2 * self.kernel_size[2],
                      self.kernel_size[0] * self.kernel_size[1] * 4),
            nn.Dropout(dropout),
            nn.Linear(self.kernel_size[0] * self.kernel_size[1] * 4, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim, n_bands, bias=True)
        )
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, patch_size ** 2, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

    def tokenizer(self, x):  # x(b,n,d)
        print('---------------------------------------tokenizer-----------------------------------------')
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        return T

    def partition_3D(self, image, kernel_size, padding, stride):
        """
        Function to partition a 3D image into smaller patches.

        Args:
            image (Tensor): Input image of shape (batch_size, channels, height, width).
            kernel_size (tuple): Size of the patches to create, in the format (height, width, depth).
            padding (tuple): Amount of padding to apply, in the format (height, width, depth).
            stride (tuple): Stride for the patch extraction, in the format (height, width, depth).

        Returns:
            Tensor: Output tensor after partitioning.
        """

        # Check the dimension of the input image
        assert image.dim() == 4, "Input image must be a 4D tensor of shape (batch_size, channels, height, width)."

        def partition_along_dimension(tensor, kernel, padding, stride):
            return nn.functional.unfold(tensor, kernel_size=kernel, padding=padding, stride=stride)

        kernel_size_h_w, kernel_size_d = (kernel_size[0], kernel_size[1]), kernel_size[2]
        padding_h_w, padding_d = (padding[0], padding[1]), padding[2]
        stride_h_w, stride_d = (stride[0], stride[1]), stride[2]

        # Partition along height and width
        trans2D = partition_along_dimension(image, kernel_size_h_w, padding_h_w, stride_h_w)
        num_patches = trans2D.shape[-1]
        trans2D = rearrange(trans2D, 'b (c p1 p2) n -> b n (p1 p2) c', p1=kernel_size[0], p2=kernel_size[1])

        # Partition along depth
        trans3D = partition_along_dimension(trans2D, (kernel_size[0] * kernel_size[1], kernel_size_d), (0, padding_d),
                                            stride_d)
        trans3D = rearrange(trans3D, 'b (n2D p p3) n3D -> b (n2D n3D) p p3 ', p=kernel_size[0] * kernel_size[1],
                            p3=kernel_size[2], n2D=num_patches)

        return trans3D

    def forward(self, x):
        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        x = x.squeeze(1)

        x = self.sub_band_with_group_cnn(x)  # (100, 320, 15 15)
        # x_reg_out = x
        # print('x.size', x.shape)
        x = self.partition_3D(x, self.kernel_size, self.padding, self.stride)
        # print('x.size', x.shape)  # (100, 500, 9, 16)
        x = self.short_term_RT(x)
        # print('x.size', x.shape)  # (100, 500, 9, 16)
        x = self.layernorm_s(x)
        x_forward = self.long_term_RT_f(x)  # (100, 9, 16, 25)
        x_forward = self.layernorm_f(x_forward)
        x_reverse = self.long_term_RT_r(torch.flip(x, dims=[1]))
        x_reverse = self.layernorm_r(x_reverse)
        x = torch.cat([x_forward, torch.flip(x_reverse, dims=[1])], dim=3)  # fixed the reversal
        x_cls = self.to_cls_token(reduce(x, 'b n p1 p2 -> b p1 p2', reduction='mean'))
        x_mlp = x_cls.view(x_cls.size(0), -1)
        x_cls_out = self.mlp_head(x_mlp)  # renamed for clarity
        x_reg_out = self.reg(x_mlp)  # renamed for clarity
        return x_cls_out, x_reg_out

    #
    #
    #     'generate different directions x_dir_1 and x_dir_3, x_dir_6 and x_dir_8, 5 and 7, 4 and 2'
    #     x_dir_1 = x
    #     x_dir_6 = torch.rot90(x_dir_1, 1, [2, 3])
    #     x_dir_3 = torch.rot90(x_dir_6, 1, [2, 3])
    #     x_dir_8 = torch.rot90(x_dir_3, 1, [2, 3])
    #     x_dir_7 = torch.flipud(x_dir_1)
    #     x_dir_2 = torch.flipud(x_dir_6)
    #     x_dir_5 = torch.flipud(x_dir_3)
    #     x_dir_4 = torch.flipud(x_dir_8)
    #
    #     out_1, reg_1 = self.one_direction_operation(x_dir_1)
    #     # out_1 = F.normalize(out_1, p=2, dim=1)
    #     # reg_1 = F.normalize(reg_1, p=2, dim=1)
    #     # out_2, reg_2= self.one_direction_operation(x_dir_2)
    #     out_3, reg_3 = self.one_direction_operation(x_dir_3)
    # #     out_4 = self.ond_dir_operation(x_dir_4)
    # #     out_5, reg_5 = self.one_direction_operation(x_dir_5)
    # #     out_6, reg_6 = self.one_direction_operation(x_dir_6)
    # #     out_7, reg_7 = self.one_direction_operation(x_dir_7)
    # #     out_8, reg_8 = self.one_direction_operation(x_dir_8)
