# -*- coding: utf-8 -*-
# Torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import visdom

vis = visdom.Visdom()
sns.set()
sns.set_style('darkgrid')
sns.set_context('notebook')
from torch.nn import init
import visdom

viz = visdom.Visdom(port=8097)
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math, copy, time
# from sklearn.cluster import SpectralClustering
# from torchvision.transforms.autoaugment import AutoAugment
# import torchvision
# from scipy.special import softmax
# from VitNew4_3d_learned_absolute_pos_encoding import multiscan


# def _weights_init(m):
#     classname = m.__class__.__name__
#     #print(classname)
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
#         init.kaiming_normal_(m.weight)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class Residual(nn.Module):
    def __init__(self, fn, dim, dropout):
        super(Residual, self).__init__()
        self.fn = fn
        self.norm = PreNorm(dim, fn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        return self.fn(self.dropout(self.norm(x)), **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-6):
        super(PreNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(dim))
        self.b_2 = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.fn = fn

    def forward(self, x, **kwargs):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.fn(self.a_2 * (x - mean) / (std + self.eps) + self.b_2, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            weight_norm(nn.Linear(dim, hidden_dim, bias=True)),
            nn.GELU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hidden_dim, dim, bias=True)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SMSA(nn.Module):
    def __init__(self, dim, heads=1, dim_head=128, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = weight_norm(nn.Linear(dim, inner_dim * 3, bias=True))
        self.to_out = FeedForward(inner_dim, dim, dropout) if project_out else nn.Identity()

    def spe_mask(self, x):
        b, n, d = x.shape
        spe_pairwise_distances = torch.cdist(x, x, p=2)
        sigma_spe = torch.mean(spe_pairwise_distances)  # adjust this parameter to control the scale of the weights
        spe_pairwise_weights = torch.exp(-spe_pairwise_distances / (2 * sigma_spe ** 2))
        # for b in range(spe_pairwise_weights.size(0)):
        #     if b == 1:
        #          plt.imshow(spe_pairwise_weights[1, :,:].cpu().detach().numpy())
        #          plt.title('batch 1')
        #          plt.show()
        #     elif b == 2:
        #         plt.imshow(spe_pairwise_weights[2,:,:].cpu().detach().numpy())
        #         plt.title('batch 2')
        #         plt.show()
        return spe_pairwise_weights.cuda()

    def spa_mask(self, x):
        b, n, d = x.shape
        x = rearrange(x, 'b (h w) d -> b h w d', h=int(n ** (1 / 2)), w=int(n ** (1 / 2)))
        b, h, w, c = x.shape
        x, y = torch.meshgrid(torch.arange(h), torch.arange(w))
        xy_grid = torch.stack((x, y), dim=-1).float()
        xy_grid = xy_grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
        # calculate coordinate difference
        coord_diff = xy_grid.view(b, -1, 1, 2) - xy_grid.view(b, 1, -1, 2)
        # calculate pairwise distances
        spa_pairwise_distances = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
        # transform distances into weights
        sigma_spa = torch.mean(spa_pairwise_distances)  # adjust this parameter to control the scale of the weights
        spa_pairwise_weights = torch.exp(-spa_pairwise_distances / (2 * sigma_spa ** 2))
        # reshape weights into a matrix of pairwise weights
        spa_pairwise_weights = spa_pairwise_weights.view(b, -1, h * w)
        return spa_pairwise_weights.cuda()

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        spe_mask = self.spe_mask(x=x)
        spe_mask = repeat(spe_mask, 'b n1 n2 -> b h n1 n2', h=self.heads, b=b)

        spa_mask = self.spa_mask(x=x)
        spa_mask = repeat(spa_mask, 'b n1 n2 -> b h n1 n2', h=self.heads, b=b)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        dots = dots * spe_mask * spa_mask  # acc:94.688, flip_aug:95.295
        # dots = dots * spe_mask #acc:93.561%
        # dots = dots * spa_mask  # acc: 93.583%

        attn = self.softmax(dots)
        attn = self.dropout(attn)

        # out = torch.matmul(attn, v)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        b, h, n, d = out.shape
        out = rearrange(out, 'b h n d -> b n (h d)', b=b, h=h, n=n, d=d) #(b, n, d)
        # print('out', out.shape)
        return self.to_out(out)


'------------RT-------------------------------------------'
class RT(nn.Module):
    def __init__(self, dim, patch_size, depth, heads, dim_head, mlp_dim, rnn_layers=1, dropout=0.2, ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.dim = dim
        self.gamma = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(dim)
        self.patch_size = patch_size
        self.mlp = Residual(FeedForward(dim, mlp_dim, dropout=dropout), dim=dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.reducedim = nn.Linear(dim * 2, dim)
        self.batchnorm = nn.BatchNorm1d(dim)
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, patch_size**2, self.dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, nn.LSTM(input_size=dim, hidden_size=dim,
                                              num_layers=rnn_layers, dropout=dropout, batch_first=True, bias=True,bidirectional=False)), dim=dim, dropout=dropout),
                Residual(PreNorm(dim, SMSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)), dim=dim,
                         dropout=dropout),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)), dim=dim, dropout=dropout)
            ]))

    def tokenizer(self, x): #x(b,n,d)
        print('---------------------------------------tokenizer-----------------------------------------')
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        return T

    def midlayers(self, x):
        b, n, d = x.shape
        for rnn, smsa, ff in self.layers:
            gamma = self.sigmoid(self.gamma)
            delta = 1 - gamma
            # x = self.tokenizer(x)
            p_rnn = rnn(x)  # you set as LSTM
            # p_rnn = self.dropout(torch.nn.functional.softmax(p_rnn)) * x + x
            s_tilde = ff(gamma * x + delta * p_rnn)
            # s_tilde = self.tokenizer(s_tilde)
            f_smsa = smsa(s_tilde)
            # f_smsa = self.dropout(torch.nn.functional.softmax(f_smsa)) * s_tilde + s_tilde
            t_out = ff(f_smsa)
            t_tilde = ff(t_out)
            x = self.layernorm(t_out + t_tilde)
            # x = ff(x)
        return x

    def forward(self, x):
        x = self.midlayers(x)
        out = self.mlp(x)
        return out


class Attention2(nn.Module):
    def __init__(self, dim, heads=1, dim_head=128, dropout=0.2):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = weight_norm(nn.Linear(dim, inner_dim * 3, bias=True))

        self.to_out = nn.Sequential(
            weight_norm(nn.Linear(inner_dim, dim, bias=True)),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        c = int((n-1)/2)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.attend(dots)
        # print('attn shape', attn.shape)
        # for b in range(attn.size(0)):
        #     if b == 1:
        #         plt.plot(attn[1, 0, c, :].cpu().detach().numpy(), marker='o')
            # elif b == 2:
            #     plt.plot(attn[2, 0, c, :].cpu().detach().numpy(), marker='x')
            #     plt.show()
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FT(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention2(dim, heads=heads, dim_head=dim_head, dropout=dropout)), dim=dim,
                         dropout=dropout),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)), dim=dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class multiTrans(nn.Module):
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.Conv2d, nn.LSTM, nn.Parameter, nn.BatchNorm1d,)):
            init.kaiming_normal_(m.weight.data)
            init.kaiming_normal_(m.bias.data )

    def __init__(self, input_channels, n_classes, patch_size=7, sub_patch_size=(1, 1), dilation=1, padding=0, stride=1,
                 emb_size=64, pool='cls', emb_dropout=0.1): #pool=cls,patch=9,dropout=0.2,emb size=64,pca=30,OA=92.912%
        #'reduce emb_size -> accuracy decrease'
        #
        super(multiTrans, self).__init__()
        self.sub_patch_size = sub_patch_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.emb_dropout = emb_dropout
        image_height, image_width = pair(patch_size)
        num_h = int((image_height - sub_patch_size[0] + 2 * padding) / stride) + 1
        num_w = int((image_width - sub_patch_size[1] + 2 * padding) / stride) + 1
        num_steps = num_h * num_w
        self.num_steps = num_steps
        patch_dim = input_channels * sub_patch_size[0] * sub_patch_size[1]
        # patch_height, patch_width = pair(sub_patch_size)
        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # num_steps = (image_height // patch_height) * (image_width // patch_width)

        # patch_dim = input_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self._overlapped_patch_2D = nn.Unfold(sub_patch_size, dilation, padding, stride)
        self.to_patch_embedding = nn.Sequential(
            # nn.BatchNorm2d(input_channels),
            Rearrange('b (c p1 p2) n -> b n (p1 p2 c)', p1=sub_patch_size[0], p2=sub_patch_size[1]),
            nn.LayerNorm(patch_dim),
            weight_norm(nn.Linear(patch_dim, emb_size, bias=True)),
            nn.LayerNorm(emb_size),
        )
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.pre_emd = nn.Linear(emb_size, emb_size, bias=True)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(emb_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(emb_size)))
        self.point_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3)
        self.depth_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,
                                      groups=input_channels)
        self.transformer_FT = FT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=False, mlp_dim=emb_size)
        self.rt_1 = RT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=self.emb_dropout, mlp_dim=emb_size,
                       patch_size=patch_size)
        self.rt_2 = RT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=self.emb_dropout, mlp_dim=emb_size,
                       patch_size=patch_size)
        self.rt_3 = RT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=self.emb_dropout, mlp_dim=emb_size,
                       patch_size=patch_size)
        self.rt_4 = RT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=self.emb_dropout, mlp_dim=emb_size,
                       patch_size=patch_size)
        self.rt_5 = RT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=self.emb_dropout, mlp_dim=emb_size,
                       patch_size=patch_size)
        self.rt_6 = RT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=self.emb_dropout, mlp_dim=emb_size,
                       patch_size=patch_size)
        self.rt_7 = RT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=self.emb_dropout, mlp_dim=emb_size,
                       patch_size=patch_size)
        self.rt_8 = RT(dim=emb_size, heads=1, depth=2, dim_head=emb_size, dropout=self.emb_dropout, mlp_dim=emb_size,
                       patch_size=patch_size)
        # self.svt = FT(dim=patch_size**2, heads=1,depth=1, dim_head=patch_size**2, dropout=self.emb_dropout, mlp_dim=patch_size**2)
        self.pos_embedding_RT = nn.Parameter(torch.randn(1, num_steps, emb_size), requires_grad=True)
        self.pos_embedding_FT = nn.Parameter(torch.randn(1, 8 + 1, emb_size), requires_grad=True)
        self.class_token = nn.Parameter(torch.randn(1, 1, emb_size), requires_grad=True)
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.to_cls_token = nn.Identity()
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(emb_size),
            nn.LayerNorm(emb_size),
            nn.GELU(),
            nn.Dropout(emb_dropout),
            nn.Linear(emb_size, int(emb_size / 2), bias=True),
            nn.BatchNorm1d(int(emb_size / 2)),
            nn.LayerNorm(int(emb_size / 2)),
            nn.GELU(),
            nn.Dropout(emb_dropout),
            nn.Linear(int(emb_size / 2), n_classes, bias=True)
        )

        self.reg = nn.Sequential(
            nn.BatchNorm1d(emb_size),
            nn.LayerNorm(emb_size),
            nn.GELU(),
            nn.Dropout(emb_dropout),
            nn.Linear(emb_size, int(input_channels / 2), bias=True),
            nn.Linear(int(input_channels / 2), input_channels, bias=True)
        )

        self.bn = nn.BatchNorm1d(emb_size * (9))
        self.fc = nn.Linear(emb_size, n_classes, bias=True)
        self.aux_loss_weight = 1

    # trans后的维度为（batchsize,(c Kernelsize[1] Kernelsize[2]),patchNum)
    # def _overlapped_patch_2D(self, x):
    #     trans = nn.functional.unfold(x, self.sub_patch_size, self.dilation, self.padding, self.stride)
    #     print('transformed image', trans.shape)
    #     return trans

    def feature_selection(self, input, type):
        if type == 'RT':
            _, n, d = input.shape
            c = int((n - 1) / 2)
            f_c = input[:, c, :]  # (64)
            f_other = input[:, :, :]
            # a_m = (f_other * f_c.unsqueeze(1)).sum(dim=-1)
            a_m = torch.matmul(f_other, f_c.unsqueeze(-1)).squeeze(-1) #torch.Size([b, n])
            # print('a_m', a_m.shape)
            a_m = F.softmax(a_m, dim=1)
            # print('d1 shape',d.shape)
            # for b in range(a_m.size(0)):
            #     if b == 1:
            #         plt.plot(a_m[1, :].cpu().detach().numpy(), marker='o')
            #     elif b == 2:
            #         plt.plot(a_m[2, :].cpu().detach().numpy(), marker='x')
            #         plt.show()
            result = input * a_m.unsqueeze(-1)
        elif type == 'FT':
            _, n, d = input.shape
            e_0 = input[:, 0, :]  # (64)
            e_other = input[:, :, :]
            # e_m = (e_other * e_0.unsqueeze(1)).sum(dim=-1)
            e_m = torch.matmul(e_other, e_0.unsqueeze(-1)).squeeze(-1) #(b, s)
            # print('e_m', e_m.shape)
            e_m = F.softmax(e_m, dim=1)
            # print('d1 shape',d.shape)
            # for b in range(e_m.size(0)):
            #     if b == 1:
            #         plt.plot(e_m[1, :].cpu().detach().numpy(), marker='o')
            #     elif b == 2:
            #         plt.plot(e_m[2, :].cpu().detach().numpy(), marker='x')
            #         plt.show()
            result = input * e_m.unsqueeze(-1)
        else:
            print('choose the type of feature selection, #type = string#')
        return result.cuda()

    def tokenizer(self, x): #x(b,n,d)
        print('---------------------------------------tokenizer-----------------------------------------')
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        return T

    def forward(self, x):  # original x (b 1 c h w)
        x = x.squeeze(1)# b, c, h, w = x.shape
        # x = self.point_conv_1(x) + self.depth_conv_1(x)
        # x = self.point_conv_1(x) + self.depth_conv_1(x)
        x = self._overlapped_patch_2D(x) #b d n

        x = self.to_patch_embedding(x)  # (b,n,d)
        num_pacth = x.size(1)
        x = rearrange(x, 'b (h w) d -> b d h w', h=int((num_pacth) ** (1 / 2)), w=int((num_pacth) ** (1 / 2))) #(b d h w)
        # print('After patch embedding', x.shape)
        'U-Turn scanning manner'
        # x = rearrange(x,'b c h w -> b h w c')
        # x1r = multiscan(x, 1).cuda()
        # x2r = multiscan(x, 2).cuda()
        # x3r = multiscan(x, 3).cuda()
        # x4r = multiscan(x, 4).cuda()
        # x5r = multiscan(x, 5).cuda()
        # x6r = multiscan(x, 6).cuda()
        # x7r = multiscan(x, 7).cuda()
        # x8r = multiscan(x, 8).cuda()  # (b c s)
        'General scanning manner'
        x1 = x
        x1r = x1.reshape(x1.shape[0], x1.shape[1], -1)

        x2 = x1r.cpu()
        x2rn = np.flip(x2.detach().numpy(), axis=2).copy()
        x2rt = torch.from_numpy(x2rn)
        x2r = x2rt.cuda()

        x3 = torch.transpose(x1, 2, 3)
        x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)

        x4 = x3r.cpu()
        x4rn = np.flip(x4.detach().numpy(), axis=2).copy()
        x4rt = torch.from_numpy(x4rn)
        x4r = x4rt.cuda()

        x5 = torch.rot90(x1, 1, (2, 3))
        x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)

        x6 = x5r.cpu()
        x6rn = np.flip(x6.detach().numpy(), axis=2).copy()
        x6rt = torch.from_numpy(x6rn)
        x6r = x6rt.cuda()

        x7 = torch.transpose(x5, 2, 3)
        x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)

        x8 = x7r.cpu()
        x8rn = np.flip(x8.detach().numpy(), axis=2).copy()
        x8rt = torch.from_numpy(x8rn)
        x8r = x8rt.cuda()# batch, fea dim, steps

        x1r = self.pre_emd(rearrange(x1r, 'b d n -> b n d'))
        x2r = self.pre_emd(rearrange(x2r, 'b d n -> b n d'))
        x3r = self.pre_emd(rearrange(x3r, 'b d n -> b n d'))
        x4r = self.pre_emd(rearrange(x4r, 'b d n -> b n d'))
        x5r = self.pre_emd(rearrange(x5r, 'b d n -> b n d'))
        x6r = self.pre_emd(rearrange(x6r, 'b d n -> b n d'))
        x7r = self.pre_emd(rearrange(x7r, 'b d n -> b n d'))
        x8r = self.pre_emd(rearrange(x8r, 'b d n -> b n d')) #(b n d)

        '----step one: Pixel Transformer----'
        x1r_out = self.feature_selection(input=self.rt_1(x1r),type='RT')
        '-----------------------'
        x2r_out = self.feature_selection(input=self.rt_2(x2r),type='RT')
        '-----------------------'
        x3r_out = self.feature_selection(input=self.rt_3(x3r),type='RT')
        '-----------------------'
        x4r_out = self.feature_selection(input=self.rt_4(x4r),type='RT')
        '-----------------------'
        x5r_out = self.feature_selection(input=self.rt_5(x5r),type='RT')
        '-----------------------'
        x6r_out = self.feature_selection(input=self.rt_6(x6r),type='RT')
        '-----------------------'
        x7r_out = self.feature_selection(input=self.rt_7(x7r),type='RT')
        '-----------------------'
        x8r_out = self.feature_selection(input=self.rt_8(x8r),type='RT')
        '-----------------------'

        '----step two: fusion Transformer----'
        #deploy central pixel (central step)#
        # c = int((x8r_out.size(1) - 1) / 2)
        # x1_c = repeat(x1r_out[:,c,:], 'b e -> b e ()')
        # x2_c = repeat(x2r_out[:,c,:], 'b e -> b e ()')
        # x3_c = repeat(x3r_out[:,c,:], 'b e -> b e ()')
        # x4_c = repeat(x4r_out[:,c,:], 'b e -> b e ()')
        # x5_c = repeat(x5r_out[:,c,:], 'b e -> b e ()')
        # x6_c = repeat(x6r_out[:,c,:], 'b e -> b e ()')
        # x7_c = repeat(x7r_out[:,c,:], 'b e -> b e ()')
        # x8_c = repeat(x8r_out[:,c,:], 'b e -> b e ()')

        #deploy mean of all steps#
        x1_c = repeat(x1r_out.mean(dim=1), 'b e -> b e ()')
        x2_c = repeat(x2r_out.mean(dim=1), 'b e -> b e ()')
        x3_c = repeat(x3r_out.mean(dim=1), 'b e -> b e ()')
        x4_c = repeat(x4r_out.mean(dim=1), 'b e -> b e ()')
        x5_c = repeat(x5r_out.mean(dim=1), 'b e -> b e ()')
        x6_c = repeat(x6r_out.mean(dim=1), 'b e -> b e ()')
        x7_c = repeat(x7r_out.mean(dim=1), 'b e -> b e ()')
        x8_c = repeat(x8r_out.mean(dim=1), 'b e -> b e ()')

        fusion_fea_center = torch.cat([x1_c, x2_c, x3_c, x4_c, x5_c, x6_c, x7_c, x8_c], dim=2) #torch.Size([100, 64, 8])
        # print('fusion_fea_center', fusion_fea_center.shape)
        fusion_fea_center = rearrange(fusion_fea_center, 'b e n -> b n e')  # (b,8,d)
        b, n, e = fusion_fea_center.shape
        # cls_tokens = repeat(self.class_token, '() n d -> b n d', b = fusion_fea_center.size(0)) #[b,1,dim]

        cls_tokens = self.class_token.expand(fusion_fea_center.shape[0], -1, -1)  # [b,1,dim]
        fusion_fea_center_cls = torch.cat((cls_tokens, fusion_fea_center), dim=1)  # [b,n+1,dim] # (b,9,e)
        # print('fusion_fea_center_cls', fusion_fea_center_cls.shape)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        fusion_fea_center_cls += self.pos_embedding_FT[:, :(n + 1)]
        fusion_fea_center_cls = self.transformer_FT(fusion_fea_center_cls)  # b n+1 d
        fusion_fea_center_cls = nn.functional.normalize(fusion_fea_center_cls, p=2, dim=1)
        fusion_fea_center_cls = self.feature_selection(input=fusion_fea_center_cls, type='FT')  # if you want use it, please select the cls token for classificaiion

        # # classification: using cls_token output
        x = fusion_fea_center_cls.mean(dim=1) if self.pool == 'mean' else fusion_fea_center_cls[:, 0]
        x = self.to_latent(x)
        # x_reg = self.reg(x)
        x_class = self.mlp_head(x)
        x_reg = self.reg(x)# batch, num_class

        # classification 2
        # x  = fusion_fea_center_cls.view(fusion_fea_center_cls.size(0), -1)
        # x = self.bn(x)
        # x = self.relu(x)
        # x = self.fc(x)

        # classification 3
        # x = x8r_out + x7r_out + x6r_out + x5r_out + x4r_out+ x3r_out + x2r_out + x1r_out
        # print('x',x.shape)
        # # x = x.permute(1,2,0).contiguous()
        # x = x.view(x.size(0),-1)
        # print('x',x.shape)
        # x = self.gru_bn_3(x)
        # x = self.tanh(x)
        # print('into fc',x.shape)
        # x = self.dropout(x)
        # x = self.fc_3(x)

        return x_class, x_reg
