import math
import torch
import numpy as np
from torch import nn
from torch.nn import init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
# import optuna
import visdom

viz = visdom.Visdom(env='plot', port=8097)
from torch.nn import TransformerDecoder, TransformerDecoderLayer

# from datasets import add_random_mask, add_gaussian_noise
'---------------------------------------------------------------------------------------------------------------------------------------------------------------------'


# Hypergraph part
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft),
                                   requires_grad=True)  # Initializing as a trainable parameter
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # Adjust for batch and heads
        x = torch.einsum('bhni,io->bhno', x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.matmul(G, x)
        return x


class HGNN_weight(nn.Module):
    def __init__(self, in_ch, n_hid, dim, dropout=0.):
        super(HGNN_weight, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, dim)
        self.normalzation1 = nn.LayerNorm(in_ch)
        self.normalzation2 = nn.LayerNorm(n_hid)
        self.normalzation3 = nn.LayerNorm(dim)

    def forward(self, x, G):
        x = self.normalzation1(x)
        x = F.relu(self.hgc1(x, G))
        x = self.normalzation2(x)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        x = self.normalzation3(x)
        return x


def _generate_G_from_H(S, W_node_edge=None):
    """
    calculate G from hypergraph incidence tensor S
    :param S: hypergraph incidence tensor S of shape (batch_size, num_heads, N, E)
    :return: G
    """
    batch_size, num_heads, N, E = S.shape
    G = torch.zeros((batch_size, num_heads, N, N), device=S.device)  # shape (batch_size, num_heads, N, N)

    # loop over the batch and heads to compute each graph adjacency matrix
    for i in range(batch_size):
        for j in range(num_heads):
            H = S[i, j]  # shape (N, E)
            # the weight of the hyperedge
            n_edge = H.shape[1]
            W_edge = torch.ones(n_edge, device=H.device)
            # the degree of the node
            # DV = torch.sum(H_weighted, dim=1)  # shape (N,)
            DV = torch.sum(H * W_edge, dim=1)  # shape (N,)

            # the degree of the hyperedge
            DE = torch.sum(H, dim=0)  # shape (E,)

            invDE = torch.diag(1.0 / DE)  # shape (E, E), inverse degree matrix of hyperedges
            DV2 = torch.diag(1.0 / torch.sqrt(DV))  # shape (N, N), square root inverse degree matrix of nodes
            HT = H.t()  # shape (E, N), transposed incidence matrix

            if W_node_edge is not None:
                W_current = W_node_edge[i, j]  # shape (N, E)
                # the weighted incidence matrix
                H_weighted = H * W_current  # shape (N, E), element-wise multiplication
                H_weighted_T = H_weighted.t()  # shape (E, N), transposed incidence matrix
                get_G = compute_G(W_current.sum(dim=0))
                G[i, j] = get_G(DV2 @ H_weighted, invDE @ H_weighted_T @ DV2)
            else:
                get_G = compute_G(W_edge)
                G[i, j] = get_G(DV2 @ H, invDE @ HT @ DV2)

            # get_G = compute_G(W_current.sum(dim=0))
            # get_G = compute_G(W_edge)
            # G[i, j] = get_G(DV2 @ H, invDE @ HT @ DV2)

    return G  # shape (batch_size, num_heads, N, N), adjacency matrices of the graphs for each head in each batch


class compute_G(nn.Module):
    def __init__(self, W):
        super(compute_G, self).__init__()
        self.W = nn.Parameter(W, requires_grad=True)

    def forward(self, DV2_H, invDE_HT_DV2):
        w = torch.diag(self.W)
        G = w @ invDE_HT_DV2
        G = DV2_H @ G
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1, sig=1000):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param batch_size: size of the batch
    :param num_heads: number of heads
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_obj = dis_mat.shape[-2]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = torch.zeros((dis_mat.size(0), dis_mat.size(1), n_obj, n_edge), device=device)

    for b in range(dis_mat.size(0)):
        for h in range(dis_mat.size(1)):
            A = torch.mean(dis_mat[b, h])
            # print('A', A.item())
            for center_idx in range(n_obj):
                dis_mat[b, h, center_idx, center_idx] = 1.0
                dis_vec = dis_mat[b, h, center_idx]
                nearest_idx = torch.argsort(dis_vec).squeeze()
                if not torch.any(nearest_idx[:k_neig] == center_idx):
                    nearest_idx[k_neig - 1] = center_idx

                for node_idx in nearest_idx[:k_neig]:
                    if is_probH:
                        H[b, h, node_idx, center_idx] = torch.exp(- sig * dis_vec[node_idx] / A)
                    else:
                        H[b, h, node_idx, center_idx] = 1.0
    return H


'-----------------------------------------------------------------------------------------------------------------------------------------------------------------------'


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, stride, dropout, max_seq_len=10000, device='cuda'):
        super(LocalRNN, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.device = device

        rnn_types = {'GRU': nn.GRU, 'LSTM': nn.LSTM, 'RNN': nn.RNN}
        self.rnn = rnn_types[rnn_type](output_dim, output_dim, batch_first=True)

        idx = [i for j in range(self.ksize - 1, max_seq_len, self.stride) for i in
               range(j - (self.ksize - 1), j + 1, 1)]
        self.select_index = torch.LongTensor(idx).to(self.device)  # Shape: (max_seq_len,)
        self.zeros = torch.zeros((self.ksize - 1, input_dim)).to(self.device)  # Shape: (ksize-1, input_dim)

    def get_K(self, x):
        batch_size, l, d_model = x.shape  # Shape of x: (batch_size, seq_len, input_dim)
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape of zeros: (batch_size, ksize-1, input_dim)
        x = torch.cat((zeros, x), dim=1)  # Shape of x after cat: (batch_size, seq_len+ksize-1, input_dim)
        new_l = 1 + (l - self.ksize) // self.stride
        key = torch.index_select(x, 1, self.select_index[
                                       :self.ksize * new_l])  # Shape of key: (batch_size, ksize*new_l, input_dim)
        key = key.reshape(batch_size, new_l, self.ksize, -1)  # Shape of key: (batch_size, new_l, ksize, input_dim)
        return key

    def forward(self, x):
        nbatches, l, input_dim = x.shape  # Shape of x: (batch_size, seq_len, input_dim)
        x = self.get_K(x)  # Shape of x after get_K: (batch_size, new_l, ksize, input_dim)
        batch, new_l, ksize, d_model = x.shape  # Shape of x: (batch_size, new_l, ksize, input_dim)
        h = self.rnn(x.view(-1, self.ksize, d_model))[:, -1, :]  # Shape of h: (batch_size*new_l, output_dim)
        return h.view(batch, new_l, d_model)  # Shape of output: (batch_size, new_l, output_dim)


class LocalRNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, stride, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, stride, dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.dropout(self.local_rnn(self.norm(x))) + x
        x = self.local_rnn(self.norm(x))
        return x


# Transformer classes--------------------------------------------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, dim, dropout, fn):
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


class Attention(nn.Module):
    def __init__(self, dim, num_f, heads, dim_head, dropout, attn_type, mask_ratio, use_dropkey=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.dim = dim
        self.attn_type = attn_type
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.num_f = num_f
        self.mask_ratio = mask_ratio
        self.use_dropkey = use_dropkey
        self.softmax = nn.Softmax()
        self.hyperconv = HGNN_weight(in_ch=dim, n_hid=dim, dim=dim)
        self.elu = nn.ELU()
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * heads),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        ) if project_out else nn.Identity()

    def spe_dis_mat_3d(self, x):
        """
        Calculate the spectral distance among each sub-cube
        :return: batch * N X N distance matrix
        """
        spe_pairwise_distances = torch.cdist(x, x, p=2)
        # spe_H = construct_hypergraph_incidence_matrix_batch(spe_pairwise_distances, 16, True, 1, 1000)
        sigma_spe = torch.mean(spe_pairwise_distances)
        spe_pairwise_weights = torch.exp(-spe_pairwise_distances ** 2 / (2 * sigma_spe ** 2))
        spe_pairwise_weights = torch.where(spe_pairwise_weights < torch.mean(spe_pairwise_weights),
                                           torch.zeros_like(spe_pairwise_weights), spe_pairwise_weights)
        # print('spe_mean', torch.mean(spe_pairwise_weights))
        return spe_pairwise_weights.cuda()

    def spa_dis_mat_3d(self, x):
        """
        Calculate the spatial distance among each sub-cube
        :return: batch * N X N distance matrix
        """
        b, num_step, d = x.shape
        num_h = (num_step // self.num_f) ** (0.5)
        num_w = num_h
        x = rearrange(x, 'b (h w f) d -> b h w f d', h=int(num_h), w=int(num_w), f=self.num_f)
        b, h, w, f, c = x.shape
        x, y, z = torch.meshgrid(torch.arange(h), torch.arange(w), torch.arange(f), indexing='ij')
        xyz_grid = torch.stack((x, y, z), dim=-1).float()
        xyz_grid = xyz_grid.view(1, h, w, f, 3).repeat(b, 1, 1, 1, 1)
        coord_diff = xyz_grid.view(b, -1, 1, 3) - xyz_grid.view(b, 1, -1, 3)
        spa_pairwise_distances = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
        # spa_H = construct_hypergraph_incidence_matrix_batch(spa_pairwise_distances, 16, True, 1, 1000)
        sigma_spa = torch.mean(spa_pairwise_distances)
        spa_pairwise_weights = torch.exp(-spa_pairwise_distances ** 2 / (2 * sigma_spa ** 2))
        spa_pairwise_weights = torch.where(spa_pairwise_weights < torch.mean(spa_pairwise_weights),
                                           torch.zeros_like(spa_pairwise_weights), spa_pairwise_weights)
        # print('spa_mean', torch.mean(spa_pairwise_weights))
        return spa_pairwise_weights.view(b, -1, h * w * f).cuda()

    def SMSA3d(self, x):
        b, num_step, d, h = *x.shape, self.heads

        spe_dis = self.spe_dis_mat_3d(x)
        spe_dis = repeat(spe_dis, 'b n1 n2 -> b h n1 n2', h=self.heads, b=b)
        # spe_H_knn = construct_H_with_KNN_from_distance(spe_dis,16)
        # print('spe h knn', spe_H_knn.shape)

        spa_dis = self.spa_dis_mat_3d(x)
        spa_dis = repeat(spa_dis, 'b n1 n2 -> b h n1 n2', h=self.heads, b=b)
        # spa_H_knn = construct_H_with_KNN_from_distance(spa_dis, 16)
        # print('spa h knn', spa_H_knn.shape)

        'if spe and spa separately, spe_spa_H'
        spe_H = torch.where(spe_dis.float() != 0, torch.tensor(1.0).to(spe_dis.device), spe_dis.float())
        spa_H = torch.where(spa_dis.float() != 0, torch.tensor(1.0).to(spa_dis.device), spa_dis.float())
        spe_spa_H = torch.cat([spe_H, spa_H], dim=-1)
        # spe_spa_H_knn = torch.cat([spe_H_knn, spa_H_knn], dim=-1)

        'if spa_spa together, overlapp_H'
        overlapp = torch.where((spe_dis.float() != 0) & (spa_dis.float() != 0), spe_dis.float(),
                               torch.zeros_like(spe_dis.float()))
        overlap_H = torch.where(overlapp.float() != 0, torch.tensor(1.0).to(overlapp.device), overlapp.float())
        # for b in range(b):
        #     if b == 1:
        #         plt.subplot(141)
        #         plt.imshow(spe_dis[b, 0, :, :].cpu().detach().numpy())
        #         plt.title('spe distance batch 1')
        #         plt.colorbar()
        #         plt.subplot(142)
        #         plt.imshow(spa_dis[b, 0, :, :].cpu().detach().numpy())
        #         plt.title('spa distance batch 1')
        #         plt.colorbar()
        #         plt.subplot(143)
        #         plt.imshow(spe_H[b, 0, :, :].cpu().detach().numpy())
        #         plt.title('spe_H batch 1')
        #         plt.colorbar()
        #         plt.subplot(144)
        #         plt.imshow(spa_H[b, 0, :, :].cpu().detach().numpy())
        #         plt.title('spa_H batch 1')
        #         plt.colorbar()
        #         plt.show()
        #         plt.imshow(spe_spa_H[b, 0, :, :].cpu().detach().numpy())
        #         plt.title('spe_spa_H batch 1')
        #         plt.colorbar()
        #         plt.show()

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(self.elu(q) + 1, (self.elu(k) + 1).transpose(-1, -2)) * self.scale
        if self.use_dropkey == True:
            m_r = torch.ones_like(dots) * self.mask_ratio
            dots = dots + torch.bernoulli(m_r) * -1e12

        dots = self.dropout(dots)

        'if spe and spa separately'
        attn_spe = self.attend(dots) * spe_dis
        attn_spa = self.attend(dots) * spa_dis
        attn_spe_spa_W = self.attend(torch.cat([attn_spe, attn_spa], dim=-1))

        'if spa_spa together'
        overlap_attn = self.attend(dots * spe_dis * spa_dis)
        overlap_attn = self.dropout(overlap_attn)
        # for b in range(b):
        #     if b == 1:
        #         plt.imshow(attn_spe_spa_W[b, 0, :, :].cpu().detach().numpy())
        #         plt.title('batch 1 attn')
        #         plt.colorbar()
        #         plt.show()

        'if spe and spa separately'
        # out = self.hyperconv(v, _generate_G_from_H(S=overlap_attn, W_node_edge=None))

        'if spa_spa together'
        out = torch.matmul(overlap_attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def spe_dis_mat_2d(self, x):
        b, n, d = x.shape
        spe_pairwise_distances = torch.cdist(x, x, p=2)
        sigma_spe = torch.mean(spe_pairwise_distances)
        spe_pairwise_weights = torch.exp(-spe_pairwise_distances / (2 * sigma_spe ** 2))
        return spe_pairwise_weights

    def spa_dis_mat_2d(self, x):
        b, n, d = x.shape
        x = rearrange(x, 'b (h w) d -> b h w d', h=int(n ** (1 / 2)), w=int(n ** (1 / 2)))
        b, h, w, c = x.shape
        x, y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        xy_grid = torch.stack((x, y), dim=-1).float()
        xy_grid = xy_grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
        coord_diff = xy_grid.view(b, -1, 1, 2) - xy_grid.view(b, 1, -1, 2)
        spa_pairwise_distances = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
        sigma_spa = torch.mean(spa_pairwise_distances)
        spa_pairwise_weights = torch.exp(-spa_pairwise_distances / (2 * sigma_spa ** 2))
        return spa_pairwise_weights.view(b, -1, h * w)

    def SMSA2d(self, x):
        b, n, d, h = *x.shape, self.heads
        spe_mask = self.spe_dis_mat_2d(x)
        spe_mask = repeat(spe_mask, 'b n1 n2 -> b h n1 n2', h=self.heads, b=b)
        spe_mask = spe_mask.float()
        spa_mask = self.spa_dis_mat_2d(x)
        spa_mask = repeat(spa_mask, 'b n1 n2 -> b h n1 n2', h=self.heads, b=b)
        spa_mask = spa_mask.float()
        'if spe and spa separately, spe_spa_H'
        spe_H = torch.where(spe_mask.float() != 0, torch.tensor(1.0).to(spe_mask.device), spe_mask.float())
        spa_H = torch.where(spa_mask.float() != 0, torch.tensor(1.0).to(spa_mask.device), spa_mask.float())
        spe_spa_H = torch.cat([spe_H, spa_H], dim=-1)
        'if spa_spa together, overlapp_H'
        overlapp = torch.where((spe_mask != 0) & (spa_mask != 0), spe_mask, torch.zeros_like(spe_mask))
        overlap_H = torch.where(overlapp.float() != 0, torch.tensor(1.0).to(overlapp.device), overlapp.float())
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        dots = dots * spe_mask * spa_mask
        if self.use_dropkey == True:
            m_r = torch.ones_like(dots) * self.mask_ratio
            dots = dots + torch.bernoulli(m_r) * -1e12
        dots = self.dropout(dots)
        attn_spe = self.attend(dots) * spe_mask
        attn_spa = self.attend(dots) * spa_mask
        attn_spe_spa_W = self.attend(torch.cat([attn_spe, attn_spa], dim=-1))
        overlap_attn = self.attend(dots) * spe_mask * spa_mask
        overlap_attn = self.dropout(overlap_attn)
        # for b in range(b):
        #     if b == 1:
        #         plt.imshow(attn_spe_spa_W[b, 0, :, :].cpu().detach().numpy())
        #         plt.title('batch 1 attn')
        #         plt.colorbar()
        #         plt.show()
        # out = torch.einsum('bhij,bhjd->bhid', overlap_attn, v)
        out = self.hyperconv(v, _generate_G_from_H(S=overlap_H, W_node_edge=overlap_attn))
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def Original(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', self.elu(q) + 1, self.elu(k) + 1) * self.scale
        # if self.use_dropkey == True:
        #     m_r = torch.ones_like(dots) * self.mask_ratio
        #     dots = dots + torch.bernoulli(m_r) * -1e12
        #     # dots = dots.masked_fill_(~m_r.bool(), float('-inf'))
        dots = self.dropout(dots)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def forward(self, x):
        if self.attn_type == 'SMSA3d':
            out = self.SMSA3d(x)
        elif self.attn_type == 'SMSA2d':
            out = self.SMSA2d(x)
        elif self.attn_type == 'Original':
            out = self.Original(x)
        else:
            raise ValueError(f'Invalid attn_type: {self.attn_type}')
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_f, dropout=0.1, attn_type='SMSA3d', mask_ratio=0.1,
                 use_dropkey=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.alpha = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.relu = nn.ReLU()
        self.identify = nn.Identity()

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, nn.LSTM(input_size=dim, hidden_size=dim,
                                     num_layers=1, dropout=dropout, batch_first=True, bias=True, bidirectional=False)),
                PreNorm(dim, Attention(dim, num_f=num_f, heads=heads, dim_head=dim_head, dropout=dropout,
                                       attn_type=attn_type, mask_ratio=mask_ratio, use_dropkey=use_dropkey)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x_1):
        for rnn, attn, ff in self.layers:
            x_2 = self.relu(rnn(x_1)) + self.identify(x_1)
            x_3 = self.relu(attn(x_2)) + self.identify(x_1)
            x_4 = self.relu(ff(x_3)) + self.identify(x_1)
            # gamma = self.sigmoid(self.gamma)
            # delta = 1 - gamma
            #
            # p_rnn = rnn(x)  # you set as LSTM
            # p_rnn = self.relu(p_rnn)
            #
            # s_tilde = ff(gamma * x + delta * p_rnn)
            #
            # s_tlide = self.identify(x) + s_tilde
            # x_2 = self.relu(s_tilde)
            #
            # f_smsa = attn(x_2)
            # f_smsa = self.relu(f_smsa)
            #
            # t_out = ff(f_smsa)
            #
            # s_tlide = self.identify(x_2) + t_out
            # x_3 = self.relu(s_tilde)
            #
            # out = ff(x_3)
            # # x = self.norm(t_out + t_tilde)
            # # x = ff(x)
        return x_4


Houston = (1905, 349, 144)
Indian_Pine = (145, 145, 200)
Pavia_U = (610, 340, 100)

subimage_patch_size = (3, 3, 20)
stride = (3, 3, 20)
padding = (0, 0, 0)
frame_patch_size = subimage_patch_size[2]  # HU:12


class ViT3d_2(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.Conv2d, nn.LSTM, nn.Parameter,
                          nn.BatchNorm1d, nn.LayerNorm, nn.Dropout, Transformer)):
            init.kaiming_normal_(m.weight.data)
            init.kaiming_normal_(m.bias.data)

    def __init__(self, n_bands, n_classes, patch_size, subimage_patch_size=subimage_patch_size,
                 frame_patch_size=frame_patch_size,
                 padding=padding, stride=stride, dim=32, depth=1, heads=1, mlp_dim=32, dim_head=32, dropout=0.):

        super(ViT3d_2, self).__init__()
        self.patch_size = patch_size
        self.kernel_size = subimage_patch_size
        self.padding = padding
        self.stride = stride
        self.dim = dim
        image_height, image_width = pair(self.patch_size)
        image_frame = n_bands
        self.num_h = int((image_height - self.kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
        self.num_w = int((image_width - self.kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
        self.num_f = int((image_frame - self.kernel_size[2] + 2 * padding[2]) / stride[2]) + 1
        self.num_spatial = self.num_h * self.num_w  ## (spatial aspect)
        self.num_patches = self.num_h * self.num_w * self.num_f  # spatial * spectral aspect
        self.patch_dim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        print('num_w:', self.num_h, 'num_w:', self.num_w, 'num_f:', self.num_f, 'num_spa:',
              self.num_spatial, 'num_patches:', self.num_patches, 'patch_dim:', self.patch_dim)
        self.partial_rnn = nn.ModuleList(
            [nn.LSTM(input_size=self.patch_dim, hidden_size=self.patch_dim, num_layers=1, batch_first=True, bias=True,
                     bidirectional=True)
             for _ in range(self.num_spatial)])
        # self.partial_rnn = nn.LSTM(input_size=self.patch_dim, hidden_size=self.patch_dim, num_layers=1, batch_first=True, bias=True, bidirectional=True)
        self.localconv = nn.Conv2d(n_bands, n_bands, 3, 1, padding=(1, 1), bias=True,
                                   groups=n_bands // self.kernel_size[2])
        self.patch_dim_to_dim = nn.Linear(self.patch_dim * 2, dim, bias=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, self.num_spatial + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(dim * self.num_f, dim, bias=True)
        self.layernorm = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(self.patch_dim)
        self.layernorm3 = nn.LayerNorm(frame_patch_size)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, self.num_f, dropout,
                                       attn_type='SMSA3d', mask_ratio=0.1, use_dropkey=True)
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, self.num_f, dropout,
                                        attn_type='Original', mask_ratio=0.1, use_dropkey=True)
        self.depth_cnn = nn.Conv3d(in_channels=1, out_channels=frame_patch_size, kernel_size=(frame_patch_size, 3, 3),
                                   padding=(0, 1, 1), stride=(frame_patch_size, 1, 1))

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_classes)
        )
        self.softmax = nn.Softmax()
        self.aux_loss_weight = 1
        self.reg = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_bands, bias=True)
        )
        self.relu = nn.ReLU6()

    def coordinate3D(self, num_patches_3D, dim, num_h, num_w, num_f, device):
        # Convert Tensors to int if necessary
        num_patches_3D = num_patches_3D.item() if isinstance(num_patches_3D, torch.Tensor) else num_patches_3D
        dim = dim.item() if isinstance(dim, torch.Tensor) else dim

        # Create tensors for the positional encodings
        pe_h = torch.zeros(1, num_patches_3D, dim).to(device)
        pe_w = torch.zeros(1, num_patches_3D, dim).to(device)
        pe_f = torch.zeros(1, num_patches_3D, dim).to(device)

        # Calculate positions using broadcasting and reshaping
        pos_h = torch.div(torch.arange(num_patches_3D, device=device), (num_w * num_f),
                          rounding_mode='trunc').unsqueeze(1)
        pos_w = torch.div(torch.remainder(torch.arange(num_patches_3D, device=device), (num_w * num_f)), num_f,
                          rounding_mode='trunc').unsqueeze(1)
        pos_f = (torch.arange(num_patches_3D, device=device) % num_f).unsqueeze(1)

        # Calculate the div_term for scaling the position values
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))

        # Compute the positional encodings
        pe_h[:, :, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, :, 1::2] = torch.cos(pos_h * div_term)
        pe_w[:, :, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, :, 1::2] = torch.cos(pos_w * div_term)
        pe_f[:, :, 0::2] = torch.sin(pos_f * div_term)
        pe_f[:, :, 1::2] = torch.cos(pos_f * div_term)

        return pe_h, pe_w, pe_f

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
        # image = self.localconv(image)
        # print('image size',image.shape)
        x = image.unsqueeze(1)
        # print('x1', x.shape)
        x = self.depth_cnn(x)
        # print('x2', x.shape)
        image = rearrange(x, 'b c d h w -> b (c d) h w')

        # print('image size',image.shape)
        # Check the validity of kernel_size, padding, and stride
        # assert all(i <= j for i, j in zip(kernel_size, image.shape[1:])), "Kernel size must be smaller than the image size in all dimensions."

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
        trans3D = rearrange(trans3D, 'b (n2D p p3) n3D -> b (n2D n3D) (p3 p)', p=kernel_size[0] * kernel_size[1],
                            p3=kernel_size[2], n2D=num_patches)
        return trans3D

    # def partialized_patch_embeddimg(self, x):
    #     x = torch.chunk(x, chunks=self.num_spatial, dim=1)
    #     reshaped_sub_images = [self.partial_rnn(self.layernorm2(sub_seq)) for sub_seq in x]
    #     x_out = torch.cat(reshaped_sub_images, dim=1)
    #     x_out = self.patch_dim_to_dim(x_out)
    #     x_out = self.layernorm(x_out)
    #     return x_out
    def partialized_patch_embeddimg(self, x):
        x = torch.chunk(x, chunks=self.num_spatial, dim=1)
        reshaped_sub_images = []
        for idx, sub_seq in enumerate(x):
            sub_seq = self.layernorm2(sub_seq)
            lstm_output = self.partial_rnn[idx](sub_seq)
            reshaped_sub_images.append(lstm_output)
        x_out = torch.cat(reshaped_sub_images, dim=1)
        x_out = self.patch_dim_to_dim(x_out)
        x_out = self.layernorm(x_out)
        return x_out

    def feature_selection(self, input, type):
        # input shape: (batch_size, seq_len, dim)
        _, n, d = input.shape
        if type == 'center':
            c = int((n - 1) / 2)
            selected = input[:, c, :]  # (batch_size, dim)
        elif type == 'class':
            selected = input[:, 0, :]  # (batch_size, dim)
        elif type == 'last':
            selected = input[:, -1, :]
        else:
            raise ValueError('Unsupported type for feature selection. Choose either "center" or "class token"')

        a_m = torch.matmul(input, selected.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
        a_m = F.softmax(a_m, dim=1)  # (batch_size, seq_len)
        result = input * a_m.unsqueeze(-1)  # (batch_size, seq_len, dim)

        return result.to(input.device)  # return tensor on the same device as input

    def one_direction_operation(self, x):
        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        # x = x.squeeze(1)
        # x = x.view(-1, 1, self.n_bands, self.patch_size, self.patch_size)

        x = self.partition_3D(x, self.kernel_size, self.padding, self.stride)  # x(b (h w f) d)
        # pe_h, pe_w, pe_f = self.coordinate3D(self.num_patches, self.dim, self.num_h, self.num_w, self.num_f, device='cuda')

        x = self.partialized_patch_embeddimg(x)  # x(b (h w f) d)
        # x = x + pe_h + pe_w + pe_f
        '------------------------------Spectral Trans: how many spectral (num_f * num_h * num_w) patches?------------------------------------'
        x += self.pos_embedding[:, :(self.num_patches)]
        x = self.dropout(x)
        x = self.transformer(x)  # (b, num_h*num_w*num_f, dim)
        x = self.layernorm(x)

        chunks = torch.chunk(x, chunks=self.num_spatial, dim=1)
        reshaped_chunks = [chunk.reshape(chunk.size(0), -1) for chunk in chunks]
        reshaped_x = torch.stack(reshaped_chunks, dim=1)  # (b, num_spatial, dim*num_f)

        x2 = self.project(reshaped_x)  # (b, num_spa, dim)
        # x2 = self.feature_selection(x2, 'center')
        x2 = self.layernorm(x2)
        '------------------------------Spatial Trans: how many spatial patches (num_h * num_w)?------------------------------------'
        cls_tokens2 = repeat(self.cls_token2, '1 1 d -> b 1 d', b=x2.size(0))
        x2 = torch.cat((cls_tokens2, x2), dim=1)
        x2 += self.pos_embedding2[:, :(self.num_spatial + 1)]
        x2 = self.dropout(x2)
        x2 = self.transformer2(x2)
        x2 = self.layernorm(x2)
        # x2 = self.feature_selection(x2, 'class')
        # x2 = self.layernorm(x2)

        x2_center = x2[:, int((self.num_spatial + 1) / 2), :]
        x2_mean = x2[:, 1:].mean(dim=1)
        x2_class = x2[:, 0]
        '------------------------------Decoder part: classification & regression ------------------------------------'
        x_regression = self.to_latent(x2_class + x2_center + x2_mean)
        out = self.to_latent(x2_class + x2_center + x2_mean)
        return out, x_regression

    def forward(self, x):
        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        x = x.squeeze(1)
        # x_new = torch.chunk(x, 4, dim=1)
        # reshaped_x = [self.conv(chunk) for chunk in x_new]
        # x = torch.cat(reshaped_x, dim=1)
        # x_new = torch.cat([x_group1.unsqueeze(1), x_group2.unsqueeze(1), x_group3.unsqueeze(1)],dim=1)
        # x = self.re_band(x_new)
        'generate different directions x_dir_1 and x_dir_3, x_dir_6 and x_dir_8, 5 and 7, 4 and 2'
        x_dir_1 = x
        x_dir_6 = torch.rot90(x_dir_1, 1, [2, 3])
        x_dir_3 = torch.rot90(x_dir_6, 1, [2, 3])
        x_dir_8 = torch.rot90(x_dir_3, 1, [2, 3])
        x_dir_7 = torch.flipud(x_dir_1)
        x_dir_2 = torch.flipud(x_dir_6)
        x_dir_5 = torch.flipud(x_dir_3)
        x_dir_4 = torch.flipud(x_dir_8)
        out_1, reg_1 = self.one_direction_operation(x_dir_1)
        # out_1 = F.normalize(out_1, p=2, dim=1)
        # reg_1 = F.normalize(reg_1, p=2, dim=1)
        # out_2, reg_2= self.one_direction_operation(x_dir_2)
        # out_3, reg_3 = self.one_direction_operation(x_dir_3)
        #     out_4 = self.ond_dir_operation(x_dir_4)
        #     out_5, reg_5 = self.one_direction_operation(x_dir_5)
        #     out_6, reg_6 = self.one_direction_operation(x_dir_6)
        #     out_7, reg_7 = self.one_direction_operation(x_dir_7)
        #     out_8, reg_8 = self.one_direction_operation(x_dir_8)
        return self.mlp_head(out_1), self.reg(reg_1)
