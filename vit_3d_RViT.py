import math
import torch
# import numpy as np
from torch import nn
from einops import rearrange, repeat, reduce
# from einops.layers.torch import Rearrange
# import matplotlib.pyplot as plt
import torch.nn.functional as F
# import seaborn as sns
# import optuna
import visdom

viz = visdom.Visdom(env='regression', port=8097)
from torch.nn import init

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
    # :param batch_size: size of the batch
    # :param num_heads: number of heads
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge

    Args:
        sig:
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


# Transformer classes---------------------------------------------------------------------------------------------------------------


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
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.use_dropkey == True:
            m_r = torch.ones_like(dots) * self.mask_ratio
            dots = dots + torch.bernoulli(m_r) * -1e12

        dots = self.dropout(dots)

        'if spe and spa separately'
        attn_spe = self.attend(dots) * spe_dis
        attn_spa = self.attend(dots) * spa_dis
        attn_spe_spa_W = self.attend(torch.cat([attn_spe, attn_spa], dim=-1))

        'if spa_spa together'
        overlap_attn = self.attend(dots @ spe_dis @ spa_dis)
        overlap_attn = self.dropout(overlap_attn)
        # for b in range(b):
        #     if b == 1:
        #         plt.imshow(attn_spe_spa_W[b, 0, :, :].cpu().detach().numpy())
        #         plt.title('batch 1 attn')
        #         plt.colorbar()
        #         plt.show()

        'if spe and spa separately'
        # out = self.hyperconv(v, _generate_G_from_H(S=spe_spa_H, W_node_edge=attn_spe_spa_W))

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
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
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


def split_qkv(x, h):
    """Splits Q, K, V from a single tensor."""
    qkv = rearrange(x, 'b n (qkv h d) -> qkv b h n d', h=h, qkv=3)
    return qkv[0], qkv[1], qkv[2]


class Recur_self_attn(nn.Module):
    def __init__(self, dim, ffn_dim, heads, dropout):
        super().__init__()

        self.alpha = None
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        # Projection layers
        self.cur_to_qkv = nn.Linear(ffn_dim, ffn_dim * heads * 3)
        self.pre_to_qkv = nn.Linear(ffn_dim, ffn_dim * heads * 3)
        self.proj = nn.Linear(ffn_dim * heads, ffn_dim)
        # self.proj = nn.LSTM(input_size=ffn_dim * heads, hidden_size=ffn_dim * heads, num_layers=1, bias=True, batch_first=True, proj_size=ffn_dim)

        # Batch normalization layers
        self.bn1 = nn.LayerNorm(ffn_dim)
        self.bn2 = nn.LayerNorm(ffn_dim)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.GELU()

        # Parameter initialization
        self.init_alpha()

    def init_alpha(self):
        # Initializing alpha values between 0 and 1
        self.alpha = nn.Parameter(torch.rand(3, self.ffn_dim * self.heads, requires_grad=True))
        nn.init.uniform_(self.alpha, 0, 1)

    def forward(self, cur_input_t, pre_hidden):
        b, n, d = cur_input_t.shape
        # print('shape:', cur_input_t.shape)
        h = self.heads

        Q_cur, K_cur, V_cur = split_qkv(self.cur_to_qkv(cur_input_t), h)
        Q_pre, K_pre, V_pre = split_qkv(self.pre_to_qkv(pre_hidden), h)
        # print('shape:', Q_pre.shape, Q_cur.shape)
        # Reshape alpha to match the heads and dim
        alpha_sigmoid = torch.sigmoid(self.alpha).view(3, h, self.ffn_dim)

        # Weighted sum of current and previous Q, K, V with sigmoid gating
        Q = (alpha_sigmoid[0].unsqueeze(0).unsqueeze(2) * Q_cur +
             (1 - alpha_sigmoid[0].unsqueeze(0).unsqueeze(2)) * Q_pre)
        K = (alpha_sigmoid[1].unsqueeze(0).unsqueeze(2) * K_cur +
             (1 - alpha_sigmoid[1].unsqueeze(0).unsqueeze(2)) * K_pre)
        V = (alpha_sigmoid[2].unsqueeze(0).unsqueeze(2) * V_cur +
             (1 - alpha_sigmoid[2].unsqueeze(0).unsqueeze(2)) * V_pre)
        # print('..', V.shape)

        # Self attention calculation
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        # print('..',attn.shape)

        A_cur_t = attn @ V
        # print('..', A_cur_t.shape)
        A_cur_t = self.bn1(A_cur_t)

        A_cur_t = rearrange(A_cur_t, 'b h n d -> b n (h d)')
        A_cur_t = self.proj(A_cur_t)
        A_cur_t = self.bn2(A_cur_t)
        A_cur_t = self.dropout(A_cur_t)

        # Residual connection
        A_cur_t += self.relu(cur_input_t)

        return A_cur_t


class RTCell(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super(RTCell, self).__init__()
        self.dim = dim
        self.pro1 = nn.Linear(dim, ffn_dim)
        self.pro2 = nn.Linear(dim, ffn_dim)
        self.attention = Recur_self_attn(dim=ffn_dim, ffn_dim=ffn_dim, heads=heads, dropout=0.)

        self.layer_norm_cur = nn.LayerNorm(ffn_dim)
        self.layer_norm_pre = nn.LayerNorm(ffn_dim)
        self.layer_norm_o = nn.LayerNorm(ffn_dim)

        self.ffn = nn.Linear(ffn_dim, ffn_dim)
        self.elu = nn.GELU()

    def forward(self, cur_input_t, pre_hidden):
        if cur_input_t.size(2) == self.dim:
            cur_input_t = self.layer_norm_cur(self.pro1(cur_input_t))
        else:
            cur_input_t = cur_input_t

        if pre_hidden is None:
            pre_hidden = torch.zeros_like(cur_input_t, device=cur_input_t.device)
        else:
            if pre_hidden.size(2) == self.dim:
                pre_hidden = self.layer_norm_pre(self.pro2(pre_hidden))
            else:
                pre_hidden = pre_hidden

        A_cur_t = self.attention(cur_input_t, pre_hidden)
        cur_hidden = pre_hidden + A_cur_t
        O_hat = self.layer_norm_o(cur_input_t + A_cur_t)
        O_cur_t = self.ffn(O_hat) + self.elu(O_hat)

        return O_cur_t, cur_hidden


class RecursiveTransformer(nn.Module):
    def __init__(self, dim, ffn_dim, depth, heads):
        super(RecursiveTransformer, self).__init__()
        self.cells = nn.ModuleList([RTCell(dim, ffn_dim, heads) for _ in range(depth)])
        self.relu = nn.ReLU()
        self.proj_dim_to_ffn_dim = nn.Linear(dim, ffn_dim)
        self.proj_ffn_dim_to_dim = nn.Linear(ffn_dim, dim)
        self.dim = dim

    def forward(self, sequence, H_pre=None):
        batch_size, n, _, d = sequence.size()

        # If hidden state not provided, initialize
        if H_pre is None:
            H_pre = [nn.Parameter(
                torch.randn(batch_size, sequence.size(1), sequence.size(2), self.dim, device=sequence.device),
                requires_grad=True) for _ in range(len(self.cells))]

        outputs = []

        for t in range(n):
            O_t = sequence[:, t, :, :]
            for l, cell in enumerate(self.cells):
                H_t = H_pre[l][:, t, :, :]
                # print("O_t shape:", O_t.shape, "H_t shape:", H_t.shape)  # Debugging
                if H_t.size(2) == self.dim:
                    O_t, H_t = cell(O_t, H_t)
                else:
                    O_t, H_t = cell(O_t, self.proj_ffn_dim_to_dim(H_t))
                # H_pre[l][:, t, :, :] = H_t
            outputs.append(O_t)

        out = torch.stack(outputs, dim=1)
        return out


# class RecursiveTransformer(nn.Module):
#     def __init__(self, dim, ffn_dim, depth, heads):
#         super(RecursiveTransformer, self).__init__()
#         self.cells = nn.ModuleList([RTCell(dim, ffn_dim, heads) for _ in range(depth)])
#         self.relu = nn.ReLU()
#
#     def forward(self, sequence, H_pre=None):
#         batch_size, n, _, d = sequence.size()
#
#         # If hidden state not provided, initialize
#         if H_pre is None:
#             H_pre = [torch.zeros(batch_size, sequence.size(2), sequence.size(3), device=sequence.device) for _ in range(len(self.cells))]
#
#         outputs = []
#
#         for t in range(n):
#             O_t = sequence[:, t, :, :]
#             for l, cell in enumerate(self.cells):
#                 # Aggregate previous hidden states
#                 aggregated_H_pre = sum(H_pre[:l+1])
#                 # Process with current cell
#                 O_t, H_pre[l] = cell(O_t, aggregated_H_pre)
#             outputs.append(O_t)
#
#         out = torch.stack(outputs, dim=1)
#         return out

# class RecursiveTransformer(nn.Module):
#     def __init__(self, dim, ffn_dim, depth, heads):
#         super(RecursiveTransformer, self).__init__()
#         self.cells = nn.ModuleList([RTCell(dim, ffn_dim, heads) for _ in range(depth)])
#         self.relu = nn.ReLU()
#
#     def forward(self, sequence, H_pre=None):
#         batch_size, n, _, d = sequence.size()
#
#         # If hidden state not provided, initialize
#         if H_pre is None:
#             H_pre = [torch.zeros(batch_size, sequence.size(2), sequence.size(3), device=sequence.device) for _ in
#                      range(len(self.cells))]
#
#         outputs = []
#
#         for t in range(n):
#             O_t = sequence[:, t, :, :]
#             for l, cell in enumerate(self.cells):
#                 # Weighted sum of previous hidden states based on linear distance
#                 weighted_sum_H_pre = torch.zeros_like(H_pre[0])
#                 for i in range(l):
#                     weight = 1 + (l - i) / l  # Weight based on linear distance
#                     weighted_sum_H_pre += weight * self.relu(H_pre[i])
#
#                 # Process with current cell
#                 O_t, H_pre[l] = cell(O_t, weighted_sum_H_pre)
#             outputs.append(O_t)
#
#         out = torch.stack(outputs, dim=1)
#         return out

# class RecursiveTransformer(nn.Module):
#     def __init__(self, dim, ffn_dim, depth, heads):
#         super(RecursiveTransformer, self).__init__()
#         self.cells = nn.ModuleList([RTCell(dim, ffn_dim, heads) for _ in range(depth)])
#
#     def forward(self, sequence, H_pre=None):
#         batch_size, N, _, d = sequence.size()
#
#         # Initialize hidden states if not provided
#         if H_pre is None:
#             H_pre = [torch.zeros(batch_size, sequence.size(2), sequence.size(3), device=sequence.device) for _ in
#                      range(N)]
#
#         outputs = []
#
#         for t in range(N):
#             O_t = sequence[:, t, :, :]
#             weighted_hidden_states = torch.zeros_like(O_t)
#
#             # Calculate weighted sum of previous hidden states
#             for i in range(t):
#                 weight_i = 1 + (t - i) / t  # Weight decreases for older hidden states
#                 weighted_hidden_states += weight_i * H_pre[i]
#
#             # Process with current cell
#             for l, cell in enumerate(self.cells):
#                 O_t, H_pre[t] = cell(O_t, weighted_hidden_states)
#
#             outputs.append(O_t)
#
#         return torch.stack(outputs, dim=1)


# Note: Ensure the RTCell is modified to accept weighted_hidden_states as an additional argument.


class Short_term_RT(nn.Module):
    def __init__(self, num=9, dim=16, ffn_dim=64, depth=4, heads=4):
        super(Short_term_RT, self).__init__()

        self.num = num
        # Create a list (or ModuleList) of RecursiveTransformer instances
        self.short_term_RTs = nn.ModuleList(
            [RecursiveTransformer(dim=dim, ffn_dim=ffn_dim, depth=depth, heads=heads) for _ in range(num)])

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
    def __init__(self, dim=25, ffn_dim=64, depth=4, heads=4):
        super(Long_term_RT, self).__init__()

        self.long_term_RT = RecursiveTransformer(dim=dim, ffn_dim=ffn_dim, depth=depth, heads=heads)
        self.long_term_RT_re = RecursiveTransformer(dim=dim, ffn_dim=ffn_dim, depth=depth, heads=heads)

    def forward(self, x_from_short_term):  # Expected input shape: (100, 9, 25, 16)
        # Rearrange the tensor dimensions. Assuming:
        # b = batch size, n = number of sequences,
        # p1 = original feature dimension, p2 = original sequence length
        # Transforming it to: b = batch size, n = number of sequences,
        # p2 = new feature dimension, p1 = new sequence length

        x_from_short_term = rearrange(x_from_short_term, 'b n p1 p2 -> b n p2 p1')

        # Assuming x_from_short_term is your tensor

        x_from_short_term_re = torch.flip(x_from_short_term, dims=[1])

        x_out = self.long_term_RT(x_from_short_term)
        # x_out_re = self.long_term_RT_re(x_from_short_term_re)
        # print('out', x_out.shape)
        # out = torch.cat([x_out, x_out_re], dim=1)
        # print('out', out.shape)
        return x_out


def partition_3D(image, kernel_size, padding, stride):
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

    x = image.unsqueeze(1)

    # x = self.depth_cnn(x)

    # x = self.batchnorm3d(x)

    image = rearrange(x, 'b c d h w -> b (c d) h w')

    def partition_along_dimension(tensor, kernel, padding, stride):
        return nn.functional.unfold(tensor, kernel_size=kernel, padding=padding, stride=stride)

    kernel_size_h_w, kernel_size_d = (kernel_size[0], kernel_size[1]), kernel_size[2]
    padding_h_w, padding_d = (padding[0], padding[1]), padding[2]
    stride_h_w, stride_d = (stride[0], stride[1]), stride[2]

    # Partition along height and width
    trans2D = partition_along_dimension(image, kernel_size_h_w, padding_h_w, stride_h_w)
    num_patches = trans2D.shape[-1]
    trans2D = rearrange(trans2D, 'b (c p1 p2) n -> b n (p1 p2) c', p1=kernel_size[0], p2=kernel_size[1])
    # example = trans2D[0,0,:,:].cpu().detach().numpy()
    # sns.heatmap(example, cmap='Greys')
    # plt.show()

    # Partition along depth
    trans3D = partition_along_dimension(trans2D, (kernel_size[0] * kernel_size[1], kernel_size_d), (0, padding_d),
                                        stride_d)
    trans3D = rearrange(trans3D, 'b (n2D p p3) n3D -> b (n2D n3D) p p3', p=kernel_size[0] * kernel_size[1],
                        p3=kernel_size[2], n2D=num_patches)

    return trans3D


Houston2013 = (1905, 349, 144)
Houston2018 = (601, 2384, 48)
Indian_Pine = (145, 145, 200)
Pavia_U = (610, 340, 100)
Salinas = (512, 217, 200)

subimage_patch_size = (3, 3, 20)
stride = (3, 3, 20)
padding = (0, 0, 0)
frame_patch_size = subimage_patch_size[2]  # HU:12


def pair(x):
    return (x, x)


class RVITBase(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.Conv2d, nn.LSTM, nn.Parameter,
                          nn.BatchNorm1d, nn.LayerNorm, nn.Dropout)):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                init.kaiming_normal_(m.bias.data)

    def __init__(self, n_bands, n_classes, patch_size, subimage_patch_size, frame_patch_size, padding, stride, dim,
                 ffn_dim, depth, heads, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.kernel_size = subimage_patch_size
        self.padding = padding
        self.stride = stride
        image_height, image_width = pair(self.patch_size)
        image_frame = n_bands
        self.num_h = int((image_height - self.kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
        self.num_w = int((image_width - self.kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
        self.num_f = int((image_frame - self.kernel_size[2] + 2 * padding[2]) / stride[2]) + 1
        self.num_spatial = self.num_h * self.num_w
        self.num_patches = self.num_h * self.num_w * self.num_f
        # patch_dim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        print('Model: TGRS_3_RVIT')
        print('num_w:', self.num_h, 'num_w:', self.num_w, 'num_f:', self.num_f, 'num_spa:', self.num_spatial,
              'num_patches:', self.num_patches)
        # Common layers and parameters
        self.layernorm_s = nn.LayerNorm([self.num_spatial, self.kernel_size[0] * self.kernel_size[1], ffn_dim])
        self.layernorm_f = nn.LayerNorm([self.num_spatial, ffn_dim, ffn_dim])
        self.dropout = nn.Dropout(dropout)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(ffn_dim ** 2),
            nn.Linear(ffn_dim ** 2, ffn_dim),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, ffn_dim // 2),
            nn.LayerNorm(ffn_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ffn_dim // 2, n_classes)
        )
        self.reg = nn.Sequential(
            nn.BatchNorm1d(ffn_dim ** 2),
            nn.Linear(ffn_dim ** 2, ffn_dim),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, ffn_dim // 2),
            nn.LayerNorm(ffn_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ffn_dim // 2, n_bands, bias=True)
        )


class RVIT_1(RVITBase):
    def __init__(self, n_bands, n_classes, patch_size, subimage_patch_size, frame_patch_size, padding, stride, dim,
                 ffn_dim, depth, heads, dropout):
        super().__init__(n_bands, n_classes, patch_size, subimage_patch_size, frame_patch_size, padding, stride, dim,
                         ffn_dim, depth, heads, dropout)
        self.short_term_RT = Short_term_RT(num=self.num_spatial, dim=self.kernel_size[2], depth=depth, heads=heads,
                                           ffn_dim=ffn_dim)
        self.long_term_RT_f = Long_term_RT(dim=self.kernel_size[0] * self.kernel_size[1], depth=depth, heads=heads,
                                           ffn_dim=ffn_dim)

    def forward(self, x):
        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        x = x.squeeze(1)
        # x = x.view(-1, 1, self.n_bands, self.patch_size, self.patch_size)
        x = partition_3D(x, self.kernel_size, self.padding, self.stride)  # (100, 500, 9, 16)
        x = self.short_term_RT(x)  # (100, 500, 9, 16)
        x = self.layernorm_s(x)
        x_forward = self.long_term_RT_f(x)  # (100, 9, 16, 25)
        x_forward = self.layernorm_f(x_forward)
        x_center = x_forward[:, ((self.num_spatial - 1) // 2), :, :]
        x_mlp = x_center.view(x_center.size(0), -1)
        x_mean = self.to_cls_token(reduce(x_forward, 'b n p1 p2 -> b p1 p2', reduction='mean'))
        x_reg = x_mean.view(x_mean.size(0), -1)
        x_cls_out = self.mlp_head(x_mlp)  # renamed for clarity
        x_reg_out = self.reg(x_reg)  # renamed for clarity
        return x_cls_out, x_reg_out


class RVIT_2(RVITBase):
    def __init__(self, n_bands, n_classes, patch_size, subimage_patch_size, frame_patch_size, padding, stride, dim,
                 ffn_dim, depth, heads, dropout):
        super().__init__(n_bands, n_classes, patch_size, subimage_patch_size, frame_patch_size, padding, stride, dim,
                         ffn_dim, depth, heads, dropout)
        self.short_term_RT = Short_term_RT(num=self.num_spatial, dim=self.kernel_size[2], depth=depth, heads=heads,
                                           ffn_dim=ffn_dim)
        self.long_term_RT_f = Long_term_RT(dim=self.kernel_size[0] * self.kernel_size[1], depth=depth, heads=heads,
                                           ffn_dim=ffn_dim)

    def forward(self, x):
        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        x = x.squeeze(1)
        x = torch.rot90(x, 1, [2, 3])
        x = torch.flipud(x)
        # x = x.view(-1, 1, self.n_bands, self.patch_size, self.patch_size)
        x = partition_3D(x, self.kernel_size, self.padding, self.stride)  # (100, 500, 9, 16)
        x = self.short_term_RT(x)  # (100, 500, 9, 16)
        x = self.layernorm_s(x)
        x_forward = self.long_term_RT_f(x)  # (100, 9, 16, 25)
        x_forward = self.layernorm_f(x_forward)
        x_center = x_forward[:, ((self.num_spatial - 1) // 2), :, :]
        x_mlp = x_center.view(x_center.size(0), -1)
        x_mean = self.to_cls_token(reduce(x_forward, 'b n p1 p2 -> b p1 p2', reduction='mean'))
        x_reg = x_mean.view(x_mean.size(0), -1)
        x_cls_out = self.mlp_head(x_mlp)  # renamed for clarity
        x_reg_out = self.reg(x_reg)  # renamed for clarity
        return x_cls_out, x_reg_out


# class RVIT_1(nn.Module):
#     @staticmethod
#     def weight_init(m):
#         if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.Conv2d, nn.LSTM, nn.Parameter,
#                           nn.BatchNorm1d, nn.LayerNorm, nn.Dropout, RecursiveTransformer, RTCell, Recur_self_attn,
#                           Short_term_RT, Long_term_RT)):
#             init.kaiming_normal_(m.weight.data)
#             init.kaiming_normal_(m.bias.data)
#
#     def __init__(self, n_bands, n_classes, patch_size, subimage_patch_size=subimage_patch_size,
#                  frame_patch_size=frame_patch_size,
#                  padding=padding, stride=stride, dim=32, ffn_dim=32, depth=2, heads=2, dropout=0.):
#         super(RVIT_1, self).__init__()
#         self.patch_size = patch_size
#         self.kernel_size = subimage_patch_size
#         self.padding = padding
#         self.stride = stride
#         image_height, image_width = pair(self.patch_size)
#         image_frame = n_bands
#         self.num_h = int((image_height - self.kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
#         self.num_w = int((image_width - self.kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
#         self.num_f = int((image_frame - self.kernel_size[2] + 2 * padding[2]) / stride[2]) + 1
#         self.num_spatial = self.num_h * self.num_w  ## (spatial aspect)
#         self.num_patches = self.num_h * self.num_w * self.num_f  # spatial * spectral aspect
#         patch_dim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
#         print('Model: TGRS_3_RVIT')
#         print('num_w:', self.num_h, 'num_w:', self.num_w, 'num_f:', self.num_f, 'num_spa:', self.num_spatial,
#               'num_patches:', self.num_patches)
#         # self.partial_rnn = nn.LSTM(input_size=patch_dim, hidden_size=patch_dim, num_layers=1, batch_first=True,
#         #                            bias=True, bidirectional=False)
#         # self.partial_rvit = RecurrentTransformer(patch_dim, depth, heads, mlp_dim, dropout)
#         self.layernorm_s = nn.LayerNorm(
#             [self.num_spatial, self.kernel_size[0] * self.kernel_size[1], ffn_dim])
#         self.layernorm_f = nn.LayerNorm(
#             [self.num_spatial, ffn_dim, ffn_dim])
#         self.layernorm_r = nn.LayerNorm(
#             [self.num_spatial, ffn_dim, ffn_dim])
#         # self.sub_band_with_group_cnn = Sub_band_with_group_cnn(dim=dim, dropout=dropout)
#         self.short_term_RT = Short_term_RT(num=self.num_spatial, dim=self.kernel_size[2], depth=depth, heads=heads,
#                                            ffn_dim=ffn_dim)
#         self.long_term_RT_f = Long_term_RT(dim=self.kernel_size[0] * self.kernel_size[1], depth=depth, heads=heads,
#                                            ffn_dim=ffn_dim)
#         self.long_term_RT_r = Long_term_RT(dim=self.kernel_size[0] * self.kernel_size[1], depth=depth, heads=heads,
#                                            ffn_dim=ffn_dim)
#         self.to_cls_token = nn.Identity()
#         self.depth_cnn = nn.Conv3d(in_channels=1, out_channels=frame_patch_size, kernel_size=(frame_patch_size, 3, 3),
#                                    padding=(0, 1, 1), stride=(frame_patch_size, 1, 1))
#
#         self.conv_short_term = nn.Conv2d(self.num_spatial, self.num_spatial, 3, 1, padding=(1, 1), bias=True)
#         self.conv_long_term = nn.Conv2d(self.num_spatial, self.num_spatial, 3, 1, padding=(1, 1), bias=True)
#         self.localconv = nn.Conv2d(n_bands, n_bands, 3, 1, padding=(1, 1), bias=True,
#                                    groups=n_bands // self.kernel_size[2])
#         self.patch_dim_to_dim = nn.Linear(patch_dim, dim, bias=True)
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
#         self.pos_embedding2 = nn.Parameter(torch.randn(1, self.num_spatial + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(dropout)
#         self.project = nn.Linear(dim * self.num_f, dim, bias=True)
#         self.layernorm = nn.LayerNorm(dim)
#         self.layernorm2 = nn.LayerNorm(patch_dim)
#         self.to_latent = nn.Identity()
#         self.batchnorm2d_s = nn.BatchNorm2d(num_features=self.num_spatial)
#         self.batchnorm2d_l = nn.BatchNorm2d(num_features=self.num_spatial)
#         self.batchnorm3d = nn.BatchNorm3d(num_features=frame_patch_size)
#         self.mlp_head = nn.Sequential(
#             nn.BatchNorm1d(ffn_dim ** 2),
#             nn.Linear(ffn_dim ** 2,
#                       ffn_dim),
#             nn.Dropout(dropout),
#             nn.Linear(ffn_dim, ffn_dim // 2),
#             nn.LayerNorm(ffn_dim // 2),
#             nn.Dropout(dropout),
#             nn.GELU(),
#             nn.Linear(ffn_dim // 2, n_classes)
#         )
#
#         self.aux_loss_weight = 1
#
#         self.reg = nn.Sequential(
#             nn.BatchNorm1d(ffn_dim ** 2),
#             nn.Linear(ffn_dim ** 2,
#                       ffn_dim),
#             nn.Dropout(dropout),
#             nn.Linear(ffn_dim, ffn_dim // 2),
#             nn.LayerNorm(ffn_dim // 2),
#             nn.Dropout(dropout),
#             nn.GELU(),
#             nn.Linear(ffn_dim // 2, n_bands, bias=True)
#         )
#
#     def forward(self, x):
#         # Make 4D data ((Batch x) Planes x Channels x Width x Height)
#         x = x.squeeze(1)
#         # x = x.view(-1, 1, self.n_bands, self.patch_size, self.patch_size)
#
#         x = partition_3D(x, self.kernel_size, self.padding, self.stride)  # (100, 500, 9, 16)
#
#         x = self.short_term_RT(x)  # (100, 500, 9, 16)
#         # x = self.conv_short_term(x)
#         # print('x', x.shape)
#
#         x = self.layernorm_s(x)
#         # x = self.batchnorm2d_s(x)
#
#         x_forward = self.long_term_RT_f(x)  # (100, 9, 16, 25)
#         # x_forward = self.conv_long_term(x_forward)
#         # print('x_forward', x_forward.shape)
#         # x_forward = self.batchnorm2d_l(x_forward)
#         x_forward = self.layernorm_f(x_forward)
#
#         # x_reverse = self.long_term_RT_r(torch.flip(x, dims=[1]))
#         # x_reverse = self.batchnorm2d_l(x_reverse)
#         # x = torch.cat([x_forward, torch.flip(x_reverse, dims=[1])], dim=3)  # fixed the reversal
#
#         # x_cls = self.to_cls_token(reduce(x_forward, 'b n p1 p2 -> b p1 p2', reduction='mean'))
#
#         x_center = x_forward[:, ((self.num_spatial - 1) // 2), :, :]
#         x_mlp = x_center.view(x_center.size(0), -1)
#         x_mean = self.to_cls_token(reduce(x_forward, 'b n p1 p2 -> b p1 p2', reduction='mean'))
#         x_reg = x_mean.view(x_mean.size(0), -1)
#
#         # x_mlp = x_forward.view(x_forward.size(0), -1)
#
#         x_cls_out = self.mlp_head(x_mlp)  # renamed for clarity
#         x_reg_out = self.reg(x_reg)  # renamed for clarity
#
#         return x_cls_out, x_reg_out


# class RVIT_2(nn.Module):
#     @staticmethod
#     def weight_init(m):
#         if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.Conv2d, nn.LSTM, nn.Parameter,
#                           nn.BatchNorm1d, nn.LayerNorm, nn.Dropout, RecursiveTransformer, RTCell,
#                           Recur_self_attn,
#                           Short_term_RT, Long_term_RT)):
#             init.kaiming_normal_(m.weight.data)
#             init.kaiming_normal_(m.bias.data)
#
#     def __init__(self, n_bands, n_classes, patch_size, subimage_patch_size=subimage_patch_size,
#                  frame_patch_size=frame_patch_size,
#                  padding=padding, stride=stride, dim=32, ffn_dim=32, depth=2, heads=2, dropout=0.):
#         super(RVIT_2, self).__init__()
#         self.patch_size = patch_size
#         self.kernel_size = subimage_patch_size
#         self.padding = padding
#         self.stride = stride
#         image_height, image_width = pair(self.patch_size)
#         image_frame = n_bands
#         self.num_h = int((image_height - self.kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
#         self.num_w = int((image_width - self.kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
#         self.num_f = int((image_frame - self.kernel_size[2] + 2 * padding[2]) / stride[2]) + 1
#         self.num_spatial = self.num_h * self.num_w  ## (spatial aspect)
#         self.num_patches = self.num_h * self.num_w * self.num_f  # spatial * spectral aspect
#         patch_dim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
#         print('Model: TGRS_3_RVIT')
#         print('num_w:', self.num_h, 'num_w:', self.num_w, 'num_f:', self.num_f, 'num_spa:', self.num_spatial,
#               'num_patches:', self.num_patches)
#         self.partial_rnn = nn.LSTM(input_size=patch_dim, hidden_size=patch_dim, num_layers=1, batch_first=True,
#                                    bias=True, bidirectional=False)
#         # self.partial_rvit = RecurrentTransformer(patch_dim, depth, heads, mlp_dim, dropout)
#         self.layernorm_s = nn.LayerNorm(
#             [self.num_spatial, self.kernel_size[0] * self.kernel_size[1], ffn_dim])
#         self.layernorm_f = nn.LayerNorm(
#             [self.num_spatial, ffn_dim, ffn_dim])
#         self.layernorm_r = nn.LayerNorm(
#             [self.num_spatial, ffn_dim, ffn_dim])
#         # self.sub_band_with_group_cnn = Sub_band_with_group_cnn(dim=dim, dropout=dropout)
#         self.short_term_RT = Short_term_RT(num=self.num_spatial, dim=self.kernel_size[2], depth=depth,
#                                            heads=heads,
#                                            ffn_dim=ffn_dim)
#         self.long_term_RT_f = Long_term_RT(dim=self.kernel_size[0] * self.kernel_size[1], depth=depth,
#                                            heads=heads,
#                                            ffn_dim=ffn_dim)
#         self.long_term_RT_r = Long_term_RT(dim=self.kernel_size[0] * self.kernel_size[1], depth=depth,
#                                            heads=heads,
#                                            ffn_dim=ffn_dim)
#         self.to_cls_token = nn.Identity()
#         self.depth_cnn = nn.Conv3d(in_channels=1, out_channels=frame_patch_size,
#                                    kernel_size=(frame_patch_size, 3, 3),
#                                    padding=(0, 1, 1), stride=(frame_patch_size, 1, 1))
#
#         self.conv_short_term = nn.Conv2d(self.num_spatial, self.num_spatial, 3, 1, padding=(1, 1), bias=True)
#         self.conv_long_term = nn.Conv2d(self.num_spatial, self.num_spatial, 3, 1, padding=(1, 1), bias=True)
#         self.localconv = nn.Conv2d(n_bands, n_bands, 3, 1, padding=(1, 1), bias=True,
#                                    groups=n_bands // self.kernel_size[2])
#         self.patch_dim_to_dim = nn.Linear(patch_dim, dim, bias=True)
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
#         self.pos_embedding2 = nn.Parameter(torch.randn(1, self.num_spatial + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(dropout)
#         self.project = nn.Linear(dim * self.num_f, dim, bias=True)
#         self.layernorm = nn.LayerNorm(dim)
#         self.layernorm2 = nn.LayerNorm(patch_dim)
#         self.to_latent = nn.Identity()
#         self.batchnorm2d_s = nn.BatchNorm2d(num_features=self.num_spatial)
#         self.batchnorm2d_l = nn.BatchNorm2d(num_features=self.num_spatial)
#         self.batchnorm3d = nn.BatchNorm3d(num_features=frame_patch_size)
#         self.mlp_head = nn.Sequential(
#             nn.BatchNorm1d(ffn_dim ** 2),
#             nn.Linear(ffn_dim ** 2,
#                       ffn_dim),
#             nn.Dropout(dropout),
#             nn.Linear(ffn_dim, ffn_dim // 2),
#             nn.LayerNorm(ffn_dim // 2),
#             nn.Dropout(dropout),
#             nn.GELU(),
#             nn.Linear(ffn_dim // 2, n_classes)
#         )
#
#         self.aux_loss_weight = 1
#
#         self.reg = nn.Sequential(
#             nn.BatchNorm1d(ffn_dim ** 2),
#             nn.Linear(ffn_dim ** 2,
#                       ffn_dim),
#             nn.Dropout(dropout),
#             nn.Linear(ffn_dim, ffn_dim // 2),
#             nn.LayerNorm(ffn_dim // 2),
#             nn.Dropout(dropout),
#             nn.GELU(),
#             nn.Linear(ffn_dim // 2, n_bands, bias=True)
#         )
#
#     def forward(self, x):
#         # Make 4D data ((Batch x) Planes x Channels x Width x Height)
#         x = x.squeeze(1)
#         x = torch.rot90(x, 1, [2, 3])
#         # x = x.view(-1, 1, self.n_bands, self.patch_size, self.patch_size)
#
#         x = partition_3D(x, self.kernel_size, self.padding, self.stride)  # (100, 500, 9, 16)
#
#         x = self.short_term_RT(x)  # (100, 500, 9, 16)
#         # x = self.conv_short_term(x)
#         # print('x', x.shape)
#
#         x = self.layernorm_s(x)
#         # x = self.batchnorm2d_s(x)
#
#         x_forward = self.long_term_RT_f(x)  # (100, 9, 16, 25)
#         # x_forward = self.conv_long_term(x_forward)
#         # print('x_forward', x_forward.shape)
#         # x_forward = self.batchnorm2d_l(x_forward)
#         x_forward = self.layernorm_f(x_forward)
#
#         # x_reverse = self.long_term_RT_r(torch.flip(x, dims=[1]))
#         # x_reverse = self.batchnorm2d_l(x_reverse)
#         # x = torch.cat([x_forward, torch.flip(x_reverse, dims=[1])], dim=3)  # fixed the reversal
#
#         # x_cls = self.to_cls_token(reduce(x_forward, 'b n p1 p2 -> b p1 p2', reduction='mean'))
#
#         x_center = x_forward[:, ((self.num_spatial - 1) // 2), :, :]
#         x_mlp = x_center.view(x_center.size(0), -1)
#         x_mean = self.to_cls_token(reduce(x_forward, 'b n p1 p2 -> b p1 p2', reduction='mean'))
#         x_reg = x_mean.view(x_mean.size(0), -1)
#
#         # x_mlp = x_forward.view(x_forward.size(0), -1)
#
#         x_cls_out = self.mlp_head(x_mlp)  # renamed for clarity
#         x_reg_out = self.reg(x_reg)  # renamed for clarity
#
#         return x_cls_out, x_reg_out

class RVIT(nn.Module):
    def __init__(self, n_bands, n_classes, patch_size, subimage_patch_size=subimage_patch_size,
                 frame_patch_size=frame_patch_size,
                 padding=padding, stride=stride, dim=32, ffn_dim=32, depth=2, heads=2, dropout=0.):
        super().__init__()
        self.dir_1 = RVIT_1(n_bands, n_classes, patch_size, subimage_patch_size, frame_patch_size, padding, stride, dim,
                            ffn_dim, depth, heads, dropout)
        self.dir_2 = RVIT_2(n_bands, n_classes, patch_size, subimage_patch_size, frame_patch_size, padding, stride, dim,
                            ffn_dim, depth, heads, dropout)
        self.aux_loss_weight = 1

    def forward(self, x):
        x1, x1reg = self.dir_1(x)
        x2, x2reg = self.dir_2(x)
        return x1 + x2, x1reg + x2reg

        # x_dir_6 = torch.rot90(x_dir_1, 1, [2, 3])
        # x_dir_3 = torch.rot90(x_dir_6, 1, [2, 3])
        # x_dir_8 = torch.rot90(x_dir_3, 1, [2, 3])
        # x_dir_7 = torch.flipud(x_dir_1)
        # x_dir_2 = torch.flipud(x_dir_6)
        # x_dir_5 = torch.flipud(x_dir_3)
        # x_dir_4 = torch.flipud(x_dir_8)

        # out_1 = F.normalize(out_1, p=2, dim=1)
        # reg_1 = F.normalize(reg_1, p=2, dim=1)
        # out_2, reg_2= self.one_direction_operation(x_dir_2)
        # out_3, reg_3 = self.one_direction_operation(x_dir_3)
        #     out_4 = self.ond_dir_operation(x_dir_4)
        #     out_5, reg_5 = self.one_direction_operation(x_dir_5)
        #     out_6, reg_6 = self.one_direction_operation(x_dir_6)
        #     out_7, reg_7 = self.one_direction_operation(x_dir_7)
        #     out_8, reg_8 = self.one_direction_operation(x_dir_8)
