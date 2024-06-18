import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _triple

class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * 4)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)

class nattention3d(nn.Module):
    def __init__(self, in_planes):
        super(nattention3d, self).__init__()
        bn_eps = 1e-3
        bn_mmt = 0.01
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        k = in_planes * 4
        self.a = nn.Conv3d(
            in_channels=in_planes,
            out_channels=k,
            kernel_size=[3, 1, 1],
            padding=[1, 0, 0],
        )
        self.bn = nn.BatchNorm3d(k)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=k,
            out_channels=in_planes,
            kernel_size=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False
        )
        self.sig = nn.Sigmoid()
        self.b.skip_init = True
        self.b.weight.data.zero_()  # to make sure the initial values
        # for the output is 1.
    def forward(self, x):
        y = x
        x = self.avgpool(x)
        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x)
        x = self.sig(x)
        y = x * y
        return y


class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=False, K=4, temperature=4):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)
        self.local = nattention3d(in_planes)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size[0], kernel_size[1], kernel_size[2]), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        y = self.local(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)
        y = y.view(1, -1, depth, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(y, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv3d(y, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'ViP_S': _cfg(crop_pct=0.9),
    'ViP_M': _cfg(crop_pct=0.9),
    'ViP_L': _cfg(crop_pct=0.875),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        #print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class ConvPermuteMLP(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.mlp_c = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1, groups=dim, bias=qkv_bias),
           # nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
        )
        self.mlp_h = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, groups=dim, bias=qkv_bias),
           # nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
        )
        self.mlp_w = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        #self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x1 = x.reshape(B, H, W, C).permute(0, 3, 1, 2).reshape(B, C, H, W)
        h = self.mlp_c(x1)
        w = self.mlp_h(x1)
        c = self.mlp_w(x1)

        a = (h + w + c).flatten(2).mean(2)

        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).permute(0, 1, 4, 2, 3)

        x = h * a[0] + w * a[1] + c * a[2]
        x = x.reshape(B, H, W, C)  
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=ConvPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        # print('input shape:', x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=15, patch_size=3, in_chans=3, embed_dim=16):
        super().__init__()
        #### for GRSS (10, 9) pu(15,9)step(2,1) XA(46,30) for in(20,16)
        self.proj1_1 = Dynamic_conv3d(in_planes=1, out_planes=4, kernel_size=(3, 3, 3), ratio=8, stride=(2, 2, 2), padding=1, )
        self.proj2_1 = Dynamic_conv3d(in_planes=4, out_planes=8, kernel_size=(3, 3, 3), ratio=8, stride=(2, 1, 1), padding=1, )
    def forward(self, x):
        x = self.proj1_1(x)
        x = self.proj2_1(x)
        B, D, H, W, C = x.shape
        x = x.reshape(B, D*H, W, C)
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3., qkv_bias=True, qk_scale=None, \
                 attn_drop=0, drop_path_rate=0., skip_lam=1.0, mlp_fn=WeightedPermuteMLP, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, \
                                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))

    blocks = nn.Sequential(*blocks)

    return blocks


class HiT(nn.Module):
    """ Vision Permutator
    """

    def __init__(self, layers, img_size=15, patch_size=3, in_chans=3, num_classes=1000,
                 embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, mlp_fn=ConvPermuteMLP):

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, skip_lam=skip_lam,
                                 mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(368, num_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))
