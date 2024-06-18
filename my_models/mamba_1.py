import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from Mamba.mamba import Mamba, MambaConfig
from multiscan_v3 import snake_flatten, to_uturn_sequence

# from mamba_ssm import Mamba

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class Mamba_1(nn.Module):
    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=128, depth=1, emb_dropout=0.):
        super(Mamba_1, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.to_patch_embedding = nn.Sequential(
            # nn.Conv2d(channels, dim, 3,1,1),
            # Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.mamba_config = MambaConfig(d_model=dim, n_layers=depth, expand_factor=2)
        self.mamba = Mamba(self.mamba_config)
        self.lstm = nn.LSTM(dim, dim, num_layers=depth, batch_first=True)
        self.linear_head = nn.Sequential(
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
        # Tokenization
        self.L = ((image_height // patch_height) * (image_width // patch_width)) // 2
        self.cT = dim
        self.token_wA = nn.Parameter(torch.empty(1, self.L, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

    def tokenizer(self, x): #x(b,n,d)
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        return T

    def forward(self, img):
        device = img.device
        img = img.squeeze(1)
        img = snake_flatten(img)
        x = self.to_patch_embedding(img)
        # x = snake_flatten(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.tokenizer(x)

        b, n, d = x.shape
        c = int(n + 1) // 2

        x = self.mamba(x)

        x_mean = torch.mean(x, dim=1)
        x_last = x[:, -1]
        x_center = x[:, c]
        x_adapool = F.adaptive_avg_pool2d(x, (1, self.dim)).view(b, -1)

        x = self.to_latent(x_adapool)
        return self.linear_head(x), self.regression(x)
