import torch
from torch import nn, einsum
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.3):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 将经过线性层的输出拆分为三块，分别是q、k、v:(b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


###MLP
class MLP(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.3):
        super(MLP,self).__init__()
        self.Layer=nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(p=dropout)
        )
    def forward(self,x):
        return self.Layer(x)

###transformer
class Block(nn.Module):
    def __init__(self, dim=768, hidden_dim=3072):
        super(Block, self).__init__()
        self.dim = dim
        self.hidden_size = hidden_dim
        self.attention_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MLP(dim,hidden_dim)
        self.attn = Attention(dim)

    def forward(self, x):
        # input x - [B,196,768]
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x
