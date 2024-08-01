from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from utils.residual import exists

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(in_channels=dim,
                                             out_channels=dim_out,
                                             kernel_size=3,
                                             padding=1)
        
        self.norm = nn.GroupNorm(num_groups=groups,
                                 num_channels=dim_out)
        
        self.act = nn.SiLU()
        
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        
        x = self.norm(x)
        
        if exists(scale_shift):
            scale, shift = scale_shift
            
            x = x * (scale + 1) + shift
            
        x = self.act(x)
        
        return x
        
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(),
                          nn.Linear(in_features=time_emb_dim,
                                    out_features=dim_out * 2)) if exists(time_emb_dim)
            else None
        )
        
        self.block1 = Block(dim=dim,
                            dim_out=dim_out,
                            groups=groups)
        
        self.block2 = Block(dim=dim_out,
                            dim_out=dim_out,
                            groups=groups)
        
        self.res_conv = nn.Conv2d(in_channels=dim,
                                  out_channels=dim_out,
                                  kernel_size=1) if dim != dim_out else nn.Identity()
        
        def forward(self, x, time_emb=None):
            scale_shift = None
            
            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                
                time_emb = rearrange(time_emb, 'b c -> b c 1 1')
                
                scale_shift = time_emb.chunk(2, dim=1)
                
            h = self.block1(x, scale_shift=scale_shift)
            h = self.block2(h)
            
            return h + self.res_conv(h)
        
class PreNorm(nn.Module):
    def __init__(self, in_dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(num_groups=1,
                                 num_channels=in_dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)                
        
        
        
        