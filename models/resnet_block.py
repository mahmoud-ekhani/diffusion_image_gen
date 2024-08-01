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
        
        
        
        
        