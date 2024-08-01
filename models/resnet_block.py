from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

class WeightStandardizedConv2d(nn.Conv2d):
    def __init__(self, weight, bias, stride, padding, dilation, groups):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        
        weight = self.weight
        
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        
        normalized_weight = (weight - mean) / (var + eps).sqrt()
        
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        
        
        
        
        
        