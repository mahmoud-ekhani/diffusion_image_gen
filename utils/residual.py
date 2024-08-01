from inspect import isfunction
import torch.nn as nn
from einops.layers.torch import Rearrange

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
    
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )
    
def Downsample(dim, dim_out=None):
    """
    downsample the height and width of the input tensor by a factor of 2.

    args:
        dim (int): number of input channels.
        dim_out (int, optional): number of output channels after the convolution. 
                                 Defaults to the value of dim if not specified.

    returns:
        nn.Sequential: A sequential container that downsamples the input tensor.
    """
    
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x