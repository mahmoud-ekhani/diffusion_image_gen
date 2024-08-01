import torch
import torch.nn as nn
from einops import einsum, rearrange

class Attention(nn.Module):
    def __init__(self, in_dim, num_heads=4, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        
        self.scale = head_dim ** -0.5
        
        hidden_dim = num_heads * head_dim
        
        self.to_qkv = nn.Conv2d(in_channels=in_dim,
                                out_channels=hidden_dim * 3,
                                kernel_size=1,
                                bias=False)
        
        self.to_out = nn.Conv2d(in_channels=hidden_dim,
                                out_channels=in_dim,
                                kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.num_heads), 
            qkv
        )
        
        q = q * self.scale
        
        sim = einsum('b h c i, b h c j -> b h i j', q, k)
        
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        
        attn = sim.softmax(dim=-1)
        
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        return self.to_out(out)
    
    
class LinearAttention(nn.Module):
    def __init__(self, in_dim, num_heads=4, head_dim=32):
        super().__init__()
        
        self.scale = head_dim ** -0.5
        
        self.num_heads = num_heads
        
        hidden_dim = num_heads * head_dim
        
        self.to_qkv = nn.Conv2d(in_channels=in_dim,
                                out_channels=hidden_dim*3,
                                kernel_size=1,
                                bias=False)
        
        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=in_dim,
                      kernel_size=1),
            nn.GroupNorm(num_groups=1,
                         num_channels=in_dim)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.num_heads),
            qkv
        )
        
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        q = q * self.scale
        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
        
        