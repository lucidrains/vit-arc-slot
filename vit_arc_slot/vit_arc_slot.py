import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from slot_attention import SlotAttention

from x_transformers import Attention, FeedForward, RMSNorm

from einops import rearrange
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# main class

class SlotViTArc(Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        channels = 3,
        depth = 6,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        dropout = 0.
    ):
        super().__init__()
        self.input_shape = (channels, image_size, image_size)
        assert divisible_by(image_size, patch_size)

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(ModuleList([
                RMSNorm(dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout, **attn_kwargs),
                RMSNorm(dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = dropout, **ff_kwargs),
            ]))

        self.layers = layers
        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        images
    ):
        assert images.shape[-3:] == self.input_shape

        tokens = self.to_tokens(images)

        tokens = rearrange(tokens, 'b h w d -> b (h w) d')

        for attn_norm, attn, ff_norm, ff in self.layers:
            tokens = attn(attn_norm(tokens)) + tokens
            tokens = ff(ff_norm(tokens)) + tokens

        return self.final_norm(tokens)
