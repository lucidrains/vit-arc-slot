import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from slot_attention import SlotAttention

from x_transformers import Attention, FeedForward, RMSNorm

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def pack_with_inverse(t, pattern):
    t, packed_shape = pack(t, pattern)

    def inverse(out, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(out, packed_shape, inverse_pattern)

    return t, inverse

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
        num_slots = 50,
        slot_attn_iterations = 3,
        dropout = 0.,
        dim_output = None
    ):
        super().__init__()
        self.input_shape = (channels, image_size, image_size)
        assert divisible_by(image_size, patch_size)

        self.slot_attn = SlotAttention(
            num_slots = num_slots,
            dim = dim,
            iters = slot_attn_iterations
        )

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

        self.to_pred = nn.Linear(dim, default(dim_output, dim))

    def forward(
        self,
        images
    ):
        assert images.shape[-3:] == self.input_shape

        tokens = self.to_tokens(images)

        tokens = rearrange(tokens, 'b h w d -> b (h w) d')

        objects = self.slot_attn(tokens)

        tokens, unpack_fn = pack_with_inverse([objects, tokens], 'b * d')

        for attn_norm, attn, ff_norm, ff in self.layers:
            tokens = attn(attn_norm(tokens)) + tokens
            tokens = ff(ff_norm(tokens)) + tokens

        tokens = self.final_norm(tokens)

        object_tokens, tokens = unpack_fn(tokens)

        return self.to_pred(tokens)
