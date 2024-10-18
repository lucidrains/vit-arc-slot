import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from slot_attention import MultiHeadSlotAttention

from x_transformers import Attention, FeedForward, RMSNorm

from einops import rearrange, pack, unpack, repeat
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
        slot_attn_heads = 4,
        dropout = 0.,
        dim_output = None,
        images_add_coords = False
    ):
        super().__init__()
        self.input_shape = (channels, image_size, image_size)
        assert divisible_by(image_size, patch_size)

        # maybe coord conv

        self.image_size = image_size
        self.images_add_coords = images_add_coords

        if images_add_coords:
            channels += 2

        # slot attention

        self.slot_attn = MultiHeadSlotAttention(
            num_slots = num_slots,
            dim = dim,
            heads = slot_attn_heads,
            iters = slot_attn_iterations
        )

        # some math

        patched_dim = image_size // patch_size
        num_patches = patched_dim ** 2
        patch_dim = channels * patch_size ** 2

        # project patches to tokens

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        # absolute axial positions

        self.width_pos_emb = nn.Embedding(patched_dim, dim)
        self.height_pos_emb = nn.Embedding(patched_dim, dim)

        # layers

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
        batch, device = images.shape[0], images.device
        assert images.shape[-3:] == self.input_shape

        # maybe add coords

        if self.images_add_coords:
            image_size_seq = torch.arange(self.image_size, device = device)
            coords = torch.stack(torch.meshgrid((image_size_seq, image_size_seq), indexing = 'ij'))
            coords = repeat(coords, '... -> b ...', b = batch)
            images = torch.cat((images, coords), dim = 1)

        # patches to tokens

        tokens = self.to_tokens(images)

        height_seq = torch.arange(tokens.shape[1], device = device)
        width_seq = torch.arange(tokens.shape[2], device = device)

        height_pos_emb = self.height_pos_emb(height_seq)
        width_pos_emb = self.width_pos_emb(width_seq)

        tokens = tokens + rearrange(height_pos_emb, 'h d -> h 1 d') + rearrange(width_pos_emb, 'w d -> 1 w d')

        tokens = rearrange(tokens, 'b h w d -> b (h w) d')

        objects = self.slot_attn(tokens)

        tokens, unpack_fn = pack_with_inverse([objects, tokens], 'b * d')

        for attn_norm, attn, ff_norm, ff in self.layers:
            tokens = attn(attn_norm(tokens)) + tokens
            tokens = ff(ff_norm(tokens)) + tokens

        tokens = self.final_norm(tokens)

        object_tokens, tokens = unpack_fn(tokens)

        return self.to_pred(tokens)
