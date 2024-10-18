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

def softclamp(t, value):
    # clamp from 0. to value
    half_value = value / 2
    t = t - half_value
    t = t / half_value
    t = t.tanh()
    t = t * half_value
    return t + half_value

def pack_with_inverse(t, pattern):
    t, packed_shape = pack(t, pattern)

    def inverse(out, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(out, packed_shape, inverse_pattern)

    return t, inverse

# relative positions to attention bias mlp
# will use this in place of their 2d alibi position with symmetry breaking by using left and right slope

class RelativePositionMLP(Module):
    def __init__(
        self,
        *,
        dim,
        heads
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        shape: tuple[int, int],
        learned_coords = None
    ):
        h, w, device = *shape, self.device

        h_seq = torch.arange(h, device = device)
        w_seq = torch.arange(w, device = device)

        coords = torch.stack(torch.meshgrid((h_seq, w_seq), indexing = 'ij'), dim = -1)

        coords = rearrange(coords, 'i j c -> (i j) c')

        if exists(learned_coords):
            coords = repeat(coords, '... -> b ...', b = learned_coords.shape[0])
            coords = torch.cat((learned_coords, coords), dim = -2)

        rel_coords = rearrange(coords, '... i c -> ... i 1 c') - rearrange(coords, '... j c -> ... 1 j c')

        attn_bias = self.mlp(rel_coords.float())

        return attn_bias

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

        self.slot_to_coords = nn.Linear(dim, 2)

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

        # relative positions

        self.rel_pos_mlp = RelativePositionMLP(dim = dim // 4, heads = heads)

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

        height_patches, width_patches = tokens.shape[1:3]

        height_seq = torch.arange(height_patches, device = device)
        width_seq = torch.arange(width_patches, device = device)

        height_pos_emb = self.height_pos_emb(height_seq)
        width_pos_emb = self.width_pos_emb(width_seq)

        tokens = tokens + rearrange(height_pos_emb, 'h d -> h 1 d') + rearrange(width_pos_emb, 'w d -> 1 w d')

        patches_dims = tuple(tokens.shape[1:3])

        tokens = rearrange(tokens, 'b h w d -> b (h w) d')

        objects = self.slot_attn(tokens)

        num_objects = objects.shape[-2]

        # eventually, will have to figure out how to determine each slot's coordinates, and also feed that into the mlp

        slot_coords = self.slot_to_coords(objects)

        # soft clamp to make sure predicted coordinates are not out of bounds

        slot_coords_height, slot_coords_width = slot_coords.unbind(dim = -1)
        slot_coords_height = softclamp(slot_coords_height, height_patches)
        slot_coords_width = softclamp(slot_coords_width, width_patches)

        attn_bias = self.rel_pos_mlp(patches_dims, learned_coords = slot_coords)

        tokens, unpack_fn = pack_with_inverse([objects, tokens], 'b * d')

        for attn_norm, attn, ff_norm, ff in self.layers:
            tokens = attn(attn_norm(tokens), attn_bias = attn_bias) + tokens
            tokens = ff(ff_norm(tokens)) + tokens

        tokens = self.final_norm(tokens)

        object_tokens, tokens = unpack_fn(tokens)

        return self.to_pred(tokens)
