import torch
from torch import nn, tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class SlotViT(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        images
    ):
        return images
