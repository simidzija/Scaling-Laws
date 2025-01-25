# Standard library

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# class Transformer(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Layers
#         self.embed = Embedding(vocab_size, d_model)
#         self.pe = PositionalEncoding(d_model, max_seq_len)
#         self.dropout = Dropout(p_drop)
#         self.blocks = nn.Sequential([Block(d_model, n_heads, p_drop)] 
#                                     for _ in range(n_blocks))
#         self.ln = LayerNorm(d_model)
#         self.deembed = DeEmbedding(d_model, vocab_size)

#     def forward(self, x: Tensor) -> Tensor:
#         # Embedding
#         x = self.embed(x)

#         # Positional Encoding
#         x = self.pe(x)

#         # Dropout
#         x = self.dropout(x)

#         # Transformer Blocks
#         x = self.blocks(x)

#         # Layer Norm
#         x = self.ln(x)

#         # De-embedding
#         x = self.deembed(x)

#         return x


##################################  Layers  ###################################

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()

        # Params
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding tensor
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int) -> None:
        if d_model % 2 != 0:
            raise ValueError(f'd_model must be even but got {d_model}')
        super().__init__()

        # Params
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # PE tensor
        ps = torch.arange(0, max_seq_len)[:, None]
        ds = torch.arange(0, d_model, 2)
        wavelen = max_seq_len ** (ds / d_model)
        weight = torch.zeros(max_seq_len, d_model)
        weight[:, 0::2] = torch.sin(ps / wavelen)
        weight[:, 1::2] = torch.cos(ps / wavelen)
        
        # Register buffer
        self.register_buffer('weight', weight)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        return x + self.weight[..., :seq_len, :]



