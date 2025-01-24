# Standard library

# Third-party
import torch
import torch.nn as nn
from torch import Tensor


# class Transformer(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Layers
#         self.embed = Embedding(vocab_size, d_model)
#         self.pe = PositionalEncoding(d_model, max_wavelen)
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
        k = 1 / torch.sqrt(d_model)
        self.emb = nn.Parameter(k * (2 * torch.randn(vocab_size, d_model) - 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.emb[x]





