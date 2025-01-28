# Standard library
from typing import Iterable, Optional

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 max_seq_len: int,
                 n_heads: int,
                 n_blocks: int,
                 p_drop: float=0.0) -> None:
        super().__init__()
        
        # Layers
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.dropout = Dropout(p_drop)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads, p_drop) 
                                      for _ in range(n_blocks)])
        self.ln = LayerNorm(d_model)
        self.deembed = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # Embedding
        x = self.embed(x)

        # Positional Encoding
        x = self.pe(x)

        # Dropout
        x = self.dropout(x)

        # Transformer Blocks
        x = self.blocks(x)

        # Layer Norm
        x = self.ln(x)

        # De-embedding
        x = self.deembed(x)

        return x


##################################  Layers  ###################################

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p_drop: float) -> None:
        super().__init__()

        # Params
        self.d_model = d_model
        self.n_heads = n_heads
        self.p_drop = p_drop

        # Layers
        self.ln_mha = LayerNorm(d_model)
        self.ln_ff = LayerNorm(d_model)
        self.dropout = Dropout(p_drop)
        self.mha = MultiheadAttention(d_model, n_heads, p_drop)
        self.ff = FeedForward(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # Multi-head attention
        y = self.ln_mha(x)
        y = self.mha(x)
        y = self.dropout(x)
        y = y + x

        # Feed Forward
        x = y
        y = self.ln_ff(x)
        y = self.ff(x)
        y = self.dropout(x) 
        y = y + x

        return y

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p_drop: int) -> None:
        super().__init__()

        # Params
        if d_model % n_heads != 0:
            raise ValueError(f'n_heads ({n_heads}) does not divide d_model {d_model}.')
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.p_drop = p_drop

        # Layers tensors
        self.wqkv = nn.Linear(d_model, 3 * d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(-1)
        self.dropout = Dropout(p_drop)
    
    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        # Batching
        batched = x.ndim == 3
        if not batched:
            x = x.unsqueeze(0)

        # Dimeansion sizes
        b = x.shape[0] if x.ndim == 3 else 1  # batch
        s = x.shape[-2]  # sequence
        h = self.n_heads  # head
        f = self.head_size  # feature

        # Q, K, V
        q, k, v = self.wqkv(x).chunk(3, dim=-1)
        q  = q.reshape(b, s, h, f).permute(0, 2, 1, 3)  # (b, h, s, f)
        kT = k.reshape(b, s, h, f).permute(0, 2, 3, 1)  # (b, h, f, s)
        v  = v.reshape(b, s, h, f).permute(0, 2, 1, 3)  # (b, h, s, f)

        # Attention
        mask = torch.full((s,s), -torch.inf).triu(1) if mask is None else mask
        scores = q @ kT / np.sqrt(f) + mask
        weights = self.softmax(scores)
        weights = self.dropout(weights)
        superpos = weights @ v  # (b, h, s, f)

        # Linear
        out = self.wo(superpos.permute(0, 2, 1, 3).reshape(b, s, h * f))

        # Return
        if batched:
            return out
        else:
            return out.squeeze(0)


class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

        # Params
        self.d_model = d_model

        # Layers
        self.layers = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                    nn.ReLU(),
                                    nn.Linear(4 * d_model, d_model))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()

        # Params
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding tensor
        self.emb = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, x: Tensor) -> Tensor:
        return self.emb[x]


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
        pe = torch.empty(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(ps / wavelen)
        pe[:, 1::2] = torch.cos(ps / wavelen)
        
        # Register buffer
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(f'seq_len ({seq_len}) is larger than max_seq_len ({self.max_seq_len}).')
        return x + self.pe[..., :seq_len, :]


class Dropout(nn.Module):
    def __init__(self, p_drop: float) -> None:
        super().__init__()

        # Params
        self. p_drop = p_drop

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = torch.rand_like(x) < self.p_drop
            return x.masked_fill(mask, 0) / (1 - self.p_drop)
        else:
            return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Iterable[int], 
                 eps: float=0.00001) -> None:
        super().__init__()

        # Params
        self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape 
        self.eps = eps
        self.n_dims = len(self.normalized_shape)
        self.dims = list(range(-self.n_dims, 0))

        # Gain and bias
        self.gain = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-self.n_dims:] != torch.Size(self.normalized_shape):
            raise ValueError(f'Expected last {self.n_dims} dims of x to have shape {self.normalized_shape} but got {x.shape[-self.n_dims:]}.')

        mean = torch.mean(x, dim=self.dims, keepdim=True)
        var = torch.var(x, dim=self.dims, unbiased=False, keepdim=True)

        return ((x - mean) / torch.sqrt(var + self.eps)) * self.gain + self.bias


        

        
