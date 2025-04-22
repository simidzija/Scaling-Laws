"""
Transformer model.

Contains:
  - Transformer: class for decoder only transformer
  - Block: class defining single transformer layer
  - MultiheadAttention: class defining attention sublayer
  - FeedForward: class defining feed forward sublayer
  - DeEmbedding: class for de-embedding layer that is tied to embedding layer
  - PositionalEncoding: class for sin/cos positional encodings

Running this as a script will train a transformer to reverse integer sequences.
This provides a quick check that the architecture (in particular the self attention) is working as it should.
"""

# Standard library
from typing import Optional

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Transformer(nn.Module):
    """Decoder only transformer model."""
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 max_seq_len: int,
                 n_heads: int,
                 n_blocks: int,
                 device: torch.device | str,
                 p_drop: float=0.0) -> None:
        super().__init__()

        # Hyperparameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.p_drop = p_drop
        self.device = torch.device(device)
        self.hyperparams_dict = {'vocab_size': vocab_size,
                                 'd_model': d_model,
                                 'max_seq_len': max_seq_len,
                                 'n_heads': n_heads,
                                 'n_blocks': n_blocks,
                                 'p_drop': p_drop,
                                 'device': device}
        
        # Layers
        self.embed = nn.Embedding(vocab_size, d_model, device=device)
        self.pe = PositionalEncoding(d_model, max_seq_len, device=device)
        self.dropout = nn.Dropout(p_drop)
        self.blocks = nn.Sequential(
            *[Block(d_model, n_heads, p_drop, device=device) 
              for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d_model, device=device)
        self.deembed = DeEmbedding(self.embed)

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
    
    @property
    def n_params(self) -> int:
        n_mha = 4 * self.d_model**2  # multi-head attention
        n_mha_ln = 2 * self.d_model  # multi-head attention layer norm
        n_ff = 8 * self.d_model**2 + 5*self.d_model  # feed-forward
        n_ff_ln = 2 * self.d_model  # feed-forward layer norm
        
        n_emb = self.vocab_size * self.d_model  # embedding
        n_block = n_mha + n_mha_ln + n_ff + n_ff_ln  # entire transformer block
        n_ln = 2 * self.d_model  # final layer norm

        return n_emb + self.n_blocks * n_block + n_ln

    @property
    def n_bytes(self) -> int:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return param_size + buffer_size

##################################  Layers  ###################################

class Block(nn.Module):
    """Single transformer layer."""
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 p_drop: float,
                 device: torch.device | str) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.p_drop = p_drop
        self.device = torch.device(device)

        # Layers
        self.ln_mha = nn.LayerNorm(d_model, device=device)
        self.ln_ff = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(p_drop)
        self.mha = MultiheadAttention(d_model, n_heads, p_drop, device=device)
        self.ff = FeedForward(d_model, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # Multi-head attention
        y = self.ln_mha(x)
        y = self.mha(y)
        y = self.dropout(y)
        y = y + x

        # Feed Forward
        x = y
        y = self.ln_ff(x)
        y = self.ff(y)
        y = self.dropout(y) 
        y = y + x

        return y

class MultiheadAttention(nn.Module):
    """Multi-head attention sublayer for transformer."""
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 p_drop: int,
                 device: torch.device | str) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f'n_heads ({n_heads}) does not divide d_model {d_model}.')

        # hyperparameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.p_drop = p_drop
        self.device = device

        # Layers tensors
        self.wqkv = nn.Linear(d_model, 3 * d_model, device=device)
        self.wo = nn.Linear(d_model, d_model, device=device)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(p_drop)
    
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
        if mask is None:
            mask = torch.full((s,s), -torch.inf, device=self.device).triu(1)
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
    """Feed forward sublayer for transformer."""
    def __init__(self, 
                 d_model: int, 
                 device: torch.device | str) -> None:
        super().__init__()

        # hyperparameters
        self.d_model = d_model
        self.device = device

        # Layers
        self.layers = nn.Sequential(nn.Linear(d_model, 4 * d_model, 
                                              device=device),
                                    nn.ReLU(),
                                    nn.Linear(4 * d_model, d_model,
                                              device=device))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DeEmbedding(nn.Module):
    """De-embedding layer for transformer. 
    Weights are tied to specified embedding layer."""
    def __init__(self, embed: nn.Embedding) -> None:
        super().__init__()
        self.embed = embed

    # De-embedding tensor - tied to embedding tensor
    @property
    def weight(self):
        return self.embed.weight.T

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight
        

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 max_seq_len: int, 
                 max_wavelen: int=10000,
                 device: torch.device | str='cpu') -> None:
        if d_model % 2 != 0:
            raise ValueError(f'd_model must be even but got {d_model}')
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_wavelen = max_wavelen
        self.device = torch.device(device)

        # Wavelengths
        wavelen = max_wavelen ** (torch.arange(d_model // 2) / d_model)

        # Positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        theta = torch.arange(max_seq_len).unsqueeze(1) / wavelen.unsqueeze(0)
        pe[:, 0::2] = torch.sin(theta)
        pe[:, 1::2] = torch.cos(theta)
        
        # Register buffer
        self.register_buffer('pe', pe.to(device))

    def forward(self, x: Tensor) -> Tensor:
        seqlen = x.shape[-2]
        if seqlen > self.max_seq_len:
            raise ValueError(f'seqlen ({seqlen}) is larger than max_seq_len ({self.max_seq_len}).')
        return x + self.pe[:seqlen, :].to(x.dtype)



################################################################################

if __name__ == '__main__':
    ### train Transformer to reverse sequence
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    max_num = 100
    n_batches = 1500
    batch_size = 32
    seq_len = 16

    model = Transformer(vocab_size=max_num,
                        d_model=128,
                        max_seq_len=16,
                        n_heads=4,
                        n_blocks=3,
                        p_drop=0.0)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters())

    losses = []
    for _ in tqdm(range(n_batches)):
        # data
        x = torch.randint(0, max_num, (batch_size, seq_len))
        y = x.flip(1)

        # prediction
        logits: Tensor = model(x)

        # loss
        loss: Tensor = loss_fn(logits.permute(0,2,1), y)
        losses.append(loss.item())

        # grad and step
        optim.zero_grad()
        loss.backward()
        optim.step()

    # causal attention means first half of output seq is wrong
    random_guessing = np.log(max_num)
    irreducible_loss = 0.5 * random_guessing

    plt.plot(losses, label='training loss')
    plt.axhline(random_guessing, color='k', linestyle='--', label='random guessing loss')
    plt.axhline(irreducible_loss, color='r', linestyle='--', label='irreducible loss')
    plt.yscale('log')
    plt.xlabel('batch')
    plt.legend()
    plt.show()









