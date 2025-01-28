# Standard Library
import sys
from pathlib import Path

# Third-party
import torch
from torch import nn

# Local
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT.absolute()/'src'))
from model import (Block, 
                   Dropout, 
                   Embedding, 
                   FeedForward, 
                   LayerNorm,
                   MultiheadAttention,
                   PositionalEncoding,
                   Transformer)


def test_Embedding():
    vocab_size = 7
    d_model = 5
    batch_size = 3

    embed = Embedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (batch_size, ))

    assert embed(x).shape == torch.Size([batch_size, d_model])

def test_PositionalEncoding():
    max_seq_len = 9
    seq_len = 7
    d_model = 6
    batch_size = 3

    pe = PositionalEncoding(d_model, max_seq_len)
    x = torch.rand(batch_size, seq_len, d_model)

    assert pe(x).shape == x.shape

def test_Dropout():
    p_drop = 0.9
    dropout = Dropout(p_drop)
    N = 10000
    x = torch.rand(N)

    # training
    dropout.train()
    y = dropout(x)
    avg_x = torch.mean(x)
    avg_y = torch.mean(y)
    n_zeros = torch.sum(y == 0, dtype=torch.float)
    n_zeros_expected = torch.tensor(N * p_drop)

    assert torch.allclose(avg_x, avg_y, rtol=0.1) # should be close
    assert torch.allclose(n_zeros, n_zeros_expected, rtol=0.1)

    # inference
    dropout.eval()
    y = dropout(x)

    assert torch.equal(x, y)

def test_LayerNorm():
    x = torch.rand([3,4,5])
    normalized_shape = [4,5]
    eps = 0.1
    ln = LayerNorm(normalized_shape, eps=eps)
    ln_torch = nn.LayerNorm(normalized_shape, eps=eps)

    assert torch.allclose(ln(x), ln_torch(x), rtol=0.001)

def test_MultiheadAttention():
    batch_size = 3
    seq_len = 7
    d_model = 6
    n_heads = 2
    p_drop = 0.1

    mha = MultiheadAttention(d_model, n_heads, p_drop)
    x1 = torch.rand(batch_size, seq_len, d_model)
    assert mha(x1).shape == x1.shape

    # check attention mechanism is causal
    mha.eval()  # no dropout

    # modify tokens >= i
    i = 3
    x2 = x1.clone()
    x2[:, i:, :].uniform_()

    # mha should give same outputs for tokens < i
    out1 = mha(x1)
    out2 = mha(x2)
    assert torch.allclose(out1[:, :i, :], out2[:, :i, :])

    # and different outputs for tokens >= i
    assert not torch.any(torch.isclose(out1[:, i:, :], out2[:, i:, :]))

test_MultiheadAttention()

def test_FeedForward():
    batch_size = 3
    seq_len = 5
    d_model = 7
    
    x = torch.rand(batch_size, seq_len, d_model)
    ff = FeedForward(d_model)

    assert ff(x).shape == x.shape

def test_Block():
    d_model = 6
    n_heads = 2
    p_drop = 0.1
    batch_size = 3
    seq_len = 5

    block = Block(d_model, n_heads, p_drop)
    x = torch.rand(batch_size, seq_len, d_model)

    assert block(x).shape == x.shape

def test_Transformer():
    vocab_size = 17
    d_model = 6
    max_seq_len = 13
    n_heads = 2
    n_blocks = 7
    p_drop = 0.1
    batch_size = 9
    seq_len = 11

    transformer = Transformer(vocab_size=vocab_size,
                              d_model=d_model,
                              max_seq_len=max_seq_len,
                              n_heads=n_heads,
                              n_blocks=n_blocks,
                              p_drop=p_drop)

    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    assert transformer(x).shape == torch.Size([batch_size, seq_len, vocab_size])
