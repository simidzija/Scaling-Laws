# Standard Library
import sys
from pathlib import Path

# Third-party
import torch

# Local
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT.absolute()/'src'))
from model import (Embedding, PositionalEncoding)


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
