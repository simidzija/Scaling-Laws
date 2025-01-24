# Standard Library
import sys
from pathlib import Path

# Third-party
import torch

# Local
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(ROOT)
from model import Embedding


def test_Embedding():
    vocab_size = 7
    d_model = 5
    batch_size = 3

    embed = Embedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (batch_size, 1))

    assert embed(x).shape == torch.Size([batch_size, d_model])


