# Standard library
import json
import sys
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local imports
from model import Transformer
from train import train

if __name__ == '__main__':
    with open(ROOT/'data/fibonacci/metadata.json', 'r') as f:
        metadata = json.load(f)
        vocab_size = metadata['max_int']
        seq_len = metadata['seq_len']
        train_path = metadata['train_path']
        n_seeds = metadata['n_seeds']
        max_int = metadata['max_int']

        # can't predict seed tokens since they are random
        irreducible_loss = (n_seeds / (seq_len - 1)) * np.log(max_int - 2)

    
    model = Transformer(vocab_size=vocab_size,
                        d_model=64,
                        max_seq_len=seq_len,
                        n_heads=2,
                        n_blocks=3)

    losses = train(model=model,
                   device='cpu',
                   train_path=train_path,
                   data_dtype=np.int16,
                   n_batches=100,
                   batch_size=32,
                   seq_len=seq_len,
                   lr=0.001,
                   print_period=10)

    plt.plot(losses, label='train loss')
    plt.axhline(irreducible_loss, color='k', ls='--', label='irreducible loss')
    plt.xlabel('batch')
    plt.yscale('log')
    plt.title(f'{n_seeds}-digit Fibonacci')
    plt.legend()
    plt.show()