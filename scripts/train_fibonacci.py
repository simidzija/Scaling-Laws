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

# Local
from model import Transformer
from train import train_from_scratch, train_from_checkpoint


if __name__ == '__main__':
    # data
    with open(ROOT/'data/fibonacci/metadata.json', 'r') as f:
        metadata = json.load(f)
        vocab_size = metadata['max_int']
        seq_len = metadata['seq_len']
        data_path = metadata['train_path']
        n_seeds = metadata['n_seeds']
        max_int = metadata['max_int']
        data_dtype = metadata['dtype']

        # can't predict seed tokens since they are random
        irreducible_loss = (n_seeds / (seq_len - 1)) * np.log(max_int - 2)
        print(f'irreducible_loss = {irreducible_loss}')

    # model
    model = Transformer(vocab_size=vocab_size,
                        d_model=64,
                        max_seq_len=seq_len,
                        n_heads=2,
                        n_blocks=3,
                        device='mps')
    
    # training hyperparams
    device = 'mps'
    total_batches = 1000
    batch_size = 32
    lr = 0.001
    print_period = 10
    checkpoint_dir = str(ROOT/'checkpoints/fibonacci')
    checkpoint_period = 100

    # # train from scratch
    # start_batch = 0
    # losses = train_from_scratch(model=model,
    #                             device=device,
    #                             data_path=data_path,
    #                             data_dtype=data_dtype,
    #                             total_batches=total_batches,
    #                             batch_size=batch_size,
    #                             seq_len=seq_len,
    #                             lr=lr,
    #                             print_period=print_period,
    #                             checkpoint_dir=checkpoint_dir,
    #                             checkpoint_period=checkpoint_period)

    # train from checkpoint
    start_batch = 200
    checkpoint_path = checkpoint_dir + f'/checkpoint_batch_{start_batch}.pt'

    losses = train_from_checkpoint(checkpoint_path=checkpoint_path,
                                   device=device,
                                   data_path=data_path,
                                   data_dtype=data_dtype,
                                   total_batches=total_batches,
                                   batch_size=batch_size,
                                   seq_len=seq_len,
                                   print_period=print_period,
                                   checkpoint_dir=checkpoint_dir,
                                   checkpoint_period=checkpoint_period)

    # plot
    batches = range(start_batch, total_batches)
    plt.plot(batches, losses, label='train loss')
    plt.axhline(irreducible_loss, color='k', ls='--', label='irreducible loss')
    n_diff_seqs = vocab_size ** n_seeds
    plt.axvline(n_diff_seqs, color='r', ls='--', label='seqs start repeating')
    plt.xlabel('batch')
    plt.xlim(start_batch, total_batches)
    plt.yscale('log')
    plt.title(f'{n_seeds}-digit Fibonacci')
    plt.legend()
    plt.show()