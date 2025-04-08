# Standard library
import json
import sys
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from model import Transformer
from train import train_from_scratch, train_from_checkpoint
from utils import set_seed


if __name__ == '__main__':
    # seed
    set_seed()

    # data
    with open(ROOT/'data/tiny_stories/tokenizer.json', 'r') as f:
        tokenizer = json.load(f)
        vocab_size = len(tokenizer['model']['vocab'])
    
    data_path = str(ROOT/'data/tiny_stories/train.memmap')
    data_dtype = np.dtype('int16')

    # training hyperparams
    device = 'mps'
    total_batches = 100
    batch_size = 32
    seq_len = 64
    lr = 0.001
    print_period = 10
    checkpoint_dir = str(ROOT/'checkpoints/tiny_stories')
    checkpoint_period = 100
    results_path = str(ROOT/'results/tiny_stories/results_2.json')
    
    # model
    model = Transformer(vocab_size=vocab_size,
                        d_model=1024,
                        max_seq_len=seq_len,
                        n_heads=8,
                        n_blocks=16,
                        device='mps')
    print(f'model size = {model.n_bytes:.3E} bytes')
    
    # train from scratch
    start_batch = 0
    losses = train_from_scratch(model=model,
                                device=device,
                                data_path=data_path,
                                data_dtype=data_dtype,
                                total_batches=total_batches,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                lr=lr,
                                print_period=print_period,
                                checkpoint_dir=checkpoint_dir,
                                checkpoint_period=checkpoint_period,
                                results_path=results_path)

    # # train from checkpoint
    # start_batch = 200
    # checkpoint_path = checkpoint_dir + f'/checkpoint_batch_{start_batch}.pt'

    # losses = train_from_checkpoint(checkpoint_path=checkpoint_path,
    #                                device=device,
    #                                data_path=data_path,
    #                                data_dtype=data_dtype,
    #                                total_batches=total_batches,
    #                                batch_size=batch_size,
    #                                seq_len=seq_len,
    #                                print_period=print_period,
    #                                checkpoint_dir=checkpoint_dir,
    #                                checkpoint_period=checkpoint_period,
    #                                results_path=results_path)

    # plot
    batches = range(start_batch, total_batches)
    random_guessing = np.log(vocab_size)
    plt.plot(batches, losses, label='train loss')
    plt.axhline(random_guessing, label='random_guessing', color='k', ls='--')
    plt.xlabel('batch')
    plt.xlim(start_batch, total_batches)
    plt.yscale('log')
    plt.title(f'Tiny stories')
    plt.legend()
    plt.show()