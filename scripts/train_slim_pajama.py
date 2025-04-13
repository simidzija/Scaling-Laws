# Standard library
import os
import sys
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import yaml

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

    # config
    with open(ROOT/'config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # run
    run_number = len(config) - 1
    run = config[run_number]

    # device
    device = 'mps'
    
    # data
    data_path = str(ROOT/'data/slim_pajama/data_10M.memmap')
    data_dtype = np.dtype('int16')

    # checkpoint
    checkpoint_dir = str(ROOT/f'checkpoints/slim_pajama_{run_number}')
    checkpoint_period = run['total_batches'] // 10

    # results
    results_path = str(ROOT/f'results/slim_pajama/results_{run_number}.json')
    if os.path.exists(results_path):
        raise RuntimeError(f"results path {results_path} already exists. Please specify another path so that existing results aren't overwritten")

    # progress
    print_period = 10
    
    # model
    model = Transformer(vocab_size=run['vocab_size'],
                        d_model=run['d_model'],
                        max_seq_len=run['seq_len'],
                        n_heads=run['n_heads'],
                        n_blocks=run['n_blocks'],
                        device='mps')
    
    # train
    print(f'Training {model.n_params:,} parameter model.')
    train_from_scratch(model=model,
                       device=device,
                       data_path=data_path,
                       data_dtype=data_dtype,
                       total_batches=run['total_batches'],
                       batch_size=run['batch_size'],
                       seq_len=run['seq_len'],
                       lr=run['lr'],
                       print_period=print_period,
                       checkpoint_dir=checkpoint_dir,
                       checkpoint_period=checkpoint_period,
                       results_path=results_path)
