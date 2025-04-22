"""
Trains a transformer on the SlimPajama dataset, created by the script 
create_slim_pajama_data.py.
The training can run on CPU, MPS, or cuda GPU.
"""

# Standard library
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch
import yaml

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from model import Transformer
from train import train_from_scratch
from utils import set_seed


if __name__ == '__main__':
    # seed
    set_seed()

    # device
    device = 'mps'

    # data
    data_path = str(ROOT/'data/slim_pajama/data_528M.memmap')
    data_dtype = np.dtype('int16')

    # print period (batches)
    print_period = 10
        
    # config dict
    with open(ROOT/'config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # fixed hyperparameters
    vocab_size = config['fixed_hyperparams']['vocab_size']
    p_drop = config['fixed_hyperparams']['p_drop']
    batch_size = config['fixed_hyperparams']['batch_size']
    seq_len = config['fixed_hyperparams']['seq_len']
    lr = config['fixed_hyperparams']['lr']

    # loop over incomplete runs
    for run in config['runs']:
        # clear cuda memory
        torch.cuda.empty_cache()

        # skip if complete
        if run.get('complete'):
            continue

        # run hyperparameters
        d_model = run['d_model']
        n_blocks = run['n_blocks']
        total_batches = run['total_batches']
        n_heads = run['n_heads']

        # print model details
        print('--------------------------------------------------------')
        print(f'Training model:')
        print(f'  d_model = {d_model}')
        print(f'  n_blocks = {n_blocks}')
        print(f'  total_batches = {total_batches}')
        print(f'  n_heads = {n_heads}')
        print()

        # run name
        name = f'{d_model}_{n_blocks}_{total_batches}'

        # checkpoint
        checkpoint_dir = str(ROOT/f'checkpoints/slim_pajama/{name}')
        checkpoint_period = total_batches // 2

        # results
        results_path = str(ROOT/f'results/slim_pajama/{name}.json')

        # model
        model = Transformer(vocab_size=vocab_size,
                            d_model=d_model,
                            max_seq_len=seq_len,
                            n_heads=n_heads,
                            n_blocks=n_blocks,
                            device=device)
        
        # train
        train_from_scratch(model=model,
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

        # mark run as complete
        run['complete'] = True
        with open(ROOT/'config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print('\nModel training complete')
        print('--------------------------------------------------------')