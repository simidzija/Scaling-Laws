"""
Running this script will print out a high-level summary of the last training 
run listed in the config.yaml configuration file, listing n_params, n_tokens, 
and n_flops of the training run.
"""


# Standard library
import sys
from pathlib import Path

# Third-party
import yaml

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from model import Transformer
from utils import get_flops

if __name__ == '__main__':
    # config
    with open(ROOT/'config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # fixed hyperparameters
    vocab_size = config['fixed_hyperparams']['vocab_size']
    batch_size = config['fixed_hyperparams']['batch_size']
    seq_len = config['fixed_hyperparams']['seq_len']

    # run hyperparameters
    run = config['runs'][-1]
    d_model = run['d_model']
    n_heads = run['n_heads']
    n_blocks = run['n_blocks']
    total_batches = run['total_batches']
    n_tokens = seq_len * batch_size * total_batches

    # create model
    model = Transformer(vocab_size=vocab_size,
                        d_model=d_model,
                        max_seq_len=seq_len,
                        n_heads=n_heads,
                        n_blocks=n_blocks,
                        device='cpu')

    # size of model
    n_params = model.n_params
    n_bytes = model.n_bytes

    # flop count
    n_flops_accurate = get_flops(vocab_size=vocab_size,
                                 d_model=d_model,
                                 n_heads=n_heads,
                                 n_blocks=n_blocks,
                                 seq_len=seq_len,
                                 batch_size=batch_size,
                                 total_batches=total_batches)
    n_flops_approx = 6 * n_params * n_tokens

    # print info
    print('---------------------- MODEL TRAINING INFO  -----------------------')
    print(f'n_params = {n_params:_}')
    print(f'n_tokens = {n_tokens:_}')
    print(f'         = {n_tokens / n_params:.3} * n_params')
    print(f'n_flops  = {n_flops_accurate:_}')
    print(f'         = {n_flops_accurate / n_flops_approx:.3} * 6ND')
