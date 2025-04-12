# Standard library
import sys
from pathlib import Path

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from model import Transformer
from utils import get_flops

if __name__ == '__main__':
    # model hyperparams
    vocab_size = 32000
    d_model = 1280
    n_heads = 10
    n_blocks = 24

    # training hyperparams
    seq_len = 256
    batch_size = 16
    total_batches = 200
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
    print(f'n_params = {n_params:,}')
    print(f'n_tokens = {n_tokens:,}')
    print(f'n_flops  = {n_flops_accurate:.2e}')
    print(f'         = {n_flops_accurate / n_flops_approx:.3} * 6ND')
