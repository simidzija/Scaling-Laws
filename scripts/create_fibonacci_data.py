"""
Script for creating dataset of generalized Fibonacci sequences.
This provides a useful toy dataset for testing sequence models.
"""

# Standard library
import json
import sys
from pathlib import Path

# Third-party
import numpy as np
from tqdm import tqdm

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

def create_fibonacci(metadata_path: str,
                     train_path: str,
                     test_path: str,
                     seq_len: int=32,
                     n_seeds: int=3,
                     sos: int=0,
                     eos: int=1,
                     max_int: int=20,
                     n_train_seqs: int=10**6,
                     n_test_seqs: int=100,
                     dtype: np.dtype | str='int16') -> None:
    """
    Dataset of integer sequences.
    Each sequence starts with n_seeds random integers, and subsequent elements in the sequence are sums of the previous n_seeds elements, modulo max_int.
    """

    def create_seq() -> np.ndarray:
        seq = np.zeros(seq_len, dtype=dtype)
        seq[0] = sos
        seq[1:n_seeds + 1] = np.random.randint(2, max_int, n_seeds, dtype=dtype)
        for i in range(n_seeds + 1, seq_len - 1):
            seq[i] = np.sum(seq[i - n_seeds: i]) % max_int
        seq[-1] = eos
        return seq

    # dtype
    dtype = np.dtype(dtype)

    # metadata
    metadata_dict = {'seq_len': seq_len,
                     'n_seeds': n_seeds,
                     'sos': sos,
                     'eos': eos,
                     'max_int': max_int,
                     'n_train_seqs': n_train_seqs,
                     'n_test_seqs': n_test_seqs,
                     'metadata_path': metadata_path,
                     'train_path': train_path,
                     'test_path': test_path,
                     'dtype': str(dtype)}
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    # train data
    print(f'Creating train data:')
    shape = (n_train_seqs * seq_len,)
    data = np.memmap(filename=train_path, dtype=dtype, mode='w+', shape=shape)
    data[:] = np.zeros(shape, dtype=dtype)
    for i in tqdm(range(n_train_seqs)):
        data[i * seq_len : (i + 1) * seq_len] = create_seq()
    data.flush()

    # test data
    print('Creating test data')
    shape = (n_test_seqs * seq_len,)
    data = np.memmap(filename=test_path, dtype=dtype, mode='w+', shape=shape)
    data[:] = np.zeros(shape, dtype=dtype)
    for i in tqdm(range(n_test_seqs)):
        data[i * seq_len : (i + 1) * seq_len] = create_seq()
    data.flush()


if __name__ == '__main__':
    create_fibonacci(metadata_path=str(ROOT / 'data/fibonacci/metadata.json'),
                     train_path=str(ROOT / 'data/fibonacci/train.memmap'),
                     test_path=str(ROOT / 'data/fibonacci/test.memmap'),
                     n_seeds=2,
                     max_int=10,
                     n_train_seqs=10**5)