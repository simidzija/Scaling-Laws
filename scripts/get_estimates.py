# Standard library
from pathlib import Path

# Third-party
import numpy as np

# Root folder
ROOT = Path(__file__).resolve().parent.parent



def get_total_flops(tflops: float=20.0, 
                    cost_per_day: float=8.0, 
                    initial_credits: float=250) -> float:
    """Total flops with given GPU budget."""
    n_days = initial_credits / cost_per_day
    flops = n_days * 24 * 60 * 60 * tflops * 10**12

    return flops

def get_total_tokens(memmap_file: str):
    """Total tokens in data file."""
    mmap_array = np.memmap(memmap_file, dtype=np.int16, mode='r')
    return mmap_array.shape[0]


################################  GPU Memory  #################################

def memory_of_model(n_params: int, bytes_per_param: int) -> int:
    return n_params * bytes_per_param 

def memory_of_data(n_tokens: int, bytes_per_token: int) -> int:
    return n_tokens * bytes_per_token

def memory_of_activations(n_tokens: int, 
                          n_layers: int, 
                          d_model: int, 
                          bytes_per_activation) -> int:
    # Factor of 9 below is due to:
    #   - 1 from input to MHA
    #   - 3 from Q, K, V
    #   - 1 from input to FF
    #   - 4 from intermediate layer of FFNN
    activations_per_layer = 9 * n_tokens * d_model

    return n_layers * activations_per_layer * bytes_per_activation

def memory_of_gradients(n_params: int, bytes_per_param: int) -> int:
    return n_params * bytes_per_param

def memory_of_adam(n_params: int, bytes_per_param: int) -> int:
    return 2 * n_params * bytes_per_param



##############################################################################

def get_model_sizes(total_compute: float, 
                    size_fractions: list[float], 
                    toks_to_params: list[int]) -> list[float]:
    """
    Estimates model size that can be trained with given compute budget.
      - size_fractions: model sizes to train as fractions of max model size
      - toks_to_params: tokens/param ratios to train each model size with
    """
    total_toks_to_params = sum(toks_to_params)
    total_toks_per_max_size = [p * total_toks_to_params for p in size_fractions]

    computes_per_max_size_squared = [6*p*t for p, t in zip(size_fractions, total_toks_per_max_size)]

    total_compute_per_max_size_squared = sum(computes_per_max_size_squared)

    max_size = np.sqrt(total_compute / total_compute_per_max_size_squared)

    model_sizes = [frac * max_size for frac in size_fractions]

    return model_sizes


if __name__ == '__main__':
    # Total tokens
    tokens = get_total_tokens(ROOT/'data/train_data.memmap')
    print(f'Total tokens: {tokens:,}')

    # Total flops
    flops = get_total_flops()
    print(f'Total flops: {flops:.1E}')

    # Model sizes
    toks_to_params=[5, 10, 20, 40]
    model_sizes = get_model_sizes(flops, 
                                  size_fractions=[0.001, 0.01, 0.1, 1.0],
                                  toks_to_params=toks_to_params)
    print(f'Can train:')
    for params in model_sizes:
        print(f'  - {params:>12,.0f} parameter model')
    print(f'each with {toks_to_params} tokens to parameters.')













