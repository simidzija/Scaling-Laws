"""
Utility methods.
"""

# Standard library
import random
from typing import Any

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

def move_module(module: nn.Module, device: torch.device | str):
    """Recursively moves module and all its children to device."""
    module.to(device)
    if hasattr(module, 'device'):
        module.device = device
    for _, child in module.named_children():
        move_module(child, device)

def move_optim_state_dict(sd: dict, device: torch.device | str):
    """Move optimizer's state tensors to device."""
    for param_group_dict in sd['state'].values():
        for name, ten in param_group_dict.items():
            if name == 'step':
                # Keeping step tensors on CPU speeds up training on MPS by 2x!
                param_group_dict[name] = ten.to('cpu')
            else:
                param_group_dict[name] = ten.to(device)

def copy(obj: Any, device: torch.device | str) -> Any:
    """Creates a copy of an object (Tensor or dict/list containing tensors) 
    to device. Returns copy."""
    if isinstance(obj, Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: copy(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [copy(x, device) for x in obj]
    return obj

def print_optim_device(optim: torch.optim.Optimizer):
    """Prints device of optim's state tensors. Useful for debugging."""
    state = optim.state_dict()['state']
    for group in range(len(state)):
        print(f'group {group}:')
        for name, ten in state[group].items():
            print(f'  {name:10} is on {ten.device}')

def set_seed(seed: int=42) -> None:
    """Sets random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_flops(vocab_size: int,
              d_model: int,
              n_heads: int,
              n_blocks: int,
              seq_len: int,
              batch_size: int,
              total_batches: int) -> int:
    """
    Number of flops during model training. See Appendix F of Chinchilla paper.
    """
    embeddings = 2 * seq_len * vocab_size * d_model
    total_attention = (
        2 * 3 * seq_len * d_model**2 +  # KQV projections
        2 * seq_len**2 * d_model +      # K @ Q logits
        3 * n_heads * seq_len**2 +      # softmax
        2 * seq_len**2 + d_model +      # softmax @ V
        2 * seq_len * d_model**2        # final linear
    )
    feed_forward = 2 * seq_len * 8 * d_model**2
    deembdeddings = 2 * seq_len * d_model * vocab_size

    total_forward_per_seq = (
        embeddings + 
        n_blocks * (total_attention + feed_forward) +
        deembdeddings
    )

    total_backward_per_seq = 2 * total_forward_per_seq

    n_seqs = batch_size * total_batches

    return n_seqs * (total_forward_per_seq + total_backward_per_seq)

def moving_average(data, window_size, mode='valid'):
    """Smooths data by taking moving average."""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode=mode)
    
def gaussian_smoothing(data, sigma, window_size=None, mode='valid'):
    """Smooths data by smoothing against a Gaussian kernel."""
    window_size = 3 * sigma if window_size is None else window_size
    window = np.exp(- 0.5*(np.arange(-window_size, window_size) / sigma)**2)
    window = window / np.sum(window)
    return np.convolve(data, window, mode=mode)
