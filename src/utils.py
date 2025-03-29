# Standard library
from typing import Any

# Third-party
import torch
import torch.nn as nn
from torch import Tensor

def move_module(module: nn.Module, device: torch.device | str):
    module.to(device)
    if hasattr(module, 'device'):
        module.device = device
    for _, child in module.named_children():
        move_module(child, device)

def move_optim_state_dict(sd: dict, device: torch.device | str):
    for param_group_dict in sd['state'].values():
        for name, ten in param_group_dict.items():
            if name == 'step':
                # Keeping step tensors on CPU speeds up training on MPS by 2x!
                param_group_dict[name] = ten.to('cpu')
            else:
                param_group_dict[name] = ten.to(device)

def copy(obj: Any, device: torch.device | str) -> Any:
    if isinstance(obj, Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: copy(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [copy(x, device) for x in obj]
    return obj

def print_optim_device(optim: torch.optim.Optimizer):
    state = optim.state_dict()['state']
    for group in range(len(state)):
        print(f'group {group}:')
        for name, ten in state[group].items():
            print(f'  {name:10} is on {ten.device}')