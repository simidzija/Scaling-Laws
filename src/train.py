# Standard library
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add source dir to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from data import MemmapDataset
from model import Transformer
from utils import copy, move_module, move_optim_state_dict

def train_from_scratch(model: Transformer,
                       device: torch.device | str,
                       data_path: str,
                       data_dtype: np.dtype | str,
                       total_batches: int,
                       batch_size: int,
                       seq_len: int,
                       lr: float=0.001,
                       print_period: Optional[int]=None,
                       checkpoint_dir: Optional[str]=None,
                       checkpoint_period: Optional[int]=None,
                       results_path: Optional[str]=None) -> None:
    
    # device and dtype
    device = torch.device(device)
    data_dtype = np.dtype(data_dtype)

    # define optimizer, lr scheduler, grad scaler
    optim = AdamW(model.parameters(), lr=lr)
    lr_scheduler = CosineAnnealingLR(optim, T_max=total_batches, eta_min=lr/10)
    scaler = GradScaler() if device.type == 'cuda' else None

    return train(model=model,
                 optim=optim,
                 lr_scheduler=lr_scheduler,
                 scaler=scaler,
                 device=device,
                 data_path=data_path,
                 data_dtype=data_dtype,
                 start_batch=0,
                 total_batches=total_batches,
                 batch_size=batch_size,
                 seq_len=seq_len,
                 print_period=print_period,
                 checkpoint_dir=checkpoint_dir,
                 checkpoint_period=checkpoint_period,
                 results_path=results_path)

def train_from_checkpoint(checkpoint_path: str,
                          device: torch.device | str,
                          data_path: str,
                          data_dtype: np.dtype | str,
                          total_batches: int,
                          batch_size: int,
                          seq_len: int,
                          print_period: Optional[int]=None,
                          checkpoint_dir: Optional[str]=None,
                          checkpoint_period: Optional[int]=None,
                          results_path: Optional[str]=None) -> None:

    # device and dtype
    device = torch.device(device)
    data_dtype = np.dtype(data_dtype)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # load batch
    start_batch = checkpoint['batch']
    print(f'Restarting training from batch {start_batch} / {total_batches}.')

    # load model
    model = Transformer(**checkpoint['model_hyperparameters'])
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)

    # load optim
    optim = AdamW(model.parameters())
    state_dict = checkpoint['optim_state_dict']
    move_optim_state_dict(state_dict, device)  # keeps step tensors on cpu
    optim.load_state_dict(state_dict)

    # load lr scheduler
    lr_scheduler = CosineAnnealingLR(optim, T_max=total_batches)
    state_dict = checkpoint['lr_scheduler_state_dict']
    lr_scheduler.load_state_dict(state_dict)

    # load scaler
    if device.type == 'cuda':
        scaler = GradScaler()
        state_dict = checkpoint['scaler_state_dict']
        scaler.load_state_dict(state_dict)
    else:
        scaler = None

    return train(model=model,
                 optim=optim,
                 lr_scheduler=lr_scheduler,
                 scaler=scaler,
                 device=device,
                 data_path=data_path,
                 data_dtype=data_dtype,
                 start_batch=start_batch,
                 total_batches=total_batches,
                 batch_size=batch_size,
                 seq_len=seq_len,
                 print_period=print_period,
                 checkpoint_dir=checkpoint_dir,
                 checkpoint_period=checkpoint_period,
                 results_path=results_path)

def train(model: Transformer,
          optim: Optimizer,
          lr_scheduler: LRScheduler,
          scaler: Optional[GradScaler],
          device: torch.device | str,
          data_path: str,
          data_dtype: np.dtype | str,
          start_batch: int,
          total_batches: int,
          batch_size: int,
          seq_len: int,
          print_period: Optional[int]=None,
          checkpoint_dir: Optional[str]=None,
          checkpoint_period: Optional[int]=None,
          results_path: Optional[str]=None) -> None:

    # device and dtype
    device = torch.device(device)
    accelerator = device.type != 'cpu'
    data_dtype = np.dtype(data_dtype)

    # move model to device and put in train mode
    move_module(model, device)
    model.train()

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # dataset
    start_seq = start_batch * batch_size
    n_seqs = (total_batches - start_batch) * batch_size
    dataset = MemmapDataset(data_path, 
                            start_seq=start_seq,
                            n_seqs=n_seqs, 
                            seq_len=seq_len, 
                            dtype=data_dtype)

    # dataloader
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=4,
                            pin_memory=accelerator,
                            persistent_workers=True)
    
    # use grad scaler only if device is cuda
    scaler = scaler if device.type == 'cuda' else None

    # loss list
    losses = []

    # training loop
    for batch, data in tqdm(enumerate(dataloader, start_batch), 
                            total=len(dataloader)):

        # move data to device
        data = data.to(device=device, non_blocking=accelerator).to(torch.int32)

        # automatically use FP16 precision when safe, FP32 otherwise
        with autocast(device_type=device.type):

            # logits (omit last token; shape (seq, token, logits))
            logits: torch.Tensor = model(data[:, :-1]).permute(0, 2, 1)

            # ground truth (omit first token)
            truth = data[:, 1:]  # shape (seq, token) 

            # loss
            loss: torch.Tensor = loss_fn(logits, truth.to(torch.int64))
            losses.append(loss.item())

        # backprop and step
        optim.zero_grad()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step()
            scaler.update()
        else:
            loss.backward()
            optim.step()

        # lr step
        lr_scheduler.step()

        # print
        if print_period and batch % print_period == 0:
            print(f'batch {batch:3d}/{total_batches}: loss = {loss.item():10.5f}')

        # checkpoint
        if checkpoint_period and (batch % checkpoint_period == 0 or 
                                  batch == total_batches - 1):
            save_checkpoint(checkpoint_dir=checkpoint_dir,
                            batch=batch,
                            model=model,
                            optim=optim,
                            lr_scheduler=lr_scheduler,
                            scaler=scaler,
                            losses=losses)

    # save results
    if results_path:
        save_results(results_path=results_path,
                    model=model,
                    optim=optim,
                    total_batches=total_batches,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    losses=losses)

    return losses


def save_checkpoint(checkpoint_dir: str,
                    batch: int,
                    model: Transformer,
                    optim: AdamW,
                    lr_scheduler: CosineAnnealingLR,
                    scaler: GradScaler,
                    losses: list[float]) -> None:

    # create directory
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # create checkpoint
    checkpoint = {
        'batch': batch,
        'model_hyperparameters': model.hyperparams_dict,
        'model_state_dict': copy(model.state_dict(), 'cpu'),
        'optim_state_dict': copy(optim.state_dict(), 'cpu'),
        'lr_scheduler_state_dict': copy(lr_scheduler.state_dict(), 'cpu'),
        'scaler_state_dict': copy(scaler.state_dict(), 'cpu') if scaler else None,
        'losses': losses
    }

    # save checkpoint
    path = checkpoint_dir + f'/checkpoint_batch_{batch}.pt'
    torch.save(checkpoint, path)


def save_results(results_path: str,
                 model: Transformer,
                 optim: torch.optim.Optimizer,
                 total_batches: int,
                 batch_size: int,
                 seq_len: int,
                 losses: list[float]):

    # create results dict
    results = {
        'model_hyperparameters': model.hyperparams_dict,
        'lr': optim.state_dict()['param_groups'][0]['lr'],
        'total_batches': total_batches,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'losses': losses
    }

    # create directory
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
