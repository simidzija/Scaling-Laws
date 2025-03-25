# Standard library
import json
import sys
from pathlib import Path
from typing import Optional

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # automatic mixed precision
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add source dir to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from data import MemmapDataset
from model import Transformer

def train(model: Transformer,
          device: torch.device,
          train_path: str,
          data_dtype: torch.dtype,
          n_batches: int,
          batch_size: int,
          seq_len: int,
          lr: float=0.001,
          print_period: Optional[int]=None) -> None:

    # move model to device
    model.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optim = AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = CosineAnnealingLR(optim, T_max=n_batches, eta_min=lr/10)

    # grad scaler
    if device == 'cuda':
        scaler = GradScaler()

    # create dataset
    n_seqs = n_batches * batch_size
    dataset = MemmapDataset(train_path, n_seqs=n_seqs, seq_len=seq_len, dtype=data_dtype)

    # create dataloader
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=4,
                            pin_memory=device == 'cuda',
                            persistent_workers=True)
    
    # loss list
    losses = []

    # training loop
    for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        # move data to device
        data = data.to(device=device, dtype=torch.int32, non_blocking=True)

        # automatically use FP16 precision when safe, FP32 otherwise
        with autocast(device_type=device):

            # logits (omit last token; shape (seq, token, logits))
            logits: torch.Tensor = model(data[:, :-1]).permute(0, 2, 1)

            # ground truth (omit first token)
            truth = data[:, 1:]  # shape (seq, token) 

            # loss
            loss: torch.Tensor = loss_fn(logits, truth.to(torch.int64))
            losses.append(loss.item())

        # backward on scaled loss
        optim.zero_grad()
        if device == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # optim step
        if device == 'cuda':
            scaler.step(optim)
        else:
            optim.step()

        # update scaler
        if device == 'cuda':
            scaler.update()

        # lr step
        lr_scheduler.step()

        # print
        if print_period and batch % print_period == 0:
            print(f'batch {batch:3d}/{n_batches}: loss = {loss.item():10.5f}')

    return losses


