# TODO: profile actual TFLOPS

# Standard library
import sys
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # automatic mixed precision
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Add source dir to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from model import create_model, Transformer

def train(model: Transformer,
          device: torch.device,
          n_tokens: int,
          batch_size: int,
          seq_len: int,
          lr: float,
          train_data: str,
          test_data: str):

    # move model to device
    model.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optim = AdamW(model.parameters(), lr=lr)

    # number of batches
    n_batches = compute_num_batches(n_tokens, batch_size, seq_len)

    # lr scheduler
    lr_scheduler = CosineAnnealingLR(optim, T_max=n_batches, eta_min=lr/10)

    # grad scaler
    scaler = GradScaler()

    # create dataset
    n_seqs = n_batches * batch_size
    dataset = MemmapDataset(train_data, n_seqs=n_seqs, seq_len=seq_len)

    # create dataloader
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=4,
                            pin_memory=device.type == 'cuda',
                            persistent_workers=True)

    # training loop
    for batch, data in enumerate(dataloader):

        # move data to device
        data = data.to(device, non_blocking=True)

        # automatically use FP16 precision when safe, FP32 otherwise
        with autocast():

            # logits (omit last token; shape (seq, token, logits))
            logits: torch.Tensor = model(data[:, :-1]).permute(0, 2, 1)

            # ground truth (omit first token)
            truth = data[:, 1:]  # shape (seq, token) 

            # loss
            loss: torch.Tensor = loss_fn(logits, truth)

        # backward on scaled loss
        optim.zero_grad()
        scaler.scale(loss).backward()

        # optim step
        scaler.step(optim)

        # update scaler
        scaler.update()

        # lr step
        lr_scheduler.step()


class MemmapDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 n_seqs: int,
                 seq_len: int) -> None:
        self.data_path = data_path
        self.data: Optional[np.memmap] = None
        self.n_seqs = n_seqs
        self.seq_len = seq_len
        self.data_shape = (n_seqs, seq_len)

    def __getitem__(self, idx: int) -> np.ndarray:
        # initialize data if not already initialized
        if self.data is None:
            self.init_data()
        
        return self.data[idx]

    def init_data(self):
        self.data = np.memmap(self.data_path, 
                              dtype=np.int16, 
                              mode='r', 
                              shape=self.data_shape)



##############################  Helper functions  ##############################

def compute_num_batches(n_tokens: int, 
                        batch_size: int,
                        seq_len: int) -> int:
    pass


