# Standard library
import sys
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Add source dir to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from model import Transformer


def train(n_params: int,
          n_tokens: int,
          batch_size: int,
          seq_len: int,
          lr: float,
          train_data: str,
          test_data: str):
    # create model
    model = create_model(n_params)

    # create optimizer
    optim = AdamW(model.parameters(), lr=lr)

    # compute number of batches
    n_batches = compute_num_batches(n_params, n_tokens, batch_size, seq_len)

    # create cosine annealing lr scheduler
    lr_scheduler = CosineAnnealingLR(optim, T_max=n_batches, eta_min=lr/10)

    # create dataset
    n_seqs = n_batches * batch_size
    dataset = MemmapDataset(train_data, n_seqs=n_seqs, seq_len=seq_len)

    # create dataloader
    data_loader = DataLoader(dataset, 
                             batch_size=batch_size, 
                             num_workers=4,
                             pin_memory=torch.cuda.is_available(),
                             persistent_workers=True)





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
        
        assert self.data.shape == self.data_shape, f'self.data.shape is {self.data.shape} and not the expected {self.data_shape}. There might have been an error loading the data.'
        
        return self.data[idx]

    def init_data(self):
        self.data = np.memmap(self.data_path, 
                              dtype=np.int16, 
                              mode='r', 
                              shape=self.data_shape)





##############################  Helper functions  ##############################

def create_model(n_params: int) -> Transformer:
    pass

def compute_num_batches(n_params: int, 
                        n_tokens: int, 
                        batch_size: int,
                        seq_len: int) -> int:
    pass


