# Standard library
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Third-party
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

################################  Dataset class  ###############################

class MemmapDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 start_seq: Optional[int],
                 n_seqs: int,
                 seq_len: int,
                 dtype: np.dtype | str) -> None:
        self.data_path = data_path
        self.data: Optional[np.memmap] = None
        self.n_seqs = n_seqs
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.start_seq = 0 if start_seq is None else start_seq
        self.offset = self.start_seq * seq_len * self.dtype.itemsize
        self.data_shape = (n_seqs, seq_len)

    def __getitem__(self, idx: int) -> np.ndarray:
        # initialize data if not already initialized
        if self.data is None:
            self.init_data()
        
        return self.data[idx].copy()  # copy so array is writeable

    def __len__(self) -> int:
        return self.n_seqs

    def init_data(self):
        self.data = np.memmap(self.data_path,
                              mode='r', 
                              offset=self.offset,
                              dtype=self.dtype,
                              shape=self.data_shape)
        

##################################  Load data  #################################

def load_memmap(path: str, dtype: np.dtype | str) -> np.memmap:

    return np.memmap(path, mode='r', dtype=np.dtype(dtype))

##############################  Data processing  ###############################

def create_tokenizer(iterable: Iterable[str], 
                     savepath: str, 
                     vocab_size: int) -> None:
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialzie trainer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", 
                                                                "[BOS]", 
                                                                "[EOS]"])

    # Train on corpus
    tokenizer.train_from_iterator(iter(iterable), trainer)

    # Save
    tokenizer.save(savepath)

def tokenize(iterable: Iterable[str], 
             savepath: str, 
             tokenizer_path: str, 
             filetype: str, 
             dtype: np.dtype | str) -> None:

    # dtype
    dtype = np.dtype(dtype)

    # Create token handler
    handlers = {
        'npy': npy_handler,
        'memmap': memmap_handler
    }
    if filetype not in handlers:
        raise ValueError(f'filetype must be in set {set(handlers)} but got {filetype}')
    handler = handlers[filetype]

    # Load tokenizer
    tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)

    # Tokenize
    tokens = []
    for text in tqdm(iterable):
        encoding = tokenizer.encode(text)
        tokens.append(tokenizer.token_to_id("[BOS]"))
        tokens.extend(encoding.ids)
        tokens.append(tokenizer.token_to_id("[EOS]"))

    # Handle tokens
    handler(tokens, savepath, dtype=dtype)

def get_hf_iterator(path: str) -> Iterator[str]:
    """Returns iterator over text in Hugging Face dataset."""
    print('Loading dataset...', end='')
    dataset = load_dataset(path)
    print('Loading complete.')
    for split in dataset.values():
        for text in split['text']:
            yield text


############################  Filetype handlers  ##############################

def npy_handler(tokens: list[int], tokens_path: str, dtype: np.dtype | str) -> None:
    # create array
    data = np.array(tokens, dtype=np.dtype(dtype))  

    # save to file
    np.save(tokens_path, data)

def memmap_handler(tokens: list[int], tokens_path: str, dtype: np.dtype | str) -> None:
    # dtype 
    dtype = np.dtype(dtype)

    # create array
    data = np.memmap(filename=tokens_path,
                     dtype=dtype,
                     mode='w+',
                     shape=(len(tokens),))

    data[:] = np.array(tokens, dtype=dtype)
    
    # save to file
    data.flush()

