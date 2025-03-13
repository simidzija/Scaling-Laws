# Standard library
import sys
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
        

##################################  Load data  #################################

def load_memmap(path: str, dtype: np.dtype) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode='r')

##############################  Data processing  ###############################

def create_tokenizer(iterable: Iterable[str], savepath: str, vocab_size: int) -> None:
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

def tokenize_hf_dataset():
    pass

def tokenize(iterable: Iterable[str], savepath: str, tokenizer_path: str, filetype: str) -> None:

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
    handler(tokens, savepath)

def get_hf_iterator(path: str) -> Iterator[str]:
    print('Loading dataset...', end='')
    dataset = load_dataset(path)
    print('Loading complete.')
    for split in dataset.values():
        for text in split['text']:
            yield text


############################  Filetype handlers  ##############################

def npy_handler(tokens: list[int], tokens_path: str) -> None:
    # create array
    data = np.array(tokens, dtype=np.int16)  # use int32 for large vocabs

    # save to file
    np.save(tokens_path, data)

def memmap_handler(tokens: list[int], tokens_path: str) -> None:
    # create array
    data = np.memmap(filename=tokens_path,
                     dtype=np.int16,
                     mode='w+',
                     shape=(len(tokens),))

    data[:] = np.array(tokens, dtype=np.int16)
    
    # save to file
    data.flush()


################################################################################

if __name__ == '__main__':
    dataset_path = "roneneldan/TinyStories"
    tokenizer_path = str(ROOT / 'data/tokenizer_ts.json')

    ### create tokenizer
    # vocab_size = 32000
    # iterator = get_hf_iterator(dataset_path)
    # create_tokenizer(iterator, tokenizer_path, vocab_size)

    ### tokenize dataset
    tokens_path = str(ROOT / 'data/tokens_ts_train.memmap')
    text_list = load_dataset(dataset_path)['train']['text']
    tokenize(text_list, tokens_path, tokenizer_path, filetype='memmap')

    marr = load_memmap(ROOT / 'data/tokens_ts_train.memmap', dtype=np.int16)
    print(marr.shape)