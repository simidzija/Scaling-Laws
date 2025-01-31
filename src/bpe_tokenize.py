# Third-party
import numpy as np
from tokenizers import Tokenizer

def tokenize(text_path: str, 
             tokens_path: str, 
             tokenizer_path: str,
             filetype: str,
             section_size: int=5*10**7) -> None:

    # Filetype handlers
    handlers = {
        'npy': npy_handler,
        'memmap': memmap_handler
    }

    # Load tokenizer
    tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load text
    with open(text_path, 'r') as f:
        text = f.read()

    # Split text into manageable pieces
    text_list = split_text(text, size=section_size)

    # Tokenize
    tokens = []
    for i, text in enumerate(text_list, 1):
        tokens.extend(tokenizer.encode(text).ids)
        print(f'Encoded text {i} / {len(text_list)} of size {len(text)}')

    # Save
    if filetype in handlers:
        handler = handlers[filetype]
        handler(tokens, tokens_path)
    else:
        raise ValueError(f'filetype must be in set {set(handlers)} but got {filetype}')


def split_text(text: str, size: int=10**8, sep='\n') -> list[str]:
    """Splits text into pieces roughly of given size, separated by sep."""
    if size >= len(text):
        return [text]

    # Find separator indices
    sep_idxs = []
    for i in range(1, len(text) // size + 1):
        start = i * size
        end = (i + 1) * size
        idx = text.find(sep, start, end)
        if idx == -1:
            raise RuntimeError(f'No string {sep!r} found in positions {start} to {end} of text.')
        sep_idxs.append(idx)
    
    # Split text at separator indices
    text_list = []
    start = 0
    for end in sep_idxs + [len(text)]:
        text_list.append(text[start:end])
        start = end

    return text_list



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
    

    

