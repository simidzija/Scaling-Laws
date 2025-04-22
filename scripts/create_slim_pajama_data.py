"""
Script for creating tokenized version of SlimPajama dataset.

SlimPajama (https://huggingface.co/datasets/cerebras/SlimPajama-627B) is a
natural language datast that is obtained by cleaning the larger RedPajama 
dataset. This script tokenizes a subset of this dataset into a desired number 
of tokens using the LLaMa-30b tokenizer (vocab size = 32k). We use this 
tokenized data for our scaling laws experiments.
"""

# Standard library
import json
import os
import sys
from pathlib import Path
from time import time

# Third-party
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

if __name__ == '__main__':

    # input
    target_tokens = 528_000_000
    write_period = 100_000_000  # tokens
    filename = 'data_528M.memmap'

    # directory
    dir = ROOT / 'data/slim_pajama/'
    os.makedirs(dir, exist_ok=True)

    # raise error if file already exists to prevent overwrites
    if os.path.exists(dir / filename):
        raise RuntimeError(f'data path "{dir/filename}" already exists.')

    # dataset
    dataset = load_dataset("cerebras/SlimPajama-627B", split='train', streaming=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
    vocab = tokenizer.get_vocab()
    sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))
    with open(dir/'tokenizer.json', 'w') as f:
        json.dump(sorted_vocab, f, indent=2)

    # print settings
    print_period = 5  # seconds
    last_print_time = time()

    # create memmap file
    data = np.memmap(filename=dir / filename,
                     dtype=np.dtype('int16'),
                     mode='w+',
                     shape=(target_tokens,))

    # tokenize
    token_count = 0
    tokenized_data = []
    last_write = 0

    for example in dataset:
        # get tokens
        tokens = tokenizer(example['text'], return_tensors="np")['input_ids'][0]

        # add to list
        tokenized_data.extend(tokens.tolist())

        # update token count
        token_count += len(tokens)

        # break when target reached
        if token_count >= target_tokens:
            remaining_tokens = target_tokens - last_write
            data[last_write:] = np.array(tokenized_data[remaining_tokens],
                                         dtype='int16')
            data.flush()
            print(f'{token_count} tokens reached -> breaking.')
            break

        # periodically write to file 
        if token_count - last_write >= write_period:
            print(f'writing {token_count - last_write:,} tokens to file.')
            data[last_write:token_count] = np.array(tokenized_data, dtype='int16')
            data.flush()
            last_write = token_count
            tokenized_data = []

        # print status
        current_time = time()
        if current_time > last_print_time + print_period:
            print(f'token_count = {token_count:,} / {target_tokens:,}')
            last_print_time = current_time
