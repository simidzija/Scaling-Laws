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
    # directory
    dir = ROOT / 'data/pile/'
    os.makedirs(dir, exist_ok=True)

    # dataset
    dataset = load_dataset("cerebras/SlimPajama-627B", split='train', streaming=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
    vocab = tokenizer.get_vocab()
    sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))
    with open(dir/'tokenizer.json', 'w') as f:
        json.dump(sorted_vocab, f, indent=2)

    # tokenize
    token_count = 0
    target_tokens = 10_000_000
    tokenized_data = []
    print_time_interval = 5  # seconds
    last_print_time = time()

    for example in dataset:
        # get tokens
        tokens = tokenizer(example['text'], return_tensors="np")['input_ids'][0]

        # add to list
        tokenized_data.extend(tokens.tolist())

        # update token count
        token_count += len(tokens)

        # print status
        current_time = time()
        if current_time > last_print_time + print_time_interval:
            print(f'token_count = {token_count:,} / {target_tokens:,}')
            last_print_time = current_time

        # break when target reached
        if token_count > target_tokens:
            break

    # create memmap file
    data = np.memmap(filename=dir/'data_10M.memmap',
                     dtype=np.dtype('int16'),
                     mode='w+',
                     shape=(len(tokenized_data),))

    # write to file
    data[:] = np.array(tokenized_data, dtype=np.dtype('int16'))

    # save
    data.flush()


    
    




