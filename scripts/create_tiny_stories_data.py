"""
Script for creating tokenized version of TinyStories dataset 
(https://huggingface.co/datasets/roneneldan/TinyStories).
This provides a clean and relatively simple natural language dataset, useful 
for prototyping LLMs.
"""

# Standard library
import sys
from pathlib import Path

# Third-party
import numpy as np
from datasets import load_dataset

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from data import create_tokenizer, get_hf_iterator, tokenize

if __name__ == '__main__':
    dataset_path = "roneneldan/TinyStories"
    tokenizer_path = str(ROOT / 'data/tiny_stories/tokenizer.json')

    ## create tokenizer
    vocab_size = 10000
    iterator = get_hf_iterator(dataset_path)
    create_tokenizer(iterator, tokenizer_path, vocab_size)

    ## tokenize
    tokens_path = str(ROOT / 'data/tiny_stories/validation.memmap')
    text_list = load_dataset(dataset_path)['validation']['text']
    tokenize(text_list, tokens_path, tokenizer_path, filetype='memmap', dtype=np.dtype('int16'))

    tokens_path = str(ROOT / 'data/tiny_stories/train.memmap')
    text_list = load_dataset(dataset_path)['train']['text']
    tokenize(text_list, tokens_path, tokenizer_path, filetype='memmap', dtype=np.dtype('int16'))