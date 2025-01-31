# Standard library
import sys
from pathlib import Path

# Add source dir to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from bpe_train import train


if __name__ == '__main__':
    corpus_path = str(ROOT/'data/data.txt')
    tokenizer_path = str(ROOT/'data/tokenizer.json')
    vocab_size = 32000  # same as Chinchilla

    train(corpus_path, tokenizer_path, vocab_size)
