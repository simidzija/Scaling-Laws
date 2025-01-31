# Standard library
import sys
from pathlib import Path

# Add source dir to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from bpe_tokenize import tokenize


if __name__ == '__main__':
    tokenizer_path = str(ROOT/'data/tokenizer.json')

    # # Test set
    # test_text_path = str(ROOT/'data/test_data.txt')
    # test_tokens_path = str(ROOT/'data/test_data.npy')
    # tokenize(text_path=test_text_path,
    #          tokens_path=test_tokens_path,
    #          tokenizer_path=tokenizer_path,
    #          filetype='npy')

    # Train set
    train_text_path = str(ROOT/'data/train_data.txt')
    train_tokens_path = str(ROOT/'data/train_data.memmap')
    tokenize(text_path=train_text_path,
             tokens_path=train_tokens_path,
             tokenizer_path=tokenizer_path,
             filetype='memmap')

