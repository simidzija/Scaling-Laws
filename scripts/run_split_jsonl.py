# Standard library
import sys
from pathlib import Path

# Add source dir to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from split_jsonl import split_jsonl


if __name__ == '__main__':
    jsonl_path = str(ROOT/'data/data.jsonl')
    train_path = str(ROOT/'data/train_data.txt')
    test_path = str(ROOT/'data/test_data.txt')
    test_size = 100  # articles
    
    split_jsonl(jsonl_path=jsonl_path,
                train_path=train_path,
                test_path=test_path,
                test_size=test_size)
