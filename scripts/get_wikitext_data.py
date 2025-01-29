# Standard library
from pathlib import Path

# Third-party
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    dataset.save_to_disk(ROOT/'data')