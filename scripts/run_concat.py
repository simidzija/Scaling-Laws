# Standard library
import sys
from pathlib import Path

# Add source dir to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from concat import concat_jsonl


if __name__ == '__main__':
    inpath = ROOT/'data/data.jsonl'
    outpath = ROOT/'data/data.txt'
    
    concat_jsonl(inpath, outpath)
