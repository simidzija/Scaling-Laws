# Standard library
import json
import os
import sys
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

if __name__ == '__main__':
    run = '448_7_16000'
    lrs = [
        '0.005',
        # '0.0005',
        # '0.00005',
        '0.0000016',
    ]
    for lr in lrs:
        with open(ROOT/f'results/slim_pajama/lr_{lr}/{run}.json', 'r') as f:
            results = json.load(f)
            losses = results['losses']
            plt.plot(losses, label=f'lr = {lr}')

    plt.title(f'run = {run}')
    plt.yscale('log')
    plt.legend()
    plt.show()
