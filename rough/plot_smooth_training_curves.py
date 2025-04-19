# Standard library
import json
import os
import re
import sys
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import sem

# Root dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT/'src'))

# Local
from utils import moving_average

if __name__ == '__main__':
    lr = '0.0000016'
    dir_path = ROOT/f'results/slim_pajama/lr_{lr}/'
    
    ### get data
    window_size = 200
    flops_and_losses = {}

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            with open(file_path, 'r') as file:
                # load results dict
                results = json.load(file)

                # params
                n_params = results['n_params']

                # flops
                total_batches = results['total_batches']
                total_flops = results['n_flops']
                flops = np.linspace(0, total_flops, total_batches)
                flops = flops[window_size//2:-window_size//2 + 1]

                # smoothed losses
                losses = results['losses']
                smooth_losses = moving_average(losses, window_size)

                # create dict entry if it doesn't exist
                if n_params not in flops_and_losses:
                    flops_and_losses[n_params] = []
                
                # add to dict
                flops_and_losses[n_params].append((flops, smooth_losses))

    ### plot data

    # create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # colourmap
    cmap = plt.cm.viridis  # 'plasma', 'inferno', 'magma', 'jet'
    norm = Normalize(vmin=min(flops_and_losses), vmax=max(flops_and_losses))

    # plot
    for n_params in flops_and_losses:
        for flops, smooth_losses in flops_and_losses[n_params]:
            ax.plot(flops, smooth_losses, color=cmap(norm(n_params)), markersize=2)
    
    # colourbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('n_params')

    # labels
    ax.set_xlabel('FLOPS')
    ax.set_ylabel('Training loss')
    ax.grid(True)

    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()






                






