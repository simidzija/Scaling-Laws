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

if __name__ == '__main__':
    lr = '0.0000016'
    dir_path = ROOT/f'results/slim_pajama/lr_{lr}/'
    
    ### get data
    avg_interval = 100
    flops_and_losses = {}

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            with open(file_path, 'r') as file:
                # load results dict
                results = json.load(file)

                # params
                n_params = results['n_params']

                # flops
                n_flops = results['n_flops']

                # final loss
                final_loss = np.mean(results['losses'][-avg_interval:]).item()

                # create dict entry if it doesn't exist
                if n_params not in flops_and_losses:
                    flops_and_losses[n_params] = {'flops': [], 'losses': []}
                
                # add to dict
                flops_and_losses[n_params]['flops'].append(n_flops)
                flops_and_losses[n_params]['losses'].append(final_loss)

    print(flops_and_losses)

    ### plot data

    # create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # colourmap
    cmap = plt.cm.viridis  # 'plasma', 'inferno', 'magma', 'jet'
    norm = Normalize(vmin=min(flops_and_losses), vmax=max(flops_and_losses))

    # plot
    for n_params in flops_and_losses:
        flops = sorted(flops_and_losses[n_params]['flops'])
        losses = sorted(flops_and_losses[n_params]['losses'], reverse=True)

        ax.scatter(flops, losses, color=cmap(norm(n_params)))
    
    # colourbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('n_params')

    # labels
    ax.set_xlabel('FLOPS')
    ax.set_ylabel('Final training loss')
    ax.grid(True)

    # log scale
    ax.set_xscale('log')
    ax.set_xscale('log')

    plt.tight_layout()
    plt.show()






                






