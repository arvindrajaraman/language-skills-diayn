import matplotlib.pyplot as plt
import numpy as np
import wandb

colormaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']

def plot_visitations(bins, x_min, x_max, y_min, y_max, skill_idx=None, log=False):
    if log:
        bins = np.where(bins != 0, np.log(bins), 0)
        label = 'log(Visitation)'
    else:
        label = 'Visitation'
    plt.close()
    plt.imshow(bins, cmap=colormaps[skill_idx % len(colormaps)] if skill_idx is not None else 'Greys',
               interpolation='nearest', extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.colorbar(label=label)
    plt.title(f'{label} Distribution')
    return wandb.Image(plt)
