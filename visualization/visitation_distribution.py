import matplotlib.pyplot as plt
import wandb

colormaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']

def plot_visitations(bins, x_min, x_max, y_min, y_max, skill_idx=None):
    plt.close()
    plt.imshow(bins, cmap=colormaps[skill_idx % len(colormaps)] if skill_idx is not None else 'Greys',
               interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label='Visitation')
    plt.title('Visitation Distribution')
    return wandb.Image(plt)
