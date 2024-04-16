from flax import linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb

from lunar_utils import normalize_freq_matrix

colormaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']

def plot_visits(bins, x_min, x_max, y_min, y_max, skill_idx=None, log=False):
    if log:
        bins = jnp.where(bins != 0, jnp.log(bins + 1e-9), 0)
        label = 'log(Visitation)'
    else:
        label = 'Visitation'
    plt.close()
    plt.imshow(bins, cmap=colormaps[skill_idx % len(colormaps)] if skill_idx is not None else 'Greys',
               interpolation='nearest', extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.colorbar(label=label)
    plt.title(f'{label} Distribution')
    return wandb.Image(plt)

def goal_skill_heatmap(freq_matrix):
    plt.close()
    row_labels = [f'skill {i}' for i in range(3)]
    col_labels = [f'goal {i}' for i in range(3)]

    confusion_matrix = normalize_freq_matrix(freq_matrix)
    plt.imshow(confusion_matrix, vmin=0.0, vmax=1.0)

    plt.xticks(jnp.arange(len(col_labels)), labels=col_labels)
    plt.yticks(jnp.arange(len(row_labels)), labels=row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            plt.text(j, i, round(confusion_matrix[i, j], 3), ha="center", va="center", color="w", bbox=dict(facecolor='black', alpha=0.2))

    plt.title('Confusion matrix of (skills, goals)')
    plt.tight_layout()
    plt.colorbar()
    return wandb.Image(plt)

def plot_pred_landscape(discrim, discrim_params, embedding_fn):
    plt.close()
    xs = jnp.arange(-1.0, 1.005, 0.005)
    ys = jnp.arange(0.0, 1.005, 0.005)

    X, Y = jnp.meshgrid(xs, ys)
    points = jnp.stack((X, Y), axis=2).reshape(-1, 2)
    points = jnp.concatenate([points, jnp.zeros((points.shape[0], 6))], axis=1)
    embeddings = embedding_fn(points)

    skill_logits = discrim.apply(discrim_params, embeddings, train=False).reshape((X.shape[0], X.shape[1], 3))
    skill_probs = nn.activation.softmax(skill_logits)
    plt.pcolormesh(X, Y, skill_probs, shading='gouraud')

    # Add legend with red, green, blue
    plt.legend(handles=[mpatches.Patch(color='#ff0000', label='Skill 0'),
                       mpatches.Patch(color='#00ff00', label='Skill 1'),
                       mpatches.Patch(color='#0000ff', label='Skill 2')])
    
    plt.title('Prediction Landscape')
    plt.xlabel('X')
    plt.ylabel('Y')

    return wandb.Image(plt)
