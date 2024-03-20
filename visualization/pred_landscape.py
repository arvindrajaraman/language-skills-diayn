import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
import wandb

from utils import to_numpy, to_torch

def plot_pred_landscape(discrim, embedding_fn):
    plt.close()
    xs = torch.arange(-1.0, 1.005, 0.005)
    ys = torch.arange(0.0, 1.005, 0.005)

    X, Y = torch.meshgrid(xs, ys)
    points = torch.stack((X, Y), dim=2).view(-1, 2)
    points = torch.cat([points, torch.zeros((points.shape[0], 6))], dim=1)
    embeddings = to_torch(embedding_fn(points))

    skill_probs = to_numpy(discrim.forward(embeddings)).reshape((X.shape[0], X.shape[1], 3))
    plt.pcolormesh(X, Y, skill_probs, shading='gouraud')

    # Add legend with red, green, blue
    plt.legend(handles=[mpatches.Patch(color='#ff0000', label='Skill 0'),
                       mpatches.Patch(color='#00ff00', label='Skill 1'),
                       mpatches.Patch(color='#0000ff', label='Skill 2')])
    
    plt.title('Prediction Landscape')
    plt.xlabel('X')
    plt.ylabel('Y')

    return wandb.Image(plt)
