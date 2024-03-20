import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix_heatmap(confusion_matrix):
    plt.close()
    row_labels = [f'skill {i}' for i in range(3)]
    col_labels = [f'goal {i}' for i in range(3)]

    # fig, ax = plt.subplots()
    plt.imshow(confusion_matrix, cmap='gist_heat', vmin=0.0, vmax=1.0)

    plt.xticks(np.arange(len(col_labels)), labels=col_labels)
    plt.yticks(np.arange(len(row_labels)), labels=row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            plt.text(j, i, round(confusion_matrix[i, j], 3), ha="center", va="center", color="w", bbox=dict(facecolor='black', alpha=0.2))

    plt.title('Confusion matrix of (skills, goals)')
    plt.tight_layout()
    plt.colorbar()
    return plt
