import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl


def plot_heatmap(data, x_labels, y_labels, x_label, y_label, filename):

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, '{:.4f}'.format(data[i, j]),
                        ha="center", va="center", color="w")

    fig.tight_layout()
    plt.savefig(filename)
    plt.clf()
