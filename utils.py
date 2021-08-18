import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib


def plot_triangle(coors_list, ax=None, title="", pts_x=None, pts_y=None,
                  path=None):
    if pts_y is None:
        pts_y = []
    if pts_x is None:
        pts_x = []
    ax_was_none = ax is None
    if ax is None:
        plt.figure()
        ax = plt.gca()

    for coors in coors_list:
        ax.add_patch(Polygon(np.array(coors), edgecolor='black',  fc=(0,0,1,
                                                                      0.5)))
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title(title)
    ax.scatter(pts_x, pts_y)

    if path:
        plt.savefig(path)
        return

    if ax_was_none:
        plt.show()


def first_ratio(N):

    n_sqrt = np.sqrt(N)
    height = int(np.ceil(n_sqrt))
    width = int(np.floor(n_sqrt))

    return height, width


def plot_N_times(list_to_plot, edge_pairs, path=None):

    h, w = first_ratio(len(list_to_plot))
    _, axes = plt.subplots(ncols=w, nrows=h)

    for i in range(h):
        for j in range(w):
            data = list_to_plot[i * w + j]
            pair = edge_pairs[i * w + j]
            edge1, edge2 = pair
            plot_triangle(data, axes[i][j], f"Edges are {edge1} and {edge2}")

    if path:
        plt.savefig(path)
    else:
        plt.show()

