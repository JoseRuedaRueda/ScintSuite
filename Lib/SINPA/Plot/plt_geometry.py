"""
Plot SINPA geometry

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 6.0.0
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def plot_geometry_element(filename=None, ax=None):
    """
    Plot a geometry element

    Jose Rueda Rueda: jrrueda@us.es

    @param filename: file with the triangles
    @param ax: axes where to plot

    Note: The use of this routine is not recomended if you use a fine mesh with
    several triangles

    """
    ntriangles = np.loadtxt(filename, max_rows=1).astype(np.int)
    x, y, z = np.loadtxt(filename, skiprows=1, unpack=True)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.tight_layout()

    for i in range(ntriangles):
        ax.plot([x[3*i], x[3*i + 1]],
                [y[3*i], y[3*i + 1]], [z[3*i], z[3*i + 1]], 'k')
        ax.plot([x[3*i+2], x[3*i + 1]],
                [y[3*i+2], y[3*i + 1]], [z[3*i+2], z[3*i + 1]], 'k')
        ax.plot([x[3*i], x[3*i + 2]],
                [y[3*i], y[3*i + 2]], [z[3*i], z[3*i + 2]], 'k')
