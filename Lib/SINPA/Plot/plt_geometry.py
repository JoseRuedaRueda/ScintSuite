"""
Plot SINPA geometry

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 6.0.0
"""
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)


def plot_geometry_element(filename: str = None, ax=None, line_params: dict = {}):
    """
    Plot a geometry element

    Jose Rueda Rueda: jrrueda@us.es

    @param filename: file with the triangles
    @param ax: axes where to plot

    Note: The use of this routine is not recomended if you use a fine mesh with
    several triangles

    """
    # --- Initialize options
    line_options = {
        'color': 'k'
    }
    line_options.update(line_params)
    ntriangles = np.loadtxt(filename, max_rows=1).astype(np.int)
    x, y, z = np.loadtxt(filename, skiprows=1, unpack=True)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.tight_layout()

    for i in range(ntriangles):
        ax.plot([x[3*i], x[3*i + 1]],
                [y[3*i], y[3*i + 1]], [z[3*i], z[3*i + 1]], **line_options)
        ax.plot([x[3*i+2], x[3*i + 1]],
                [y[3*i+2], y[3*i + 1]], [z[3*i+2], z[3*i + 1]], **line_options)
        ax.plot([x[3*i], x[3*i + 2]],
                [y[3*i], y[3*i + 2]], [z[3*i], z[3*i + 2]], **line_options)


def plot_geometry(GeomID: str = 'Test0', ax=None, line_params: dict = {}):
    """
    Plot all plates from the SINPA geomerty

    Jose Rueda: jrrueda@us.es

    Note: Note recomended for dense geometries with lot of triangles

    @param runID: GeomID of the SINPA geometry
    @param ax: axis where to plot, if None, new ones will be created.
    """
    # --- Open the figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.tight_layout()
    # --- Initialize options
    line_options = {
        'color': 'k'
    }
    line_options.update(line_params)
    # --- Plot the elements
    names = ['Scintillator.txt', 'Collimator.txt', 'Foil.txt']
    for n in names:
        file = os.path.join(paths.SINPA, 'Geometry', GeomID, n)
        plot_geometry_element(filename=file, ax=ax, line_params=line_options)
