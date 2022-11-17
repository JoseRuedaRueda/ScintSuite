"""3D plotting"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
__all__ = ['plot_3D_revolution', 'axisEqual3D', 'clean3Daxis']

# -----------------------------------------------------------------------------
# --- 3D Plotting
# -----------------------------------------------------------------------------


def plot_3D_revolution(r, z, phi_min: float = 0.0, phi_max: float = 1.57,
                       nphi: int = 25, ax=None, label: str = None,
                       color=[0.5, 0.5, 0.5], alpha: float = 0.75):
    """
    Plot a revolution surface with the given cross-section

    Jose Rueda: ruejo@ipp.mpg.de

    :param  r: Array of R's of points defining the cross-section
    :param  z: Array of Z's of points defining the cross-section
    :param  phi_min: minimum phi to plot, default 0
    :param  phi_max: maximum phi to plot, default 1.57
    :param  nphi: Number of points in the phi direction, default 25
    :param  color: Color to plot the surface, default, light gray [0.5,0.5,0.5]
    :param  alpha: transparency factor, default 0.75
    :param  ax: 3D axes where to plot, if none, a new window will be opened
    :return ax: axes where the surface was drawn
    """
    # --- Section 0: Create the coordinates to plot
    phi_array = np.linspace(phi_min, phi_max, num=nphi)
    # Create matrices
    X = np.tensordot(r, np.cos(phi_array), axes=0)
    Y = np.tensordot(r, np.sin(phi_array), axes=0)
    Z = np.tensordot(z, np.ones(len(phi_array)), axes=0)

    # --- Section 1: Plot the surface
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    label=label)

    return ax


def axisEqual3D(ax):
    """
    Set aspect ratio to equal in a 3D plot.

    Taken from:

    https://stackoverflow.com/questions/8130823/
    set-matplotlib-3d-plot-aspect-ratio

    :param  ax: axes to be changed
    """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                       for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    return ax


def clean3Daxis(ax):
    """
    Get rid of colored axes planes and grid placed in 3D matplotlib plots

    (https://stackoverflow.com/questions/11448972/
    changing-the-background-color-of-the-axes-planes-of
    -a-matplotlib-3d-plot)

    :param  ax: axes to be 'cleaned'
    """
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    return ax
