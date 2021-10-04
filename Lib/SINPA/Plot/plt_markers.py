"""
Plot the strike points from SINPA simulations

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 6.0.0
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def axisEqual3D(ax):
    """
    Set the aspect ratio of a 3D plot to equal.

    Extracted from:
    https://stackoverflow.com/questions/8130823/
        set-matplotlib-3d-plot-aspect-ratio
    @param ax: 3D axis where to put the aspect ratio
    """
    extents = \
        np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)





def plot_orbits3D(orb, per=0.1, ax=None, line_params={}, imax=10):
    """
    Plot the strike points in a 3D axis as scatter points

    Jose Rueda: jrrueda@us.es

    @param strikes: diccionary with the strike points (created by reading.py)
    @param per: ratio of markers to be plotted (1=all of them)
    @param ax: axes where to plot
    @param mar_params: Dictionary with the parameters for the markers
    """
    # --- Default markers
    line_options = {
        'color': 'k'
    }
    line_options.update(line_params)
    # --- Create the axes
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        created = True
    else:
        created = False

    # --- Plot the markers:
    nlines = len(orb)

    for i in range(nlines):
        ax.plot(orb['data'][i][:imax, 0], orb['data'][i][:imax, 1],
                orb['data'][i][:imax, 2], **line_options)
    # --- Set properly the axis
    if created:
        # Get rid of colored axes planes
        # (https://stackoverflow.com/questions/11448972/changing-the-background
        # -color-of-the-axes-planes-of-a-matplotlib-3d-plot)
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
