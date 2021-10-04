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


def plot_strikes3D(strikes, per=0.1, ax=None, mar_params={},
                   iig=None, iia=None, NBI_pos=False, scint_system=False):
    """
    Plot the strike points in a 3D axis as scatter points

    Jose Rueda: jrrueda@us.es

    @param strikes: diccionary with the strike points (created by reading.py)
    @param per: ratio of markers to be plotted (1=all of them)
    @param ax: axes where to plot
    @param mar_params: Dictionary with the parameters for the markers
    """
    # --- Default markers
    mar_options = {
        'marker': '.',
        'color': 'k'
    }
    mar_options.update(mar_params)
    # --- Create the axes
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        created = True
    else:
        created = False

    # --- Plot the markers:
    nalpha, ngyr = strikes['counters'].shape
    minx = +100.
    miny = +100.0
    minz = +100.0
    maxx = -300.0
    maxy = -300.0
    maxz = -300.0
    if iig is None:
        for ig in range(ngyr):
            for ia in range(nalpha):
                if strikes['counters'][ia, ig] > 0:
                    flags = np.random.rand(strikes['counters'][ia, ig]) < per
                    if NBI_pos:
                        x = strikes['data'][ia, ig][flags, 3]
                    elif scint_system:
                        x = strikes['data'][ia, ig][flags, 15]
                    else:
                        x = strikes['data'][ia, ig][flags, 0]
                    minx = min(minx, x.min())
                    maxx = max(maxx, x.max())
                    if NBI_pos:
                        y = strikes['data'][ia, ig][flags, 4]
                    elif scint_system:
                        y = strikes['data'][ia, ig][flags, 16]
                    else:
                        y = strikes['data'][ia, ig][flags, 1]
                    miny = min(miny, y.min())
                    maxy = max(maxy, y.max())
                    if NBI_pos:
                        z = strikes['data'][ia, ig][flags, 5]
                    elif scint_system:
                        z = strikes['data'][ia, ig][flags, 17]
                    else:
                        z = strikes['data'][ia, ig][flags, 2]
                    minz = min(minz, z.min())
                    maxz = max(maxz, z.max())
                    ax.scatter(x, y, z, **mar_options)
    else:
        gg = np.array([iig])
        aa = np.array([iia])
        for ig in gg:
            for ia in aa:
                if strikes['counters'][ia, ig] > 0:
                    flags = np.random.rand(strikes['counters'][ia, ig]) < per
                    x = strikes['data'][ia, ig][flags, 0]
                    minx = min(minx, x.min())
                    maxx = max(maxx, x.max())
                    y = strikes['data'][ia, ig][flags, 1]
                    miny = min(miny, y.min())
                    maxy = max(maxy, y.max())
                    z = strikes['data'][ia, ig][flags, 2]
                    minz = min(minz, z.min())
                    maxz = max(maxz, z.max())
                    ax.scatter(x, y, z, **mar_options)
    if created:
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_zlim(minz, maxz)
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
