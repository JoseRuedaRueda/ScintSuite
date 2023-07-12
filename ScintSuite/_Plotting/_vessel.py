"""Plot the tokamak vessel"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ScintSuite.LibData as ssdat
from ScintSuite._Plotting._3D_plotting import plot_3D_revolution
__all__ = ['plot_vessel']


def plot_vessel(projection: str = 'pol', units: str = 'm', h: float = None,
                color='k', linewidth=0.5, ax=None, shot: int = 30585,
                params3d: dict = {},
                tor_rot: float = -np.pi/8.0*3.0):
    """
    Plot the tokamak vessel

    Jose Rueda: jrrueda@us.es

    :param  projection: 'tor' or 'toroidal', '3D', else, poloidal view
    :param  units: 'm' or 'cm' accepted
    :param  h: z axis coordinate where to plot (in the case of 3d axes), if none
    a 2d plot will be used
    :param  color: color to plot the vessel
    :param  linewidth: linewidth to be used
    :param  ax: axes where to plot, if none, a new figure will be created
    :param  shot: shot number, only usefull for the case of the poloidal vessel
    of ASDEX Upgrade
    :param  params3d: optional parameters for the plot_3D_revolution method,
    except for the axes
    :param  tor_rot: rotation parameter to properly set the origin of the phi=0
    for the toroidal plot
    :return ax: the axis where the vessel has been drawn
    """
    # --- Section 0: conversion factors
    if units == 'm':
        fact = 1.0
    elif units == 'cm':
        fact = 100.0
    else:
        raise Exception('Units not understood')
    # --- Section 0: get the coordinates:
    if projection == 'tor' or projection == 'toroidal':
        # get the data
        vessel = ssdat.toroidal_vessel(rot=tor_rot) * fact
    else:
        if projection.lower() != '3d':
            vessel = ssdat.poloidal_vessel(shot=shot) * fact
        else:
            vessel = ssdat.poloidal_vessel(simplified=True) * fact
    # --- Section 1: Plot the vessel
    # open the figure if needed:
    if ax is None:
        if (h is None) and (projection.lower() != '3d'):
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
    # Plot the vessel:
    if (h is None) and (projection.lower() != '3d'):
        ax.plot(vessel[:, 0], vessel[:, 1], color=color, linewidth=linewidth)
    elif h is not None:
        height = h * np.ones(len(vessel[:, 1]))
        ax.plot(vessel[:, 0], vessel[:, 1], height, color=color,
                linewidth=linewidth)
    else:
        ax = plot_3D_revolution(vessel[:, 0], vessel[:, 1],
                                ax=ax, **params3d)

    return ax
