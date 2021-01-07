"""Module to plot"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from LibMachine import machine
if machine == 'AUG':
    import LibDataAUG as ssdat


# Basic 1D plotting
def p1D(ax, x, y, param_dict: dict = None):
    """
    Create basic 1D plot

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param ax: Axes. The axes to draw to
    @param x: The x data
    @param y: The y data
    @param param_dict: dict. Dictionary of kwargs to pass to ax.plot
    @return out: ax.plot with the applied settings
    """
    if param_dict is None:
        param_dict = {}
    ax.plot(x, y, **param_dict)
    return ax


def p1D_shaded_error(ax, x, y, u_up, color='k', alpha=0.1, u_down=-10001):
    """
    Plot confidence intervals

    Jose Rueda: jose.rueda@ipp.mpg.de

    Basic shaded region between y + u_up and y - u_down. If no u_down is
    provided, u_down is taken as u_up

    @param ax: Axes. The axes to draw to
    @param x: The x data
    @param y: The y data
    @param u_up: The upper limit of the error to be plotted
    @param u_down: (Optional) the bottom limit of the error to be plotter
    @param color: (Optional) Color of the shaded region
    @param alpha: (Optional) Transparency parameter (0: full transparency,
    1 opacity)
    @return ax with the applied settings
    """
    if u_down == -10001:
        u_down = u_up

    ax.fill_between(x, (y - u_down), (y + u_up), color=color, alpha=alpha)
    return ax


def axis_beauty(ax, param_dict: dict):
    """
    Modify axis labels, title, ....

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param ax: Axes. The axes to be modify
    @param param_dict: Dictionary with all the fields
    @return ax: Modified axis
    """
    # Define fonts
    font = {}
    if 'fontname' in param_dict:
        font['fontname'] = param_dict['fontname']
    if 'fontsize' in param_dict:
        font['size'] = param_dict['fontsize']
        labelsize = param_dict['fontsize']
        # ax.tick_params(labelsize=param_dict['fontsize'])
    if 'xlabel' in param_dict:
        ax.set_xlabel(param_dict['xlabel'], **font)
    if 'ylabel' in param_dict:
        ax.set_ylabel(param_dict['ylabel'], **font)
    if 'yscale' in param_dict:
        ax.set_yscale(param_dict['yscale'])
    if 'xscale' in param_dict:
        ax.set_xscale(param_dict['xscale'])
    if 'tickformat' in param_dict:
        ax.ticklabel_format(style=param_dict['tickformat'], scilimits=(-2, 2),
                            useMathText=True)
        if 'fontsize' in param_dict:
            ax.yaxis.offsetText.set_fontsize(param_dict['fontsize'])
        if 'fontname' in param_dict:
            ax.yaxis.offsetText.set_fontname(param_dict['fontname'])
    if 'grid' in param_dict:
        if param_dict['grid'] == 'both':
            ax.grid(True, which='minor', linestyle=':')
            ax.minorticks_on()
            ax.grid(True, which='major')
        else:
            ax.grid(True, which=param_dict['grid'])
    if 'ratio' in param_dict:
        ax.axis(param_dict['ratio'])
    # Arrange ticks a ticks labels
    if 'fontsize' in param_dict:
        ax.tick_params(which='both', direction='in', color='k', bottom=True,
                       top=True, left=True, right=True, labelsize=labelsize)
    else:
        ax.tick_params(which='both', direction='in', color='k', bottom=True,
                       top=True, left=True, right=True)
    return ax


def Gamma_II(n=256):
    """
    Gamma II colormap

    This function creates the colormap that coincides with the
    Gamma_II_colormap of IDL.

    @param n: numbers of levels of the output colormap
    """
    cmap = LinearSegmentedColormap.from_list(
        'mycmap', ['black', 'blue', 'red', 'yellow', 'white'], N=n)
    return cmap


def plot_3D_revolution(r, z, phi_min: float = 0.0, phi_max: float = 1.57,
                       nphi: int = 25, ax=None,
                       color=[0.5, 0.5, 0.5], alpha: float = 0.75):
    """
    Plot a revolution surface with the given cross-section

    Jose Rueda: ruejo@ipp.mpg.de

    @param r: Array of R's of points defining the cross-section
    @param z: Array of Z's of points defining the cross-section
    @param phi_min: minimum phi to plot, default 0
    @param phi_max: maximum phi to plot, default 1.57
    @param nphi: Number of points in the phi direction, default 25
    @param color: Color to plot the surface, default, light gray [0.5,0.5,0.5]
    @param alpha: transparency factor, default 0.75
    @param ax: 3D axes where to plot, if none, a new window will be opened
    @return ax: axes where the surface was drawn
    """
    # --- Section 0: Create the coordiates to plot
    phi_array = np.linspace(phi_min, phi_max, num=nphi)
    # Create matrices
    X = np.tensordot(r, np.cos(phi_array), axes=0)
    Y = np.tensordot(r, np.sin(phi_array), axes=0)
    Z = np.tensordot(z, np.ones(len(phi_array)), axes=0)

    # --- Section 1: Plot the surface
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

    return ax


def plot_vessel(projection: str = 'pol', units: str = 'm', h: float = None,
                color='k', linewidth=0.5, ax=None, shot: int = 30585,
                shaded3d: bool = 'False', params3d: dict = {},
                tor_rot: float = -np.pi/8.0*3.0):
    """
    Plot the tokamak vessel

    Jose Rueda: jrrueda@us.es

    @param projection: 'tor' or 'toroidal', else, poloidal view
    @param units: 'm' or 'cm' accepted
    @param h: z axis coorditate where to plot (in the case of 3d axes), if none
    a 2d plot will be used
    @param color: color to plot the vessel
    @param linewidth: linewidth to be used
    @param ax: axes where to plot, if none, a new figure will be created
    @param shot: shot number, only usefull for the case of the poloidal vessel
    of ASDEX Upgrade
    @param shaded3d: if true a 3d basic representation will be plotted.
    @param params3d: optional parameters for the plot_3D_revolution method,
    except for the axes
    @param tor_rot: rotation parameter to properly set the origin of the phi=0
    for the toroidal plot
    @return ax: the axis where the vessel has been drawn
    """
    # --- Section 0: conversion factors
    if units == 'm':
        fact = 1.0
    elif units == 'cm':
        fact = 10.0
    # --- Section 0: get the coordinates:
    if projection == 'tor' or projection == 'toroidal':
        # get the data
        vessel = ssdat.toroidal_vessel() * fact
    else:
        if shaded3d is not True:
            vessel = ssdat.poloidal_vessel(shot=shot) * fact
        else:
            vessel = ssdat.poloidal_vessel(simplified=True) * fact
    # --- Section 1: Plot the vessel
    # open the figure if needed:
    if ax is None:
        if (h is None) and (shaded3d is not True):
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
    # Plot the vessel:
    if (h is None) and (shaded3d is not True):
        ax.plot(vessel[:, 0], vessel[:, 1], color=color, linewidth=linewidth)
    elif h is not None:
        height = h * np.ones(len(vessel[:, 1]))
        ax.plot(vessel[:, 0], vessel[:, 1], height, color=color,
                linewidth=linewidth)
    elif shaded3d:
        ax = plot_3D_revolution(vessel[:, 0], vessel[:, 1], ax=ax, **params3d)

    return ax
