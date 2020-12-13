"""Module to plot"""

import matplotlib.pyplot as plt
# import numpy as np
from matplotlib.colors import LinearSegmentedColormap


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
    # Arrange ticks a ticks labels
    ax.tick_params(which='both', direction='in', color='k', bottom=True,
                   top=True, left=True, right=True, labelsize=labelsize)
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
