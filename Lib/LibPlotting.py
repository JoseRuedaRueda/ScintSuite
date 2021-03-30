"""Module to plot"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from LibMachine import machine
if machine == 'AUG':
    import LibDataAUG as ssdat


# -----------------------------------------------------------------------------
# --- 1D Plotting
# -----------------------------------------------------------------------------
def p1D(ax, x, y, param_dict: dict = None):
    """
    Create basic 1D plot

    Jose Rueda: jrrueda@us.es

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


def p1D_shaded_error(ax, x, y, u_up, color='k', alpha=0.1, u_down=None,
                     line_param={}, line=True):
    """
    Plot confidence intervals

    Jose Rueda: jrrueda@us.es

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
    @param line: If true, the line x,y will be also plotted
    @param line_param: (optional) Line parameters to plot the central line
    @return ax with the applied settings
    """
    if u_down is None:
        u_down = u_up

    ax.fill_between(x, (y - u_down), (y + u_up), color=color, alpha=alpha)
    if line:
        ax.plot(x, y, **line_param)
    return ax


# -----------------------------------------------------------------------------
# --- Axis tuning and colormap
# -----------------------------------------------------------------------------
def axis_beauty(ax, param_dict: dict):
    """
    Modify axis labels, title, ....

    Jose Rueda: jrrueda@us.es

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


# -----------------------------------------------------------------------------
# --- 3D Plotting
# -----------------------------------------------------------------------------
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
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

    return ax


# -----------------------------------------------------------------------------
# --- Vessel plot
# -----------------------------------------------------------------------------
def plot_vessel(projection: str = 'pol', units: str = 'm', h: float = None,
                color='k', linewidth=0.5, ax=None, shot: int = 30585,
                params3d: dict = {},
                tor_rot: float = -np.pi/8.0*3.0):
    """
    Plot the tokamak vessel

    Jose Rueda: jrrueda@us.es

    @param projection: 'tor' or 'toroidal', '3D', else, poloidal view
    @param units: 'm' or 'cm' accepted
    @param h: z axis coordinate where to plot (in the case of 3d axes), if none
    a 2d plot will be used
    @param color: color to plot the vessel
    @param linewidth: linewidth to be used
    @param ax: axes where to plot, if none, a new figure will be created
    @param shot: shot number, only usefull for the case of the poloidal vessel
    of ASDEX Upgrade
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
        fact = 100.0
    # --- Section 0: get the coordinates:
    if projection == 'tor' or projection == 'toroidal':
        # get the data
        vessel = ssdat.toroidal_vessel(rot=tor_rot) * fact
    else:
        if projection != '3D':
            vessel = ssdat.poloidal_vessel(shot=shot) * fact
        else:
            vessel = ssdat.poloidal_vessel(simplified=True) * fact
    # --- Section 1: Plot the vessel
    # open the figure if needed:
    if ax is None:
        if (h is None) and (projection != '3D'):
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
    # Plot the vessel:
    if (h is None) and (projection != '3D'):
        ax.plot(vessel[:, 0], vessel[:, 1], color=color, linewidth=linewidth)
    elif h is not None:
        height = h * np.ones(len(vessel[:, 1]))
        ax.plot(vessel[:, 0], vessel[:, 1], height, color=color,
                linewidth=linewidth)
    else:
        ax = plot_3D_revolution(vessel[:, 0], vessel[:, 1], ax=ax, **params3d)

    return ax

# -----------------------------------------------------------------------------
# --- Flux surfaces plot.
# -----------------------------------------------------------------------------
def plot_flux_surfaces(shotnumber: int, time: float, ax = None, 
                       linewidth:float = 1.2, 
                       diag: str = 'EQH', exp: str = 'AUGD', ed: int = 0, 
                       levels: float = None, label_surf: bool = True,
                       coord_type: str = 'rho_pol'):
    """
    Plots the flux surfaces of a given shot in AUG for a given time point.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param shotnumber: shot number to get the equilibrium.
    @param time: time point to retrieve the equilibrium.
    @param ax: axis to plot the flux surfaces.
    @param linewidth: line width for flux surfaces.
    @param diag: equilibrium diagnostic for the equilibrium. EQH by default.
    @param exp: experiment in AUG where the EQ is stored. AUGD by default
    @param ed: Edition of the equilibrium. The latest is taken by default.
    @param levels: number of flux surfaces to plot.
    @param label_surf: states whether the flux surfaces should be labelled by
    their value.
    @param coord_type: type of coordinate to be used for the map. rho_pol by
    default. Also available 'rho_tor'
    """
    
    if ax is None:
        fig, ax = plt.subplots(1)
        
    if levels is None:
        if coord_type == 'rho_pol':
            levels = np.arange(start=0.2, stop=1.2, step = 0.2)
        else:
            levels = np.arange(start=0.2, stop=1.0, step = 0.2)
        
    #--- Getting the equilibrium
    R = np.linspace(1.03, 2.65, 128)
    z = np.linspace(-1.224, 1.10, 256)
    
    Rin, zin = np.meshgrid(R, z)    
    rho = ssdat.get_rho(shot=shotnumber, time=time,
                        Rin=Rin.flatten(), zin=zin.flatten(),
                        diag=diag, exp=exp, ed=ed, coord_out=coord_type)
    
    rho = np.reshape(rho, (256, 128))
    #--- Plotting the flux surfaces.
    CS=ax.contour(R, z, rho, levels, linewidth=linewidth)
    if label_surf:
        ax.clabel(CS, inline=1, fontsize=10)
    
    return ax

# -----------------------------------------------------------------------------
# --- Plotting ECE
# -----------------------------------------------------------------------------
def plot2D_ECE(ecedata: dict, rType: str = 'rho_pol', downsample: int = 2, 
               ax = None, fig = None, cmap = None, which: str='norm',
               cm_norm: str = 'linear'):
    """
    Plots the ECE data into a contour 2D plot.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param ecedata: data as obtained by routine @see{get_ECE}
    @param rType: X axis to use. Choice between rho_pol, rho_tor, Rmaj 
    and channels.
    @param downsample: downsampling ratio. If the signal is large, the 
    plotting routines may saturate your computer.
    @param ax: axis to plot the data.
    @param fig: figure handler where the figure is.
    @param cmap: colormap to use. If None is provide, plasma colormap is used.
    @param which: type of plot. Norm will plot T_e/<T_e> and total, the whole 
    ECE signal.
    @param cm_norm: colormap normalization. Optional to be chosen between 
    linear, sqrt and log.
    @return ax: return the axis used.
    """
    
    if ax is None:
        fig, ax = plt.subplots(1)
        
    if cmap is None:
        cmap = matplotlib.cm.plasma
    
    ntime = len(ecedata['time'])
    downsample_flag = np.arange(start=0, stop=ntime, step=downsample)
    ntime2 = len(downsample_flag)
    if ecedata['fast_rhop']:
        if rType == 'rho_pol':
            R = np.tile(ecedata['rhop'], (ntime2, 1))
        elif rType == 'rho_tor':
            R = np.tile(ecedata['rhot'], (ntime2, 1))
        elif rType == 'Rmaj':
            R = np.tile(ecedata['r'], (ntime2, 1))
        elif rType == 'channels':
            R = np.tile(ecedata['channels'], (ntime2, 1))
    else:
        if rType == 'rho_pol':
            R = ecedata['rhop'][downsample_flag]
        elif rType == 'rho_tor':
            R = ecedata['rhot'][downsample_flag]
        elif rType == 'Rmaj':
            R = ecedata['r'][downsample_flag]
        elif rType == 'channels':
            R = np.tile(ecedata['channels'], (ntime2, 1))
    
    tbasis = np.tile(ecedata['time'][downsample_flag], (R.shape[1], 1)).T
    if which == 'norm':
        A = ecedata['Trad_norm'][downsample_flag, :]
    elif which == 'total':
        A = ecedata['Trad'][downsample_flag, :]
    
    cont_opts ={'cmap': cmap,
                'shading':'gouraud',
                'antialiased': True
               }
    
    if cm_norm == 'sqrt':
        cont_opts['norm'] = colors.PowerNorm(gamma=0.50)
    elif cm_norm == 'log':
        cont_opts['norm'] = colors.LogNorm(A.min(), A.max())
    
    im1 = ax.pcolormesh(tbasis, R, A, **cont_opts)
    
    fig.colorbar(im1, ax=ax)
    
    return ax