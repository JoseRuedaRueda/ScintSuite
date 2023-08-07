"""Flux surface, ECE and other non categorised methods

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ScintSuite.LibData as ssdat
__all__ = ['plot_flux_surfaces', 'plot2D_ECE']


# -----------------------------------------------------------------------------
# --- Flux surfaces plot.
# -----------------------------------------------------------------------------
def plot_flux_surfaces(shotnumber: int, time: float, ax=None,
                       linewidth: float = 1.2,
                       diag: str = 'EQH', exp: str = 'AUGD', ed: int = 0,
                       levels: float = None, label_surf: bool = True,
                       coord_type: str = 'rho_pol',
                       axis_ratio: str = 'equal',
                       units: str = 'm', color=None, view: str = 'pol'):
    """
    Plot the flux surfaces of a given shot in AUG for a given time point.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    Note: @todo, as this has diag and exp as input, maybe we need to rewrite it
    once we move to MAST-U or SMART, I do not know their format... (jrueda)

    :param  shotnumber: shot number to get the equilibrium.
    :param  time: time point to retrieve the equilibrium.
    :param  ax: axis to plot the flux surfaces.
    :param  linewidth: line width for flux surfaces.
    :param  diag: equilibrium diagnostic for the equilibrium. EQH by default.
    :param  exp: experiment in AUG where the EQ is stored. AUGD by default
    :param  ed: Edition of the equilibrium. The latest is taken by default.
    :param  levels: rho values to plot. If none, from 0 to 1
    :param  label_surf: states whether the flux surfaces should be labeled by
    their value.
    :param  coord_type: type of coordinate to be used for the map. rho_pol by
    default. Also available 'rho_tor'
    :param  axis_ratio: axis ratio, 'auto' or 'equal'
    :param  color: if present,all the lines will be plotted in this color
    :param  units: units for the R, Z axis, only cm and m supported
    
    Note: labeling does not work with toroidal view, neither colors, WIP @ TODO
    """
    if units == 'm':
        factor = 1.
    elif units == 'cm':
        factor = 100.
    else:
        raise Exception('Not understood unit')

    if ax is None:
        fig, ax = plt.subplots(1)
        heredado = False
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        heredado = True

    if levels is None:
        if coord_type.lower() == 'rho_pol':
            levels = np.arange(start=0.2, stop=1.25, step=0.2)
        else:
            levels = np.arange(start=0.2, stop=1.05, step=0.2)

    # --- Getting the equilibrium
    R = np.linspace(1.03, 2.65, 128)
    z = np.linspace(-1.224, 1.10, 256)

    Rin, zin = np.meshgrid(R, z)
    rho = ssdat.get_rho(shot=shotnumber, time=time,
                        Rin=Rin.flatten(), zin=zin.flatten(),
                        diag=diag, exp=exp, ed=ed, coord_out=coord_type)

    rho = np.reshape(rho, (256, 128))
    # --- Plotting the flux surfaces.
    if view.lower() == 'pol' or view.lower() == 'poloidal':
        CS = ax.contour(factor * R, factor * z, rho, levels,
                        linewidths=linewidth,
                        colors=color)
        if label_surf:
            ax.clabel(CS, inline=1, fontsize=10)
    else:
        raise Exception('Not yet implemented')
            
    ax.set_aspect(axis_ratio)
    if heredado:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


# -----------------------------------------------------------------------------
# --- Plotting ECE
# -----------------------------------------------------------------------------
def plot2D_ECE(ecedata: dict, rType: str = 'rho_pol', downsample: int = 2,
               ax=None, fig=None, cmap=None, which: str = 'norm',
               cm_norm: str = 'linear'):
    """
    Plot the ECE data into a contour 2D plot.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  ecedata: data as obtained by routine @see{get_ECE}
    :param  rType: X axis to use. Choice between rho_pol, rho_tor, Rmaj
    and channels.
    :param  downsample: downsampling ratio. If the signal is large, the
    plotting routines may saturate your computer.
    :param  ax: axis to plot the data.
    :param  fig: figure handler where the figure is.
    :param  cmap: colormap to use. If None is provide, plasma colormap is used.
    :param  which: type of plot. Norm will plot T_e/<T_e> and total, the whole
    ECE signal.
    :param  cm_norm: colormap normalization. Optional to be chosen between
    linear, sqrt and log.

    :return ax: The used axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    if cmap is None:
        cmap = mpl.cm.plasma

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

    cont_opts = {
        'cmap': cmap,
        'shading': 'gouraud',
        'antialiased': True
    }

    if cm_norm == 'sqrt':
        cont_opts['norm'] = colors.PowerNorm(gamma=0.50)
    elif cm_norm == 'log':
        cont_opts['norm'] = colors.LogNorm(A.min(), A.max())

    im1 = ax.pcolormesh(tbasis, R, A, **cont_opts)

    fig.colorbar(im1, ax=ax)

    return ax
