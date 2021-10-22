"""Routines to plot the FIDASIM data."""
import numpy as np
import Lib.SimulationCodes.FIDASIM.read as read
import matplotlib.pyplot as plt


def plt_spec(spec: dict = None, filename: str = None):
    """
    Plot FIDASIM spectra

    @param spec: Dictionary created by read spec,
    @param filename: name of the folder with FIDASIM the results, if you want
    to read
    """
    if spec is None:
        spec = read.read_spec(filename)
    plt.plot(spec['lambda'], np.sum(spec['halo'], axis=2)[0, :])
    plt.show()
    return


def plt_neutrals(neutrals: dict = None, grid: dict = None, diag: dict = None,
                 filename: str = None):
    """
    Plot FIDASIM neutral density

    @param filename: name of the folder with FIDASIM the results
    """
    if neutrals is None:
        neutrals = read.read_neutrals(filename)
    if grid is None:
        grid = read.read_grid(filename)
    plt.contour(grid['xx'][0], grid['yy'][0],
                np.sum(neutrals['fdens'], axis=1)[3, :, :])
    if diag is None:
        diag = read.read_diag(filename)
    for i in range(diag['nchan']):
        plt.plot([diag['xyzlos'][0, i], diag['xyzhead'][0, i]],
                 [[diag['xyzlos'][1, i]], [diag['xyzhead'][1, i]]])
    # field = read_field(filename)
    # ii = np.argmin(np.abs(grid['zz'][0]))
    # IPython.embed()
    # # convert from r,z to x,y
    # plt.contour(grid['xx'], grid['yy'], field['rho_grid'])
    plt.show()
    return


def plt_profiles(profiles: dict = None, filename: str = None,
                 clr='k', label=''):
    """
    Plot FIDASIM profiles

    @param filename: name of the folder with FIDASIM the results
    @param clr: the color
    @param label: The label for the line plot
    """
    if profiles is None:
        profiles = read.read_profiles(filename)
    # Temperature
    plt.subplot(221)
    plt.plot(profiles['rho'], profiles['te'], color=clr, label=label)
    plt.plot(profiles['rho'], profiles['ti'], ls='--', color=clr)
    plt.ylabel('Temperature')
    # Density
    plt.subplot(222)
    plt.plot(profiles['rho'], profiles['dene'], color=clr)
    plt.plot(profiles['rho'], profiles['denp'], ls='--', color=clr)
    plt.ylabel('Density')
    # Vtor
    plt.subplot(223)
    plt.plot(profiles['rho'], profiles['vtor'], ls='--', color=clr)
    plt.ylabel('v_tor')
    # Zeff
    plt.subplot(224)
    plt.plot(profiles['rho'], profiles['zeff'], ls='--', color=clr)
    plt.ylabel('z_eff')
    return


def plt_fida(fida: dict = None, filename: str = None):
    """
    Plot FIDA spectrum

    @param filename: name of the folder with FIDASIM the results
    """
    if fida is None:
        fida = read.read_fida(filename)
    plt.plot(fida['lambda'], np.sum(fida['afida'], axis=2)[0, :])
    plt.show()
    return


def plt_grid(grid: dict = None, filename: str = None, view='tor',
             clr='red', alpha=1.0):
    """
    Plot FIDASIM grid

    @param filename: name of the folder with FIDASIM the results
    """
    if grid is None:
        grid = read.read_grid(filename)

    if view == 'tor':
        (xmin, xmax, ymin, ymax) =\
            (grid['xmin'], grid['xmax'], grid['ymin'], grid['ymax'])
        (xmin, xmax, ymin, ymax) = (xmin*0.01, xmax*0.01, ymin*0.01, ymax*0.01)
        plt.plot([xmin, xmin], [ymin, ymax], color=clr, alpha=alpha)
        plt.plot([xmax, xmax], [ymin, ymax], color=clr, alpha=alpha)
        plt.plot([xmin, xmax], [ymin, ymin], color=clr, alpha=alpha)
        plt.plot([xmin, xmax], [ymax, ymax], color=clr, alpha=alpha)
    if view == 'pol':
        (rmin, rmax, zmin, zmax) =\
            (grid['rmin'], grid['rmax'], grid['zmin'], grid['zmax'])
        (rmin, rmax, zmin, zmax) = (rmin*0.01, rmax*0.01, zmin*0.01, zmax*0.01)
        plt.plot([rmin, rmin], [zmin, zmax], color=clr, alpha=alpha)
        plt.plot([rmax, rmax], [zmin, zmax], color=clr, alpha=alpha)
        plt.plot([rmin, rmax], [zmin, zmin], color=clr, alpha=alpha)
        plt.plot([rmin, rmax], [zmax, zmax], color=clr, alpha=alpha)
    return
