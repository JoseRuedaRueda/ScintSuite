"""
Deposition library for iHIBPsim.

This library handles the deposition file generated by the iHIBPsim code with
the secondary birth profile.

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import RectBivariateSpline
from Lib.LibData import get_rho
import Lib.SimulationCodes.iHIBPsim.hibp_utils as utils
logger = logging.getLogger('ScintSuite.iHIBPsim')
try:
    import xarray as xr
except ModuleNotFoundError:
    logger.warning('10: Xarray not found. Needed for iHIBPsim.')


# -----------------------------------------------------------------------------
# Variables.
# -----------------------------------------------------------------------------
variables_name = np.array(('ID', 'Rmajor', 'Z', 'phi', 'vR',  'vPhi', 'vZ',
                           'mass', 'charge', 'weight', 'time', 'rhopol0',
                           'Rmajor0', 'Z0', 'scint_X', 'scint_Y', 'scint_Z',
                           'Lambda', 'tPerp', 'Angle', 'intensity'))


# -----------------------------------------------------------------------------
# Auxiliar functions to read the files.
# -----------------------------------------------------------------------------
def read_deposition_header(filename: str, version: int = 1):
    """
    Get the header from the deposition file according to the code version.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param filename: filename of the file with the deposition.
    """
    if version == 1:
        with open(filename, 'rb') as fid:
            nch   = np.fromfile(fid, 'int32', 1)[0]
            nmark = np.fromfile(fid, 'int32', 1)[0]

        header = { 'N': nmark,
                   'nch': nch,
                   'offset': 8,
                   'names': variables_name[:nch]
                 }
    else:
        txt = 'The next version of deposition is not yet implementeds'
        raise NotImplementedError(txt)

    return header

# -----------------------------------------------------------------------------
# Class to access the deposition.
# -----------------------------------------------------------------------------
class deposition:
    """
    This class controls the access to the deposition profile generated by the
    iHIBPsim fortran code.
    """
    def __init__(self, path: str, ver: int=1, shot: int=None, time: float=None):
        """
        Reads the header of the file with the deposition and allows to plot and
        read the data easily.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param path: filename of the deposition file generated by the code.
        @param ver: version of the deposition file. 1 -> Up to iHIBPsim version
        0.2.4, 2 thereon.
        """

        if not os.path.isfile(path):
            raise FileNotFoundError('Cannot find %s'%path)

        ## Loading the header.
        self.header = read_deposition_header(path, version=ver)
        self.fn     = path

        ## ---------------------------------------------------------
        if ver == 1:
            if (shot is not None) and (time is not None):
                self.set_magnetics(shotnumber=shot, timepoint=time)

    def set_magnetics(self, shotnumber: int, timepoint: float):
        """
        Loads the corresponding magnetic field and magnetic flux coordinates
        from the AUG database.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param shotnumber: shot to map the coordinates.
        @param timepoint: timepoint in the shot to retrieve.
        """
        Rin = np.linspace(1.03, 2.65, 513)
        zin = np.linspace(-1.24, 1.05, 512)

        Rin2, zin2 = np.meshgrid(Rin, zin)

        try:
            self.rhopol = get_rho(shot=shotnumber, time=timepoint,
                                  Rin=Rin2.flatten(), zin=zin2.flatten())
            self.rhopol = np.reshape(self.rhopol, (512, 513)).T
        except:
            txt = 'Cannot get the equilibrium for shot #%05d'%shotnumber
            raise ValueError(txt)

        self.rhop_interp = RectBivariateSpline(Rin, zin, self.rhopol)

    def read(self):
        """
        Reads the deposition file and returns the data.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        # Total number of data to read.
        ntotal = self.header['N'] * self.header['nch']

        with open(self.fn, 'rb') as fid:
            fid.seek(self.header['offset'], 0)
            data = np.fromfile(fid, dtype='float64',
                               count=ntotal).reshape((self.header['nch'],
                                                      self.header['N']),
                                                      order='F')
        # Returns the data as an xarray.
        output = xr.DataArray(data, dims=('variable', 'marker'),
                              coords=(self.header['names'],
                                      np.arange(self.header['N'])))

        return output

    def plot1d(self, xaxis: str='rmajor', ax=None, bins: int=None, **line_params):
        """
        Plot the deposition profile as a function either from major radius or
        the rhopol.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param xaxis: axis to plot the birth profile. Either 'rmajor' or 'rhopol'.
        @param ax: axis to plot the data. If none, new ones will be created.
        @param bins: number of bins to plot the deposition.
        @param line_params: keyword arguments to pass down to the matplotlib
        plot function.
        """

        # Let's read the data from the file.
        data = self.read()

        ## Let's generate the axis to plot.
        ax_was_none = ax is None
        if ax_was_none:
            fig, ax = plt.subplots(1)

        ## Making the 1D histogram.

        R = data.sel(variable='Rmajor').values
        w = np.exp(data.sel(variable='weight').values)
        if xaxis.lower() == 'rmajor':
            grr, H = utils.hist1d(R, w, bins=bins)

            xlabel = 'Major radius (m)'
            ylabel = 'Ion birth density($m^{-3}/m$)'
        elif xaxis.lower() == 'rhopol':
            z = data.sel(variable='Z').values
            if ('rhop_interp' not in self.__dict__):
                raise ValueError('Cannot plot rhopol if magnetic'+\
                                 'are not loaded.')
            rhop = self.rhop_interp(R, z, grid=False)
            grr, H = utils.hist1d(rhop, w, bins=bins)


            xlabel = '$\\rho_{pol}$ (-)'
            ylabel = 'Ion birth density($m^{-3}$/(unit $\\rho_{pol}$))'
        else:
            raise ValueError('Axis = %s not recognized'%xaxis)

        dr = grr[1] - grr[0]
        H /= dr

        ax.plot(grr, H, **line_params)

        if ax_was_none:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        return ax

    def plot2d(self, view: str='pol', ax=None, vessel: bool=False,
               bins: int=None, **spec_params):
        """
        Plot the deposition as a 2D figure either in the poloidal or toroidal
        view and allows to include the vessel plot.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param view: either 'tor' or 'pol' for toroidal or poloidal, respectively.
        @param ax: axis to plot the data. If none, new ones will be created.
        @param bins: number of bins to plot the deposition.
        @param line_params: keyword arguments to pass down to the matplotlib
        imshow function.
        """

        # Let's read the data from the file.
        data = self.read()

        ## Let's generate the axis to plot.
        ax_was_none = ax is None
        if ax_was_none:
            fig, ax = plt.subplots(1)

        ## Making the 1D histogram.

        R = data.sel(variable='Rmajor').values
        z = data.sel(variable='Z').values
        phi = data.sel(variable='phi').values
        w = np.exp(data.sel(variable='weight').values)
        if view.lower() == 'pol':
            grr, gzz, H = utils.hist2d(R, z, w, bins=bins)

            xlabel = 'Major radius (m)'
            ylabel = 'Height (m)'
        elif view.lower() == 'tor':
            x = R*np.cos(phi)
            y = R*np.sin(phi)
            grr, gzz, H = utils.hist2d(x, y, w, bins=bins)

            xlabel = 'X (m)'
            ylabel = 'Y (m)'

        else:
            raise ValueError('View = %s not recognized'%view)

        dr = grr[1] - grr[0]
        dz = gzz[1] - gzz[0]
        H /= dr*dz
        im = ax.imshow(H, extent=[grr[0], grr[-1], gzz[0], gzz[-1]],
                       origin='lower', **spec_params)

        if ax_was_none:
            zlabel = 'Ion birth density($m^{-3}/m^2$)'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            cbar = fig.colorbar(mappable=im, ax=ax)
            cbar.set_label(zlabel)

        return ax

    def plot3d(self, ax=None, **plt_params):
        """"
        Plots the deposition using a 3D scatter plot.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param ax: axis to plot the deposition.
        """
        # Let's read the data from the file.
        data = self.read()

        ## Let's generate the axis to plot.
        ax_was_none = ax is None
        if ax_was_none:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')

        R = data.sel(variable='Rmajor').values
        z = data.sel(variable='Z').values
        phi = data.sel(variable='phi').values
        w = np.exp(data.sel(variable='weight').values)

        ## To Cartesian coordinates.
        x = R*np.cos(phi)
        y = R*np.sin(phi)

        return ax.scatter(x, y, z, c=w)