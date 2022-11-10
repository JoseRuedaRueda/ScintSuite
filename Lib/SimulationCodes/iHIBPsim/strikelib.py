"""
Library to read the strike map and strike line for the iHIBPsim simulations.

Pablo Oyola - poyola@us.es
"""

import numpy as np
import xarray as xr
from scipy.constants import elementary_charge as ec
from scipy.constants import physical_constants
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from Lib._Paths import Path
from Lib._Scintillator._mainClass import Scintillator
from Lib._Mapping._Common import transform_to_pixel
import os
import Lib.errors as errors
import random
from Lib.SimulationCodes.iHIBPsim.strikes import strikeLine

pa = Path()
amu2kg = physical_constants['atomic mass constant'][0]

# ------------------------------------------------------------------------------
# STRIKE DATA.
# ------------------------------------------------------------------------------

__header0 = { 0: { 'name': 'id',      'longName': 'Marker ID',
                'shortName': 'ID', 'units': ''
              },
           1: { 'name': 'R',       'longName': 'Major radius',
                'shortName': 'R',  'units': 'm'
              },
           2: { 'name': 'z',       'longName': 'Vertical position',
                'shortName': 'z',  'units': 'm'
              },
           3: { 'name': 'phi',          'longName': 'Toroidal angle',
                'shortName': '$\\phi$', 'units': 'rad'
              },
           4: { 'name': 'vr',         'longName': 'Radial velocity',
                'shortName': '$v_R$', 'units': 'm/s'
              },
           5: { 'name': 'vz',         'longName': 'Vertical velocity',
                'shortName': '$v_z$', 'units': 'm/s'
              },
           6: { 'name': 'vphi',           'longName': 'Toroidal velocity',
                'shortName': '$v_\\phi$', 'units': 'm/s'
              },
           7: { 'name': 'mass',   'longName': 'Mass of the species',
                'shortName': 'm', 'units': 'amu'
              },
           8: { 'name': 'charge',  'longName': 'Charge of the species',
                'shortName': 'q',  'units': 'e'
              },
           9: { 'name': 'weight', 'longName': 'Logarithm of the marker weight',
                'shortName': 'log(w)',  'units': '$log(m^{-3})$'
              },
           10: { 'name': 'time',  'longName': 'Time',
                 'shortName': 't', 'units': 's'
              },
           11: { 'name': 'rhopol0',
                 'longName': 'Initial radial location in mag. coords.',
                 'shortName': '$\\rho_pol{pol}$', 'units': ''
              },
           12: { 'name': 'R0',       'longName': 'Initial major radius',
                 'shortName': '$R_0$',   'units': 'm'
              },
           13: { 'name': 'z0',       'longName': 'Initial vertical position',
                 'shortName': '$z_0$',     'units': 'm'
              },
           14: { 'name': 'x1',
                 'longName': 'Scintillator coordinate (horz.)',
                 'shortName': '$x_1$',     'units': 'm'
              },
           15: { 'name': 'x2',
                 'longName': 'Scintillator coordinate (vert.)',
                 'shortName': '$x_2$',     'units': 'm'
              },
           16: { 'name': 'x3',
                 'longName': 'Scintillator coordinate (perp.)',
                 'shortName': '$x_3$',     'units': 'm'
              },
           17: { 'name': 'lambda',
                 'longName': 'Coordinate along the strikeline',
                 'shortName': '$\\lambda$',     'units': 'm'
              },
           18: { 'name': 'tperp',
                 'longName': 'Coordinate perpendicular to strikeline',
                 'shortName': '$t$',     'units': 'm'
              },
           19: { 'name': 'angle',
                 'longName': 'Collision angle',
                 'shortName': '$\\alpha$',     'units': 'rad'
              },
           20: { 'name': 'intensity',
                 'longName': 'Intensity hitting the scintillator',
                 'shortName': 'I',     'units': 'A/m'
              },
         }

STRIKES_HEADER = { 'v0': __header0
                 }

def read_strike_files(fn: str, version: int=0):
    """
    Reads the strikes file containing the data of the markers when it hits the
    scintillator.

    Pablo Oyola - poyola@us.es

    :param fn: filename of the file with the strike data.
    :param version: version of the strikes data.
    """
    # Checking if the file exists
    if not os.path.isfile(fn):
        raise FileNotFoundError('Cannot find the file %s with strikes'%fn)

    # Retrieving the file structure.
    label = 'v%d'%version
    if label not in STRIKES_HEADER:
        raise errors.NotValidInput('The version %d is not available'%version)

    structure = STRIKES_HEADER[label]

    with open(fn, 'rb') as fid:
        nstrikes = np.fromfile(fid, 'int32', 1)[0]
        nch      = np.fromfile(fid, 'int32', 1)[0]
        data     = np.fromfile(fid, 'float64', nch*nstrikes)

        # In FORTRAN, the strikes are saved as (nch, nstrikes).
        # To read that from python, we can either use the order='F' or
        # load the data in the transposed order.
        data = np.reshape(data, (nstrikes, nch)).T

        # The output will be an ordered xarray.Dataset with all the info.
        output = xr.Dataset()
        output['filename'] = fn
        output['nCh'] = nch
        output['nmarkers'] = nstrikes

        # output['ID'] = xr.DataArray(data[0, :])


        for ikey in structure:
            if ikey == 'id':
                continue
            iname = structure[ikey]['name']
            attrs = { 'short_name': structure[ikey]['shortName'],
                      'long_name': structure[ikey]['longName'],
                      'units': structure[ikey]['units']
                    }
            output[iname] = xr.DataArray(data[ikey, :],
                                         dims=('ID',),
                                         coords = {'ID': data[0, :]},
                                         attrs = attrs)
    return output


# ------------------------------------------------------------------------------
# STRIKE CLASS
# ------------------------------------------------------------------------------
class strikes:
    """
    This class contains all the API to access the strikes file and operate with
    it easily.

    Pablo Oyola - poyola@us.es
    """

    def __init__(self, filename: str, cal=None, scintillator=None):
        """
        Load the data from a strike file and stores the data into the class.

        Pablo Oyola - poyola@us.es

        :param filename: path to the file with the strikes.
        :param cal: when calibration is provided, transformation back and
        forth from the pixel world is possible.
        """

        self.data = read_strike_files(filename, 0)

        # Changing the weight from log(w) to weight.
        self.data['weight'] = np.exp(self.data['weight'])
        self.data['weight'].attrs['units'] = '$m^{-3}$'
        self.data['weight'].attrs['long_name'] = 'Particle weight'
        self.data['weight'].attrs['short_name'] = '$w_i$'

        # Setting distances in the scintillator in cm.
        self.data['x1'] = self.data['x1']*100.0
        self.data['x1'].attrs['units'] = '$cm$'
        self.data['x1'].attrs['long_name'] = 'X Scintillator'
        self.data['x1'].attrs['short_name'] = '$X_{scint}$'

        self.data['x2'] = - self.data['x2']*100.0
        self.data['x2'].attrs['units'] = '$cm$'
        self.data['x2'].attrs['long_name'] = 'Y Scintillator'
        self.data['x2'].attrs['short_name'] = '$Y_{scint}$'

        self.data['x3'] = self.data['x3']*100.0
        self.data['x3'].attrs['units'] = '$cm$'
        self.data['x3'].attrs['long_name'] = 'Z Scintillator'
        self.data['x3'].attrs['short_name'] = '$Z_{scint}$'


        # From the data stored in the file, we compute useful derived values.
        self.data['velocity'] = np.sqrt(self.data['vr'] ** 2 + \
                                        self.data['vz'] ** 2 + \
                                        self.data['vphi'] ** 2)
        self.data['velocity'].attrs['units'] = 'm/s'
        self.data['velocity'].attrs['short_name'] = 'v'
        self.data['velocity'].attrs['long_name'] = 'Speed'

        self.data['energy'] = self.data['velocity']**2 * \
                              self.data['mass']/2.0 * amu2kg / ec *1e3
        self.data['energy'].attrs['units'] = 'keV'
        self.data['energy'].attrs['short_name'] = 'E'
        self.data['energy'].attrs['long_name'] = 'Kinetic energy'

        self.data['flux'] = self.data['velocity'] * self.data['weight']
        self.data['flux'].attrs['units'] = 'ion/s'
        self.data['flux'].attrs['short_name'] = '$\\Phi_{ion}$'
        self.data['flux'].attrs['long_name'] = 'Particle flux'

        # Setting internally the calibration and scintillator.
        self.scint = None
        self.calib = None

        self.set_scintillator(scintillator)
        self.set_calibration(cal)

    def set_scintillator(self, scint: Scintillator=None):
        """
        Sets internally the scintillator object. If the scintillator is equiped
        with a camera calibration, this will check whether the calibrations are
        consistent.

        Pablo Oyola - poyola@us.es

        :param scint: scintillator object.
        """

        if scint is None:
            return

        if scint.CameraCalibration is not None:
            if self.calib is not None:
                if self.CameraCalibration != scint.CameraCalibration:
                    raise errors.InconsistentData('Calibration is not consistent!')
            else:
                self.calib = scint.CameraCalibration

        self.scint = scint

    def set_calibration(self, calib=None):
        """
        Sets internally the calibration object. If the Scintillator is already
        set inside the class, then this will also change internally the
        calibration of the scintillator.

        Pablo Oyola - poyola@us.es

        :param scint: scintillator object.
        """

        if calib is None:
            return

        self.calib = calib
        self.scint.calculate_pixel_coordinates(cal=self.calib)

        # With the calibration, we can compute the strikes positions in
        # distorted pixel data.
        xcam, ycam = transform_to_pixel(self.data['x1']/100.0,
                                        self.data['x2']/100.0,
                                        self.calib)

        self.data['xcam'] = xcam
        self.data['xcam'].attrs['units'] = 'pix'
        self.data['xcam'].attrs['long_name'] = 'X in pix. coords.'
        self.data['xcam'].attrs['short_name'] = '$X_{pix}$'

        self.data['ycam'] = ycam
        self.data['ycam'].attrs['units'] = 'pix'
        self.data['ycam'].attrs['long_name'] = 'Y in pix. coords.'
        self.data['ycam'].attrs['short_name'] = '$Y_{pix}$'


    def hist1d(self, name: str, bins: int=51, ranges: float=None,
               get_centers: bool=True, weight_name: str='weight'):
        """
        We make a 1D histogram with the data in the strike data.

        Pablo Oyola - poyola@us.es

        :param args: a list of parameters to pass down to the histogram makers.
        :param bins: number of bins to divide the data.
        :param ranges: ranges for the data. If None, the (min, max) values along
        each direction are taken.
        :param get_centers: whether to return the edges or the centers of the
        bins histograms. Returns the centers by default (True).
        """

        if name not in self.data.keys():
            raise errors.NotValidInput('The variable %s is not available'%name)
        if weight_name not in self.data.keys():
            raise errors.NotValidInput('The variable %s is not available'%weight_name)

        data = self.data[name].values
        weights = self.data[weight_name].values
        if ranges is None:
            ranges = [data.min(), data.max()]

        ranges = np.atleast_1d(ranges)

        W, xc = np.histogram(data, weights=weights, bins=bins, range=ranges)

        if get_centers:
            xx = (xc[1:] + xc[:-1])/2.0
        else:
            xx = xc

        # Dividing the histogram by the step size.
        dx = xc[1] - xc[0]
        W /= dx

        return xx, W

    def hist2d(self, name_x: str, name_y: str, bins: int=51, ranges: float=None,
               get_centers: bool=True, weight_name: str = 'weight'):
        """
        We make a 1D histogram with the data in the strike data.

        Pablo Oyola - poyola@us.es

        :param args: a list of parameters to pass down to the histogram makers.
        :param bins: number of bins to divide the data.
        :param ranges: ranges for the data. If None, the (min, max) values along
        each direction are taken.
        :param get_centers: whether to return the edges or the centers of the
        bins histograms. Returns the centers by default (True).
        """

        if name_x not in self.data.keys():
            raise errors.NotValidInput('The variable %s is not available'%name_x)

        if name_y not in self.data.keys():
            raise errors.NotValidInput('The variable %s is not available'%name_y)

        if weight_name not in self.data.keys():
            raise errors.NotValidInput('The variable %s is not available'%weight_name)

        datax = self.data[name_x].values
        datay = self.data[name_y].values
        weights = self.data[weight_name].values
        if ranges is None:
            ranges = [[datax.min(), datax.max()], [datay.min(), datay.max()]]

        ranges = np.atleast_1d(ranges)

        W, xc, yc = np.histogram2d(datax, datay, weights=weights,
                                    bins=bins, range=ranges)

        if get_centers:
            xx = (xc[1:] + xc[:-1])/2.0
            yy = (yc[1:] + yc[:-1])/2.0
        else:
            xx = xc
            yy = yc

        # Dividing the histogram by the step size.
        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        W /= dx * dy

        return xx, yy, W

    def hist(self, *args, bins=51, ranges=None, get_centers: bool=True,
             weight_name : str = 'weight'):
        """
        Wraps around the internal histogram routines.

        Pablo Oyola - poyola@us.es

        :param args: a list of parameters to pass down to the histogram makers.
        :param bins: number of bins to divide the data.
        :param ranges: ranges for the data. If None, the (min, max) values along
        each direction are taken.
        :param get_centers: whether to return the edges or the centers of the
        bins histograms. Returns the centers by default (True).
        """

        if len(args) == 1:
            return self.hist1d(args[0], bins=bins, ranges=ranges,
                               get_centers=get_centers)
        elif len(args) == 2:
            return self.hist2d(args[0],args[1], bins=bins, ranges=ranges,
                               get_centers=get_centers)
        else:
            raise NotImplementedError('Histograms for more than 2D are not' + \
                                      ' yet supported')

    def plot(self, *args, bins: int=51, ranges=None, ax=None,
             weight_name:str='weights', **plt_params):
        """
        Routine to plot 1D/2D data from histograms from the strike data.

        Pablo Oyola - poyola@us.es

        :param args: a list of parameters to pass down to the histogram makers.
        :param bins: number of bins to divide the data.
        :param ranges: ranges for the data. If None, the (min, max) values along
        each direction are taken.
        :param get_centers: whether to return the edges or the centers of the
        bins histograms. Returns the centers by default (True).
        """

        get_centers = True
        if len(args) == 2:
            get_centers = False

        data = self.hist(*args, bins=bins, ranges=ranges,
                         get_centers=get_centers, weight_name=weight_name)


        ax_was_none = ax is None
        if ax_was_none:
            fig, ax = plt.subplots(1)
        else:
            fig = ax.figure

        labels = [self.data[iname].short_name + \
                   '[%s]'%self.data[iname].units for iname in args]

        histlabel = self.data[weight_name].short_name + \
                   '[%s]'%self.data[weight_name].units

        if len(args) == 1:
            x = data[0]
            y = data[1]

            if 'label' not in plt_params:
                im = ax.plot(x, y, label=histlabel, **plt_params)
            else:
                im = ax.plot(x, y, **plt_params)

            ax.set_xlabel(labels[0])
            ax.set_ylabel(histlabel)

        elif len(args) == 2:
            x = data[0]
            y = data[1]
            z = data[2]
            extent = [x.min(), x.max(), y.min(), y.max()]

            if 'label' not in plt_params:
                im = ax.imshow(z.T, extent=extent, origin='lower',
                               label=histlabel, **plt_params)
            else:
                im= ax.plot(x, y, extent=extent, origin='lower', **plt_params)

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

            cbar = fig.colorbar(mappable=im, ax=ax)
            cbar.set_label(histlabel)
        return ax, im

    def plot3d(self, weighted: bool = False, npoints:int = 2000,
               ax=None, **plt_params):
        """
        Plot strikes in a 3D axis.

        Pablo Oyola - poyola@us.es

        :param weighted: whether to plot the data colour-coded with the weight or
        with a given color.
        :param npoints: maximum number of points to draw. They are randomly
        taken from the deck of strikes.
        :param ax: axis to plot the 3D object. If not provided, it will be
        automatically created.
        :param plt_params: extra parameters to pass down to the plt.scatter.
        """
        ## Let's generate the axis to plot.
        ax_was_none = ax is None
        if ax_was_none:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')

        npoints = min(npoints, self.data.R.size)
        idx     = random.sample(range(self.data.R.size),
                                   npoints)

        R = self.data.R[idx]
        z = self.data.z[idx]
        phi = self.data.phi[idx]

        if weighted:
            weights = self.data.intensity[idx]
        else:
            weights = None

        x = R * np.cos(phi)
        y = R * np.sin(phi)

        im = ax.scatter(x, y, z, 'o', c=weights, **plt_params)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        if weighted:
            cbar = ax.figure.colorbar(mappable=im, ax=ax)
            cbar.set_label('Intensity')

        return im, ax



    def plotScintillator(self, dx: float = 0.1, dy: float = 0.1, ax=None,
                         cmap=cm.plasma, kindplot: str = 'intensity',
                         norm=None, ax_options: dict = {},
                         min_cb: float = None, max_cb: float = None):

        """
        Plot the scintillator image to the provided axis. If None are provided,
        new ones are created.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param dx: bin size in the Y-direction [cm]. Default to 100 um
        :param dy: bin size in the Y-direction [cm]. Default to 100 um
        :param ax: Axis to plot the scintillator image.
        :param cmap: Colormap to plot. By default, set to plasma.
        :param kindplot: type of plot to make: flux, intensity, density.
        :param norm: kind of norm for the colorbar. Linear if None is provided
        :param ax_options: dictionary with extract axis options.
        @return ax: axis with the scintillator image.
        """

        # Check if the user loaded the scintillator coordinates.
        if self.scint is None:
            raise errors.NotDataPreloaded('The scintillator was not loaded!')

        if kindplot.lower() == 'flux':
            weights_name = 'flux'
        elif kindplot.lower() == 'intensity':
            weights_name = 'intensity'
        elif kindplot.lower() == 'weight':
            weights_name = 'weight'
        else:
            raise errors.NotValidInput('The option %s is not available'%kindplot)

        # Setting up the options.
        cont_opts = {'cmap': cmap}

        if norm == 'log':
            cont_opts['norm'] = colors.LogNorm()
        elif norm == 'sqrt':
            cont_opts['norm'] = colors.PowerNorm(gamma=0.5)

        if (min_cb is not None) and (min_cb is not None):
            cont_opts['vmin'] = min_cb
            cont_opts['vmax'] = max_cb

        # Checking the size of the scintillator.
        xlims = np.array([self.scint._coord_real['x1'].min(),
                          self.scint._coord_real['x1'].max()])
        ylims = np.array([self.scint._coord_real['x2'].min(),
                          self.scint._coord_real['x2'].max()])

        ranges = np.array([xlims, ylims])

        dx_tot = xlims[1] - xlims[0]
        dy_tot = ylims[1] - ylims[0]

        bins = np.array([np.ceil(dx_tot/dx), np.ceil(dy_tot/dy)], dtype=int)

        ax, im = self.plot('x1', 'x2', bins=bins, ranges=ranges, ax=ax,
                            weight_name=weights_name, **cont_opts)


        # Plotting on top the scintillator.
        self.scint.plot_real(ax=ax, line_params={'color' : 'w'})
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        return ax, im

    def plot_frame(self, pix_x: int, pix_y: int, ax=None, cmap=cm.plasma,
                   kindplot: str = 'intensity', norm=None, ax_options: dict = {},
                   min_cb: float = None, max_cb: float = None):

        """
        Plots the camera frame using the internal calibration, which will
        produce a view of the scintillator distorted.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param pix_x: number of pixels along the X-direction.
        :param pix_y: number of pixels along the Y-direction.
        :param ax: Axis to plot the scintillator image.
        :param cmap: Colormap to plot. By default, set to plasma.
        :param kindplot: type of plot to make: flux, intensity, density.
        :param norm: kind of norm for the colorbar. Linear if None is provided
        :param ax_options: dictionary with extract axis options.
        @return ax: axis with the scintillator image.
        """

        # Checking whether a calibration was provided.
        if self.calib is None:
            raise errors.NotFoundCameraCalibration('Calibration is not set!')

        if kindplot.lower() == 'flux':
            weights_name = 'flux'
        elif kindplot.lower() == 'intensity':
            weights_name = 'intensity'
        elif kindplot.lower() == 'weight':
            weights_name = 'weight'
        else:
            raise errors.NotValidInput('The option %s is not available'%kindplot)

        # Setting up the options.
        cont_opts = {'cmap': cmap}

        if norm == 'log':
            cont_opts['norm'] = colors.LogNorm()
        elif norm == 'sqrt':
            cont_opts['norm'] = colors.PowerNorm(gamma=0.5)

        if (min_cb is not None) and (min_cb is not None):
            cont_opts['vmin'] = min_cb
            cont_opts['vmax'] = max_cb

        # If the calibration is set, then we expect the data is already
        # computed in the self.data xarray.
        if ('xcam' not in self.data) or ('ycam' not in self.data):
            self.set_calibration(self.calib)

        # With this all the calibration is set self-consistently. We compute
        # now the histogram using the pixel coordinates.
        ranges = [[0, pix_x], [0, pix_y]]
        bins   = [pix_x + 1, pix_y + 1]
        ax, im =  self.plot('xcam', 'ycam', bins=bins, ranges=ranges,
                            weight_name=weights_name, ax=ax, **cont_opts)


        # Plotting on top the scintillator.
        self.scint.plot_pix(ax=ax, line_params={'color' : 'w'})


        ax.set_xlim([0, pix_x])
        ax.set_ylim([0, pix_y])

        return ax, im