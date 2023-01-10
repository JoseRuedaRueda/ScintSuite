"""
This library is used to handle the strikeline data from the iHIBPsim simulation.

Strikeline or strikemap in iHIBPsim are done differently to the FILD/INPA, since
the main parameters is just the initial position along the beam coordinates.

Pablo Oyola - poyola@us.es
"""

import numpy as np
import xarray as xr
import os
from scipy.constants import elementary_charge as ec
from typing import Union
from Lib._Mapping._Common import transform_to_pixel
from Lib._Scintillator._mainClass import Scintillator
from Lib._Plotting._1D_plotting import colorline
import scipy.interpolate as sinterp
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import Lib.LibData.AUG.DiagParam as diagparams

import Lib.errors as errors
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# STRIKELINE/STRIKEMAP DATA.
# ------------------------------------------------------------------------------
def bytes_to_str(data: bytes, **kwargs):
    """
    Transform bytes to string.

    Pablo Oyola - poyola@us.es

    :param data: string encoded as bytes.
    """

    return (b''.join(data).strip()).decode('utf8', errors='ignore').strip()

def j2keV_fun(data: float,**kwargs):
    """
    Transform the energy units into keV.

    Pablo Oyola - poyola@us.es

    :param data: energy data.
    :param units: units to transform
    """

    data /= ec
    return data * 1.0e3

def beam_model_to_str(data: int, **kwargs):
    """
    Transform the beam model number code into a human readable string.

    Pablo Oyola - poyola@us.es

    :param data: read value of the beam model.
    :param kwargs: other arguments to be ignored
    """

    if data == 0:
        return 'Beam line'
    elif data == 1:
        return 'Finite beam width'
    elif data == 2:
        return 'Divergent finite beam'
    elif data == 3:
        return 'Energy spread and divergent finite beam'
    else:
        raise ValueError('Not supported option %d'%data)


def efield_mode_to_str(data: int, **kwargs):
    """
    Transform the beam model number code into a human readable string.

    Pablo Oyola - poyola@us.es

    :param data: read value of the electric field model.
    :param kwargs: other arguments to be ignored
    """

    return 'No E-field model'

def map_method_to_str(data: int, **kwargs):
    """
    Transform the mapping method to a human readable string.

    Pablo Oyola - poyola@us.es

    :param data: read value of the mapping method.
    :param kwargs: other arguments to be ignored
    """

    if data == 0:
        return 'Nothing'
    elif data == 1:
        return 'Weight density'
    elif data == 2:
        return 'Intensity'
    else:
        raise ValueError('Not supported mapping method %d'%data)

def strline_mode_to_str(data: int, **kwargs):
    """
    Transform the strikeline mode to human readable string.

    Pablo Oyola - poyola@us.es

    :param data: read value of the mapping method.
    :param kwargs: other arguments to be ignored
    """

    if data == 1:
        return 'rhopol'  # Using the major radius as a mapping coordinate.
    elif data == 2:
        return 'R' # Using the rhopol as a mapping coordinate.
    elif data == 3:
        return 'R,z' # Using both (R, z) to map the coordinates on scintillator
    else:
        raise ValueError('Not supported mapping coords. method %d'%data)

def int_to_bool(data: int, **kwargs):
    """
    Boolean data is encoded as int32 numbers into binary files. This just
    transform the int32 into python-bool.

    Pablo Oyola - poyola@us.es

    :param data: integer to transform into boolean.
    """

    return data != 0

GENERIC_ATTRS_STRLINE = { 'x1': { 'shortName': '$x_1$',
                                  'longName': 'Horizontal axis',
                                  'units': 'm',
                                 },
                          'x2': { 'shortName': '$x_2$',
                                  'longName': 'Vertical axis',
                                  'units': 'm',
                                 },
                          'dx1': { 'shortName': '$\\delta x_1$',
                                  'longName': 'Horizontal axis uncertainty',
                                  'units': 'm',
                                 },
                          'dx2': { 'shortName': '$\\delta x_2$',
                                  'longName': 'Vertical axis uncertainty',
                                  'units': 'm',
                                 },

                          'w': { 'shortName': 'w',
                                  'longName': 'Weight along strikeline',
                                  'units': '',
                                 }
                         }

def mapping_text_to_attrs(typ: str):
    """"
    Transform the type of mapping coordinate to a set of attributes to be set
    on the dictionary.

    Pablo Oyola - poyola@us.es

    :param typ: string indicating the type of mapping coordinate.
    """

    if typ.lower() == 'r':
        return {'units': 'm', 'longName': 'Initial major radius',
                'shortName': '$R_0$'}
    elif typ.lower() == 'rhopol':
        return {'units': '', 'longName': 'Normalized magn. coordinate',
                'shortName': '$\\rho_{pol}$'}
    elif typ.lower() == 'r,z':
        return {'units': 'm', 'longName': '(R, z)',
                'shortName': '$(R, z)$'}
    else:
        raise NotImplementedError('The method %s is not yet implemented'%typ)

def weight_mapping_text_to_attrs(typ: str):
    """"
    Transform the type of mapping weight to a set of attributes to be set
    on the dictionary.

    Pablo Oyola - poyola@us.es

    :param typ: string indicating the type of mapping weight.
    """

    if typ.lower() == 'nothing':
        return {'units': '', 'longName': 'Non-weighted',
                'shortName': '$w=1$'}
    elif typ.lower() == 'weight density':
        return {'units': 'm^{-3}', 'longName': 'Weight density',
                'shortName': '$n_{ion}$'}
    elif typ.lower() == 'intensity':
        return {'units': 'A', 'longName': 'Particle current',
                'shortName': '$I$'}
    else:
        raise NotImplementedError('The method %s is not yet implemented'%typ)

__header0_line = { 0: { 'name': 'version',
                        'lenght': 3,
                        'type': 'int32',
                        'transform': ('version_major',
                                      'version_minor',
                                      'version_fix'),
                        'units': '',
                        'longName': 'Code version of the iHIBPsim code.',
                        'shortName': 'version',
                      },
                  1: { 'name': 'inj_angles',
                       'lenght': 2,
                       'type': 'float64',
                       'transform': ('tor_angle',
                                     'pol_angle'),
                       'units': 'rad',
                       'longName': 'Injection angles',
                       'shortName': ('$\\beta$', '$\\theta_{inj}$')
                    },
                  2: { 'name': 'beam_model',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': beam_model_to_str,
                       'units': '',
                       'longName': 'Beam model used',
                       'shortName': 'Beam model'
                    },
                  3: { 'name': 'mapping_method',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': map_method_to_str,
                       'units': '',
                       'longName': 'Scintillator mapping method',
                       'shortName': 'Map method'
                    },
                  4: { 'name': 'E',
                       'lenght': 1,
                       'type': 'float64',
                       'transform': j2keV_fun,
                       'units': 'keV',
                       'longName': 'Injection energy',
                       'shortName': 'E'
                    },
                  5: { 'name': 'fwhm_e',
                       'lenght': 1,
                       'type': 'float64',
                       'transform': j2keV_fun,
                       'units': '',
                       'longName': 'Width of the Gaussian for the energy',
                       'shortName': '$\\sigma_E$'
                    },
                  6: { 'name': 'div',
                       'lenght': 1,
                       'type': 'float64',
                       'transform': None,
                       'units': 'rad',
                       'longName': 'Divergency of the beam',
                       'shortName': '$\\alpha$'
                    },
                  7: { 'name': 'mass',
                       'lenght': 1,
                       'type': 'float64',
                       'transform': None,
                       'units': 'kg',
                       'longName': 'Beam mass',
                       'shortName': 'mass'
                    },
                  8: { 'name': 'intensity',
                       'lenght': 1,
                       'type': 'float64',
                       'transform': None,
                       'units': 'A',
                       'longName': 'Initial intensity',
                       'shortName': 'I'
                    },
                  9: { 'name': 'origin',
                       'lenght': 3,
                       'type': 'float64',
                       'transform': None,
                       'units': 'm',
                       'longName': 'Origin of the port',
                       'shortName': '$\\vec{r}_0$'
                    },
                  10: { 'name': 'strline_mode',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': strline_mode_to_str,
                       'units': '',
                       'longName': 'Strikeline-building mode',
                       'shortName': 'Strikeline-building mode'
                    },
                  11: { 'name': 'efieldMode',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': efield_mode_to_str,
                       'units': '',
                       'longName': 'Implementation of the Efield',
                       'shortName': 'E-field mode'
                    },
                  12: { 'name': 'Bcoils_flag',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': int_to_bool,
                       'units': '',
                       'longName': 'Bcoils active',
                       'shortName': 'Bcoils active'
                    },
                  13: { 'name': 'errorfield_set',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': int_to_bool,
                       'units': '',
                       'longName': 'VACFIELD Error field computed',
                       'shortName': 'VACFIELD Error field computed'
                    },
                  14: { 'name': 'mag_exp',
                       'lenght': 4,
                       'type': 'byte',
                       'transform': bytes_to_str,
                       'units': '',
                       'longName': 'Magnetics experiment',
                       'shortName': 'Magnetics experiment'
                    },
                  15: { 'name': 'mag_diag',
                       'lenght': 3,
                       'type': 'byte',
                       'transform': bytes_to_str,
                       'units': '',
                       'longName': 'Magnetics diagnostic',
                       'shortName': 'Magnetics diagnostic'
                    },
                  16: { 'name': 'mag_ed',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': None,
                       'units': '',
                       'longName': 'Magnetics edition',
                       'shortName': 'Magnetics edition'
                    },
                  17: { 'name': 'shot_mag',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': None,
                       'units': '',
                       'longName': 'Shotnumber for the magnetics',
                       'shortName': 'Shotnumber for the magnetics'
                    },
                  18: { 'name': 'mp_exp',
                       'lenght': 4,
                       'type': 'byte',
                       'transform': bytes_to_str,
                       'units': '',
                       'longName': 'External MP data experiment',
                       'shortName': 'External MP data experiment'
                    },
                  19: { 'name': 'mp_diag',
                       'lenght': 3,
                       'type': 'byte',
                       'transform': bytes_to_str,
                       'units': '',
                       'longName':  'External MP data diagnostic',
                       'shortName':  'External MP data diagnostic',
                    },
                  20: { 'name': 'mp_ed',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': None,
                       'units': '',
                       'longName':  'External MP data edition',
                       'shortName': 'External MP data edition',
                    },
                  21: { 'name': 'mp_shot',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': bytes_to_str,
                       'units': '',
                       'longName': 'Shotnumber for the MPs',
                       'shortName': 'Shotnumber for the MPs'
                    },
                  22: { 'name': 'prof_exp',
                       'lenght': 4,
                       'type': 'byte',
                       'transform': bytes_to_str,
                       'units': '',
                       'longName': 'Profiles experiment',
                       'shortName': 'Profiles experiment'
                    },
                  23: { 'name': 'prof_diag',
                       'lenght': 3,
                       'type': 'byte',
                       'transform': bytes_to_str,
                       'units': '',
                       'longName': 'Profiles diagnostic',
                       'shortName': 'Profiles diagnostic'
                    },
                  24: { 'name': 'prof_ed',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': None,
                       'units': '',
                       'longName': 'Profile edition',
                       'shortName': 'Profile edition'
                    },
                  25: { 'name': 'prof_shot',
                       'lenght': 1,
                       'type': 'int32',
                       'transform': None,
                       'units': '',
                       'longName': 'Shotnumber for the profile',
                       'shortName': 'Shotnumber for the profile'
                    },
                  'flip_y' : True,
                  'flip_x' : False
                 }

STRIKELINE_HEADER = { 'v0': __header0_line
                     }

def read_strikeline_file(fn: str, version: int = 0):
    """
    Read the strike line file according to a given version.

    Pablo Oyola - poyola@us.es
    """
    # Checking if the file exists
    if not os.path.isfile(fn):
        raise FileNotFoundError('Cannot find the file %s with strikes'%fn)

    # Retrieving the file structure.
    label = 'v%d'%version
    if label not in STRIKELINE_HEADER:
        raise errors.NotValidInput('The version %d is not available'%version)

    structure = STRIKELINE_HEADER[label]

    file_size = os.path.getsize(fn)

    with open(fn, 'rb') as fid:
        # Number of points in the code.
        npoints = np.fromfile(fid, 'int32', 1)[0]
        # The output will be an ordered xarray.Dataset with all the info.
        properties = xr.Dataset()
        properties['npoints'] = npoints
        properties['filename'] = fn

        ikeys = sorted([ikey for ikey in structure if isinstance(ikey, int)])

        for ikey in ikeys:
            iname  = structure[ikey]['name']
            lenght = structure[ikey]['lenght']
            typ    = structure[ikey]['type']
            func   = structure[ikey]['transform']

            tmp = np.fromfile(fid, dtype=typ, count=lenght)
            if func is not None:
                if callable(func):
                    tmp = func(tmp, **structure[ikey])
                    properties[iname] = xr.DataArray(tmp)
                else:
                    try:
                        _ = iter(func)
                    except TypeError:
                        print('Cannot parse option %s'%iname)
                    else:
                        for ii, ielem in enumerate(func):
                            properties[ielem] =  xr.DataArray(tmp[ii])

        # Checking whether we need to flip some directions.
        flip_x = 1.0
        flip_y = 1.0
        if 'flip_x' in structure:
            if structure['flip_x']:
                flip_x = -1.0
        if 'flip_y' in structure:
            if structure['flip_y']:
                flip_y = -1.0

        attrs_s = mapping_text_to_attrs(str(properties.strline_mode.values).lower())

        output = list()
        while True:
            pos = fid.tell()
            if(pos == file_size):
                break
            tmp = {}
            tmp['timestamp'] = np.fromfile(fid, 'float64', 1)
            tmp['nStrikes'] = np.fromfile(fid, 'int32', 1)
            tmp['map_s'] = np.fromfile(fid, 'float64', npoints)
            tmp['x1'] = flip_x*np.fromfile(fid, 'float64', npoints)
            tmp['x2'] = flip_y*np.fromfile(fid, 'float64', npoints)
            tmp['dx1'] = np.fromfile(fid, 'float64', npoints)
            tmp['dx2'] = np.fromfile(fid, 'float64', npoints)
            tmp['w'] = np.fromfile(fid, 'float64', npoints)

            # Deleting the NaN channels.
            flags = np.logical_not(np.isnan(tmp['x1']))

            tmp['map_s'] = tmp['map_s'][flags]
            tmp['x1']     = tmp['x1'][flags]
            tmp['x2']     = tmp['x2'][flags]
            tmp['dx1']    = tmp['dx1'][flags]
            tmp['dx2']    = tmp['dx2'][flags]
            tmp['w']      = tmp['w'][flags]

            # Creating a new dataset to store this map.
            data = xr.Dataset()

            data['map_s'] = xr.DataArray(tmp['map_s'], dims=('s',))
            data['map_s'].attrs.update(attrs_s)
            data['time'] = tmp['timestamp']
            data['time'].attrs = { 'shortName': 't',
                                   'longName': 'Time',
                                   'units': 's'
                                 }
            data['nstrikes'] = tmp['nStrikes']
            for ikey in ('x1', 'dx1', 'x2', 'dx2'):
                data[ikey]    = xr.DataArray(tmp[ikey], coords=(data['map_s'],),
                                             attrs=GENERIC_ATTRS_STRLINE[ikey])

            attrs_w = weight_mapping_text_to_attrs(str(properties.mapping_method.values))
            data['w'] = xr.DataArray(tmp['w'], coords=(data['map_s'],),
                                     attrs=attrs_w)

            output.append(data)
    return properties, output


class strikeLine:
    """
    Class handling a single strikeline.

    DO NOT USE DIRECTLY THIS CLASS. CREATE A STRIKELINECOLLECTION INSTEAD AND
    WORK FROM THAT.

    Pablo Oyola - poyola@us.es
    """
    def __init__(self, data, properties):
        """
        Initializes the class by providing the full data of the strike line and
        the properties as read by the parent class.

        Pablo Oyola - poyola@us.es

        :param data: map data as read from the file in an xarray.
        :param properties: general properties read from the file.
        """

        self.prop = properties

        for ikey in data.to_dict()['data_vars'].keys():
            self.__dict__[ikey] = data[ikey]


    def plot(self, weighted_color: bool=False, units: str='cm',
             cal=None, ax=None, cbar_sci: bool = False, **line_options):
        """
        Plot the result of the strike line (x1, x2).

        Pablo Oyola - poyola@us.es

        :param weighted_color: plot the line with colors given by the arriving
        flux.
        :param units: units for the plot. Either cm, m, mm or inch
        :param cal: calibration to distort the strikeline. If not provided,
        the un-distorted image is used instead.
        :param ax: axis to plot the data. If None, new axes are created.
        :param line_options: dictionary with arguments to pass down to the
        function.
        """

        ax_was_none = ax is None
        if ax_was_none:
            fig, ax = plt.subplots(1)

        factor = { 'm': 1.0,
                   'mm': 1000.0,
                   'cm': 100.0,
                   'inch': 100.0/2.54
                 }.get(units)

        if ('linestyle' in line_options) and \
           (line_options['linestyle'] == 'none') and  weighted_color:
               raise ValueError('Cannot set the color weight along with no lines!')



        if cal is None:
            x = self.x1 * factor
            y = self.x2 * factor
            xlabel = 'X [%s]' % units
            ylabel = 'Y [%s]' % units

        else:
            x, y = transform_to_pixel(self.x1, self.x2, cal)
            xlabel = 'X [pix]'
            ylabel = 'Y [pix]'

        xlim = diagparams.IHIBP_scintillator_X
        ylim = diagparams.IHIBP_scintillator_Y

        if weighted_color:
            if 'cmap' not in line_options:
                line_options['cmap'] = 'plasma'
            if cbar_sci:
                magn = math.floor(math.log10(np.max(self.w)))
                w = self.w/(10**magn)
                histlabel = 'Intensity ' + r'[$10^{%.2d}$ A]'%(magn)
            else:
                w = self.w
                histlabel = 'Intensity [A]'
            line = colorline(ax, x, y, w, **line_options)
        else:
            line = ax.plot(x, y, **line_options)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax.grid('both')

        if weighted_color:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="15%", pad=0.05)
            cbar = ax.figure.colorbar(mappable=line, cax=cax)
            cbar.set_label(histlabel)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)

        return ax, line

    def plot_weight(self, ax=None, **line_options):
        """
        Presents the weight along the strike line using as reference the
        initial parameter chosen in the simulation.

        Pablo Oyola - poyola@us.es

        :param ax: axis to use. If None, new one is created
        :param line_options: keyword arguments to send down to the plt.plot.
        """

        ax_was_none = ax is None
        if ax_was_none:
            fig, ax = plt.subplots(1)

        x, y = self.map_s, self.w

        line = ax.plot(x.values, y.values, **line_options)

        xlabel = '%s %s [%s]'%(self.map_s.longName, self.map_s.shortName,
                               self.map_s.units)
        ax.set_xlabel(xlabel)

        ylabel = '%s %s [%s]'%(self.w.longName, self.w.shortName,
                               self.w.units)
        ax.set_ylabel(ylabel)

        ax.grid('both')
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
        plt.tight_layout()

        return ax, line


    def project(self, data: float, xgrid: float, ygrid: float,
                kind: str='cubic', cal=None):
        """
        Projects a given data defined as z = z(x, y) using interpolators.

        Pablo Oyola - poyola@us.es

        :param data: 2D data to project.
        :param xgrid: grid along the horizontal direction.  First axis of the
        data.
        :param ygrid: grid along the vertical direction. Second axis of the
        data.
        :param kind: kind of the interpolator to project to the line.
        """

        if data.shape[0] != xgrid.size:
            raise ValueError('The first axis must be X size')
        if data.shape[1] != ygrid.size:
            raise ValueError('The second axis must be Y size')

        f = sinterp.RegularGridInterpolator((xgrid, ygrid), data,
                                            bounds_error=False,
                                            fill_value=np.nan)

        if cal is None:
            proj = f((self.x1.values*100.0, self.x2.values*100.0))
            proj = np.array(proj)

        else:
            x, y = transform_to_pixel(self.x1, self.x2, cal)
            proj = f((x, y))
            proj = np.array(proj)

        # The output will be an xarray.
        output = xr.DataArray(proj, dims=self.x1.dims,
                              coords=self.x1.coords,
                              attrs={'desc': 'Projection along strikeline'})

        return output

class strikeLineCollection:
    """
    This handles a collection of strike lines.

    Pablo Oyola - poyola@us.es
    """

    def __init__(self, filename: str, scint: Union[str, Scintillator]=None,
                 calib=None):
        """
        Initializes a strike line collection as they are written to a file.

        Pablo Oyola - poyola@us.es

        :param filename: filename where the strikemap is located.
        """

        # Checking whether the file exists.
        if not os.path.isfile(filename):
            raise FileNotFoundError('Cannot find the object %s'%filename)

        # Reading the full file.
        self.prop, maps = read_strikeline_file(filename)

        # Checking how many strikelines are available.
        self.ntime = len(maps)

        # We prepare now the internal timebasis.
        self.ntime = len(maps)
        self.time  = np.array([maps[ii]['time'].values[0] \
                               for ii in range(self.ntime)])

        self.maps = [strikeLine(data=maps[0], properties = self.prop)]

    def getTimeIndex(self, time: float):
        """
        Return the index closes to the time point indicated.

        Pablo Oyola - poyola@us.es

        :param time: time to get from the database of strikelines.
        """
        if time < self.time.min():
            raise ValueError('The time %.3f is below the time limit!'%time)
        elif time > self.time.max():
            raise ValueError('The time %.3f is above the time limit!'%time)

        try:
            _ = iter(time)
        except TypeError:
            idx = np.abs(self.time - time).argmin()
        else:
            idx = np.array([np.abs(self.time - itime).argmin() \
                            for itime in time], dtype=int)

        return np.atleast_1d(idx)


    # Useful shortcuts through Python builtin operators.
    def __getitem__(self, t: Union[float, int]):
        """
        Retrieves an item via the [...] operator provided either the
        timestamp or the access.

        Pablo Oyola - poyola@us.es

        :param idx: either a floating point indicating time point or the index.
        """

        if isinstance(t, int):
            idx = np.atleast_1d(t)
        else:
            idx = self.getTimeIndex(time = t)

        if len(idx) == 1:
            return self.maps[idx[0]]
        else:
            return np.array([self.maps[ii] for ii in idx])

    def plot(self, time: Union[float, int]=None, weighted_color: bool=False,
             units: str='cm', cal=None, ax=None, **line_options):
        """
        Plot a given strikeline provided the time point or the time index.
        Just acts as a wrap of the strikeLine plot function.

        Pablo Oyola - poyola@us.es

        :param time: time point to plot. If an integer is provided, then it is
        interpreted as the index in the timebase. Otherwise, if floating point,
        the nearest time point is taken.
        :param weighted_color: plot the line with colors given by the arriving
        flux.
        :param units: units for the plot. Either cm, m, mm or inch
        :param cal: calibration to distort the strikeline. If not provided,
        the un-distorted image is used instead.
        :param ax: axis to plot the data. If None, new axes are created.
        :param line_options: dictionary with arguments to pass down to the
        function.
        """

        if time is None:
            time = np.arange(self.ntime).astype(dtype=int)

        if isinstance(time, int):
            idx = np.atleast_1d(time)
        else:
            idx = self.getTimeIndex(time)

        # Using the internal plotting.
        if len(idx) == 1:
            idx = idx[0]
            if 'label' not in line_options:
                label = 't = %.3f s'%self.time[idx]
                ax, lc = self[idx].plot(weighted_color=weighted_color,
                                        units=units, cal=cal, ax=ax,
                                        label=label, **line_options)
            else:
                ax, lc = self[idx].plot(weighted_color=weighted_color,
                                        units=units, cal=cal, ax=ax,
                                        **line_options)
        else:
            lc = list()
            for ii in idx:
                if 'label' not in line_options:
                    label = 't = %.3f s'%self.time[ii]
                    ax, tmp = self[ii].plot(weighted_color=weighted_color,
                                             units=units, cal=cal, ax=ax,
                                             label=label, **line_options)
                else:
                    ax, tmp = self[ii].plot(weighted_color=weighted_color,
                                             units=units, cal=cal, ax=ax,
                                             **line_options)
                lc.append(tmp)

        return ax, lc

    def plot_weight(self, time: Union[int, float]=None, ax=None, **line_options):
        """
        Plot a given weight profile for input time(s) or time index(indices).

        Pablo Oyola - poyola@us.es

        :param time: time point to plot. If an integer is provided, then it is
        interpreted as the index in the timebase. Otherwise, if floating point,
        the nearest time point is taken.
        :param ax: axis to plot the data. If None, new axes are created.
        :param line_options: dictionary with arguments to pass down to the
        function.
        """

        if time is None:
            time = np.arange(self.ntime).astype(dtype=int)

        if isinstance(time, int):
            idx = np.atleast_1d(time)
        else:
            idx = self.getTimeIndex(time)

        # Using the internal plotting.
        if len(idx) == 1:
            idx = idx[0]
            if 'label' not in line_options:
                label = 't = %.3f s'%self.time[idx]
                ax, lc = self[idx].plot_weight(ax=ax, label=label,
                                               **line_options)
            else:
                ax, lc = self[idx].plot_weight(ax=ax,  **line_options)
        else:
            lc = list()
            for ii in idx:
                if 'label' not in line_options:
                    label = 't = %.3f s'%self.time[ii]
                    ax, tmp = self[ii].plot_weight(ax=ax, label=label,
                                                   **line_options)
                else:
                    ax, tmp = self[ii].plot_weight(ax=ax, **line_options)
                lc.append(tmp)

        return ax, lc



    # Properties.
    @property
    def size(self):
        """
        Return the number of time points stored in the collection.
        """

        return self.ntime

    @property
    def shape(self):
        """
        Gets the shape of the stored maps:
            0 -> Number of maps
            1 -> Strikepoints per map.
        """

        return np.array([self.size, self.prop['npoints']], dtype=int)
