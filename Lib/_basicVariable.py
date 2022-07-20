"""
Basic variable class

Jose Rueda Rueda: jrrueda@us.es
"""
import numpy as np
import Lib.errors as errors
import matplotlib.pyplot as plt
import xarray as xr
from scipy.fft import rfft, rfftfreq
from scipy import signal
import Lib._Plotting as ssplt
import logging
logger = logging.getLogger('ScintSutie.BasicVariables')

class BasicVariable():
    """
    Simple class to contain the data of a given variable, with metadata

    Jose Rueda Rueda: jrrueda@us.es
    """

    def __init__(self, name: str = None, units: str = None,
                 data: np.ndarray = None):
        """
        Just store everything on place

        @param name: Name atribute
        @param units: Physical units of the variable
        @param data: array with the data values
        """
        self.name = name
        self.units = units
        self.data = data

        # try to get the shape of the data
        try:
            self.shape = data.shape
            self.size = data.size
        except AttributeError:
            self.shape = None
            self.size = None

        # Deduce the label for plotting
        if (name is not None) and (units is not None):
            self.plot_label = '%s [%s]' % (name.lower().capitalize(), units)
        else:
            self.plot_label = None

    def __getitem__(self, item):
        return self.data[item]
    
    def mean(self, **kwargs):
        return self.data.mean(**kwargs)

    def std(self, **kwargs):
        return self.std.mean(**kwargs)
    
    def sum(self, **kwargs):
        return self.data.sum(**kwargs)
    

# class BasicTimeVariable(BasicVariable):
#     """
#     Simple class to contain the data of a signal (or signal group)

#     Jose Rueda: jrrueda@us.es
#     """
#     def __init__(self, name: str = None, units: str = None,
#                  data: np.ndarray = None, timebase: np.ndarray = None,
#                  unitsTimebase: str = None):
#         """
#         Initialise the class

#         @param name: Name atribute
#         @param units: Physical units of the variable
#         @param data: array with the data values
#         @param timebase: should be a 1D array with nt points, where nt is just
#             the number of points
#         @param unitsTimebase: units of the timebase variable
#         """
#         # First check if the order is the proper one (just for the case of the
#         # non-empty data)
#         if data is not None:
#             datashape = data.shape
#             timebaseShape = timebase.size
#             if timebaseShape != datashape[-1]:
#                 text = 'Time dimenssion should be the last one of data'
#                 raise errors.NotValidInput(text)

#         BasicVariable.__init__(self, name=name, units=units, data=data)
#         self.timebase = BasicVariable(name = 'Time', data=timebase, 
#                                       units=unitsTimebase)
    
#     def plot(self):
#         """
#         Basic plot.

#         This is far to be a complete and usable plot routine, and it actually
#         non intended for it. This is just something quick to see the variable
#         not more
#         """
#         fig, ax = plt.subplots()
#         ax.plot(self.timebase[:], self[:].T)
#         return ax


class BasicSignalVariable():
    """
    Basic signal variable

    This is just a basic class to serve as default for all time dependent 
    signals or signals groups

    It is not intended to be initialised by itself but to serve as parent class

    @add time units
    """
    def __init__(self):
        self._data = xr.Dataset()
    

    def calculate_fft(self, params: dict = {}):
        """
        Calculate the fft of the time trace

        Jose Rueda Rueda: jrrueda@us.es

        Only the fft of all the signals in the dataset

        @param    params: Dictionary containing optional arguments for scipyfft
        see scipy.fft.rfft for full details
        @type:    dict

        @return:  nothing, just fill self.fft
        """
        if 'freq_fft' in self.keys():
            logger.warning('11: Overwritting fft data')
            self._data.drop_dims('freq_fft')
        N = len(self['t'].values)
        freq_fft = rfftfreq(N, (self['t'][2] - self['t'][1]).values)

        self._data = self._data.assign_coords({'freq_fft': freq_fft})
        self._data['freq_fft'].attrs['long_name'] = 'Frequency'
        for k in self.keys():
            if not k.startswith('fft') and not k.startswith('spec'):
                # The if is to do not catch other fft or spectra qhich can be
                # present and just catch the signals
                self._data['fft_' + k] = xr.DataArray(
                    rfft(self[k].values, **params), dims='freq_fft')
                self._data['fft_' + k].attrs['long_name'] = 'Fast Fourier Trans'
    
    def calculate_spectrogram(self, **kargs):
        """
        Calculate the spectrogram of the time trace

        Jose Rueda Rueda: jrrueda@us.es

        Only the spec of the sum of the counts in the roi is calculated, if you
        want others to be calculated, open a request in the GitLab

        @param    params: Dictionary containing optional arguments for the
        spectrogram, see scipy.signal.spectrogram for the full details
        @type:    dict

        @return:  nothing, just fill self.spec
        """
        if 'time_spec' in self.keys():
            logger.warning('11: Overwritting spectrogram data')
            self._data.drop_dims(('time_spec', 'freq_spec'))
        sampling_freq = 1 / (self['t'][1] - self['t'][0]).values
        for k in self.keys():
            if not k.startswith('fft') and not k.startswith('spec'):
                # The if is to do not catch other fft or spectra qhich can be
                # present and just catch the signals
                freq_spec, time_spec, Sxx = \
                    signal.spectrogram(self[k].values, sampling_freq,**kargs)
                self._data['spec_' + k] = xr.DataArray(
                    Sxx, dims=('freq_spec', 'time_spec'), 
                    coords={'freq_spec': freq_spec, 
                            'time_spec': time_spec + self['t'][0].values})

        self._data['time_spec'].attrs['long_name'] = 'Time'
        self._data['freq_spec'].attrs['long_name'] = 'Frequency'
    
    def plot_fft(self, data='sum_of_roi', options: dict = {}):
        """
        Plot the fft of the TimeTrace

        Jose Rueda: jrrueda@us.es

        @param options: options for the axis_beauty method
        @return fig: figure where the fft is plotted
        @return ax: axes where the fft is plotted
        """
        if 'grid' not in options:
            options['grid'] = 'both'
        if 'xlabel' not in options:
            options['xlabel'] = 'Frequency [Hz]'
        if 'ylabel' not in options:
            options['ylabel'] = 'Amplitude'
        line_options = {'linewidth': 2, 'color': 'r'}

        fig, ax = plt.subplots()
        ax.plot(self['freq_fft'], abs(self['fft_' + data]), **line_options)
        ax = ssplt.axis_beauty(ax, options)
        plt.show()
        return ax

    def plot_spectrogram(self, data='sum_of_roi', options: dict = {}):
        """
        Plot the spectrogram

        Jose Rueda: jrrueda@us.es

        @param options: options for the axis_beauty method
        @return fig: figure where the fft is plotted
        @return ax: axes where the fft is plotted
        """
        if 'grid' not in options:
            options['grid'] = 'both'
        if 'ylabel' not in options:
            options['ylabel'] = 'Frequency [Hz]'
        if 'xlabel' not in options:
            options['xlabel'] = 'Time [s]'

        fig, ax = plt.subplots()
        cmap = ssplt.Gamma_II()
        ax.pcolormesh(self['time_spec'], self['freq_spec'],
                      np.log(self['spec_' + data]), shading='gouraud', 
                      cmap=cmap)
        ax = ssplt.axis_beauty(ax, options)
        plt.show()
        return ax


    def keys(self):
        return self._data.keys()
    
    def __getitem__(self, item):
        return self._data[item]