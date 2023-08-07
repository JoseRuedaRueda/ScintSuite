"""
Basic variable class

Jose Rueda Rueda: jrrueda@us.es
"""
import numpy as np
import scipy.signal as sp  # signal processing
import ScintSuite.errors as errors
import matplotlib.pyplot as plt
import xarray as xr
from scipy.fft import rfft, rfftfreq
from scipy import signal
import ScintSuite._Plotting as ssplt
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

        :param  name: Name atribute
        :param  units: Physical units of the variable
        :param  data: array with the data values
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
    

class BasicSignalVariable():
    """
    Basic signal variable

    This is just a basic class to serve as default for all time dependent 
    signals or signals groups

    It is not intended to be initialised by itself but to serve as parent class

    Please save always the time variable as the last index, even if axxary
    handle easily this, the filter/fft/spectrogram routines are not yet ready
    for this

    @add time units
    """
    def __init__(self):
        self._data = xr.Dataset()
    
    # --------------------------------------------------------------------------
    # --- Filtering
    # --------------------------------------------------------------------------
    def filter(self, signals: list = None, method: str = 'savgol',
               filter_params={}):
        """
        Filter the time dependent variables

        :param  signals: list of signal to be filtered. If none, all signals will
            be filtered

        @ ToDo: implement some fancy ... index to handle dimensions
        """
        # --- Initialisation and settings checking
        filters = {
            'savgol': sp.savgol_filter,
            'median': sp.medfilt,
        }
        settings = {
            'savgol': {
                'window_length': 201,
                'polyorder': 3
            },
            'median': {
                'kernel_size': 201,
            }
        }
        if method.lower() not in filters.keys():
            raise errors.NotImplementedError('Method not implemented')
        filter_options = settings[method.lower()].copy()
        filter_options.update(filter_params)
        if signals is None:
            signals = []
            for k in self.keys():
                if not k.startswith('fft') and not k.startswith('spec'):
                    # Neglect fft or spectrum:
                    signals.append(k)
        
        for k in signals:
            if len(self[k].shape) == 1:  # Variable with just time axis
                self._data['filtered_' + k] = xr.DataArray(
                        filters[method](self[k].values, **filter_options),
                        dims='t')
            elif len(self[k].shape) == 2:  # Variable with time + something
                self._data['filtered_' + k] = self._data[k].copy()
                for i in range(self[k].shape[0]):
                    self._data['filtered_' + k].values[i, :] = \
                        filters[method](self[k].values[i, :], **filter_options)
                        # xr.DataArray(
                        # filters[method](self[k].values[i, :], **kargs),
                        # dims='t')
            else:
                raise errors.NotImplementedError('To be done')


    
    # --------------------------------------------------------------------------
    # --- Frequency anasylis
    # --------------------------------------------------------------------------
    def calculate_fft(self, signals: list = None, **kargs):
        """
        Calculate the fft of the time trace

        Jose Rueda Rueda: jrrueda@us.es

        Only the fft of all the signals in the dataset
        :param  signals: list of signal for whcih we want the fft. If none, 
            all signals will be considered
        :param     kargs: optional arguments for scipyfft
        see scipy.fft.rfft for full details


        :return:  nothing, just fill self.fft
        """
        # --- Object cleaning:
        if 'freq_fft' in self.keys():
            logger.warning('11: Overwritting fft data')
            self._data.drop_dims('freq_fft')
        # --- Prepare the signal names for the fft
        if signals is None:
            signals = []
            for k in self.keys():
                if not k.startswith('fft') and not k.startswith('spec'):
                    # Neglect fft or spectrum:
                    signals.append(k)
        # --- Prepare the fft axis
        N = len(self['t'].values)
        freq_fft = rfftfreq(N, (self['t'][2] - self['t'][1]).values)

        self._data = self._data.assign_coords({'freq_fft': freq_fft})
        self._data['freq_fft'].attrs['long_name'] = 'Frequency'

        # --- Proceed with the fft
        for k in signals:
            if len(self[k].shape) == 1:  # Variable with just time axis
                self._data['fft_' + k] = xr.DataArray(
                    rfft(self[k].values, **kargs), dims='freq_fft')
            elif len(self[k].shape) == 2:  # Variable with time + something
                dummy = np.zeros((self[k].shape[0], freq_fft.size))
                for i in range(self[k].shape[0]):
                    dummy[i, :] = rfft(self[k].values[i, :], **kargs)
                self._data['fft_' + k] = xr.DataArray(
                    dummy, dims=(self[k].dims[0], 'freq_fft'))
            else:
                raise errors.NotImplementedError('To be done')

            self._data['fft_' + k].attrs['long_name'] = 'Fast Fourier Trans'
    
    def calculate_spectrogram(self, signals: list = None, **kargs):
        """
        Calculate the spectrogram of the time trace

        Jose Rueda Rueda: jrrueda@us.es

        Only the spec of the sum of the counts in the roi is calculated, if you
        want others to be calculated, open a request in the GitLab

        :param     params: Dictionary containing optional arguments for the
        spectrogram, see scipy.signal.spectrogram for the full details
        @type:    dict

        :return:  nothing, just fill self.spec
        """
        # --- Object cleaning
        if 'time_spec' in self.keys():
            logger.warning('11: Overwritting spectrogram data')
            self._data.drop_dims(('time_spec', 'freq_spec'))
        # --- Settings initialization
        if signals is None:
            signals = []
            for k in self.keys():
                if not k.startswith('fft') and not k.startswith('spec'):
                    # Neglect fft or spectrum:
                    signals.append(k)
        # --- Spectogram calculation
        sampling_freq = 1 / (self['t'][1] - self['t'][0]).values
        for k in signals:
            if len(self[k].shape) == 1:
                freq_spec, time_spec, Sxx = \
                    sp.spectrogram(self[k].values, sampling_freq, **kargs)
                self._data['spec_' + k] = xr.DataArray(
                    Sxx, dims=('freq_spec', 'time_spec'), 
                    coords={'freq_spec': freq_spec, 
                            'time_spec': time_spec + self['t'][0].values})
            elif len(self[k].shape) == 2:
                spectra = []
                for i in range(self[k].shape[0]):
                    spectra.append(sp.spectrogram(self[k].values[i,:], 
                                                  sampling_freq, **kargs))
                dummy = np.zeros((self[k].shape[0], spectra[0][2].shape[0], 
                                  spectra[0][2].shape[1]))
                for i in range(self[k].shape[0]):
                    dummy[i, ...] = spectra[i][2]
                self._data['spec_' + k] = xr.DataArray(
                    dummy, dims=(self[k].dims[0], 'freq_spec', 'time_spec'),
                    coords={'freq_spec': spectra[0][0], 
                            'time_spec': spectra[0][1] + self['t'][0].values})
            else:
                raise errors.NotImplementedError('To be done')   

        self._data['time_spec'].attrs['long_name'] = 'Time'
        self._data['freq_spec'].attrs['long_name'] = 'Frequency'
    
    # --------------------------------------------------------------------------
    # --- Plotting
    # --------------------------------------------------------------------------
    def plot_fft(self, data='sum_of_roi', options: dict = {}):
        """
        Plot the fft of the TimeTrace

        Jose Rueda: jrrueda@us.es

        :param  options: options for the axis_beauty method
        :return fig: figure where the fft is plotted
        :return ax: axes where the fft is plotted
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

        :param  options: options for the axis_beauty method
        :return fig: figure where the fft is plotted
        :return ax: axes where the fft is plotted
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

    # --------------------------------------------------------------------------
    # --- Properties and custom access layers 
    # --------------------------------------------------------------------------
    def keys(self):
        return self._data.keys()
    
    def __getitem__(self, item):
        return self._data[item]