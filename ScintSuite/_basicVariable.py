"""
Basic variable class

Jose Rueda Rueda: jrrueda@us.es
"""
import time
import numpy as np
import scipy.signal as sp  # signal processing
import ScintSuite.errors as errors
import matplotlib.pyplot as plt
import xarray as xr
from scipy.fft import rfft, rfftfreq
from scipy import signal
import ScintSuite._Plotting as ssplt
from typing import Optional, Union
import logging
logger = logging.getLogger('ScintSutie.BasicVariables')

class BasicVariable():
    """
    Simple class to contain the data of a given variable, with metadata

    Jose Rueda Rueda: jrrueda@us.es
    """

    def __init__(self, name: Optional[str] = None, units: Optional[str] = None,
                 data: Optional[np.ndarray] = None):
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

    Please save always the time variable as the last index, even if xarray
    handle easily this, 

    @ToDo add time units
    """
    def __init__(self):
        self._data = xr.Dataset()
    
    # --------------------------------------------------------------------------
    # %% Baseline correction
    # --------------------------------------------------------------------------
    def baseline_correction(self, signals: Optional[list] = None,
                            t1: Optional[float] = 0.1, t2: Optional[float] = 0.2):
        """
        Correct the baseline of the signals

        Jose Rueda Rueda: jrruedaru@uci.edu
        
        :param signals: list of signals to be corrected. If none, all signals
            will be corrected
        :param t1: initial time for the baseline calculation
        :param t2: final time for the baseline calculation
        """
        if signals is None:
            signals = []
            for k in self.keys():
                if not k.startswith('fft') and not k.startswith('spec') and not k.startswith('baseline'):
                    # Neglect fft or spectrum:
                    signals.append(k)
        for k in signals:
            baseline = self[k].sel(t=slice(t1, t2)).mean(dim='t')
            self._data['baseline_' + k] = baseline.copy()
            self._data[k] -= baseline
        return
            
    
    
    # --------------------------------------------------------------------------
    # %% Filtering
    # --------------------------------------------------------------------------
    def filter(self, signals: Optional[list] = None, 
               method: Optional[str] = 'savgol',
               filter_params: Optional[dict] = {}):
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
                if not k.startswith('fft') and not k.startswith('spec') and not k.startswith('baseline'):
                    # Neglect fft or spectrum:
                    signals.append(k)
        
        for k in signals:
            if len(self[k].shape) == 1:  # Variable with just time axis
                self._data['filtered_' + k] = xr.DataArray(
                        filters[method](self[k].values, **filter_options),
                        dims='t')
            elif len(self[k].shape) == 2:  # Variable with time + something
                if self[k].dims[1] == 't':
                    channeldim = 0
                elif self[k].dims[0] == 't':
                    channeldim = 1
                else:
                    logger.error('Time dimension not found')
                    raise errors.CalculationError('Time dimension not found')
                self._data['filtered_' + k] = self._data[k].copy()
                for i in range(self[k].shape[channeldim]):
                    if channeldim == 0:
                        self._data['filtered_' + k].values[i, :] = \
                            filters[method](self[k].values[i, :], **filter_options)
                    else:
                        self._data['filtered_' + k].values[:, i] = \
                            filters[method](self[k].values[:, i], **filter_options)
                        # xr.DataArray(
                        # filters[method](self[k].values[i, :], **kargs),
                        # dims='t')
            else:
                raise errors.NotImplementedError('To be done')

    # --------------------------------------------------------------------------
    # %% Frequency anasylis
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
                if not k.startswith('fft') and not k.startswith('spec') and not k.startswith('baseline'):
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
    
    def calculate_spectrogram(self, signals: Optional[list] = None,
                              window: Union[str, np.ndarray] = 'gaussian',
                              winParams: Optional[dict] = {'Nx': 1000},
                              **kargs):
        """
        Calculate the spectrogram of the data

        Jose Rueda Rueda: jruedaru@uci.edu
        
        :param signals: list of signal for whcih we want the fft. If none,
        all signals present in self._data will be considered
        :param window: window for the spectrogram, if a numpy array is given,
        it will be used as the window. If a string is given, the window will be
        selected from the scipy.signal.get_window function
        :param winParams: dictionary with the parameters for the window (if needed)
        :param kargs: parameters for the scipy.signal.ShortTimeFFT function

        :return:  nothing, just fill self._data.spec_*
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
        if 'Nx' not in winParams:
            logger.warning('Nx not defined, using 1000')
            winParams['Nx'] = 1000
        # --- Window selection
        if type(window) == np.ndarray:
            win = window
        else:
            if window.lower() == 'gaussian':
                # Gaussian window cannot be returned with the generic get window
                if 'std' not in winParams:
                    logger.warning('std not defined, using 7')
                    winParams['std'] = 7
                win = signal.windows.gaussian(winParams['Nx'], std=winParams['std'])
            else:
                win = signal.windows.get_window(window, **winParams)
        # --- Spectogram calculation
        sampling_freq = 1 / np.mean(np.diff(self['t']))
        for k in signals:
            if len(self[k].shape) == 1:
                # If the signal is just a 1D time signal, we can perform the 
                # Sepctrogram directly
                SFT = signal.ShortTimeFFT(win, fs=sampling_freq, **kargs)
                Sx = SFT.stft(self[k].values)
                freq_spec = SFT.f
                t_0, t_1 = SFT.extent(self._data.t.size)[:2]  # time range of plot
                time_spec = np.arange(t_0, t_1, SFT.delta_t)
                # Safety check
                dt1 = np.mean(np.diff(time_spec))
                dt2 = SFT.delta_t
                if np.abs(dt1 - dt2) > 1e-6:
                    logger.error('Time step mismatch: %f vs %f' % (dt1, dt2))
                    raise errors.CalculationError('Time step mismatch')
                self._data['spec_' + k] = xr.DataArray(
                    Sx, dims=('freq_spec', 'time_spec'), 
                    coords={'freq_spec': freq_spec, 
                            'time_spec': time_spec})
                # Add the factors to the attributes
                self._data['spec_' + k].attrs['fac_magnitude'] = SFT.fac_magnitude
                self._data['spec_' + k].attrs['fac_psd'] = SFT.fac_psd
                self._data['spec_' + k].attrs['fft_mode'] = SFT.fft_mode
                self._data['spec_' + k].attrs['invertible'] = SFT.invertible
                self._data['spec_' + k].attrs['scaling'] = SFT.scaling
                self._data['spec_' + k].attrs['window'] = SFT.win
            elif len(self[k].shape) == 2:
                # In this case, we need to perform the calculationn for each
                # of the rows inside the signal group, for example, for 
                # different channels
                # First find in which dimension is the time
                if self[k].dims[1] == 't':
                    channeldim = 0
                elif self[k].dims[0] == 't':
                    channeldim = 1
                else:
                    logger.error('Time dimension not found')
                    raise errors.CalculationError('Time dimension not found')
                
                spectra = []
                metadata = {'fac_magnitude': [], 'fac_psd': [], 'fft_mode': [],
                            'invertible': [], 'scaling': [], 'window': []}
                for i in range(self[k].shape[channeldim]):
                    # Get the spectrogram for each of the channels\
                    SFT = signal.ShortTimeFFT(win, fs=sampling_freq, **kargs)
                    if channeldim == 0:
                        Sx = SFT.stft(self[k].values[i, :])
                    else:
                        Sx = SFT.stft(self[k].values[:, i])
                    # Save the metadata
                    metadata['fac_magnitude'].append(SFT.fac_magnitude)
                    metadata['fac_psd'].append(SFT.fac_psd)
                    metadata['fft_mode'].append(SFT.fft_mode)
                    metadata['invertible'].append(SFT.invertible)
                    metadata['scaling'].append(SFT.scaling)
                    metadata['window'].append(SFT.win)
                    # Accumulate the spectra in the list
                    spectra.append(Sx)
                # Save everything to the dataarray, as they have the same time
                # base and window, they share the frequency axis and tiem axis
                freq_spec = SFT.f
                t_0, t_1 = SFT.extent(self._data.t.size)[:2]  # time range of plot
                time_spec = np.arange(t_0, t_1, SFT.delta_t)
                # Safety check
                dt1 = np.mean(np.diff(time_spec))
                dt2 = SFT.delta_t
                if np.abs(dt1 - dt2) > 1e-6:
                    logger.error('Time step mismatch: %f vs %f' % (dt1, dt2))
                    raise errors.CalculationError('Time step mismatch')
                if time_spec.size != Sx.shape[1]:
                    logger.error('Time size mismatch: %i vs %i, discarting the last point and trying again' % (time_spec.size, Sx.shape[1]))
                    time_spec = time_spec[:-1]
                    if time_spec.size != Sx.shape[1]:
                        logger.error('Time size mismatch: %i vs %i, aborting' % (time_spec.size, Sx.shape[1]))
                        raise errors.CalculationError('Time size mismatch')
                nchannels = self[k].shape[channeldim]
                dummy = np.zeros((nchannels, Sx.shape[0], Sx.shape[1]),
                                 dtype=complex)
                for i in range(self[k].shape[0]):
                    dummy[i, ...] = spectra[i]
                self._data['spec_' + k] = xr.DataArray(
                    dummy, dims=(self[k].dims[channeldim], 'freq_spec', 'time_spec'),
                    coords={'freq_spec': freq_spec, 
                            'time_spec': time_spec})
                self._data['spec_' + k].attrs = metadata
            else:
                raise errors.NotImplementedError('To be done')   

        self._data['time_spec'].attrs['long_name'] = 'Time'
        self._data['freq_spec'].attrs['long_name'] = 'Frequency'
    
    # --------------------------------------------------------------------------
    # %% Plotting
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

    def plot_signal_ss(self, signal='sum_of_roi', ax=None,
                       color: Optional[str] = 'k', 
                       alpha: Optional[float] = 0.05,
                       percent: Optional[float] = 0.05,
                       nchannel: Optional[int] = None,
                       rawLine: Optional[bool] = False):
        """
        Plot a signal shaded and smooth

        Jose Rueda: jruedaru@uci.edu
        
        This function will plot the real data as a shaded collection of points
        and the smoothed data as a line
        
        If you want to plot just the data or the smooth, call the plot routines
        from the xarray itself, will be faster
        
        :param signal: name of the signal to be plotted
        :param ax: axes where the data will be plotted. If None, a new figure
            will be created
        :param nchannel: number of channel to be plotted, in case the signal have different channels
        """
        # Dimmension check
        if len(self[signal].shape) == 2 and nchannel is None:
            logger.error('Signal has more than one channel, please specify the channel')
            raise errors.NotValidInput('Channel not specified')
        # First, check if the fildered (smoothed) data is present, if not, smoothe it
        if 'filtered_' + signal not in self.keys():
            logger.warning('Filtered data not found, filtering with default parameters')
            self.filter([signal,])
        # Now check the figure
        if ax is None:
            fig, ax = plt.subplots()
        # Plot the shaded data
        # Get the flag of points to plot (percent)
        nt = self['t'].size
        flags = np.random.rand(nt) < percent
        # Plot the shaded data
        if rawLine:
            ls = '-'
            marker = 'None'
        else:
            ls = 'None'
            marker = 'o'
        if len(self[signal].shape) == 1:
            self[signal][flags].plot(ax=ax, color=color, alpha=alpha, 
                                     marker=marker, linestyle=ls, markersize=2)
        else:
            self[signal].sel(channel=nchannel)[flags].plot(ax=ax, color=color, 
                                                    marker=marker, markersize=2,
                                                    alpha=alpha, linestyle=ls)
        # Plot the smoothed data
        if len(self[signal].shape) == 1:
            self['filtered_' + signal].plot(ax=ax, color=color, linewidth=0.5)
        else:
            self['filtered_' + signal].sel(channel=nchannel).plot(ax=ax,
                                                                  color=color,
                                                                  linewidth=0.5)
        return ax
        
    # --------------------------------------------------------------------------
    # %% Saving
    # --------------------------------------------------------------------------
    def toFile(self, filename: str, signals: Optional[list] = None, ):
        """
        Save the data to a file

        Jose Rueda Rueda: jruedaru@uci.edu
        
        Nete: The format will be H5
        
        :param filename: name of the file to save the data
        :param signals: list of signals to be saved. If none, all signals will
            be saved. Please, this is not recomended for fast channel data,
            as you will save the several GB of data coming from the pmts, which
            is already in the database
        """      
        if signals is None:
            # Save everything
            self._data.to_netcdf(filename, engine='h5netcdf',invalid_netcdf=True)
        else:
            # Save only the selected signals
            self._data[signals].to_netcdf(filename, engine='h5netcdf',invalid_netcdf=True)
        return
        
        
        


    # --------------------------------------------------------------------------
    # %% Properties and custom access layers 
    # --------------------------------------------------------------------------
    def keys(self):
        return self._data.keys()
    
    def __getitem__(self, item):
        return self._data[item]