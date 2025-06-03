"""
Basic variable class

Jose Rueda Rueda: jrrueda@us.es
"""
import time
import numpy as np
import scipy.signal as sp  # signal processing
import ScintSuite.errors as errors
import ScintSuite._Plotting as ssplt
import matplotlib.pyplot as plt
import xarray as xr
from scipy.fft import rfft, rfftfreq
from ScintSuite._SideFunctions import detrend
from scipy import signal
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
        return self.data.std.mean(**kwargs)
    
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
        self._data = xr.DataTree()
    
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
        # --- Initialise the signals to be corrected:
        if signals is None:
            signals = self.getSignalNames()
        # --- Get and correct the baseline:
        if 'baselines' not in self._data['signals'].keys():
            basDs = xr.Dataset()
            self._data['signals']['baselines'] = xr.DataTree(name='baselines',
                                    dataset=basDs)
            
        for k in signals:
            baseline = self._data['signals'][k].sel(t=slice(t1, t2)).mean(dim='t')
            self._data['signals']['baselines'][k] = baseline.copy()
            self._data['signals'][k] -= baseline
        return
            
    
    # --------------------------------------------------------------------------
    # %% Filtering
    # --------------------------------------------------------------------------
    def filter(self, signals: Optional[list] = None, 
               method: Optional[str] = 'savgol',
               filter_params: Optional[dict] = {}):
        """
        Filter the time dependent variables

        :param  signals: list of signal to be filtered. If none, all signals
            will be filtered

        @ ToDo: implement some fancy ... index to handle dimensions
        """
        # --- Initialisation and settings checking
        filters = {
            'savgol': sp.savgol_filter,
            'median': sp.medfilt,
        }
        settings = {
            'savgol': {
                'window_length': 2001,
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
            signals = self.getSignalNames()
        # --- Create the children in the signal tree
        if 'filtered' not in self._data['signals'].keys():
            filteredDs = xr.Dataset()
            self._data['signals']['filtered'] = xr.DataTree(name='filtered',
                                                 dataset=filteredDs)
        # --- Proceed with the filtering
        logger.info('Filtering signals with method %s' % method)
        for k in signals:
            axis = self._data['signals'][k].dims.index('t')
            self._data['signals']['filtered'][k] = xr.DataArray(
                filters[method](self[k].values, axis=axis, **filter_options),
                dims=self._data['signals'][k].dims,
                coords=self._data['signals'][k].coords
            )
            # Add the attributes to the filtered signals
            self._data['signals']['filtered'][k].attrs['filter_method'] = method
            for key, value in filter_options.items():
                self._data['signals']['filtered'][k].attrs[key] = value
            try:
                self._data['signals']['filtered'][k].attrs['units'] = self._data['signals'][k].attrs['units']
            except KeyError:
                pass
        logger.info('Filtering done')

    def detrend(self, signals: Optional[list] = None, type: str = 'linear',
                detrendSizeInterval: Optional[float] = 0.001):
        """
        Filter the time dependent variables

        :param  signals: list of signal to be filtered. If none, all signals 
            will be filtered
        :param  detrendSizeInterval: size of the interval to be used for the detrend
        """
        if signals is None:
            signals = self.getSignalNames()
        # Allocate the detrended tree
        if 'detrended' not in self._data['signals'].keys():
            detrendDs = xr.Dataset()
            self._data['signals']['detrended'] = xr.DataTree(name='detrend',
                                                 dataset=detrendDs)
        # --- Proceed with the detrending
        logger.info('Detrending signals with size interval %f s' % detrendSizeInterval)
        for k in signals:
            self._data['signals']['detrended'][k] = detrend(self._data['signals'][k],type=type,detrendSizeInterval=detrendSizeInterval)
            # Add the attributes to the detrended signals
            self._data['signals']['detrended'][k].attrs['detrend_method'] = type
            self._data['signals']['detrended'][k].attrs['detrendSizeInterval'] = detrendSizeInterval
            try:
                self._data['signals']['detrended'][k].attrs['units'] = self._data['signals'][k].attrs['units']
            except KeyError:
                pass


    # --------------------------------------------------------------------------
    # %% Frequency anasylis
    # --------------------------------------------------------------------------
    def calculate_fft(self, signals: Optional[list] = None, **kargs):
        """
        Calculate the fft of the time trace

        Jose Rueda Rueda: jrrueda@us.es

        Only the fft of all the signals in the dataset
        :param  signals: list of signal for whcih we want the fft. If none, 
            all signals will be considered
        :param     kargs: optional arguments for scipyfft
        see scipy.fft.rfft for full details


        :return:  nothing, just fill the tree node
        """
        # --- Object allocation:
        if 'fft' not in self._data['signals'].keys():
            detrendDs = xr.Dataset()
            self._data['fft'] = xr.DataTree(name='fft',
                                                 dataset=detrendDs)
        # --- Prepare the signal names for the fft
        if signals is None:
            signals = self.getSignalNames()
        # --- Prepare the fft axis
        N = len(self['t'].values)
        freq_fft = rfftfreq(N, (self['t'][2] - self['t'][1]).values)

        self._data['fft'].dataset = self._data['fft'].dataset.assign_coords({'f': freq_fft})
        self._data['fft']['f'].attrs['long_name'] = 'Frequency'

        # --- Proceed with the fft
        for k in signals:
            # Find the time axis
            axis = self._data['signals'][k].dims.index('t')
            # get the coordinates but remove the time
            coords = dict(self._data['signals'][k].coords)
            if 't' in coords:
                coords.pop('t')
            # Now add the frequency coordinate
            coords['f'] = freq_fft

            self._data['fft'][k] = xr.DataArray(
                rfft(self[k].values, axis=axis, **kargs),
                dims=self._data['signals'][k].dims[:axis] + ('f',) +
                     self._data['signals'][k].dims[axis + 1:],
                coords=coords
            )
            # Add the attributes to the fft signals
            self._data['fft']['f'].attrs['long_name'] = 'Fast Fourier Trans'
    

    
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
                       rawLine: Optional[bool] = False,
                       t1: Optional[float] = 0.0, 
                       t2: Optional[float] = 10.0):
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
            sig = self[signal][flags].sel(t=slice(t1, t2))
            nt = sig.size
            flags = np.random.rand(nt) < percent
            sig[flags].plot(ax=ax, color=color, alpha=alpha, 
                                     marker=marker, linestyle=ls, markersize=2)
        else:
            sig = self[signal].sel(channel=nchannel).sel(t=slice(t1,t2))
            nt = sig.size
            flags = np.random.rand(nt) < percent
            sig[flags].plot(ax=ax, color=color, marker=marker, markersize=2,
                                                    alpha=alpha, linestyle=ls)
        # Plot the smoothed data
        if len(self[signal].shape) == 1:
            self['filtered_' + signal].sel(t=slice(t1,t2)).plot(ax=ax, color=color, linewidth=0.5)
        else:
            self['filtered_' + signal].sel(channel=nchannel).sel(t=slice(t1,t2)).plot(ax=ax,
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
        return self._data.signals[item]
    
    def getSignalNames(self):
        """
        Get the names of the signals in the dataset

        :return: list of signal names
        """
        return [k for k in self._data['signals'].dataset.keys()]