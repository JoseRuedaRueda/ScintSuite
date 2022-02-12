"""
Analyze signal from the fast channel

Nowadays only a simple reading of the fast channel and some smoothing is
implemented. In the future, things like correlations with other diagnostics
will be included
"""
import numpy as np
import matplotlib.pyplot as plt
import Lib.LibPlotting as ssplt
import Lib.LibData as ssdat
import Lib.LibFrequencyAnalysis as ssfq
import Lib.errors as errors
import scipy.signal as sp  # signal processing


# -----------------------------------------------------------------------------
# --- Classes
# -----------------------------------------------------------------------------
class FastChannel:
    """To interact with signals from the fast channel"""

    def __init__(self, diag, diag_ID, channels, shot):
        """Initialize the class, see get_fast_channel for inputs description"""
        ## Experimental data (time and channel signals)
        self.raw_data = \
            ssdat.get_fast_channel(diag, diag_ID, channels, shot)
        ## Filtered data:
        self.filtered_data = None
        ## Spectras
        self.spectra = None
        ## Diagnostic name
        self.diag = diag
        ## Diagnostic number
        self.diag_ID = diag_ID
        ## Shot number
        self.shot = shot

    def filter(self, method='savgol', params: dict = {}):
        """
        Smooth the signal

        Jose Rueda: jrrueda@us.es

        @params method: Smooth to be applied. Up to now supported:
            - savgol: Savitzky-Golay (see scipy savgol_filter doc)
            - median:  Median filter (See scipy.signal.medfilt)
        """
        # --- Initialise the settings
        options_filter = {
            'savgol': {
                'window_length': 51,
                'polyorder': 3,
            },
            'median': {
                'kernel_size': None
            }
        }
        filters = {
            'savgol': sp.savgol_filter,
            'median': sp.medfilt,
        }
        # --- Perform the filter
        if method not in options_filter.keys():
            raise errors.NotImplementedError('Method not implemented')
        filtered_data = self.raw_data['data'].copy()
        options = options_filter[method]
        options.update(params)
        for i in self.raw_data['channels']:
            dummy = filters[method](self.raw_data['data'][i - 1], **options)
            filtered_data[i - 1] = dummy.copy()
            del dummy
        self.filtered_data = {
            'time': self.raw_data['time'],
            'data': filtered_data,
            'channels': self.raw_data['channels']
        }
        return

    def calculate_spectrogram(self, method: str = 'scipy', params: dict = {},
                              timeResolution: float = 0.002):
        """
        Calculate spectrograms of loaded signals

        Jose Rueda: jrrueda@us.es

        @todo: if 'gauss' timewindow is passed as parameter, my call to
        get_nfft is wrong

        @param method: method to perfor the fourier transform:
            - sfft
            - stft: hort-Time Fourier Transform, Giovanni implementation.
            - stft2: Short-Time Fourier Transform, scipy.signal implementation.
            - scipy: just call the scipy spectrogram function (recomended)
        @param params: dictionary containing the optional parameters of those
                       methos (see Lib.LibFrequencyAnalysis)
        @param timeResolution: set the time resolution of the spectogram, if
                               sfft, stft2 or stft are selected this would be
                               the time resolution parameter, normalise to one,
                               see fast channel library for more information.
                               If you select the scipy method, this will be
                               just the time window you want to use in each
                               point to calculate the fourier transform. Notice
                               that if you include manually 'nperseg' in the
                               params dict, this timeResolution will be ignored
        """
        # --- Just select the desited method
        if method != 'scipy':
            if method == 'stft':
                spec = ssfq.stft
            elif method == 'sfft':
                spec = ssfq.sfft
            elif method == 'stft2':
                spec = ssfq.stft2
            else:
                raise errors.NotImplementedError('Method not understood')
            # --- Perform the spectogram for each channel:
            ch = np.arange(len(self.raw_data['data'])) + 1
            spectra = []
            # Estimate the window size for the ft
            dt = self.raw_data['time'][1] - self.raw_data['time'][0]
            nfft = int(ssfq.get_nfft(timeResolution, method,
                                     self.raw_data['time'].size,
                                     'hann', dt))
            for ic in ch:
                if self.raw_data['data'][ic - 1] is not None:
                    s, fvec, tvec = \
                        spec(self.raw_data['time'],
                             self.raw_data['data'][ic - 1],
                             nfft, **params)
                    dummy = {
                        'spec': abs(s),
                        'fvec': fvec.copy(),
                        'tvec': tvec.copy(),
                    }
                    spectra.append(dummy)
                else:
                    spectra.append(0)
            self.spectra = spectra
        else:
            # --- Time spacing of the data points
            dt = self.raw_data['time'][1] - self.raw_data['time'][0]

            # --- default options for the spectrogram
            options = {
                'window': ('tukey', 0.),
                'fs': 1.0 / dt
            }
            options.update(params)

            # --- estimate the number of points we need:
            npoints = int(timeResolution/dt)
            print(npoints)
            if 'nperseg' not in options:
                options['nperseg'] = npoints
            # --- Perform the spectogram for each channel:
            ch = np.arange(len(self.raw_data['data'])) + 1
            spectra = []
            for ic in ch:
                if self.raw_data['data'][ic - 1] is not None:
                    fvec, tvec, s = \
                        sp.spectrogram(self.raw_data['data'][ic - 1],
                                       **options)
                    dummy = {
                        'spec': np.abs(s).T,
                        'fvec': fvec.copy(),
                        'tvec': tvec.copy(),
                    }
                    spectra.append(dummy)
                else:
                    spectra.append(0)
            self.spectra = spectra
        return

    def plot_channels(self, ch_number=None, line_params: dict = {},
                      ax_params: dict = {}, ax=None, normalise=True,
                      ptype: str = 'raw', max_to_plot: int = 7500):
        """
        Plot the fast channel signals

        Jose Rueda: jrrueda@us.es

        Note: A basic correction of the baseline using the last 100 points of
        the signal will be applied

        @param ch_number: channels to plot, np arrays accepted, if none, all
        channels will be plotted
        @param line_params: params for the lines (linewdith, etc)
        @param ax_param: parameters for the axis beauty plot
        @param ax: axes where to plot, if none, new axis will be opened
        @param normalise: If true, signal will be normalised to one to
        @param ptype: Type of plot to perform:
            - 'raw': Just the line with the raw data
            - 'smooth': Just the line with the smooth data
            - 'cloud': The line with the smoothed data plus the raw as points.
                       color property can't be pass as input if this kind of
                       plot is selecteed
        @param max_to_plot: maximum number of points to be plotted

        @return ax: axes with the time traces plotted
        """
        # Initialize the plotting options:
        line_settings = {
            'linewidth': 2.0
        }
        ax_settings = {
            'ylabel': 'Signal [a.u.]'
        }
        line_settings.update(line_params)
        ax_settings.update(ax_params)
        if ch_number is None:
            ch = self.raw_data['channels']
        else:
            # See if the desired number of channels is an array:
            try:    # If we received a numpy array, all is fine
                ch_number.size
                ch = ch_number
            except AttributeError:  # If not, we need to create it
                ch = np.array([ch_number])
                # nch_to_plot = ch.size

        # Open the figure, if needed:
        if ax is None:
            fig, ax = plt.subplots()
        # Plot the traces:
        for ic in ch:
            if 'label' not in line_settings:
                label = 'Ch{0:02}'.format(ic)
            else:
                label = line_settings['label']
                del line_settings['label']

            if self.raw_data['data'][ic - 1] is not None:
                per = max_to_plot / self.raw_data['data'][ic - 1].size
                flag = np.random.rand(self.raw_data['data'][ic - 1].size) < per
                bline = self.raw_data['data'][ic - 1][-100:-1].mean()
                if ptype == 'raw':
                    if normalise:
                        factor = self.raw_data['data'][ic - 1][flag].max()\
                            - bline
                    else:
                        factor = 1.0
                    ax.plot(self.raw_data['time'][flag],
                            (self.raw_data['data'][ic - 1][flag] - bline)/factor,
                            label=label, **line_settings,
                            alpha=0.5)
                elif ptype == 'smooth':
                    if normalise:
                        factor =\
                            self.filtered_data['data'][ic - 1][flag].max()\
                            - bline
                    else:
                        factor = 1.0
                    ax.plot(self.filtered_data['time'][flag],
                            (self.filtered_data['data'][ic - 1][flag] - bline) / factor,
                            label=label, **line_settings)
                elif ptype == 'cloud':
                    if normalise:
                        factor = self.raw_data['data'][ic - 1][flag].max() -bline
                    else:
                        factor = 1.0
                    [points] = \
                        ax.plot(self.raw_data['time'][flag],
                                (self.raw_data['data'][ic - 1][flag] - bline) / factor,
                                '.', alpha=0.1,
                                label='_nolegend_', **line_settings)
                    ax.plot(self.filtered_data['time'][flag],
                            (self.filtered_data['data'][ic - 1][flag] - bline) / factor,
                            '--', alpha=0.5, label=label,
                            color=points.get_color(), **line_settings)
            else:
                print('Channel ', ic, 'requested but not loaded, skipping!')
        ax = ssplt.axis_beauty(ax, ax_settings)
        plt.legend()
        plt.tight_layout()
        return

    def plot_spectra(self, ch_number=None,
                     ax_params: dict = {}, scale='log',
                     cmap=None):
        """
        Plot the fast channel spectrograms

        Jose Rueda: jrrueda@us.es

        @param ch_number: channels to plot, np arrays accepted, if none, all
        channels will be plotted
        @param ax_param: parameters for the axis beauty plot
        @param scale: 'linear', 'sqrt', 'log'
        """
        # Initialize the plotting options:
        ax_settings = {
            'ylabel': 'Freq. [kHz]'
        }
        ax_settings.update(ax_params)
        if cmap is None:
            cmap = ssplt.Gamma_II()
        if ch_number is None:
            ch = self.raw_data['channels']
        else:
            # See if the desired number of channels is an array:
            try:    # If we received a numpy array, all is fine
                ch_number.size
                ch = ch_number
            except AttributeError:  # If not, we need to create it
                ch = np.array([ch_number])
                # nch_to_plot = ch.size

        # Open the figure:
        nchanels = ch.size
        if nchanels < 4:
            nn = nchanels
            ncolumns = 1
        else:
            nn = 4
            ncolumns = int(nchanels/nn) + 1
        fig, ax = plt.subplots(nn, ncolumns, sharex=True)
        if nn == 1 and ncolumns == 1:
            ax = np.array(ax)
        if ncolumns == 1:
            ax = ax.reshape(nn, 1)
        # Plot the traces:
        counter = 0
        for ic in ch:
            if self.spectra[ic - 1] is not None:
                # Scale the data
                if scale == 'sqrt':
                    data = np.sqrt(self.spectra[ic - 1]['spec'])
                elif scale == 'log':
                    data = np.log10(self.spectra[ic - 1]['spec'])
                elif scale == 'linear':
                    data = self.spectra[ic - 1]['spec']
                else:
                    raise errors.NotValidInput('Not understood scale')
                # Limit for the scale
                tmin = self.spectra[ic - 1]['tvec'][0]
                tmax = self.spectra[ic - 1]['tvec'][-1]

                fmin = self.spectra[ic - 1]['fvec'][0] / 1000.
                fmax = self.spectra[ic - 1]['fvec'][-1] / 1000.
                # Look row and colum
                column = int(counter/nn)
                row = counter - nn * column
                ax[row, column].imshow(data.T, extent=[tmin, tmax, fmin, fmax],
                                       cmap=cmap, origin='lower', aspect='auto'
                                       )
                ax[row, column] = ssplt.axis_beauty(ax[row, column],
                                                    ax_settings)
                ax[row, column].set_title('Ch ' + str(ic))
                if row == nn-1:
                    ax[row, column].set_xlabel('Time [s]')
                counter += 1
            else:
                print('Channel ', ic, 'requested but not loaded, skipping!')
        return
