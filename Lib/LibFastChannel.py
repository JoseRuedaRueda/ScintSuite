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
import scipy.signal as sp  # signal processing


# -----------------------------------------------------------------------------
# --- Classes
# -----------------------------------------------------------------------------
class FastChannel:
    """To interact with signals from the fast channel"""

    def __init__(self, diag, diag_number, channels, shot):
        """Initialize the class, see get_fast_channel for inputs description"""
        ## Experimental data (time and channel signals)
        self.raw_data = \
            ssdat.get_fast_channel(diag, diag_number, channels, shot)

    def filter(self, method, params: dict = {}):
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
            raise Exception('Method not implemented')
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
        filtered_data
        return

    def plot_channels(self, ch_number=None, line_params: dict = {},
                      ax_params: dict = {}, ax=None, normalise=True,
                      ptype: str = 'cloud', max_to_plot: int = 7500):
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
            if self.raw_data['data'][ic - 1] is not None:
                per = max_to_plot / self.raw_data['data'][ic - 1].size
                flag = np.random.rand(self.raw_data['data'][ic - 1].size) < per
                bline = self.raw_data['data'][ic - 1][-100:-1].mean()
                if ptype == 'raw':
                    if normalise:
                        factor = self.raw_data['data'][ic - 1][flag].max() - bline
                    else:
                        factor = 1.0
                    ax.plot(self.raw_data['time'][flag],
                            (self.raw_data['data'][ic - 1][flag] - bline)/factor,
                            label='Ch{0:02}'.format(ic), **line_settings,
                            alpha=0.5)
                elif ptype == 'smooth':
                    if normalise:
                        factor = self.filtered_data['data'][ic - 1][flag].max() - bline
                    else:
                        factor = 1.0
                    ax.plot(self.filtered_data['time'][flag],
                            (self.filtered_data['data'][ic - 1][flag] - bline) / factor,
                            label='Ch{0:02}'.format(ic), **line_settings)
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
                            '--', alpha=0.5, label='Ch{0:02}'.format(ic),
                            color=points.get_color(), **line_settings)
            else:
                print('Channel ', ic, 'requested but not loaded, skipping!')
        ax = ssplt.axis_beauty(ax, ax_settings)
        plt.legend()
        plt.tight_layout()
