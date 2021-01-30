"""
Analyse signal from the fast channel

Nowadays only a simple reading of the fast channel and some smothing is
implemented. In the future, things like correlations with other diagnostics
will be included
"""
import numpy as np
import matplotlib.pyplot as plt
import LibPlotting as ssplt
from LibMachine import machine
if machine == 'AUG':
    import LibDataAUG as ssdat


# -----------------------------------------------------------------------------
# --- Classes
# -----------------------------------------------------------------------------
class Fast:
    """To interact with signals from the fast channel"""

    def __init__(self, diag, diag_number, channels, shot):
        """Initialise the class, see get_fast_channel for inputs description"""
        ## Experimental data (time and channel signals)
        self.data = ssdat.get_fast_channel(diag, diag_number, channels, shot)

    def plot_channels(self, ch_number, line_params: dict = {},
                      ax_param: dict = {}, ax=None):
        """
        Plot the fast channel signals

        Jose Rueda: jrrueda@us.es

        @param ch_number: channels to plot, np arrays accepted
        @param line_params: params for the lines (linewdith, etc)
        @param ax_param: parameters for the axis beauty plot
        @param ax: axes where to plot, if none, new axis will be opened
        @preturn ax: axes with the time traces plotted
        """
        # Initialise the plotting options:
        if 'linewidth' not in line_params:
            line_params['linewidth'] = 1.5
        if 'fontsize' not in ax_param:
            ax_param['fontsize'] = 14
        if 'xlabel' not in ax_param:
            ax_param['xlabel'] = 'Time [s]'
        if 'ylabel' not in ax_param:
            ax_param['ylabel'] = 'Signal [a.u.]'

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
            if self.data['signal'][ic - 1] is not None:
                ax.plot(self.data['time'], self.data['signal'][ic - 1],
                        label='Ch{0:02}'.format(ic), **line_params)
            else:
                print('Channel ', ic, 'requested but not loaded, skipping!')
        ax = ssplt.axis_beauty(ax, ax_param)
        plt.legend()
        plt.tight_layout()
