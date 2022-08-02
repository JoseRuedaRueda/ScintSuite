"""Class to read and plot pySpecView data"""

import numpy as np
from Lib._IO import ask_to_open
import Lib._Plotting as ssplt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
__all__ = ['pySpecView']


class pySpecView:
    """
    Class to handle pySpecView output files

    Jose Rueda: jrrueda@us.es
    """

    def __init__(self, file: str):
        """

        @param file: path to the file, if none, a grpahical window will pop-up

        """
        if file is None:
            self.file = ask_to_open()
        else:
            self.file = file
        dummy = np.load(self.file)
        self.description = dummy.get('description')
        self._dat = {}
        for k in dummy.files:
            self._dat[k] = dummy.get(k)

    def plot(self, ax=None, scale='log', cmap=None,
             IncludeColorbar: bool = True, vmax=1.0e6, vmin=1.0e3):
        """
        Plot the spectogram

        Jose Rueda: jrrueda@us.es
        """
        if ax is None:
            fig, ax = plt.subplots()

        # --- Prepare the scale:
        if scale == 'sqrt':
            extra_options = {'norm': colors.PowerNorm(0.5, vmin=vmin)}
        elif scale == 'log':
            extra_options = {'norm': colors.LogNorm(vmin=vmin, vmax=vmax,
                                                    clip=True)}
        else:
            extra_options = {}
        # --- Prepare the colormap
        if cmap is None:
            cmap = ssplt.Gamma_II()
        img = ax.imshow(
            self._dat['spect'],
            extent=[self._dat['tvec'][0], self._dat['tvec'][-1],
                    self._dat['fvec'][0]/1000.0, self._dat['fvec'][-1]/1000.0],
            origin='lower', aspect='auto', cmap=cmap, **extra_options)
        if IncludeColorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, cax=cax)
        # ---- Set the axis labels
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [kHz]')
        ax.text(0.05, 0.90, self.description, horizontalalignment='left',
                color='w', verticalalignment='bottom',
                transform=ax.transAxes)
        return ax
