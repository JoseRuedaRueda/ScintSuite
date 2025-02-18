"""Class to read and plot pySpecView data"""
import logging
import matplotlib
import numpy as np
import xarray as xr
import ScintSuite._Plotting as ssplt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Optional
from ScintSuite._IO import ask_to_open
from scipy.stats.mstats import mquantiles
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
__all__ = ['pySpecView']
# --- Init auxiliary objects
logger = logging.getLogger('ScintSuite.SpecView')

# ------------------------------------------------------------------------------
# --- Auxiliary class for colorbars
# ------------------------------------------------------------------------------
class colorize:
    """
    Taken from Giovanni's code
    """

    def __init__(self, hue, invert=True):
        # hue is a list of hue values for which the colors will be evaluated
        from colorsys import hsv_to_rgb
        self.colors = np.array([hsv_to_rgb(h, 1, 1) for h in hue])
        self.invert = invert

    def __call__(self, index, r):
        value = r / np.amax(r)

        # get fully saturated colors
        color = self.colors[index, :]

        # reduce the lightness
        color *= np.asarray(value)[..., None]

        # invert colors
        if self.invert:   color = 1 - color

        # remove a rounding errors
        color[color < 0] = 0
        color[color * 256 > 255] = 255. / 256

        return color


# ------------------------------------------------------------------------------
# --- Main class of the pySpecView plotting
# ------------------------------------------------------------------------------
class pySpecView:
    """
    Class to handle pySpecView output files

    Jose Rueda: jrrueda@us.es
    """

    def __init__(self, file: str):
        """
        Read the data from the pyspecview file

        :param  file: path to the file, if none, a grpahical window will pop-up
        """
        if file is None:
            self.file = ask_to_open()
        else:
            self.file = file
        dummy = np.load(self.file)
        self.description = dummy.get('description')
        self._dat = {}
        for k in dummy.files:
            try:
                self._dat[k] = dummy.get(k)
            except ValueError:
                pass
        if len(self._dat['spect'].shape) == 3:
            logger.info('We have cross-phasogram information')
            self._dat['cross-phasogram'] = \
                (self._dat['spect'][0, ...].squeeze(),
                 self._dat['spect'][1, ...].squeeze().astype(int))
            self._dat['spect'] = self._dat['spect'][0, ...].squeeze()
        if 'radial_prof_rho' in self._dat.keys():
            logger.info('We have radial profile information')
            radial_profile = xr.DataArray(
                    self._dat['radial_prof_Ampl'], dims=('harmonic', 'rho'),
                    coords={'harmonic': [1, 2, 3, 4],
                            'rho': self._dat['radial_prof_rho']})
            radial_profile.attrs['fmin'] = self._dat['radial_prof_freq'][0]
            radial_profile.attrs['fmax'] = self._dat['radial_prof_freq'][1]
            radial_profile.attrs['tmin'] = self._dat['radial_prof_time'][0]
            radial_profile.attrs['tmax'] = self._dat['radial_prof_time'][1]
            self._dat = {k: v for k, v in self._dat.items() if not k.startswith(
                    'radial_prof')}
            self._dat['radial_profile'] = radial_profile

        # -- Auxiliary data for latter plot
        self.gamma = 0.0

    def plot(self, ax=None, scale='log', cmap=None,
             IncludeColorbar: bool = True, vmax=1.0e6, vmin=1.0e3,
             cross_phasogram: bool = False, 
             gamma: float = 0.5,
             label: bool = True,
             n_cross_phasogram: Optional[int] = None, 
             interpolation='nearest'):
        """
        Plot the spectogram or the cross_phasogram

        Jose Rueda: jrrueda@us.es

        :param  ax: axes where to plot, if None, new ones will be created
        :param  scale: log, sqrt or linear. Only used for the spectogram plot
        :param  cmap: color map to plot. Only used for the spectogram plot
        :param  vmax: maximum value for the color bar, if spectogram is plotted
        :param  vmin: minimum value for the color bar, if spectogram is plotted
        :param  cross_phasogram: flag to plot the cross_phasogram (if present)
        :param  gamma: plotting factor. gamma=0 plot the full phasogram,
            gamma = 1 only plot the most prominent modes
        :param  label: boolean flag to add a labbel with the magnetic coil used
            for the spectra calculation
        """
        # --- Set the parameters
        self.gamma = gamma
        # --- Prepare the axis
        if ax is None:
            fig, ax = plt.subplots()

        # --- Prepare the scale:
        if not cross_phasogram:
            if scale == 'sqrt':
                extra_options = {'norm': colors.PowerNorm(0.5, vmin=vmin,
                                                          vmax=vmax)}
            elif scale == 'log':
                extra_options = {'norm': colors.LogNorm(vmin=vmin, vmax=vmax,
                                                        clip=True)}
            else:
                extra_options = {}
        else:
            extra_options = {}

        # --- Prepare the colormap
        if cmap is None and not cross_phasogram:
            cmap = ssplt.Gamma_II()
        if cross_phasogram:
            # Taken from Giovanni's code
            N_min = self._dat['mode colorbar'][0].min()
            N_max = self._dat['mode colorbar'][0].max()

            color_for_cbar = colorize(
                np.linspace(0, 1, N_max - N_min + 1, endpoint=False))
            cmap = matplotlib.colors.ListedColormap(
                color_for_cbar(self._dat['mode colorbar'][1], 1))
            cmap.set_bad('white')
            cmap.set_under('white')
        # --- Plot the figure
        if not cross_phasogram:
            img = ax.imshow(
                self._dat['spect'],
                extent=[self._dat['tvec'][0], self._dat['tvec'][-1],
                        self._dat['fvec'][0]/1000.0, self._dat['fvec'][-1]/1000.0],
                origin='lower', aspect='auto', cmap=cmap, **extra_options)
        else:
            dummy0 = self._dat['cross-phasogram']
            dummy = self._gammaTransform(dummy0)
            # # Taken from Giovanni's code
            N_min = self._dat['mode colorbar'][0].min()
            N_max = self._dat['mode colorbar'][0].max()

            color_for_cbar = colorize(
                    np.linspace(0, 1, N_max - N_min + 1, endpoint=False))
            z = color_for_cbar(dummy[1], dummy[0])
            if n_cross_phasogram is not None:
                j = self._dat['mode colorbar'][0, :]==n_cross_phasogram
                j = self._dat['mode colorbar'][1, j]
                mask = dummy0[1] == j
                z[~mask] = np.nan
                z = np.ma.array (z, mask=np.isnan(z))
                extra_options['interpolation'] = 'nearest'
            img = ax.imshow(
                z,
                extent=[self._dat['tvec'][0], self._dat['tvec'][-1],
                        self._dat['fvec'][0]/1000.0, self._dat['fvec'][-1]/1000.0],
                origin='lower', aspect='auto', cmap=cmap, **extra_options)
        if IncludeColorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if not cross_phasogram:
                plt.colorbar(img, cax=cax)
            else:
                # --- Prepare the scale
                bounds = np.linspace(N_min-.5, N_max+.5, N_max-N_min+2)
                cax.cla()
                cax.xaxis.set_major_formatter(NullFormatter())
                cax.yaxis.set_major_formatter(NullFormatter())
                cax.set_ylim(0, 1)
                nn = N_max - N_min + 1
                matplotlib.colorbar.ColorbarBase(cax,
                                                 cmap=cmap,
                                                 boundaries=bounds,
                                                 extendfrac='auto',
                                                 values=np.linspace(0, 1,nn),
                                                 ticks=np.arange(N_min,
                                                                 N_max + 1),
                                                 spacing='uniform')
                cax.set_xlabel('#')
                cax.xaxis.set_label_position('top')
        # ---- Set the axis labels
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [kHz]')
        if label:
            ax.text(0.05, 0.90, self.description, horizontalalignment='left',
                    color='w', verticalalignment='bottom',
                    transform=ax.transAxes)
        return ax

    def plot_radial_profile(self, ax=None, line_params: dict = {},
                            harmonic: int = 1):
        """
        Plot the radial profile

        :param  ax:
        :param  line_params:

        """
        line_options = {
            'color': 'k',
        }
        line_options.update(line_params)
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self._dat['radial_profile'].rho,
                self._dat['radial_profile'].values[harmonic-1, :],
                **line_options)

    def _gammaTransform(self, z, typ='log'):
        """
        Apply the gamma transformation to a matrix, to 'polish' the figure

        :param z:
        :param typ:

        :return:
        """
        if type(z) is tuple:
            return self._gammaTransform(z[0], typ=typ), z[1]
        if np.iscomplexobj(z):
            z = np.abs(z)

        if typ == 'log' and z.size > 1:
            # transformation by log
            ind = np.random.randint(0, z.size - 1, size=10000)
            vmin = mquantiles(z.flat[ind], .5)[0]
            z_norm = np.tan(np.pi / 2.000001 * self.gamma) * np.nanmean(z)
            out = np.log1p(
                np.maximum(0, z - vmin * np.sqrt(self.gamma * 2)) / z_norm)

            return out

        if typ == 'gamma':  # standard definition of gamma
            return z ** self.gamma
