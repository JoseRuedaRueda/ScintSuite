"""
Analyze signal from the fast channel

Nowadays only a simple reading of the fast channel and some smoothing is
implemented. In the future, things like correlations with other diagnostics
will be included
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ScintSuite._Plotting as ssplt
import ScintSuite.LibData as ssdat
import ScintSuite._FrequencyAnalysis as ssfq
import ScintSuite.errors as errors
import scipy.signal as sp  # signal processing
try:
    from d3dsuite.LibData.D3D._FastChannels import FastSignal as _fast
except ImportError:
    from ScintSuite._basicVariable import BasicSignalVariable as _fast
from ScintSuite._TimeTrace import roipoly

# -----------------------------------------------------------------------------
# --- Classes
# -----------------------------------------------------------------------------
# class FastChannel:
#     """To interact with signals from the fast channel"""


#     def plot_channels(self, ch_number=None, line_params: dict = {},
#                       ax_params: dict = {}, ax=None, normalise: bool = True,
#                       ptype: str = 'raw', max_to_plot: int = 7500):
#         """
#         Plot the fast channel signals

#         Jose Rueda: jrrueda@us.es

#         Note: A basic correction of the baseline using the last 100 points of
#         the signal will be applied

#         :param  ch_number: channels to plot, np arrays accepted, if none, all
#         channels will be plotted
#         :param  line_params: params for the lines (linewdith, etc)
#         :param  ax_param: parameters for the axis beauty plot
#         :param  ax: axes where to plot, if none, new axis will be opened
#         :param  normalise: If true, signal will be normalised to one to
#         :param  ptype: Type of plot to perform:
#             - 'raw': Just the line with the raw data
#             - 'smooth': Just the line with the smooth data
#             - 'cloud': The line with the smoothed data plus the raw as points.
#                        color property can't be pass as input if this kind of
#                        plot is selecteed
#         :param  max_to_plot: maximum number of points to be plotted

#         :return ax: axes with the time traces plotted
#         """
#         # Initialize the plotting options:
#         line_settings = {
#             'linewidth': 2.0
#         }
#         ax_settings = {
#             'ylabel': 'Signal [a.u.]'
#         }
#         line_settings.update(line_params)
#         ax_settings.update(ax_params)
#         if ch_number is None:
#             ch = self.raw_data['channels']
#         else:
#             # See if the desired number of channels is an array:
#             try:    # If we received a numpy array, all is fine
#                 ch_number.size
#                 ch = ch_number
#             except AttributeError:  # If not, we need to create it
#                 ch = np.array([ch_number])
#                 # nch_to_plot = ch.size

#         # Open the figure, if needed:
#         if ax is None:
#             fig, ax = plt.subplots()
#         # Plot the traces:
#         for ic in ch:
#             if 'label' not in line_settings:
#                 label = 'Ch{0:02}'.format(ic)
#             else:
#                 label = line_settings['label']
#                 del line_settings['label']

#             if self.raw_data['data'][ic - 1] is not None:
#                 per = max_to_plot / self.raw_data['data'][ic - 1].size
#                 flag = np.random.rand(self.raw_data['data'][ic - 1].size) < per
#                 bline = self.raw_data['data'][ic - 1][-100:-1].mean()
#                 if ptype == 'raw':
#                     if normalise:
#                         factor = self.raw_data['data'][ic - 1][flag].max()\
#                             - bline
#                     else:
#                         factor = 1.0
#                     ax.plot(self.raw_data['time'][flag],
#                             (self.raw_data['data'][ic - 1][flag] - bline)/factor,
#                             label=label, **line_settings,
#                             alpha=0.5)
#                 elif ptype == 'smooth':
#                     if normalise:
#                         factor =\
#                             self.filtered_data['data'][ic - 1][flag].max()\
#                             - bline
#                     else:
#                         factor = 1.0
#                     ax.plot(self.filtered_data['time'][flag],
#                             (self.filtered_data['data'][ic - 1][flag] - bline) / factor,
#                             label=label, **line_settings)
#                 elif ptype == 'cloud':
#                     if normalise:
#                         factor = self.raw_data['data'][ic - 1][flag].max() -bline
#                     else:
#                         factor = 1.0
#                     [points] = \
#                         ax.plot(self.raw_data['time'][flag],
#                                 (self.raw_data['data'][ic - 1][flag] - bline) / factor,
#                                 '.', alpha=0.1,
#                                 label='_nolegend_', **line_settings)
#                     ax.plot(self.filtered_data['time'][flag],
#                             (self.filtered_data['data'][ic - 1][flag] - bline) / factor,
#                             '--', alpha=0.5, label=label,
#                             color=points.get_color(), **line_settings)
#             else:
#                 print('Channel ', ic, 'requested but not loaded, skipping!')
#         ax = ssplt.axis_beauty(ax, ax_settings)
#         plt.legend()
#         plt.tight_layout()
#         return ax

#     def plot_spectra(self, ch_number=None,
#                      ax_params: dict = {}, scale: str = 'log',
#                      cmap=None):
#         """
#         Plot the fast channel spectrograms

#         Jose Rueda: jrrueda@us.es

#         :param  ch_number: channels to plot, np arrays accepted, if none, all
#         channels will be plotted
#         :param  ax_param: parameters for the axis beauty plot
#         :param  scale: 'linear', 'sqrt', 'log'
#         """
#         # Initialize the plotting options:
#         ax_settings = {
#             'ylabel': 'Freq. [kHz]'
#         }
#         ax_settings.update(ax_params)
#         if cmap is None:
#             cmap = ssplt.Gamma_II()
#         if ch_number is None:
#             ch = self.raw_data['channels']
#         else:
#             # See if the desired number of channels is an array:
#             try:    # If we received a numpy array, all is fine
#                 ch_number.size
#                 ch = ch_number
#             except AttributeError:  # If not, we need to create it
#                 ch = np.array([ch_number])
#                 # nch_to_plot = ch.size

#         # Open the figure:
#         nchanels = ch.size
#         if nchanels < 4:
#             nn = nchanels
#             ncolumns = 1
#         else:
#             nn = 4
#             ncolumns = int(nchanels/nn) + 1
#         fig, ax = plt.subplots(nn, ncolumns, sharex=True)
#         if nn == 1 and ncolumns == 1:
#             ax = np.array(ax)
#         if ncolumns == 1:
#             ax = ax.reshape(nn, 1)
#         # Plot the traces:
#         counter = 0
#         for ic in ch:
#             if self.spectra[ic - 1] is not None:
#                 # Scale the data
#                 if scale == 'sqrt':
#                     data = np.sqrt(self.spectra[ic - 1]['spec'])
#                 elif scale == 'log':
#                     data = np.log10(self.spectra[ic - 1]['spec'])
#                 elif scale == 'linear':
#                     data = self.spectra[ic - 1]['spec']
#                 else:
#                     raise errors.NotValidInput('Not understood scale')
#                 # Limit for the scale
#                 tmin = self.spectra[ic - 1]['tvec'][0]
#                 tmax = self.spectra[ic - 1]['tvec'][-1]

#                 fmin = self.spectra[ic - 1]['fvec'][0] / 1000.
#                 fmax = self.spectra[ic - 1]['fvec'][-1] / 1000.
#                 # Look row and colum
#                 column = int(counter/nn)
#                 row = counter - nn * column
#                 ax[row, column].imshow(data.T, extent=[tmin, tmax, fmin, fmax],
#                                        cmap=cmap, origin='lower', aspect='auto'
#                                        )
#                 ax[row, column] = ssplt.axis_beauty(ax[row, column],
#                                                     ax_settings)
#                 ax[row, column].set_title('Ch ' + str(ic))
#                 if row == nn-1:
#                     ax[row, column].set_xlabel('Time [s]')
#                 counter += 1
#             else:
#                 print('Channel ', ic, 'requested but not loaded, skipping!')
#         return ax


class FastChannel(_fast):
    """
    Class for the fast channel signals
    
    This is just a wrapper for the fast channel signals.
    """
    def __init__(self, diag, diag_ID, channels=None, shot: int = 199439, exp: str = 'd3d', **kwargs):
        _fast.__init__(self)
        self._data['signals'] = xr.DataTree(name='signals')
        raw_data = \
             ssdat.get_fast_channel(diag, diag_ID, channels=channels, 
                                    shot=shot, exp=exp,**kwargs)
        if type(raw_data) == xr.core.dataarray.DataArray:
            # The get fast channel is the new routine, nothing to do here
            self._data['signals'][diag + str(diag_ID)] = raw_data
        else: # Old format from AUG
            self._data['signals'][diag + str(diag_ID)] = xr.DataArray(np.array(raw_data['data']),
                dims=('channel', 't'), coords={'channel': raw_data['channels'],
                                            't': raw_data['time']})

class FastChanneltoCamera:
    """
    Class to read the calibration of the PMTs and plot them in the camera space
    
    
    """
    def __init__(self, calibrationFile:str):
        """
        Read the calibration file and store the data in a dictionary
        """
        self._readCalibrationFile(calibrationFile)
    
    def _readCalibrationFile(self, calibrationFile:str):
        """
        Read the calibration file and store the data in a dictionary
        """
        # Read the line to know which type of calibration we have
        caltype = np.loadtxt(calibrationFile, max_rows=1, skiprows=1)
        if caltype == 0:
            # We have a circular calibration file
            cx, cy, r = np.loadtxt(calibrationFile, skiprows=3, unpack=True, delimiter=',')
            self.calibration = {
                'type': 'circular',
                'cx': cx,
                'cy': cy,
                'r': r
            }
    def plot_pix(self, ax=None, **kwargs):
        """
        Plot the calibration in the camera space
        """
        if self.calibration['type'] == 'circular':
            # Plot the calibration in the camera space
            if ax is None:
                fig, ax = plt.subplots()
            phi = np.linspace(0, 2 * np.pi, 100)
            for i in range(len(self.calibration['cx'])):
                xcir = self.calibration['cx'][i] + self.calibration['r'][i] * np.cos(phi)
                ycir = self.calibration['cy'][i] + self.calibration['r'][i] * np.sin(phi)
                ax.plot(xcir, ycir, **kwargs)
            ax.set_aspect('equal')
        return ax
    
    def getROIs(self,):
        """
        Get an array of ROIpoly objects, one for each channel
        """
        if self.calibration['type'] == 'circular':
            rois = []
            phi = np.linspace(0, 2 * np.pi, 100)
            for i in range(len(self.calibration['cx'])):
                if self.calibration['r'][i] > 0:
                    xroi = self.calibration['cx'][i] + self.calibration['r'][i] * np.cos(phi)
                    yroi = self.calibration['cy'][i] + self.calibration['r'][i] * np.sin(phi)
                    path = np.array([xroi, yroi]).T
                    rois.append(roipoly(path=path))
                else:
                    rois.append(None)
        else:
            raise errors.NotImplementedError('Calibration type not implemented')
        self.rois = rois
        return rois
        
    def getMask(self, currentImage=None, imageShape=None):
        """
        Get the binary mask from the selected ROI points.
        
        Taken from the toupy codecode https://github.com/jcesardasilva/toupy

        :param  currentImage: Image (matrix) for which we want the ROI ignored 
            if imageShape is passed
        :param  imageShape: Shape of the image (ny, nx) for which we want the ROI 
        """
        masks = []
        for roi in self.rois:
            if roi is not None:
                masks.append(roi.getMask(currentImage=currentImage, imageShape=imageShape))
            else:
                masks.append(None)
        self.masks = masks
        return masks