"""
File with the TimeTrace object
"""
import time
import logging
import datetime
import numpy as np
import xarray as xr
import Lib._IO as ssio
import Lib._Plotting as ssplt
import matplotlib.pyplot as plt
from tqdm import tqdm
from Lib._basicVariable import BasicSignalVariable


# --- Initialise the auxiliary objects
logger = logging.getLogger('ScintSuite.TimeTrace')


# ------------------------------------------------------------------------------
# --- Trace calculators
# ------------------------------------------------------------------------------
def trace(frames, mask):
    """
    Calculate the trace of a region given by a mask

    Jose Rueda: ruejo@ipp.mpg.de

    Given the set of frames, frames[px,py,number_frames] and the mask[px,py]
    this function just sum all the pixels inside the mask for all the frames

    :param  frames: 3D array containing the frames[px,py,number_frames]
    :param  mask: Binary matrix which the pixels which must be consider,
    generated by roipoly

    :return sum_of_roi: numpy array with the sum of the pixels inside the mask
    :return mean_of_roi: numpy array with the mean of the pixels inside the mask
    :return std_of_roi: numpy array with the std of the pixels inside the mask
    :return max_of_roi: numpy array with the max of the pixels inside the mask
    """
    
    nt = frames.shape[-1]
    sum_of_roi = np.zeros((nt,), dtype='float64')
    mean_of_roi = np.zeros((nt,), dtype='float64')
    std_of_roi = np.zeros((nt,), dtype='float64')
    max_of_roi = np.zeros((nt,), dtype='float64')
    
    nmask_valid = frames[mask, 0].size
    
    for it in range(nt):
        sum_of_roi[it]  = frames[mask, it].astype('float64').sum()
        mean_of_roi[it] = sum_of_roi[it] / nmask_valid
        std_of_roi[it]  = np.std(frames[mask, it].astype('float64'))
        max_of_roi[it]  = np.nanmax(frames[mask, it].astype('float64'))
        
    return sum_of_roi, mean_of_roi, std_of_roi, max_of_roi


def time_trace_cine(cin_object, mask, t1=0, t2=10):
    """
    Calculate the time trace from a cin file

    Jose Rueda: jrrueda@us.es

    Note, this solution is not ideal, I should include something relaying in
    the LibVideo library, because if in the future we made som upgrade to the
    .cin routines... but for the time this looks like the most efficient way
    of doing it, as there is no need of opening the video in each frame

    :param  cin_object: cin file object (see class Video of LibVideoFiles.py)
    :param  mask: binary mask defining the roi
    :param  t1: initial time to calculate the timetrace [in s]
    :param  t2: final time to calculate the timetrace [in s]

    :return sum_of_roi: numpy array with the sum of the pixels inside the mask
    :return mean_of_roi: numpy array with the mean of the pixels inside the mask
    :return std_of_roi: numpy array with the std of the pixels inside the mask
    :return max_of_roi: numpy array with the max of the pixels inside the mask
    """
    # --- Section 0: Initialise the arrays
    # Look for the index in the time base
    i1 = (np.abs(cin_object.timebase - t1)).argmin()
    i2 = (np.abs(cin_object.timebase - t2)).argmin()
    # Initialise the arrays
    time_base = cin_object.timebase[i1:i2]
    sum_of_roi = np.zeros(i2 - i1)
    mean_of_roi = np.zeros(i2 - i1)
    std_of_roi = np.zeros(i2 - i1)
    max_of_roi = np.zeros(i2 - i1)
    # I could call for each time the read_cine_image, but that would imply to
    # open and close several time the file as well as navigate trough it... I
    # will create temporally a copy of that routine, this need to be
    # rewritten properly, but it should work and be really fast
    # ---  Section 1: Get frames position
    # Open file and go to the position of the image header
    logger.info('Opening .cin file and reading data')
    tic = time.time()
    fid = open(cin_object.file, 'r')
    fid.seek(cin_object.header['OffImageOffsets'])
    # Position of the frames
    if cin_object.header['Version'] == 0:  # old format
        position_array = np.fromfile(fid, 'int32',
                                     int(cin_object.header['ImageCount']))
    else:
        position_array = np.fromfile(fid, 'int64',
                                     int(cin_object.header['ImageCount']))
    # 8-bits or 16 bits frames:
    size_info = cin_object.settings['RealBPP']
    if size_info <= 8:
        # BPP = 8  # Bits per pixel
        data_type = 'uint8'
    else:
        # BPP = 16  # Bits per pixel
        data_type = 'uint16'
    # Image size (in bytes)
    # img_size_header = int(cin_object.imageheader['biWidth'] *
    #                     cin_object.imageheader['biHeight']) * BPP / 8
    # Number of pixels
    npixels = cin_object.imageheader['biWidth'] * \
        cin_object.imageheader['biHeight']
    itt = 0  # Index to cover the array during the loop
    # --- Section 2: Read the frames
    logger.info('Calculating the timetrace... ')
    for i in tqdm(range(i1, i2)):
        #  Go to the position of the file
        iframe = i  # - cin_object.header['FirstImageNo']
        fid.seek(position_array[iframe])
        #  Skip header of the frame
        length_annotation = np.fromfile(fid, 'uint32', 1)
        fid.seek(position_array[iframe] + length_annotation - 4)
        #  Read frame
        np.fromfile(fid, 'uint32', 1)  # Image size in bytes
        dummy = np.reshape(np.fromfile(fid, data_type,
                                       int(npixels)),
                           (int(cin_object.imageheader['biWidth']),
                            int(cin_object.imageheader['biHeight'])),
                           order='F').transpose()
        sum_of_roi[itt] = np.sum(dummy[mask])
        mean_of_roi[itt] = np.mean(dummy[mask])
        std_of_roi[itt] = np.std(dummy[mask])
        max_of_roi[itt] = dummy[mask].max()
        itt = itt + 1
    fid.close()
    toc = time.time()
    logger.info('Elapsed time [s]: ', toc - tic)
    return time_base, sum_of_roi, mean_of_roi, std_of_roi, max_of_roi


# ------------------------------------------------------------------------------
# --- TimeTrace class
# ------------------------------------------------------------------------------
class TimeTrace(BasicSignalVariable):
    """Class with information of the time trace"""

    def __init__(self, video=None, mask=None, t1: float = None,
                 t2: float = None, filename: str = None,
                 ROIname: str = None):
        """
        Initialise the TimeTrace

        Jose Rueda Rueda: jrrueda@us.es

        There are 3 ways of initalise the TT object:
            - Option 1: Loaded from a file. just give the path to the netCDF
            file using the arguments filename
            - Option 2: pass the video, t1 and t2. The trace will be calculated
            using the video frames from t1 to t2
            - Option 3: pass just the vide, the trace will be calculated
            using all the frames

        :param  video: Video object used for the calculation of the trace
        :param  mask: mask to calculate the trace
        :param  t1: Initial time if None, the loaded frames in the video will be
            used
        :param  t2: Final time if None, the loaded frames in the video will be
            used
        :param  ROIname: name of the trace, if present, it will be used as label in
            the legend plotting. (Useful to overplot different traces). If the
            trace is loaded from file, this argument will be ignored
        """
        BasicSignalVariable.__init__(self)
        if filename is None:
            # Initialise the times to look for the time trace
            if video is not None:
                if t1 is None and t2 is None:
                    if video.exp_dat is None:
                        aa = 'Frames are not loaded, use t1 and t2'
                        raise Exception(aa)
                elif t1 is None and t2 is not None:
                    raise Exception('Only one time was given!')
                elif t1 is not None and t2 is None:
                    raise Exception('Only one time was given!')
                shot = video.shot
            else:
                shot = None
            # Initialise the different arrays
            self._data.attrs['shot'] = shot

            ## Binary mask defining the roi
            self.mask = mask
            ## roiPoly object (not initialised by default!!!)
            self.roi = None
            ## Spectrogram data
            self.spec = {'taxis': None, 'faxis': None, 'data': None}
            ## fft data
            self.fft = {'faxis': None, 'data': None}
            # Calculate the time trace
            if video is not None:
                if t1 is None:
                    time_base = video.exp_dat['t'].squeeze()
                    sum_of_roi, mean_of_roi, std_of_roi,\
                        max_of_roi = trace(video.exp_dat['frames'].values, mask)
                else:
                    if video.type_of_file == '.cin':
                        time_base, sum_of_roi, mean_of_roi,\
                            std_of_roi, max_of_roi =\
                            time_trace_cine(video, mask, t1, t2)
                    else:
                        raise Exception('Still not implemented, contact ruejo')
                # Save the trace in the data structure
                self._data['sum_of_roi'] = \
                    xr.DataArray(sum_of_roi, dims=('t',),
                                 coords={'t':time_base})
                self._data['mean_of_roi'] = xr.DataArray(mean_of_roi, dims='t')
                self._data['std_of_roi'] = xr.DataArray(std_of_roi, dims='t')
                self._data['max_of_roi'] = xr.DataArray(max_of_roi, dims='t')
            self._ROIname = ROIname
            if self._ROIname is not None:
                self._data.attrs['ROIname'] = ROIname
            else:
                self._data.attrs['ROIname'] = ''
        else:
            self._data = xr.load_dataset(filename)
            self.mask = self._data['mask'].values.copy()
            self._data.drop('mask')
            try:  # old traces does not have this field
                self._ROIname = self._data.attrs['ROIname']
                if self._ROIname == '':
                    self._ROIname = None
            except KeyError:
                self._ROIname = None
                self._data.attrs['ROIname'] = ''

    def export_to_ascii(self, filename: str = None, precision=3):
        """
        Export time trace to acsii

        Jose Rueda: jrrueda@us.es

        :param  self: the TimeTrace object
        :param  filename: file where to write the data
        :param  precision: number of digints after the decimal point

        :return None. A file is created with the information
        """
        # --- check if file exist
        if filename is None:
            filename = ssio.ask_to_save(ext='*.txt')
            if filename == '' or filename == ():
                print('You canceled the export')
                return
        else:
            filename = ssio.check_save_file(filename)
        # --- Prepare the header
        date = datetime.datetime.now()
        try:
            shot = self._data.attrs['shot']
        except KeyError:
            shot = 0
        line = 'Time trace: ' + date.strftime("%d-%b-%Y (%H:%M:%S.%f)") + \
               ' shot %i' % shot + '\n' + \
               'Time [s]    ' + \
               'Counts in Roi     ' + \
               'Mean in Roi      Std Roi'
        length = self['t'].values.size
        # Save the data
        np.savetxt(filename,
                   np.hstack((self['t'].values.reshape(length, 1),
                             self['sum_of_roi'].values.reshape(length, 1),
                             self['mean_of_roi'].values.reshape(length, 1),
                             self['std_of_roi'].values.reshape(length, 1))),
                   delimiter='   ,   ', header=line,
                   fmt='%.'+str(precision)+'e')

    def export_to_netcdf(self, filename: str):
        """
        Export the time trace to a netCDF file

        :param  filename: str, name of the nc file to be created
        """
        dummy = self._data.copy()
        dummy['mask'] = xr.DataArray(self.mask)
        dummy.to_netcdf(filename)


    def plot_single(self, data: str = 'sum', ax_params: dict = {},
                    line_params: dict = {}, normalised: bool = False, ax=None,
                    correct_baseline: str = 'end'):
        """
        Plot the total number of counts in the ROI

        Jose Rueda: jrrueda@us.es

        :param  data: select which timetrace to plot:
            - sum is the total number of counts in the ROI
            - std its standard deviation
            - mean, the mean
            - sum/max: is the sum of the counts divided by the absolute maximum

        :param  ax_par: Dictionary containing the options for the axis_beauty
        function.
        :param  line_par: Dictionary containing the line parameters
        :param  normalised: if normalised, plot will be normalised to one.
        :param  ax: axes where to draw the figure, if none, new figure will be
        created
        :param  correct_baseline: str to correct baseline. If 'end' the last
        mean of the last 15 points of the time trace will be substracted to the
        tt. (minus the very last one, because some time this points is off for
        AUG CCDs). If 'ini' the first 15 points. Else, no correction
        """
        # default plotting options
        ax_options = {
            'xlabel': 't [s]'
        }
        line_options = {
        }
        if self._ROIname is not None:
            line_options['label'] = self._ROIname
        # --- Select the proper data:
        if data == 'sum':
            y = self['sum_of_roi'].values.copy()
            if 'ylabel' not in ax_params:
                if normalised:
                    ax_params['ylabel'] = 'Counts [a.u.]'
                else:
                    ax_params['ylabel'] = 'Counts'
        elif data == 'std':
            y = self['std_of_roi'].values.copy()
            if 'ylabel' not in ax_params:
                if normalised:
                    ax_params['ylabel'] = '$\\sigma [a.u.]$'
                else:
                    ax_params['ylabel'] = '$\\sigma$'
        elif data == 'mean/absmax':
            y = self['mean_of_roi'].values.copy() \
                / self['max_of_roi'].values.max()
            if 'ylabel' not in ax_params:
                if normalised:
                    ax_params['ylabel'] = '$Mean/absmax [a.u.]$'
                else:
                    ax_params['ylabel'] = '$Mean/absmax$'
        elif data == 'max':
            y = self['max_of_roi'].values.copy()
            if 'ylabel' not in ax_params:
                if normalised:
                    ax_params['ylabel'] = '$Max$'
                else:
                    ax_params['ylabel'] = '$Max [a.u.]$'
        else:
            y = self['mean_of_roi'].values.copy()
            if 'ylabel' not in ax_params:
                if normalised:
                    ax_params['ylabel'] = 'Mean [a.u.]'
                else:
                    ax_params['ylabel'] = 'Mean'
        # --- Normalize the data:
        if correct_baseline == 'end':
            baseline_level = y[-15:-2].mean().astype(y.dtype)
            y += -baseline_level
            logger.info('Baseline corrected using the last points')
            logger.info('Baseline level: %i' % round(baseline_level))

        elif correct_baseline == 'ini':
            baseline_level = y[3:8].mean().astype(y.dtype)
            y += -baseline_level
            logger.info('Baseline corrected using the initial points')
            logger.info('Baseline level: %i' % round(baseline_level))
        else:
            logger.info('Not applying any baseline correction')

        if normalised:
            logger.debug('Normalising the data')
            y /= y.max()

        # create and plot the figure
        if ax is None:
            fig, ax = plt.subplots()
            created_ax = True
        else:
            created_ax = False
        line_options.update(line_params)
        ax.plot(self['t'], y, **line_options)
        ax_options.update(ax_params)
        ax = ssplt.axis_beauty(ax, ax_options)

        if created_ax:
            plt.tight_layout()
            fig.show()
        plt.draw()
        return ax

    def plot_all(self, ax_par: dict = {}, line_par: dict = {}):
        """
        Plot the sum time trace, the average timetrace and the std ones.

        Jose Rueda: jrrueda@us.es

        Plot the sum, std and average of the roi

        :param  options: Dictionary containing the options for the axis_beauty
        function. Notice, the y and x label are fixed, if present in the
        options, they will be ignored

        :return axes: list of axes where the lines have been plotted
        """
        # Initialise the options for the plotting
        ax_options = {
            'grid': 'both'
        }
        ax_options.update(ax_par)
        line_options = {
            'linewidth': 2.0,
            'color': 'r'
        }
        line_options.update(line_par)
        # Open the figure
        fig_tt, [ax_tt1, ax_tt2, ax_tt3] = plt.subplots(3, sharex=True)
        # Plot the sum of the counts in the roi
        ax_tt1.plot(self['t'], self['sum_of_roi'].values, **line_options)
        ax_options['ylabel'] = 'Counts'
        ax_tt1 = ssplt.axis_beauty(ax_tt1, ax_options)

        # plot the mean of the counts in the roi
        ax_tt2.plot(self['t'], self['mean_of_roi'].values, **line_options)
        ax_options['ylabel'] = 'Mean'
        ax_tt2 = ssplt.axis_beauty(ax_tt2, ax_options)

        # plot the std of the counts in the roi
        ax_tt3.plot(self['t'], self['std_of_roi'].values, **line_options)
        ax_options['xlabel'] = 't [s]'
        ax_options['ylabel'] = '$\\sigma$'
        ax_tt3 = ssplt.axis_beauty(ax_tt3, ax_options)
        plt.tight_layout()
        plt.show()

        return [ax_tt1, ax_tt2, ax_tt3]

    @property
    def mean_of_roi(self):
        """
        Return the mean of the ROI.

        Pablo Oyola - poyola@us.es
        """

        return self._data['mean_of_roi'].copy()

    @property
    def sum_of_roi(self):
        """
        Return the mean of the ROI.

        Pablo Oyola - poyola@us.es
        """

        return self._data['sum_of_roi'].copy()

    @property
    def std_of_roi(self):
        """
        Standard deviation of the ROI.

        Pablo Oyola - poyola@us.es
        """

        return self._data['std_of_roi'].copy()
