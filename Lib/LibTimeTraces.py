"""
Package to calculate time traces

Contains all the routines to calculate time traces. The RoiPoly module must be
installed. If you use Spyder as Python IDE, please install the RoiPoly version
which is compatible with Spyder.
(See the Read-me of the project to get the link)

"""
import numpy as np
import datetime
import time
import warnings
from scipy.fft import rfft, rfftfreq
from scipy import signal
import matplotlib.pyplot as plt
import Lib.LibPlotting as ssplt
import Lib.LibIO as ssio
from tqdm import tqdm
try:
    from roipoly import RoiPoly
except ImportError:
    warnings.warn('You cannot calculate time traces, roipoly not found',
                  category=UserWarning)


def trace(frames, mask):
    """
    Calculate the trace of a region given by a mask

    Jose Rueda: ruejo@ipp.mpg.de

    Given the set of frames, frames[px,py,number_frames] and the mask[px,py]
    this function just sum all the pixels inside the mask for all the frames

    @param frames: 3D array containing the frames[px,py,number_frames]
    @param mask: Binary matrix which the pixels which must be consider,
    generated by roipoly
    @return tr: numpy array with the sum of the pixels inside the mask
    """
    # Allocate output array
    n = frames.shape[2]
    sum_of_roi = np.zeros(n)
    std_of_roi = np.zeros(n)
    mean_of_roi = np.zeros(n)
    # calculate the trace
    print('Calculating the timetrace... ')
    for iframe in tqdm(range(n)):
        dummy = frames[:, :, iframe].squeeze()
        sum_of_roi[iframe] = np.sum(dummy[mask])
        mean_of_roi[iframe] = np.mean(dummy[mask])
        std_of_roi[iframe] = np.std(dummy[mask])
    return sum_of_roi, std_of_roi, mean_of_roi


def create_roi(fig, re_display=False):
    """
    Wrap for the RoiPoly features

    Jose Rueda: jrrueda@us.es

    Just a wrapper for the roipoly capabilities which allows for the reopening
    of the figures

    @param fig: fig object where the image is found
    @return fig: The figure with the roi plotted
    @return roi: The PloyRoi object
    """
    # Define the roi
    print('Please select the vertex of the roi in the figure')
    print('Select each vertex with left click')
    print('Once you finished, right click')
    roi = RoiPoly(color='r', fig=fig)
    print('Thanks')
    # Show again the image with the roi
    if re_display:
        fig.show()
        roi.display_roi()
    return fig, roi


def time_trace_cine(cin_object, mask, t1=0, t2=10):
    """
    Calculate the time trace from a cin file

    Jose Rueda: jrrueda@us.es

    Note, this solution is not ideal, I should include something relaying in
    the LibVideo library, because if in the future we made som upgrade to the
    .cin routines... but for the time this looks like the most efficient way
    of doing it, as there is no need of opening the video in each frame

    @param cin_object: cin file object (see class Video of LibVideoFiles.py)
    @param mask: binary mask defining the roi
    @param t1: initial time to calculate the timetrace [in s]
    @param t2: final time to calculate the timetrace [in s]
    @return tt: TimeTrace object
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
    # I could call for each time the read_cine_image, but that would imply to
    # open and close several time the file as well as navigate trough it... I
    # will create temporally a copy of that routine, this need to be
    # rewritten properly, but it should work and be really fast
    # ---  Section 1: Get frames position
    # Open file and go to the position of the image header
    print('Opening .cin file and reading data')
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
    print('Calculating the timetrace... ')
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
        itt = itt + 1
    fid.close()
    toc = time.time()
    print('Elapsed time [s]: ', toc - tic)
    return time_base, sum_of_roi, mean_of_roi, std_of_roi


class TimeTrace:
    """Class with information of the time trace"""

    def __init__(self, video=None, mask=None, t1: float = None,
                 t2: float = None):
        """
        Initialise the TimeTrace

        Jose Rueda Rueda: jrrueda@us.es

        If no times are given, it will use the frames loaded in the video
        object, if t1 and t2 are present, it will load the corresponding frames
        If no argument are giving, and empty timetrace object will be created,
        to be filled by the reading routines

        @param video: Video object used for the calculation of the trace
        @param mask: mask to calculate the trace
        @param t1: Initial time if None, the loaded frames in the video will be
        used
        @param t2: Final time if None, the loaded frames in the video will be
        used
        """
        # Initialise the times to look for the time trace
        if video is not None:
            if t1 is None and t2 is None:
                if video.exp_dat['tframes'] is None:
                    aa = 'Frames are not loaded, use t1 and t2'
                    raise Exception(aa)
            elif t1 is None and t2 is not None:
                raise Exception('Only one time was given!')
            elif t1 is not None and t2 is None:
                raise Exception('Only one time was given!')
            self.shot = video.shot
        else:
            self.shot = None
        # Initialise the different arrays
        ## Numpy array with the time base
        self.time_base = None
        ## Numpy array with the total counts in the ROI
        self.sum_of_roi = None
        ## Numpy array with the mean of counts in the ROI
        self.mean_of_roi = None
        ## Numpy array with the std of counts in the ROI
        self.std_of_roi = None
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
                self.time_base = video.exp_dat['tframes'].squeeze()
                self.sum_of_roi, self.mean_of_roi, self.std_of_roi\
                    = trace(video.exp_dat['frames'], mask)
            else:
                if video.type_of_file == '.cin':
                    self.time_base, self.sum_of_roi, self.mean_of_roi,\
                        self.std_of_roi = time_trace_cine(video, mask, t1, t2)
                else:
                    raise Exception('Still not implemented, contact ruejo')

    def export_to_ascii(self, filename: str = None, precision=3):
        """
        Export time trace to acsii

        Jose Rueda: jrrueda@us.es

        @param self: the TimeTrace object
        @param filename: file where to write the data
        @param precision: number of digints after the decimal point

        @return None. A file is created with the information
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
        line = 'Time trace: ' + date.strftime("%d-%b-%Y (%H:%M:%S.%f)") + \
               ' shot ' + str(self.shot) + '\n' + \
               'Time [s]    ' + \
               'Counts in Roi     ' + \
               'Mean in Roi      Std Roi'
        length = self.time_base.size
        # Save the data
        np.savetxt(filename, np.hstack((self.time_base.reshape(length, 1),
                                        self.sum_of_roi.reshape(length, 1),
                                        self.mean_of_roi.reshape(length, 1),
                                        self.std_of_roi.reshape(length, 1))),
                   delimiter='   ,   ', header=line,
                   fmt='%.'+str(precision)+'e')

    def calculate_fft(self, params: dict = {}):
        """
        Calculate the fft of the time trace

        Jose Rueda Rueda: jrrueda@us.es

        Only the fft of the sum of the counts in the roi is calculated, if you
        want others to be calculated, open a request in the GitLab

        @param    params: Dictionary containing optional arguments for scipyfft
        see scipy.fft.rfft for full details
        @type:    dict

        @return:  nothing, just fill self.fft
        """
        N = len(self.time_base)
        self.fft['faxis'] = rfftfreq(N, self.time_base[2] - self.time_base[1])
        self.fft['data'] = rfft(self.sum_of_roi, **params)
        return

    def calculate_spectrogram(self, params: dict = {}):
        """
        Calculate the spectrogram of the time trace

        Jose Rueda Rueda: jrrueda@us.es

        Only the spec of the sum of the counts in the roi is calculated, if you
        want others to be calculated, open a request in the GitLab

        @param    params: Dictionary containing optional arguments for the
        spectrogram, see scipy.signal.spectrogram for the full details
        @type:    dict

        @return:  nothing, just fill self.spec
        """
        sampling_freq = 1 / (self.time_base[1] - self.time_base[0])
        # print(sampling_freq)
        f, t, Sxx = signal.spectrogram(self.sum_of_roi, sampling_freq)
        self.spec['faxis'] = f
        self.spec['taxis'] = t + self.time_base[0]
        self.spec['data'] = Sxx
        return

    def plot_single(self, data: str = 'sum', ax_params: dict = {},
                    line_params: dict = {}, normalised: bool = False, ax=None,
                    correct_baseline: str = 'end'):
        """
        Plot the total number of counts in the ROI

        Jose Rueda: jrrueda@us.es

        @param data: select which timetrace to plot: sum is the total number of
        counts in the ROI, std its standard deviation and mean, the mean
        @param ax_par: Dictionary containing the options for the axis_beauty
        function.
        @param line_par: Dictionary containing the line parameters
        @param normalised: if normalised, plot will be normalised to one.
        @param ax: axes where to draw the figure, if none, new figure will be
        created
        @param correct_baseline: str to correct baseline. If 'end' the last
        mean of the last 15 points of the time trace will be substracted to the
        tt. (minus the very last one, because some time this points is off for
        AUG CCDs). If 'ini' the first 15 points. Else, no correction
        """
        # default plotting options
        ax_options = {
            'fontsize': 16.0,
            'grid': 'both',
            'xlabel': 't [s]'
        }
        line_options = {
        }
        # --- Select the proper data:
        if data == 'sum':
            y = self.sum_of_roi
            if 'ylabel' not in ax_params:
                if normalised:
                    ax_params['ylabel'] = 'Counts [a.u.]'
                else:
                    ax_params['ylabel'] = 'Counts'
        elif data == 'std':
            y = self.std_of_roi
            if 'ylabel' not in ax_params:
                if normalised:
                    ax_params['ylabel'] = '$\\sigma [a.u.]$'
                else:
                    ax_params['ylabel'] = '$\\sigma$'
        else:
            y = self.mean_of_roi
            if 'ylabel' not in ax_params:
                if normalised:
                    ax_params['ylabel'] = 'Mean [a.u.]'
                else:
                    ax_params['ylabel'] = 'Mean'
        # --- Normalize the data:
        if correct_baseline == 'end':
            y += -y[-15:-2].mean()
        elif correct_baseline == 'ini':
            y += -y[3:8].mean()

        if normalised:
            y /= y.max()

        # create and plot the figure
        if ax is None:
            fig, ax = plt.subplots()
        line_options.update(line_params)
        ax.plot(self.time_base, y, **line_options)
        ax_options.update(ax_params)
        ax = ssplt.axis_beauty(ax, ax_options)
        plt.tight_layout()
        return ax

    def plot_all(self, ax_par: dict = {}, line_par: dict = {}):
        """
        Plot the sum time trace, the average timetrace and the std ones

        Jose Rueda: jrrueda@us.es

        Plot the sum, std and average of the roi
        @param options: Dictionary containing the options for the axis_beauty
        function. Notice, the y and x label are fixed, if present in the
        options, they will be ignored
        @return fig_tt: figure where the time trace has been plotted
        @return axes: list of axes where the lines have been plotted
        """
        # Initialise the options for the plotting
        ax_options = {
            'fontsize': 16.0,
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
        ax_tt1.plot(self.time_base, self.sum_of_roi, **line_options)
        ax_options['ylabel'] = 'Counts'
        ax_tt1 = ssplt.axis_beauty(ax_tt1, ax_options)

        # plot the mean of the counts in the roi
        ax_tt2.plot(self.time_base, self.mean_of_roi, **line_options)
        ax_options['ylabel'] = 'Mean'
        ax_tt2 = ssplt.axis_beauty(ax_tt2, ax_options)

        # plot the std of the counts in the roi
        ax_tt3.plot(self.time_base, self.std_of_roi, **line_options)
        ax_options['xlabel'] = 't [s]'
        ax_options['ylabel'] = '$\\sigma$'
        ax_tt3 = ssplt.axis_beauty(ax_tt3, ax_options)
        plt.tight_layout()
        plt.show()

        return fig_tt, [ax_tt1, ax_tt2, ax_tt3]

    def plot_fft(self, options: dict = {}):
        """
        Plot the fft of the TimeTrace

        Jose Rueda: jrrueda@us.es

        @param options: options for the axis_beauty method
        @return fig: figure where the fft is plotted
        @return ax: axes where the fft is plotted
        """
        if 'fontsize' not in options:
            options['fontsize'] = 16.0
        if 'grid' not in options:
            options['grid'] = 'both'
        if 'xlabel' not in options:
            options['xlabel'] = 'Frequency [Hz]'
        if 'ylabel' not in options:
            options['ylabel'] = 'Amplitude'
        line_options = {'linewidth': 2, 'color': 'r'}

        fig, ax = plt.subplots()
        ax.plot(self.fft['faxis'], abs(self.fft['data']), **line_options)
        ax = ssplt.axis_beauty(ax, options)
        plt.show()
        return fig, ax

    def plot_spectrogram(self, options: dict = {}):
        """
        Plot the spectrogram

        Jose Rueda: jrrueda@us.es

        @param options: options for the axis_beauty method
        @return fig: figure where the fft is plotted
        @return ax: axes where the fft is plotted
        """
        if 'fontsize' not in options:
            options['fontsize'] = 16.0
        if 'grid' not in options:
            options['grid'] = 'both'
        if 'ylabel' not in options:
            options['ylabel'] = 'Frequency [Hz]'
        if 'xlabel' not in options:
            options['xlabel'] = 'Time [s]'

        fig, ax = plt.subplots()
        cmap = ssplt.Gamma_II()
        ax.pcolormesh(self.spec['taxis'], self.spec['faxis'],
                      self.spec['data'], shading='gouraud', cmap=cmap)
        ax = ssplt.axis_beauty(ax, options)
        plt.show()
        return fig, ax
