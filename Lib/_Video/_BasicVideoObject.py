"""
Contain the Basic Video Object (BVO)

Jose Rueda Rueda: jrrueda@us.es

Contain the main class of the video object from where the other video object
will be derived. Each of these derived video object will contain the individual
routines to remap iHIBP, FILD or INPA data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm                      # For waitbars
from scipy import ndimage                  # To filter the images
import Lib._IO as _ssio
import Lib._Video._CinFiles as cin
import Lib._Video._PNGfiles as png
import Lib._Video._PCOfiles as pco
import Lib._Video._MP4files as mp4
import Lib._Video._TIFfiles as tif
import Lib._Video._AuxFunctions as aux
import Lib.LibData as ssdat
import Lib._Utilities as ssutilities
import Lib._Plotting as ssplt
import Lib._TimeTrace as sstt
import Lib._GUIs as ssGUI
import Lib.errors as errors
import tkinter as tk
import logging
logger = logging.getLogger('ScintSuite.Video')


class BVO:
    """
    Basic Video Class.

    Parent class for INPA, FILD, VRT and iHIBP videos. Allows to read the
    frames, filter them an perform the basic plotting

    Public Methods:
        - read_frame: Load a given range of frames
        - subtract noise: Use a range of times to average de noise and subtract
            it to all frames
        - filter_frames: apply filters such as median, gaussian, etc
        - cut frames: cut the frames
        - average_frames: average frames under certain windows
        - generate_average_window: generate the windows to average the frames
        - return to the original frames: remove the noise subtraction etc
        - plot_number_saturated_counts: plot the total number of saturated
            counts in each frame
        - GUI_frames: display a GUI to explore the video
        - getFrameIndex: get the frame number associated to a given time
        - getFrame: return the frame associated to a given time
        - plot_ frame: plot a given frame

    Public Properties:
        - size: Size (number of elements) of the video
        - shape: npixelX, npixelY, nframes
    """

    def __init__(self, file: str = None, shot: int = None,
                 empty: bool = False, adfreq: float = None,
                 t_trig: float = None):
        """
        Initialise the class

        @param file: For the initialization, file (full path) to be loaded,
            if the path point to a .cin or .mp4 file, the .cin file will be
            loaded. If the path points to a folder, the program will look for
            png files or tiff files inside (tiff coming soon). You can also
            point to a png or tiff file. In this case, the folder name will be
            deduced from the file. If none, a window will be open to select
            a file
        @param shot: Shot number, if is not given, the program will look for it
            in the name of the loaded file
        @param empty: if true, just an empty video object will be created, this
            is to latter use routines of the child objects such as load_remap.

        Note: The shot parameter is important for latter when loading data from
        the database to remap, etc. See FILDVideoObject to have an examples of
        more details.
        """
        # If no file was given, open a graphical user interface to select it.
        if (file is None) and (not empty):
            filename = _ssio.ask_to_open()
            if filename == '':
                raise Exception('You must select a file!!!')
            # If we select a png or tif, we need the folder, not the file
            if filename.endswith('.png') or filename.endswith('.tif'):
                file, name_png = os.path.split(filename)
            else:
                file = filename
        # Initialise some variables
        ## Type of video
        self.type_of_file = None
        ## Loaded experimental data
        self.exp_dat = {
            'frames': None,   # Loaded frames
            'tframes': None,  # Timebase of the loaded frames
            'nframes': None,  # Frame numbers of the loaded frames
        }
        ## Remapped data
        self.remap_dat = None
        ## Averaged data
        self.avg_dat = None
        ## Shot number
        self.shot = shot
        if not empty:
            if shot is None:
                self.shot = aux.guess_shot(file, ssdat.shot_number_length)
            # Fill the object depending if we have a .cin file or not
            if os.path.isfile(file):
                print('Looking for the file: ', file)
                ## Path to the file and filename
                self.path, self.file_name = os.path.split(file)
                ## Name of the file (full path)
                self.file = file

                # Check if the file is actually a .cin file
                if file.endswith('.cin') or file.endswith('.cine'):
                    ## Header dictionary with the file info
                    self.header = cin.read_header(file)
                    ## Settings dictionary
                    self.settings = cin.read_settings(file,
                                                      self.header['OffSetup'])
                    ## Image Header dictionary
                    self.imageheader = cin.read_image_header(file, self.header[
                        'OffImageHeader'])
                    ## Time array
                    self.timebase = cin.read_time_base(file, self.header,
                                                       self.settings)
                    self.type_of_file = '.cin'
                elif file.endswith('.png') or file.endswith('.tif'):
                    file, name = os.path.split(file)
                elif file.endswith('.mp4'):
                    dummy = mp4.read_file(file, verbose=False)
                    # mp4 files are handle different as they are suppose to be
                    # just a dummy temporal format, not the one to save our
                    # real exp data, so the video will be loaded all from here
                    # and not reading specific frame will be used
                    self.timebase = dummy['tframes']
                    self.exp_dat['frames'] = dummy['frames']
                    self.exp_dat['tframes'] = dummy['tframes']
                    self.exp_dat['nframes'] = np.arange(dummy['nf'])
                    self.type_of_file = '.mp4'
                else:
                    raise Exception('Not recognised file extension')
            else:
                print('Looking in the folder: ', file)
                if not os.path.isdir(file):
                    raise Exception(file + ' not found')
                ## path to the file
                self.path = file
                # Do a quick run for the folder looking of .tiff or .png files
                f = []
                for (dirpath, dirnames, filenames) in os.walk(self.path):
                    f.extend(filenames)
                    break

                # To stablish the format, count to 3 to make sure there are not
                # other types of files randomly inserted in the same folder
                # that mislead the type_of_file
                count_png = 0
                count_tif = 0
                count_pco = 0
                for i in range(len(f)):
                    if f[i].endswith('.png'):
                        count_png += 1
                        if count_png == 3:
                            self.type_of_file = '.png'
                            print('Found PNG files!')
                            break
                    elif f[i].endswith('.tif'):
                        count_tif += 1
                        if count_tif == 3:
                            self.type_of_file = '.tif'
                            print('Found tif files!')
                            break
                    elif f[i].endswith('.b16'):
                        count_pco += 1
                        if count_pco == 3:
                            self.type_of_file = '.b16'
                            print('Found PCO files!')
                            break
                    # if we do not have .png or tiff, give an error
                supported_type = ['.png', '.tif', '.b16']
                if self.type_of_file not in supported_type:
                    print(self.type_of_file)
                    raise Exception('No .pgn, .tiff nor .b16 files found')

                # If we have a .png file, a .txt must be present with the
                # information of the exposure time and from a basic frame we
                # can load the file size
                if self.type_of_file == '.png':
                    self.header, self.imageheader, self.settings,\
                        self.timebase = png.read_data(self.path)
                elif self.type_of_file == '.b16':
                    self.header, self.imageheader, self.settings,\
                        self.timebase = pco.read_data(
                            self.path, adfreq, t_trig)
                elif self.type_of_file == '.tif':
                    self.header, self.imageheader, self.settings,\
                        self.timebase = tif.read_data(self.path)
            if self.type_of_file is None:
                raise Exception('Not file found!')

    def read_frame(self, frames_number=None, limitation: bool = True,
                   limit: int = 2048, internal: bool = True, t1: float = None,
                   t2: float = None, threshold_saturation=0.95,
                   read_from_loaded: bool = False, verbose: bool = True):
        """
        Call the read_frame function.

        Just a wrapper to call the read_frame function, depending of the
        format in which the experimental data has been recorded

        @param frames_number: np array with the frame numbers to load
        @param limitation: maximum size allowed to the output variable,
        in Mbytes, to avoid overloading the memory trying to load the whole
        video of 100 Gb
        @param limit: bool flag to decide if we apply the limitation or if we
        operate in YOLO mode
        @param internal: If True, the frames will be stored in the 'frames'
        variable of the video object. Else, it will be returned just as output
        (usefull if you need to load another frame and you do not want to
        overwrite your frames already loaded)
        @param t1: Initial time to load frames (alternative to frames number)
        @param t2: Final time to load frames (alternative to frames number), if
        just t1 is given , only one frame will be loaded
        @param read_from_loaded: If true, it will return the frame closer to t1
        , independently of the flag 'internal' (usefull to extract noise
        corrected frames from the video object :-))
        @param verbose: flag to print the numer of saturated frames found

        @return M: 3D numpy array with the frames M[px,py,nframes] (if the
            internal flag is set to false)
        """
        # --- Select frames to load
        if not read_from_loaded:
            if (frames_number is not None) and (t1 is not None):
                raise Exception('You cannot give frames number and time')
            elif (t1 is not None) and (t2 is None):
                frames_number = np.array([np.argmin(abs(self.timebase-t1))])
            elif (t1 is not None) and (t2 is not None):
                tmin_video = self.timebase.min()
                if t1 < tmin_video:
                    text = 'T1 was not in the video file:' +\
                        'Taking %.3f as initial point' % tmin_video
                    logger.warninig('8: %s' % text)
                tmax_video = self.timebase.max()
                if t2 > tmax_video:
                    text = 'T2 was not in the video file:' +\
                        'Taking %.3f as initial point' % tmax_video
                    logger.warninig('8: %s' % text)
                it1 = np.argmin(abs(self.timebase-t1))
                it2 = np.argmin(abs(self.timebase-t2))
                frames_number = np.arange(start=it1, stop=it2+1, step=1)
            # else:
            #     raise Exception('Something went wrong, check inputs')

            if self.type_of_file == '.cin':
                if internal:
                    self.exp_dat['frames'] = \
                        cin.read_frame(self, frames_number,
                                       limitation=limitation, limit=limit)
                    self.exp_dat['tframes'] = self.timebase[frames_number]
                    self.exp_dat['nframes'] = frames_number
                    self.exp_dat['dtype'] = self.exp_dat['frames'].dtype
                else:
                    M = cin.read_frame(self, frames_number,
                                       limitation=limitation, limit=limit)
                    return M.squeeze()
            elif self.type_of_file == '.png':
                if internal:
                    self.exp_dat['frames'] = \
                        png.read_frame(self, frames_number,
                                       limitation=limitation, limit=limit)
                    self.exp_dat['tframes'] = \
                        self.timebase[frames_number].flatten()
                    if frames_number is None:
                        nx, ny, nf = self.exp_dat['frames'].shape
                        frames_number = np.arange(nf) + 1
                    self.exp_dat['nframes'] = frames_number
                    self.exp_dat['dtype'] = self.exp_dat['frames'].dtype
                else:
                    M = png.read_frame(self, frames_number,
                                       limitation=limitation, limit=limit)
                    return M.squeeze()
            elif self.type_of_file == '.tif':
                if internal:
                    self.exp_dat['frames'] = \
                        tif.read_frame(self, frames_number,
                                       limitation=limitation, limit=limit)
                    self.exp_dat['tframes'] = \
                        self.timebase[frames_number].flatten()
                    if frames_number is None:
                        nx, ny, nf = self.exp_dat['frames'].shape
                        frames_number = np.arange(nf) + 1
                    self.exp_dat['nframes'] = frames_number
                    self.exp_dat['dtype'] = self.exp_dat['frames'].dtype
                else:
                    M = tif.read_frame(self, frames_number,
                                       limitation=limitation, limit=limit)
                    return M.squeeze()
            elif self.type_of_file == '.b16':
                if internal:
                    try:
                        self.exp_dat['frames'] = \
                            pco.read_frame(self, frames_number,
                                           limitation=limitation, limit=limit)
                    except TypeError:
                        raise Exception('Please insert frame number as array')
                    self.exp_dat['tframes'] = \
                        self.timebase[frames_number].flatten()
                    if frames_number is None:
                        nx, ny, nf = self.exp_dat['frames'].shape
                        frames_number = np.arange(nf) + 1
                    self.exp_dat['nframes'] = frames_number
                    self.exp_dat['dtype'] = self.exp_dat['frames'].dtype
                else:
                    M = tif.read_frame(self, frames_number,
                                       limitation=limitation, limit=limit)
                    return M.squeeze()
            else:
                raise Exception('Not initialised / not implemented file type?')
            # Count saturated pixels
            max_scale_frames = 2 ** self.settings['RealBPP'] - 1
            threshold = threshold_saturation * max_scale_frames
            print('Counting "saturated" pixels')
            print('The threshold is set to: ', threshold, ' counts')
            n_pixels_saturated = \
                np.sum(self.exp_dat['frames'] >= threshold, axis=(0, 1))
            self.exp_dat['n_pixels_gt_threshold'] = \
                n_pixels_saturated.astype('int32')
            self.exp_dat['threshold_for_counts'] = threshold_saturation
            if verbose:
                print('Maximum number of saturated pixels in a frame: '
                      + str(self.exp_dat['n_pixels_gt_threshold'].max()))
        else:
            it = np.argmin(abs(self.exp_dat['tframes'] - t1))
            M = self.exp_dat['frames'][:, :, it].squeeze()
            return M
        return

    def subtract_noise(self, t1: float = None, t2: float = None, frame=None,
                       flag_copy: bool = False):
        """
        Subtract noise from camera frames.

        Jose Rueda: jrrueda@us.es

        This function subtract the noise from the experimental camera frames.
        Two main ways exist: if t1 and t2 are provided, the noise will be
        be considered as the average in this range. If 'frame' is given,
        the noise to be subtracted will be considered to be directly 'frame'

        A new variable: 'original frames' will be created, where the original
        frames will be loaded, in case one wants to revert the noise
        subtraction

        @param t1: Minimum time to average the noise
        @param t2: Maximum time to average the noise
        @param frame: Optional, frame containing the noise to be subtracted
        @param flag_copy: If true, a copy of the frame will be stored
        @param  return_noise: If True, the average frame used for the noise
        will be returned
        """
        if self.exp_dat is None:
            raise errors.NoFramesLoaded('Load the frames first')
        print('.--. ... ..-. -')
        print('Substracting noise')
        if t1 > t2:
            print('t1: ', t1)
            print('t2: ', t2)
            raise errors.NotValidInput('t1 is larger than t2!!!')
        # Get shape and data type of the experimental data
        nx, ny, nt = self.exp_dat['frames'].shape
        original_dtype = self.exp_dat['frames'].dtype
        # Get the initial and final time loaded in the video:
        t1_vid = self.exp_dat['tframes'][0]
        t2_vid = self.exp_dat['tframes'][-1]
        # Calculate the noise frame, if needed:
        if (t1 is not None) and (t2 is not None):
            if (t1 < t1_vid and t2 < t1_vid) or (t1 > t2_vid and t2 > t2_vid):
                raise Exception('Requested interval does not overlap with'
                                + ' the loaded time interval')
            if t1 < t1_vid:
                text = 'Initial time loaded: %5.3f \n' % t1_vid +\
                    'Initial time requested for noise substraction: %5.3f \n' \
                    % t1 +\
                    'Taking %5.3f as initial point' % t1
                t1 = t1_vid
                logger.warning('8: %s' % text)
            if t2 > t2_vid:
                text = 'Final time loaded: %5.3f \n' % t2_vid +\
                    'Final time requested for noise substraction: %5.3f \n' \
                    % t2 +\
                    'Taking %5.3f as finaal point' % t2
                logger.warning('8: %s' % text)
                t2 = t2_vid
            it1 = np.argmin(abs(self.exp_dat['tframes'] - t1))
            it2 = np.argmin(abs(self.exp_dat['tframes'] - t2))
            self.exp_dat['t1_noise'] = t1
            self.exp_dat['t2_noise'] = t2
            print('Using frames from the video')
            print(str(it2 - it1 + 1), ' frames will be used to average noise')
            frame = np.mean(self.exp_dat['frames'][:, :, it1:(it2 + 1)],
                            dtype=original_dtype, axis=2)
            self.exp_dat['frame_noise'] = frame
        else:
            print('Using noise frame provider by the user')
            nxf, nyf = frame.shape
            if (nxf != nx) or (nyf != ny):
                raise Exception('The noise frame has not the correct shape')
            self.exp_dat['frame_noise'] = frame
        # Create the original frame array:
        if 'original_frames' not in self.exp_dat and flag_copy:
            self.exp_dat['original_frames'] = self.exp_dat['frames'].copy()
        # Subtract the noise
        frame = frame.astype(float)  # Get the average as float to later
        #                                 subtract and not have issues with < 0
        self.exp_dat['frames'] = (self.exp_dat['frames'].astype(float)
                                  - frame[..., None])
        self.exp_dat['frames'][self.exp_dat['frames'] < 0] = 0.0
        self.exp_dat['frames'] = self.exp_dat['frames'].astype(original_dtype)
        print('-... -.-- . / -... -.-- .')

        return frame.astype(original_dtype)

    def filter_frames(self, method: str = 'median', options: dict = {},
                      flag_copy: bool = False):
        """
        Filter the camera frames

        @param method: method to be used:
            -# jrr: neutron method of the extra package (not recomended,
                extremelly slow)
            -# median: median filter from the scipy.ndimage package
            -# gaussian: gaussian filter from the scipy.ndimage package
        @param options: options for the desired filter (dictionary), defaults:
            -# jrr:
                nsigma: 3 Number of sigmas to consider a pixel as neutron
            -# median:
                size: 4, number of pixels considered
        @param make copy: flag to copy or not the original frames
        """
        print('Filtering frames')
        # default options:
        jrr_options = {
            'nsigma': 3
        }
        median_options = {
            'size': 2
        }
        gaussian_options = {
            'sigma': 1
        }
        if ('original_frames' not in self.exp_dat) and flag_copy:
            self.exp_dat['original_frames'] = self.exp_dat['frames'].copy()
        else:
            print('Not making a copy')
        # Filter frames
        nx, ny, nt = self.exp_dat['frames'].shape
        if method == 'jrr':
            print('Removing pixels affected by neutrons')
            jrr_options.update(options)
            for i in tqdm(range(nt)):
                self.exp_dat['frames'][:, :, i] = \
                    ssutilities.neutron_filter(self.exp_dat['frames'][:, :, i],
                                               **jrr_options)
        elif method == 'median':
            print('Median filter selected!')
            # if footprint is present in the options given by user, delete size
            # from the default options, to avoid issues in the median filter
            if 'footprint' in options:
                median_options['size'] = None
            # Now update the options
            median_options.update(options)
            for i in tqdm(range(nt)):
                self.exp_dat['frames'][:, :, i] = \
                    ndimage.median_filter(self.exp_dat['frames'][:, :, i],
                                          **median_options)
        elif method == 'gaussian':
            print('Gaussian filter selected!')
            gaussian_options.update(options)
            for i in tqdm(range(nt)):
                self.exp_dat['frames'][:, :, i] = \
                    ndimage.gaussian_filter(self.exp_dat['frames'][:, :, i],
                                            **gaussian_options)
        print('-... -.-- . / -... -.-- .')
        return

    def cut_frames(self, px_min: int, px_max: int, py_min: int, py_max: int,
                   flag_copy: bool = False):
        """
        Cut the frames in the box: [px_min, px_max, py_min, py_max]

        Jose Rueda: jrrueda@us.es

        @param px_min: min pixel in the x direction to cut
        @param px_max: max pixel in the x direction to cut
        @param px_min: min pixel in the x direction to cut
        @param px_max: max pixel in the x direction to cut
        @param make flag_copy: flag to copy or not the original frames

        Note exp_dat['frames'] are repaced for these cut ones
        """
        if ('original_frames' not in self.exp_dat) and flag_copy:
            self.exp_dat['original_frames'] = self.exp_dat['frames'].copy()
        else:
            print('Not making a copy of the original frames')
        frames = \
            self.exp_dat['frames'][px_min:(px_max+1), py_min:(py_max+1), :]
        self.exp_dat['frames'] = frames
        return

    def average_frames(self, window):
        """
        Average the frames on a given time window

        Jose Rueda: jrrueda@us.es

        The window is generated by the method 'generate_average_window'
        """
        nw, dummy = window.shape
        nx, ny, nt = self.exp_dat['frames'].shape
        frames = np.zeros((nx, ny, nw))
        time = np.zeros(nw)
        for i in range(nw):
            flags = (self.exp_dat['tframes'] >= window[i, 0])\
                * (self.exp_dat['tframes'] < window[i, 1])
            frames[..., i] = self.exp_dat['frames'][..., flags].mean(axis=-1)
            time[i] = 0.5 * (window[i, 0] + window[i, 1])
        self.avg_dat = {
            'tframes': time,
            'frames': frames,
            'nframes': time.size
        }

    def generate_average_window(self, step: float = None, trace: float = None):
        """
        Generate the windows to average the frame

        Jose Rueda Rueda: jrrueda@us.es

        @param step: width of the averaged window
        @param trace: dictionary containing the timetrace to plot and use to
            create the windows (see below). It should contain 't', the timebase
            and 'data' the 1 dimensional trace to plot

        Note:
            - If step is not None, the windows will have a width of 'step',
            ranging from the first time point loaded from the video to the
            last time point loaded on it
            - If step is None and trace is not none, the timetrace contained
            in the dictionary will be plotted and the user will be able to
            select as many points as wanted. The windows will be created
            between the point t_i and t_[i+1]. Notice that the first window
            will alway be between the first point loaded of the video and the
            first point of the database. The last window between the last point
            of the database and the last of the video
        """
        if step is not None:
            dummy = np.arange(self.exp_dat['tframes'][0],
                              self.exp_dat['tframes'][-1] + step, step)
            window = np.zeros((dummy.size-1, 2))
            window[:, 0] = dummy[:-1]
            window[:, 1] = dummy[1:]
        else:
            # Plot the trace to select the desired time points
            fig, ax = plt.subplots()
            ax.plot(trace['t'], trace['data'])
            plt.axvline(x=self.exp_dat['tframes'][0], color='k')
            plt.axvline(x=self.exp_dat['tframes'][-1], color='k')
            points = plt.ginput(-1)
            points = np.array(points)[:, 0]
            # Now fill the windows
            window = np.zeros((points.size + 1, 2))
            window[0, 0] = self.exp_dat['tframes'][0]
            window[1:, 0] = points
            window[-1, 1] = self.exp_dat['tframes'][-1]
            window[:-1, 1] = points
        return window

    def return_to_original_frames(self):
        """
        Place in self.exp_dat['frames'] the real experimental frames

        Jose Rueda: jrrueda@us.es

        Useful if some operation was performed and we want to place
        again the original frames at self.exp_dat['frames']
        """
        if 'original_frames' not in self.exp_dat:
            raise Exception('A copy of the original frames was not found!')
        else:
            self.exp_dat['frames'] = self.exp_dat['original_frames'].copy()
        return

    def plot_number_saturated_counts(self, ax_params: dict = {},
                                     line_params: dict = {},
                                     threshold=None,
                                     ax=None):
        """
        Plot the nuber of camera pixels larger than a given threshold.

        Jose Rueda: jrrueda@us.es

        @param ax_params: ax param for the axis_beauty
        @param line_params: line parameters
        @param threshold: If none, it will plot the data calculated when
        reading the camera frames (by the function self.read_frames) if it is
        a value [0,1] it willrecalculate this number
        @param ax: axis where to plot, if none, a new figure will pop-up
        """
        # Default plot parameters:
        ax_options = {
            'grid': 'both',
            'xlabel': 'Time [s]',
            'ylabel': '# saturated pixels',
            'yscale': 'log'
        }
        ax_options.update(ax_params)
        line_options = {
            'linewidth': 1.0
        }
        line_options.update(line_params)
        # Select x,y data
        x = self.exp_dat['tframes']
        if threshold is None:
            y = self.exp_dat['n_pixels_gt_threshold']
            print('Threshold was set to: ',
                  self.exp_dat['threshold_for_counts'] * 100, '%')
        else:
            max_scale_frames = 2 ** self.settings['RealBPP'] - 1
            thres = threshold * max_scale_frames
            print('Counting "saturated" pixels')
            print('The threshold is set to: ', thres, ' counts')
            n_pixels_saturated = \
                np.sum(self.exp_dat['frames'] >= thres, axis=(0, 1))
            y = n_pixels_saturated.astype('int32')
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, y, **line_options)  # Plot the data
        # Plot the maximum posible (the number of pixels)
        npixels = self.imageheader['biWidth'] * self.imageheader['biHeight']
        ax.plot([x[0], x[-1]], [npixels, npixels], '--',
                **line_options)
        ax = ssplt.axis_beauty(ax, ax_options)
        return ax

    @property
    def size(self):
        """Get the size of the loaded frames."""
        return self.exp_dat['frames'].size

    @property
    def shape(self):
        """Shape of the loaded frames"""
        return self.exp_dat['frames'].shape

    def GUI_frames(self, flagAverage: bool = False):
        """Small GUI to explore camera frames"""
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print('--. ..- ..')
        print(text)
        print('-... . ..- - -.--')
        root = tk.Tk()
        root.resizable(height=None, width=None)
        if flagAverage:
            ssGUI.ApplicationShowVid(root, self.avg_dat, self.remap_dat)
        else:
            ssGUI.ApplicationShowVid(root, self.exp_dat, self.remap_dat)
        root.mainloop()
        root.destroy()

    def getFrameIndex(self, t: float = None, frame_number: int = None,
                      flagAverage: bool = False):
        """
        Return the index of the frame inside the loaded frames

        Jose Rueda Rueda: jrrueda@us.es

        @param t: desired time (in the same units of the video database)
        @param frame_index: frame index to load, relative to the camera,
            ignored if time is present
        @param flagAverage: flag to look at the average o raw frames

        @return it: index of the frame in the loaded array
        """
        it = None
        if t is not None:
            if not flagAverage:
                it = np.argmin(np.abs(self.timebase - t))
            else:
                it = np.argmin(np.abs(self.avg_dat - t))
        else:
            if not flagAverage:
                it = np.where(self.exp_dat['nframes'] == frame_number)[0]
            else:
                it = np.where(self.avg_dat['nframes'] == frame_number)[0]
        return it

    def getFrame(self, t: float, flagAverage: bool = False):
        """
        Return the frame of the closest in time to the desired time

        Jose Rueda Rueda: jrrueda@us.es

        @param t: desired time (in the same units of the video database)
        @param flagAverage: flag to look at the average or raw frames

        @return it: index of the frame in the loaded array
        """
        it = self.getFrameIndex(t)
        if not flagAverage:
            frame = self.exp_dat['frames'][..., it].squeeze()
        else:
            frame = self.avg_dat['frames'][..., it].squeeze()

        return frame

    def getTime(self, it: int, flagAverage: bool = False):
        """
        Get the time corresponding to a loaded frame number

        Jose Rueda: jrrueda@us.es

        @param it: frame number (relative to the loaded frames)
        @param flagAverage: flag to look at the averaged or raw frames
        """
        if not flagAverage:
            t = float(self.exp_dat['tframes'][it])
        else:
            t = float(self.avg_dat['tframes'][it])
        return t

    def plot_frame(self, frame_number=None, ax=None, ccmap=None,
                   t: float = None,
                   verbose: bool = True,
                   vmin: int = 0, vmax: int = None,
                   xlim: float = None, ylim: float = None,
                   scale: str = 'linear',
                   alpha: float = 1.0, IncludeColorbar: bool = True,
                   RemoveAxisTicksLabels: bool = False,
                   flagAverage: bool = False):
        """
        Plot a frame from the loaded frames

        Jose Rueda Rueda: jrrueda@us.es

        Notice, you can plot a given frame giving its frame number or giving
        its time

        @param frame_number: Number of the frame to plot
        @param ax: Axes where to plot, is none, just a new axes will be created
        @param ccmap: colormap to be used, if none, Gamma_II from IDL
        @param verbose: If true, info of the theta and phi used will be printed
        @param vmin: Minimum value for the color scale to plot
        @param vmax: Maximum value for the color scale to plot
        @param xlim: tuple with the x-axis limits
        @param ylim: tuple with the y-axis limits
        @param scale: Scale for the plot: 'linear', 'sqrt', or 'log'
        @param alpha: transparency factor, 0.0 is 100 % transparent
        @param RemoveAxisTicksLabels: boolean flag to remove the numbers in the
            axis

        @return ax: the axes where the frame has been drawn
        """
        # --- Check inputs:
        if (frame_number is not None) and (t is not None):
            raise Exception('Do not give frame number and time!')
        if (frame_number is None) and (t is None):
            raise Exception("Didn't you want to plot something?")
        # --- Prepare the scale:
        if scale == 'sqrt':
            extra_options = {'norm': colors.PowerNorm(0.5)}
        elif scale == 'log':
            extra_options = {'norm': colors.LogNorm(clip=True)}
        else:
            extra_options = {}
        # --- Load the frames
        # If we use the frame number explicitly
        if frame_number is not None:
            frame_index = self.getFrameIndex(frame_number=frame_number,
                                             flagAverage=flagAverage)
            tf = self.getTime(frame_index, flagAverage)
            dummy = self.getFrame(tf, flagAverage)
        # If we give the time:
        if t is not None:
            frame_index = self.getFrameIndex(t, flagAverage)
            tf = self.getTime(frame_index, flagAverage)
            dummy = self.getFrame(t, flagAverage)
        # --- Check the colormap
        if ccmap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = ccmap
        # --- Check the axes to plot
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        if vmax is None:
            vmax = dummy.max()
        if scale == 'log':  # If we use log scale, just avoid zeros
            # we are here mixing a bit integers and float... but python will
            # provide
            dummy[dummy < 1.0] = 1.0e-5

        img = ax.imshow(dummy, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        alpha=alpha, **extra_options)
        # Set the axis limit
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        if IncludeColorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, label='Counts', cax=cax)
        ax.text(0.05, 0.9, '#' + str(self.shot),
                horizontalalignment='left',
                color='w', verticalalignment='bottom',
                transform=ax.transAxes)
        plt.text(0.95, 0.9, 't = ' + str(round(tf, 3)) + (' s'),
                 horizontalalignment='right',
                 color='w', verticalalignment='bottom',
                 transform=ax.transAxes)
        if RemoveAxisTicksLabels:
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
        # Shot the figure
        if created:
            fig.show()
            plt.tight_layout()
        return ax

    def getTimeTrace(self, t: float = None, mask=None):
        """
        Calculate the timeTrace of the video

        Jose Rueda Rueda: jrrueda@us.es

        @param t: time of the frame to be plotted for the selection of the roi
        @param mask: bolean mask of the ROI

        If mask is present, the t argument will be ignored

        @returns timetrace: a timetrace object
        """
        if mask is None:
            # - Plot the frame
            ax_ref = self.plot_frame(t=t)
            fig_ref = plt.gcf()
            # - Define roi
            roi = sstt.roipoly(fig_ref, ax_ref)
            # Create the mask
            mask = roi.getMask(self.exp_dat['frames'][:, :, 0].squeeze())

        return sstt.TimeTrace(self, mask)
