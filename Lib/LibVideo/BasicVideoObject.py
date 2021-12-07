"""
Contain the Basic Video Object (BVO)

Jose Rueda Rueda: jrrueda@us.es

Contain the main class of the video object from where the other video object
will be derived. Each of these derived video object will contain the individual
routines to remap iHIBP, FILD or INPA data
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm                      # For waitbars
from scipy import ndimage                  # To filter the images
from Lib.LibIO import ask_to_open
import Lib.LibVideo.CinFiles as cin
import Lib.LibVideo.PNGfiles as png
import Lib.LibVideo.MP4files as mp4
import Lib.LibVideo.AuxFunctions as aux
import Lib.LibData as ssdat
import Lib.LibUtilities as ssutilities
import Lib.LibPlotting as ssplt


class BVO:
    """
    Basic Video Class.

    Just read the frames and filter them
    """

    def __init__(self, file: str = None, shot: int = None):
        """
        Initialise the class

        @param file: For the initialization, file (full path) to be loaded,
        if the path point to a .cin or .mp4 file, the .cin file will be loaded.
        If the path points to a folder, the program will look for png files or
        tiff files inside (tiff coming soon). You can also point to a png or
        tiff file. In this case, the folder name will be deduced from the file.
        If none, a window will be open to select a file
        @param shot: Shot number, if is not given, the program will look for it
        in the name of the loaded file

        Note: The shot parameter is important for latter when loading data from
        the database to remap etc
        """
        # If no file was given, open a graphical user interface to select it.
        if file is None:
            filename = ask_to_open()
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
        ## Time traces: space reservation for the future
        self.time_trace = None
        ## Shot number
        self.shot = shot
        if shot is None:
            self.shot = aux.guess_shot(file, ssdat.shot_number_length)

        # Fill the object depending if we have a .cin file or not
        print('Looking for the file: ', file)
        if os.path.isfile(file):
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
            elif file[-4:] == '.mp4':
                dummy = mp4.read_file(file, verbose=False)
                # mp4 files are handle different as they are suppose to be just
                # a dummy temporal format, not the one to save our real exp
                # data, so the video will be loaded all from here and not
                # reading specific frame will be used
                self.timebase = dummy['tframes']
                self.exp_dat['frames'] = dummy['frames']
                self.exp_dat['tframes'] = dummy['tframes']
                self.exp_dat['nframes'] = np.arange(dummy['nf'])
                self.type_of_file = '.mp4'
            else:
                raise Exception('Not recognised file extension')
        else:
            if not os.path.isdir(file):
                raise Exception(file + ' not found')
            ## path to the file
            self.path = file
            # Do a quick run for the folder looking of .tiff or .png files
            f = []
            for (dirpath, dirnames, filenames) in os.walk(self.path):
                f.extend(filenames)
                break

            for i in range(len(f)):
                if f[i].endswith('.png') == '.png':
                    self.type_of_file = '.png'
                    print('Found PNG files!')
                    break
                elif f[i].endswith('.tif') == '.tif':
                    self.type_of_file = '.tif'
                    print('Found tif files!')
                    print('Tif support still not implemented, sorry')
                    break
            # if we do not have .png or tiff, give an error
            if self.type_of_file != '.png' and self.type_of_file != '.tif':
                raise Exception('No .pgn ror .tiff files found')

            # If we have a .png file, a .txt must be present with the
            # information of the exposure time and from a basic frame we can
            # load the file size
            if self.type_of_file == '.png':
                self.header, self.imageheader, self.settings,\
                    self.timebase = png.read_data(self.path)
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
        corrected frames from the video object :-)
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
                    self.exp_dat['nframes'] = frames_number
                    self.exp_dat['dtype'] = self.exp_dat['frames'].dtype
                else:
                    M = png.read_frame(self, frames_number,
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
        print('.--. ... ..-. -')
        print('Substracting noise')
        if t1 > t2:
            print('t1: ', t1)
            print('t2: ', t2)
            raise Exception('t1 is larger than t2!!!')
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
                print('Initial time loaded: ', t1_vid)
                print('Initial time requested for noise substraction: ', t1)
                t1 = t1_vid
                warnings.warn('Taking ' + str(t1_vid) + 'as initial point',
                              category=UserWarning)
            if t2 > t2_vid:
                print('Final time loaded: ', t2_vid)
                print('Final time requested for noise substraction: ', t2)
                t2 = t2_vid
                warnings.warn('Taking ' + str(t2_vid) + 'as final point',
                              category=UserWarning)
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
        frame = frame.astype(np.float)  # Get the average as float to later
        #                                 subtract and not have issues with < 0
        self.exp_dat['frames'] = (self.exp_dat['frames'].astype(np.float)
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
