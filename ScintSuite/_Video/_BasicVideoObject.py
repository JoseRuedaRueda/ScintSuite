"""
Contain the Basic Video Object (BVO)

Jose Rueda Rueda: jrrueda@us.es

Contain the main class of the video object from where the other video object
will be derived. Each of these derived video object will contain the individual
routines to remap iHIBP, FILD or INPA data.
"""
import os
import f90nml
import logging
import numpy as np
import xarray as xr
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ScintSuite._IO as _ssio
import ScintSuite._GUIs as ssGUI
import ScintSuite.LibData as ssdat
import ScintSuite.errors as errors
import ScintSuite._Plotting as ssplt
import ScintSuite._TimeTrace as sstt
import ScintSuite._Video._CinFiles as cin
import ScintSuite._Video._PNGfiles as png
import ScintSuite._Video._PCOfiles as pco
import ScintSuite._Video._MP4files as mp4
import ScintSuite._Video._TIFfiles as tif
import ScintSuite._Video._NetCDF4files as ncdf
import ScintSuite._Utilities as ssutilities
import ScintSuite._Video._AuxFunctions as aux
from tqdm import tqdm                      # For waitbars
from scipy import ndimage                  # To filter the images
from ScintSuite._Paths import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, Button, RadioButtons


# --- Initialise the auxiliary objects
logger = logging.getLogger('ScintSuite.Video')


# ------------------------------------------------------------------------------
# --- Main class of the video object
# ------------------------------------------------------------------------------
class BVO:
    """
    Basic Video Class.

    Parent class for INPA, FILD, VRT and iHIBP videos. Allows to read the
    frames, filter them and perform the basic plotting

    Public Methods:
        - read_frame: Load a given range of frames
        - subtract noise: Use a range of times to average de noise and subtract
            it to all frames
        - filter_frames: apply filters such as median, gaussian, etc
        - average_frames: average frames under certain windows
        - generate_average_window: generate the windows to average the frames
        - return_to_original_frames: remove the noise subtraction etc
        - plot_number_saturated_counts: plot the total number of saturated
            counts in each frame
        - plot_ frame: plot a given frame
        - GUI_frames: display a GUI to explore the video
        - getFrameIndex: get the frame number associated to a given time
        - getFrame: return the frame associated to a given time
        - getTime: return the time associated to a frame index
        - getTimeTrace: calculate a video timetrace
        - exportVideo: save the dataframes to a netCDF

    Public Properties:
        - size: Size (number of elements) of the video
        - shape: npixelX, npixelY, nframes
    """

    def __init__(self, file: str = None, shot: int = None,
                 empty: bool = False, adfreq: float = None,
                 t_trig: float = None, YOLO: bool = False):
        """
        Initialise the class

        :param  file: For the initialization, file (full path) to be loaded,
            if the path point to a .cin, a .nc or .mp4 file, the .cin file will be
            loaded. If the path points to a folder, the program will look for
            png files or tiff files inside (tiff coming soon). You can also
            point to a png or tiff file. In this case, the folder name will be
            deduced from the file. If none, a window will be open to select
            a file
        :param  shot: Shot number, if is not given, the program will look for it
            in the name of the loaded file
        :param  empty: if true, just an empty video object will be created, this
            is to latter use routines of the child objects such as load_remap.
        :param  adfreq: acquisition frequency of the video. This is needed just
            for the video saved in .b16 format, where this information is not
            saved in the video and must be provided externally
        :param  t_trig: trigger time. Again, this is just needed for the .b16
            format
        :param  YOLO: flag to ignore wrong timed frames. With old AUG adquisition
            system, sometimes the timebase get corrupt after a given point. if
            YOLO is false, the program will interact with the user,
            shown him/her which frames are wrong and create an ad-hoc
            time base if needed. This is not ideal for the case of automatic
            remaps etc when the program is runing in the background, as the user
            is needed. YOLO=True disable this and just ignore these frames


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
        self.exp_dat = xr.Dataset()
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
                logger.info('Looking for the file: %s' % file)
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
                    self.path = file
                elif file.endswith('.nc'):
                    dummy, self.header, self.imageheader, self.settings,\
                         = ncdf.read_file_anddata(file)
                    # self.properties['width'] = dummy['width']
                    # self.properties['height'] = dummy['height']
                    # self.properties['fps'] = dummy['fps']
                    # self.properties['exposure'] = dummy['exp']
                    # self.properties['gain'] = dummy['gain']
                    self.timebase = dummy['timebase']
                    self.exp_dat = xr.Dataset()
                    nx, ny, nt = dummy['frames'].shape
                    px = np.arange(nx)
                    py = np.arange(ny)

                    self.exp_dat['frames'] = \
                        xr.DataArray(dummy['frames'], dims=('px', 'py', 't'),
                                     coords={'px': px,
                                             'py': py,
                                             't': self.timebase.squeeze()})
                    self.exp_dat['frames'] = self.exp_dat['frames']
                    self.type_of_file = '.nc'
                elif file.endswith('.mp4'):
                    if not 'properties' in self.__dict__:
                        self.properties = {}
                    dummy = mp4.read_file(file, **self.properties)

                    frames = dummy.pop('frames')
                    self.properties.update(dummy)
                    self.exp_dat = xr.Dataset()
                    nt, nx, ny = frames.shape
                    px = np.arange(nx)
                    py = np.arange(ny)

                    self.exp_dat['frames'] = \
                        xr.DataArray(frames, dims=('t', 'px', 'py'),
                                     coords={'t': self.timebase.squeeze(),
                                             'px': px,
                                             'py': py})
                    self.exp_dat['frames'] = \
                        self.exp_dat['frames'].transpose('px', 'py', 't')
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

                # To establish the format, count to 3 to make sure there are no
                # other types of files randomly inserted in the same folder
                # that mislead the type_of_file.
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
                    elif f[i].endswith('.b16'):
                        count_pco += 1
                        if count_pco == 3:
                            self.type_of_file = '.b16'
                            print('Found PCO files!')
                            break
                    elif f[i].endswith('.tif'):
                        count_tif += 1
                        if count_tif == 1:
                            self.type_of_file = '.tif'
                            print('Found tif files!')
                            break
                else:
                    raise Exception('Type of f variable not found. Please revise code.')
                 # if we do not have .png or tiff, give an error
                supported_type = ['.png', '.tif', '.b16', '.nc']
                if self.type_of_file not in supported_type:
                    print(self.type_of_file)
                    raise Exception('No .pgn, .tiff, .nc nor .b16 files found')

                # If we have a .png file, a .txt must be present with the
                # information of the exposure time and from a basic frame we
                # can load the file size
                if self.type_of_file == '.png':
                    self.header, self.imageheader, self.settings,\
                        self.timebase = png.read_data(self.path, YOLO)
                elif self.type_of_file == '.b16':
                    self.header, self.imageheader, self.settings,\
                        self.timebase = pco.read_data(
                            self.path, adfreq, t_trig)
                elif self.type_of_file == '.nc':
                    self.header, self.imageheader, self.settings,\
                        self.timebase = ncdf.read_data(
                            self.path, adfreq, t_trig)
                elif self.type_of_file == '.tif':
                    self.header, self.imageheader, self.settings,\
                        self.timebase = tif.read_data(self.path)
            if self.type_of_file is None:
                raise Exception('Not file found!')
        ## Geometry of the diagnostic head used to record the video
        self.geometryID = None
        ## Camera calibration to relate the scintillator and the camera sensor,
        # each diagnostic will read its camera calibration from its database
        self.CameraCalibration = None
        self.CameraData = None
        ## Scintillator plate:
        self.scintillator = None

    # --------------------------------------------------------------------------
    # --- Manage Frames
    # --------------------------------------------------------------------------
    def read_frame(self, frames_number=None, limitation: bool = True,
                   limit: int = 2048, internal: bool = True, t1: float = None,
                   t2: float = None, threshold_saturation: float = 0.95,
                   verbose: bool = True):
        """
        Read the video frames

        Just a wrapper to call the read_frame function, depending on the
        format in which the experimental data has been recorded

        :param  frames_number: array or list with the frame numbers to load
        :param  limitation: bool flag to decide if we apply the limitation or if
            we operate in YOLO mode
        :param  limit: maximum size allowed for the output variable,
            in Mbytes, to avoid overloading the memory trying to load the whole
            video of 100 Gb
        :param  internal: If True, the frames will be stored in the 'frames'
            variable of the video object. Else, it will be returned just as
            output (useful if you need to load another frame and you do not
            want to overwrite your frames already loaded)
        :param  t1: Initial time to load frames (alternative to frames number)
        :param  t2: Final time to load frames (alternative to frames number), if
            just t1 is given , only one frame will be loaded
        :param  verbose: flag to print the numer of saturated frames found

        :return M: 3D numpy array with the frames M[px,py,nframes] (if the
            internal flag is set to false)

        :Example:
        >>> # Load a video from a diagnostic
        >>> import Lib as ss
        >>> vid = ss.vid.INPAVideo(shot=41090)
        >>> vid.read_frame()  # To load all the video
        """
        # --- Clean video if needed
        if 't' in self.exp_dat and internal:
            self.exp_dat = xr.Dataset()
        # --- Select frames to load
        if (frames_number is not None) and (t1 is not None):
            raise errors.NotValidInput('You cannot give frames number and time')
        elif (t1 is not None) and (t2 is None):
            frames_number = np.array([np.argmin(abs(self.timebase-t1))])
        elif (t1 is not None) and (t2 is not None):
            tmin_video = self.timebase.min()
            if t1 < tmin_video:
                text = 'T1 was not in the video file:' +\
                    'Taking %.3f as initial point' % tmin_video
                logger.warning('18: %s' % text)
            tmax_video = self.timebase.max()
            if t2 > tmax_video:
                text = 'T2 was not in the video file:' +\
                    'Taking %.3f as final point' % tmax_video
                logger.warning('18: %s' % text)
            it1 = np.argmin(abs(self.timebase-t1))
            it2 = np.argmin(abs(self.timebase-t2))
            frames_number = np.arange(start=it1, stop=it2+1, step=1)
        logger.info('Reading frames: ')
        if self.type_of_file == '.cin':
            M = cin.read_frame(self, frames_number,
                               limitation=limitation, limit=limit)
        elif self.type_of_file == '.png':
            M = png.read_frame(self, frames_number,
                               limitation=limitation, limit=limit)
        elif self.type_of_file == '.tif':
            M = tif.read_frame(self, frames_number,
                               limitation=limitation, limit=limit)
        elif self.type_of_file == '.b16':
            M = pco.read_frame(self, frames_number,
                               limitation=limitation, limit=limit,
                               verbose=verbose)
        elif self.type_of_file == '.nc':
            M = ncdf.read_frame(self, frames_number,
                               limitation=limitation, limit=limit,
                               verbose=verbose)
        else:
            raise Exception('Not initialised / not implemented file type?')
        # --- End here if we just want the frame
        if not internal:
            return M
        # --- Save the stuff in the structure
        # Get the spatial axes
        nx, ny, nt = M.shape
        px = np.arange(nx)
        py = np.arange(ny)
        # Get the time axis
        tframes = self.timebase[frames_number]
        # Get the frames number
        if frames_number is None:
            nframes = np.arange(nt) + 1
        else:
            nframes = frames_number
        # Quick solve for the case we have just one frame
        try:
            if len(tframes) != 1:
                tbase = tframes.squeeze()
                nbase = nframes.squeeze()
            else:
                tbase = tframes
                nbase = nframes
        except TypeError:
            tbase = np.array([tframes])
            nbase = np.array([nframes])
        if len(tbase.shape) == 2:
            tbase = tbase.squeeze()
            nbase = nbase.squeeze()
        # Get the data-type
        dtype = M.dtype
        # Apply the neccesary transformations
        if self.CameraData is not None:
            # Crop the frames if needed
            if 'xmin' in self.CameraData:
                xmax = self.CameraData['xmax']
                xmin = self.CameraData['xmin']
                px = np.arange(xmax - xmin)
                M = M[xmin:xmax, :, :]
            if 'ymin' in self.CameraData:
                ymax = self.CameraData['ymax']
                ymin = self.CameraData['ymin']
                px = np.arange(ymax - ymin)
                M = M[:, ymin:ymax, :]
            # invert the frames if needed
            if 'invertx' in self.CameraData:
                if self.CameraData['invertx']:
                    M = M[::-1, :, :]
            if 'inverty' in self.CameraData:
                if self.CameraData['inverty']:
                    M = M[:, ::-1, :]
            # Exchange xy if needed:
            if 'exchangexy' in self.CameraData:
                if self.CameraData['exchangexy']:
                    M = M.transpose((1, 0, 2))
                    px_tmp = px.copy()
                    px = py.copy()
                    py = px_tmp

        # Storage it
        self.exp_dat['frames'] = \
            xr.DataArray(M, dims=('px', 'py', 't'), coords={'t': tbase,
                                                            'px': px, 'py': py})
        self.exp_dat['nframes'] = xr.DataArray(nbase, dims=('t'))
        self.exp_dat.attrs['dtype'] = dtype
        # --- Count saturated pixels
        max_scale_frames = 2 ** self.settings['RealBPP'] - 1
        threshold = threshold_saturation * max_scale_frames
        logger.info('Counting "saturated" pixels')
        logger.info('The threshold is set to: %f counts', threshold)
        n_pixels_saturated = \
            np.sum(self.exp_dat['frames'].values >= threshold, axis=(0, 1))
        self.exp_dat['n_pixels_gt_threshold'] = xr.DataArray(
            n_pixels_saturated.astype('int32'), dims=('t'))
        self.exp_dat.attrs['threshold_for_counts'] = threshold_saturation
        logger.info('Maximum number of saturated pixels in a frame: %f',
                    n_pixels_saturated.max())

    def subtract_noise(self, t1: float = None, t2: float = None,
                       frame: np.ndarray = None, flag_copy: bool = False):
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

        :param  t1: Minimum time to average the noise
        :param  t2: Maximum time to average the noise
        :param  frame: Optional, frame containing the noise to be subtracted
        :param  flag_copy: If true, a copy of the frame will be stored

        :return  frame: the frame used for the noise subtraction

        :Example:
        >>> # Load a video from a diagnostic
        >>> import Lib as ss
        >>> vid = ss.vid.INPAVideo(shot=41090)
        >>> vid.read_frame()  # To load all the video
        >>> # Average the frames between 0.1 and 0.2 and use them as noise frame
        >>> vid.subtract_noise(0.1, 0.2)
        """
        # --- Check the inputs
        if self.exp_dat is None:
            raise errors.NoFramesLoaded('Load the frames first')
        logger.info('.--. ... ..-. -')
        logger.info('Substracting noise')
        if frame is None:
            if t1 > t2:
                print('t1: ', t1)
                print('t2: ', t2)
                raise errors.NotValidInput('t1 is larger than t2!!!')
        # --- Get the shapes and indexes
        # Get shape and data type of the experimental data
        nx = self.exp_dat['px'].size
        ny = self.exp_dat['py'].size
        nt = self.exp_dat['t'].size
        original_dtype = self.exp_dat['frames'].dtype
        # Get the initial and final time loaded in the video:
        t1_vid = self.exp_dat['t'].values[0]
        t2_vid = self.exp_dat['t'].values[-1]
        # --- Get the nise frame
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
                logger.warning('18: %s' % text)
            if t2 > t2_vid:
                text = 'Final time loaded: %5.3f \n' % t2_vid +\
                    'Final time requested for noise substraction: %5.3f \n' \
                    % t2 +\
                    'Taking %5.3f as finaal point' % t2
                logger.warning('18: %s' % text)
                t2 = t2_vid

            it1 = np.argmin(np.abs(self.exp_dat.t.values - t1))
            it2 = np.argmin(np.abs(self.exp_dat.t.values - t2))

            logger.info('Using frames from the video')
            logger.info('%i frames will be used to average noise', it2 - it1 + 1)
            frame = self.exp_dat['frames'].isel(t=slice(it1, it2)).mean(dim='t')
            #frame = np.mean(self.exp_dat['frames'].values[:, :, it1:(it2 + 1)],
            #                dtype=original_dtype, axis=2)

        else:  # The frame is given by the user
            logger.info('Using noise frame provider by the user')
            try:
                nxf = frame.px.size
                nyf = frame.py.size
            except AttributeError:
                nxf, nyf = frame.shape
            if (nxf != nx) or (nyf != ny):
                print(nx, nxf, ny, nyf)
                text = 'The noise frame has not the correct shape'
                raise errors.NotValidInput(text)

        # Save the frame in the structure
        self.exp_dat['frame_noise'] = xr.DataArray(frame.squeeze(),
                                                   dims=('px', 'py'))
        if t1 is not None:
            self.exp_dat['frame_noise'].attrs['t1_noise'] = t1
            self.exp_dat['frame_noise'].attrs['t2_noise'] = t2
        else:
            self.exp_dat['frame_noise'].attrs['t1_noise'] = -150.0
            self.exp_dat['frame_noise'].attrs['t2_noise'] = -150.0
        # --- Copy the original frame array:
        if 'original_frames' not in self.exp_dat and flag_copy:
            self.exp_dat['original_frames'] = self.exp_dat['frames'].copy()
        # --- Subtract the noise
        frame = frame.astype(float)  # Get the average as float to later
        #                              subtract and not have issues with < 0
        frameDA = xr.DataArray(frame, dims=('px', 'py'),
                               coords = {'px': self.exp_dat['px'],
                                         'py': self.exp_dat['py']})
        #dummy = \
        #    (self.exp_dat['frames'].values.astype(float) - frame[..., None])
        dummy = self.exp_dat['frames'].astype(float) - frameDA
        dummy.values[dummy.values < 0] = 0.0  # Clean the negative values
        self.exp_dat['frames'].values = dummy.astype(original_dtype)

        logger.info('-... -.-- . / -... -.-- .')
        return frame.astype(original_dtype)

    def filter_frames(self, method: str = 'median', options: dict = {},
                      flag_copy: bool = False):
        """
        Filter the camera frames

        :param  method: method to be used:
            -# jrr: neutron method of the extra package (not recommended,
                extremelly slow)
            -# median: median filter from the scipy.ndimage package
            -# gaussian: gaussian filter from the scipy.ndimage package
        :param  options: options for the desired filter (dictionary), defaults:
            -# jrr:
                nsigma: 3 Number of sigmas to consider a pixel as neutron
            -# median:
                size: 4, number of pixels considered
        :param  flag_copy: flag to copy or not the original frames

        :Example:
        >>> # Load a video from a diagnostic
        >>> import Lib as ss
        >>> vid = ss.vid.INPAVideo(shot=41090)
        >>> vid.read_frame()  # To load all the video
        >>> # Average the frames between 0.1 and 0.2 and use them as noise frame
        >>> vid.subtract_noise(0.1, 0.2)
        >>> # Filter the frames
        >>> vid.filter_frames(method='median', options={'size': 2})
        """
        logger.info('Filtering frames')
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
            logger.info('Not making a copy')
        # Filter frames
        nx, ny, nt = self.exp_dat['frames'].shape
        if method == 'jrr':
            logger.info('Removing pixels affected by neutrons')
            jrr_options.update(options)
            for i in tqdm(range(nt)):
                self.exp_dat['frames'][:, :, i] = \
                    ssutilities.neutron_filter(self.exp_dat['frames'].values[:, :, i],
                                               **jrr_options)
        elif method == 'median':
            logger.info('Median filter selected!')
            # if footprint is present in the options given by user, delete size
            # from the default options, to avoid issues in the median filter
            if 'footprint' in options:
                median_options['size'] = None
            # Now update the options
            median_options.update(options)
            for i in tqdm(range(nt)):
                self.exp_dat['frames'][:, :, i] = \
                    ndimage.median_filter(self.exp_dat['frames'].values[:, :, i],
                                          **median_options)
        elif method == 'gaussian':
            logger.info('Gaussian filter selected!')
            gaussian_options.update(options)
            for i in tqdm(range(nt)):
                self.exp_dat['frames'][:, :, i] = \
                    ndimage.gaussian_filter(self.exp_dat['frames'].values[:, :, i],
                                            **gaussian_options)
        logger.info('\\n-... -.-- . / -... -.-- .')
        return

    def average_frames(self, window):
        """
        Average the frames on a given time window

        Averaged frames will be saved in a Dataset in the attribute: 'avg_dat'
        of the video object

        Jose Rueda: jrrueda@us.es

        The window must be generated by the method 'generate_average_window'

        :param  window: window generated by 'generate_average_window'
        """
        # --- Get the shape and allocate the variables
        nw, dummy = window.shape
        nx, ny, nt = self.exp_dat['frames'].shape
        frames = np.zeros((nx, ny, nw))
        time = np.zeros(nw)
        # --- average the frames
        for i in range(nw):
            flags = (self.exp_dat['t'].values >= window[i, 0])\
                * (self.exp_dat['t'].values < window[i, 1])
            frames[..., i] = self.exp_dat['frames'][..., flags].mean(axis=-1)
            time[i] = 0.5 * (window[i, 0] + window[i, 1])
        # --- Save the data in the dataset
        self.avg_dat = xr.Dataset()  # Initialise the dataset
        # Prepare the axis
        px = np.arange(nx)
        py = np.arange(ny)
        # Save the frames
        self.avg_dat['frames'] = \
            xr.DataArray(frames, dims=('px', 'py', 't'),
                         coords={'t': time.squeeze(), 'px': px, 'py': py})
        self.avg_dat['nframes'] = \
            xr.DataArray(np.arange(nw) + 1, dims=('t'))

    def generate_average_window(self, step: float = None, trace: float = None):
        """
        Generate the windows to average the frame

        Jose Rueda Rueda: jrrueda@us.es

        :param  step: width of the average window
        :param  trace: xr.DataArray containing the timetrace to plot and use to
            create the windows (see below). It should contain 't', as a
            coordinate and be 1D

        Note:
            - If step is not None, the windows will have a width of 'step',
            ranging from the first time point loaded from the video to the
            last time point loaded on it
            - If step is None and trace is not none, the timetrace contained
            in the dictionary will be plotted and the user will be able to
            select as many points as wanted. The windows will be created
            between the point t_i and t_[i+1]. Notice that the first window
            will always be between the first point loaded of the video and the
            first point of the database. The last window between the last point
            of the database and the last of the video
        """
        if step is not None:
            dummy = np.arange(self.exp_dat['t'].values[0],
                              self.exp_dat['t'].values[-1] + step, step)
            window = np.zeros((dummy.size-1, 2))
            window[:, 0] = dummy[:-1]
            window[:, 1] = dummy[1:]
        else:
            # Plot the trace to select the desired time points
            fig, ax = plt.subplots()
            ax.plot(trace['t'].values, trace.values)
            plt.axvline(x=self.exp_dat['t'].values[0], color='k')
            plt.axvline(x=self.exp_dat['t'].values[-1], color='k')
            ax.set_lim(self.exp_dat['t'].values[0]*0.95,
                       self.exp_dat['t'].values[-1]*1.05)
            points = plt.ginput(-1)
            points = np.array(points)[:, 0]
            # Now fill the windows
            window = np.zeros((points.size + 1, 2))
            window[0, 0] = self.exp_dat['t'].values[0]
            window[1:, 0] = points
            window[-1, 1] = self.exp_dat['t'].values[-1]
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

    # --------------------------------------------------------------------------
    # --- Plotting and GUIs
    # --------------------------------------------------------------------------
    def plot_number_saturated_counts(self, ax_params: dict = {},
                                     line_params: dict = {},
                                     threshold=None,
                                     ax=None):
        """
        Plot the nuber of camera pixels larger than a given threshold.

        Jose Rueda: jrrueda@us.es

        :param  ax_params: ax param for the axis_beauty
        :param  line_params: line parameters
        :param  threshold: If none, it will plot the data calculated when
        reading the camera frames (by the function self.read_frames) if it is
        a value [0,1] it willrecalculate this number
        :param  ax: axis where to plot, if none, a new figure will pop-up
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
        x = self.exp_dat['t'].values
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

    def plot_frame(self, frame_number: int=None, ax=None, ccmap=None,
                   t: float = None,
                   verbose: bool = True,
                   vmin: int = 0, vmax: int = None,
                   xlim: float = None, ylim: float = None,
                   scale: str = 'linear', 
                   alpha: float = 1.0, IncludeColorbar: bool = True,
                   RemoveAxisTicksLabels: bool = False,
                   flagAverage: bool = False, normalise=None, extent: float=None, tround: int = 3,
                   rotate_frame: bool = False,):
        """
        Plot a frame from the loaded frames

        Jose Rueda Rueda: jrrueda@us.es
        Hannah Lindl: hannah.lindl@ipp.mpg.de

        Notice, you can plot a given frame giving its frame number or giving
        its time

        :param frame_number: Number of the frame to plot (option 1).
               If array: will average over the given frame frange
        :param ax: Axes where to plot, is none, just a new axes will be created
        :param ccmap: colormap to be used, if none, Gamma_II from IDL
        :param t: time point to select the frame (option 2)
                  If array: will average over the given frame frange
        :param verbose: If true, info of the theta and phi used will be printed
        :param vmin: Minimum value for the color scale to plot
        :param vmax: Maximum value for the color scale to plot
        :param xlim: tuple with the x-axis limits
        :param ylim: tuple with the y-axis limits
        :param scale: Scale for the plot: 'linear', 'sqrt', or 'log'
        :param tround: Number of decimals that we will round the time value to
        :param alpha: transparency factor, 0.0 is 100 % transparent
        :param IncludeColorbar: flag to include a colorbar
        :param RemoveAxisTicksLabels: boolean flag to remove the numbers in the
            axis
        :param  flagAverage: flag to pick the axis from the experimental or the
            averaged frames
        :param  normalise: parameter to normalise the frame when plotting:
            if normalise == 1 it would be normalised to the maximum
            if normalise == <number> it would be normalised to this value
            if normalise == None, nothing will be done
        param rotate_frame: boolean flag to rotate the frame the rotation angle of the optical parameters

        :return ax: the axes where the frame has been drawn
        """
        # --- Check inputs:
        if (frame_number is not None) and (t is not None):
            raise Exception('Do not give frame number and time!')
        if (frame_number is None) and (t is None):
            raise Exception("Didn't you want to plot something?")

        # --- Load the frames
        # If we use the frame number explicitly
        if frame_number is not None:
            flag_time_range = False
            try:
                _ = len(frame_number)
            except TypeError:
                frame_index = self.getFrameIndex(frame_number=frame_number,
                                                 flagAverage=flagAverage)
                tf = self.getTime(frame_index, flagAverage)
                dummy = self.getFrame(tf, flagAverage)
            else:
                if len(frame_number) == 1:
                    frame_number = frame_number[0]
                    frame_index = self.getFrameIndex(frame_number=frame_number,
                                                     flagAverage=flagAverage)
                    tf = self.getTime(frame_index, flagAverage)
                    dummy = self.getFrame(tf, flagAverage)
                elif len(frame_number) == 2:
                    flag_time_range = True
                    frames = self.exp_dat['frames'].isel(t = slice(frame_number[0], frame_number[1]))
                    dummy = frames.mean(dim = 't')
                    t = np.zeros(2)
                    t[0] = self.getTime(self.getFrameIndex(frame_number = frame_number[0]))
                    t[1] = self.getTime(self.getFrameIndex(frame_number = frame_number[1]))
                else:
                    raise ValueError('wrong shape of framenumber. Should not be larger than two')
        # If we give the time:
        if t is not None:
            flag_time_range = False
            try:
                _ = len(t)
            except TypeError:
                frame_index = self.getFrameIndex(t, flagAverage)
                tf = self.getTime(frame_index, flagAverage)
                dummy = self.getFrame(t, flagAverage)
            else:
                if len(t) ==1:
                    frame_index = self.getFrameIndex(t, flagAverage)
                    tf = self.getTime(frame_index, flagAverage)
                    dummy = self.getFrame(t, flagAverage)
                elif len(t)==2:
                    logger.debug('Plotting averaged frames')
                    flag_time_range =True
                    frames = self.exp_dat['frames'].where((self.exp_dat['t']>t[0]) & \
                                                          (self.exp_dat['t']<t[1]),
                                                          drop = True)
                    t[0] = min(frames.t)
                    t[1] = max(frames.t)
                    dummy = frames.mean(dim = 't')
                else:
                    raise ValueError('wrong shape of time. Should not be larger than two')

        if normalise is not None:
            if normalise == 1:
                dummy = dummy.astype('float64') / dummy.max()
            else:
                dummy = dummy.astype('float64') / normalise
        if vmax is None:
            if normalise is not None:
                vmax = 1.0
            else:
                vmax = dummy.max()
        # --- Prepare the scale:
        if scale == 'sqrt':
            extra_options = {'norm': colors.PowerNorm(0.5, vmax=vmax,
                                                      vmin=vmin)}
        elif scale == 'log':
            extra_options = {'norm': colors.LogNorm(clip=True, vmax=vmax,
                                                    vmin=vmin)}
        else:
            extra_options = {'vmin': vmin, 'vmax': vmax}
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

        if scale == 'log':  # If we use log scale, just avoid zeros
            # we are here mixing a bit integers and float... but python3 will
            # provide
            dummy[dummy < 1.0] = 1.0e-5

        if extent is not None:
            extra_options['extent'] = extent

        if rotate_frame:
            imgR = ndimage.rotate(dummy, -self.CameraCalibration.deg, reshape=False)
            img = ax.imshow(imgR, cmap=cmap,
                        alpha=alpha, **extra_options)
        else:
            img = ax.imshow(dummy, origin='lower', cmap=cmap,
                        alpha=alpha, **extra_options)
        
        # Set the axis limit
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        if flag_time_range == False:
            tf = f'%.{tround}f'%tf
        else:
            tf = '(%.3f, %.3f)' %(t[0],t[1])

        if IncludeColorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(img, label='Counts', cax=cax)
            cbar.set_label(label='Counts [a.u.]')
        ax.text(0.05, 0.9, '#' + str(self.shot),
                horizontalalignment='left',
                color='w', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.text(0.95, 0.9, 't = ' + tf + (' s'),
                 horizontalalignment='right',
                 color='w', verticalalignment='bottom',
                 transform=ax.transAxes)
        if RemoveAxisTicksLabels:
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        else:
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Pixel')
        # Shot the figure
        if created:
            fig.show()
            plt.tight_layout()
        return ax


    def GUI_frames(self, flagAverage: bool = False, mask=None):
        """
        Small GUI to explore camera frames

        :param  flagAverage: flag to decide if we need to use the averaged frames
            of the experimental ones in the GUI
        :param  mask: to plot a small coloured mask on top of the image
        """
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        logger.info('--. ..- ..')
        logger.info(text)
        logger.info('-... . ..- - -.--')
        root = tk.Tk()
        root.resizable(height=None, width=None)
        if flagAverage:
            ssGUI.ApplicationShowVid(root, self.avg_dat, self.remap_dat,
                                     self.geometryID,
                                     self.CameraCalibration,
                                     shot=self.shot, mask=mask)
        else:
            ssGUI.ApplicationShowVid(root, self.exp_dat, self.remap_dat,
                                     GeomID=self.geometryID,
                                     calibration=self.CameraCalibration,
                                     scintillator=self.scintillator,
                                     shot=self.shot, mask=mask)
        root.mainloop()
        root.destroy()

    def GUI_frames_simple(self, flagAverage: bool = False, **kwargs):
        """
        Small GUI to explore camera frames using matplotlib widgets.

        Pablo Oyola - poyola@us.es

        :param flagAverage: flag to decide if we need to use the averaged frames
            of the experimental ones in the GUI
        """
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        logger.info('--. ..- ..')
        logger.info(text)
        logger.info('-... . ..- - -.--')

        # Generating the figure.
        fig, ax = plt.subplots(1)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        slider_ax = div.append_axes('bottom', pad='15%', size='3%')

        # Generating the slider.
        if flagAverage:
            tframes = self.avg_dat['t']
            frames  = self.avg_dat['frames']
        else:
            tframes = self.exp_dat['t']
            frames  = self.exp_dat['frames']

        # Plotting the initial frame (t=0)
        extent = [frames.py.min(), frames.py.max(), frames.px.min(), frames.px.max()]
        im = ax.imshow(frames.sel(t=0, method='nearest'), origin='lower',
                       extent=extent, **kwargs)
        slider = Slider(slider_ax, 'Time', tframes.values[0], tframes.values[-1],
                        valinit=tframes.sel(t=0, method='nearest').values,
                        valstep=tframes.values, orientation='horizontal',
                        initcolor='red')

        # Drawing the scintillator.
        if self.scintillator is not None:
            self.scintillator.plot_pix(ax=ax)

        # Setting up the colorbar.
        fig.colorbar(im, cax=cax, label='Counts')

        def update(val):
            im.set_data(frames.sel(t=val, method='nearest'))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

        return ax, slider


    # --------------------------------------------------------------------------
    # --- Properties
    # --------------------------------------------------------------------------
    @property
    def size(self):
        """Get the size of the loaded frames."""
        return self.exp_dat['frames'].size

    @property
    def shape(self):
        """Shape of the loaded frames"""
        return self.exp_dat['frames'].shape

    # --------------------------------------------------------------------------
    # --- Others
    # --------------------------------------------------------------------------
    def getFrameIndex(self, t: float = None, frame_number: int = None,
                      flagAverage: bool = False):
        """
        Return the index of the frame inside the loaded frames

        Jose Rueda Rueda: jrrueda@us.es

        :param  t: desired time (in the same units of the video database)
        :param  frame_index: frame index to load, relative to the camera,
            ignored if time is present
        :param  flagAverage: flag to look at the average o raw frames

        :return it: index of the frame in the loaded array
        """
        it = None
        if t is not None:
            if not flagAverage:
                it = np.argmin(np.abs(self.exp_dat['t'].values - t))
            else:
                it = np.argmin(np.abs(self.exp_dat['t'].values - t))
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

        :param  t: desired time (in the same units of the video database)
        :param  flagAverage: flag to look at the average or raw frames

        :return it: index of the frame in the loaded array
        """
        it = self.getFrameIndex(t)
        if not flagAverage:
            frame = self.exp_dat['frames'][..., it].squeeze()
        else:
            frame = self.avg_dat['frames'][..., it].squeeze()

        return frame.copy()

    def getTime(self, it: int, flagAverage: bool = False, videoFrame:bool = False):
        """
        Get the time corresponding to a loaded frame number

        Jose Rueda: jrrueda@us.es

        :param  it: frame number 
        :param  flagAverage: flag to look at the averaged or raw frames
        :param  videoFrame: flag to decide whether we look for the frame in the video
            or the array. If videoFrame==True getTime will return the time corresponding
            to the recorded frame it. If == False, it will return the time corresponding
            to the loaded frame it.
        
        Example, imaging that you read the frames 300-350 from the recorded video
        >>> vid.getTime(301, videoFrame=True) would yield the same that
        >>> vid.getTime(1)

        """
        if not flagAverage:
            if not videoFrame:
                t = float(self.exp_dat['t'].values[it])
            else:
                fi = self.getFrameIndex(frame_number=it)
                t = float(self.exp_dat.t.isel(t=fi).values)
        else:
            if not videoFrame:
                t = float(self.avg_dat['tframes'][it])
            else:
                raise Exception('Video frame has no meaning when loading averages')
        return t

    def getTimeTrace(self, t: float = None, mask=None, ROIname: str =None, vmax: int=None):
        """
        Calculate the timeTrace of the video

        Jose Rueda Rueda: jrrueda@us.es

        :param  t: time of the frame to be plotted for the selection of the roi
        :param  mask: bolean mask of the ROI

        If mask is present, the t argument will be ignored

        :returns timetrace: a timetrace object
        """
        if mask is None:
            # - Plot the frame
            ax_ref = self.plot_frame(t=t, vmax=vmax)
            fig_ref = plt.gcf()
            # - Define roi
            roi = sstt.roipoly(fig_ref, ax_ref)
            # Create the mask
            mask = roi.getMask(self.exp_dat['frames'][:, :, 0].squeeze())

        return sstt.TimeTrace(self, mask, ROIname=ROIname), mask

    def getCameraData(self, file: str = ''):
        """
        Read the camera data.

        Jose Rueda Rueda

        :param file: if empty, we will load the camera datafile from the Data
            folder taing into account the camera name from the calibration
            database.
        """
        if file=='':
            file = os.path.join(Path().ScintSuite, 'Data',
                                'CameraGeneralParameters',
                                self.CameraCalibration.camera + '.txt')
        logger.info('Reading camera data: %s', file)
        self.CameraData = f90nml.read(file)['camera']

    # --------------------------------------------------------------------------
    # --- Export
    # --------------------------------------------------------------------------
    def exportVideo(self, filename: str = None, flagAverage: bool = False):
        """
        Export video file

        Notice: This will create a netcdf with the exp_dat xarray, this is not
        intended as a replacement of the database, as camera settings and
        metadata will not be exported. But allows to quickly export the video
        to netCDF format to be easily shared among computers

        :param  file: Path to the file where to save the results, if none, a
            GUI will appear to select the results
        :param  flagAverage: flag to indicate if we want to save the averaged
            frames

        :Example:
        >>> # Load a video from a diagnostic
        >>> import Lib as ss
        >>> vid = ss.vid.INPAVideo(shot=41090)
        >>> vid.read_frame()  # To load all the video
        >>> # Export the video
        >>> vid.exportVideo()
        """
        if filename is None:
            filename = _ssio.ask_to_save(ext='*.nc')
            if filename == '' or filename == ():
                print('You canceled the export')
                return
        print('Saving video in: ', filename)
        # Write the data:
        if not flagAverage:
            self.exp_dat.to_netcdf(filename)
        else:
            self.avg_dat.to_netcdf(filename)

    def save(self, fn: str, t0: float=None, t1: float=None,
             flagAverage: bool=False, mode: str=None):
        """
        Writes to a video file the data in exp_dat / avg_dat. The type of video
        is decided upon the filename extension or from the mode parameter.

        Pablo Oyola - poyola@us.es

        :param fn: filename of the video. If an extension is detected, the
        corresponding ? is used.
        :param t0: initial time to save the video. If None, the video is written
        starting from the starting available data.
        :param t1: final time to save the video. If None, the video is written
        unitl the end available data.
        :param flagAverage: use the averaged data instead.
        :param mode: if an extension is not detected in the input filename, the
        user is required to say which kind of video file is to used.
        """

        if not fn.endswith('mp4') and (mode != 'mp4'):
            raise NotImplementedError('Only MP4 writing is supported.')

        # Saving to file.
        if not fn.endswith('mp4'):
            fn = fn + '.mp4'


        if flagAverage:
            data = self.avg_dat.frames.values
        else:
            data = self.exp_dat.frames.values

        # Retrieving the properties of the video.
        try:
            bits_size = self.properties['bits_size']
        except KeyError:
            bits_size = 16

        # Setting the enconding.
        if data.ndim > 3:
            encoding = 'rgb'
        else:
            encoding = 'grey'

        # Getting the FPS.
        dt = self.exp_dat.t.values[1] - self.exp_dat.t.values[0]
        fps = int(1/dt)

        # Saving
        logger.debug(f'Saving to file {fn}')
        mp4.write_file(fn=fn, video=data, bit_size=bits_size,
                       encoding=encoding, fps=fps)
