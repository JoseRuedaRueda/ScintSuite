"""
iHIBP-AUG geometry.

This library allows for the calculation of the beam deflection coming from the
stray magnetic field using the video cameras for that.

Pablo Oyola - poyola@us.es

copied & adapted from Balazs Tal scripts: balazs.tal@ipp.mpg.de
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as et
from Lib._Video import BVO
import Lib.errors as errors

try:
    from yaml import CLoader as yaml_load
except:
    from yaml import Loader as yaml_load


import logging
logger = logging.getLogger('ScintSuite.AUG.iHIBP')

try:
    import ffmpeg
    __ffmpeg_loaded = True
except ModuleNotFoundError:
    logger.error('Cannot load the FFMPEG package. ' +
                 'Install it using pip install ffmpeg-python')
    raise ModuleNotFoundError('FFMPEG not found')

    __ffmpeg_loaded = False




def process_time_file(filename_time: str=None):
    """
    Loads the time base from an XML file for the AUG implementation.

    Pablo Oyola - poyola@us.es

    :param filename_time: file with the time data.
    """

    if not os.path.isfile(filename_time):
        raise FileNotFoundError('Filename with time could not be found!')
    root=et.parse(filename_time).getroot()

    if 'ts6Time' in root.keys():
        TS6 = int(root.attrib['ts6Time'], 0)
        time_vec = np.array([int(r.attrib['time'], 0) for r in root])
    elif 'ts6' in root.keys():
        TS6=int(root.attrib['ts6'],0)
        for r in root:
            if r.tag=='frames':
                frames=r
        time_vec=np.array([int(r.attrib['timestamp'], 0) for r in frames])
    else:
        raise Exception('Neither ts6 nor ts6Time is in the xml file.')


    time_vec = (time_vec - TS6)*1.0e-9 # From nanoseconds to seconds.
    return time_vec


def read_mp4_ffmpeg(filename: str, fn_time: str=None):
    """
    Using the FFMPEG package to read directly the frames.

    Balazs Tal - balazs.tal@ipp.mpg.de

    :param filename: file with the MP4 video data.
    :param fn_time: filename of the XML containing the time data.
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError('MP4 file cannot be found: %s'%filename)

    # Retrieving the data from the file.
    probe  = ffmpeg.probe(filename)
    height = probe['streams'][0]['height']
    width  = probe['streams'][0]['width']

    if probe['streams'][0]['pix_fmt'] == 'yuv444p10le':  #Tilmann's camera
        pix_fmt = 'gray16le'
        dtype = np.uint16
        pix_shift = 6
        shape = [-1, height, width]
    elif width==720 and height==540:  # Michael's camera
        pix_fmt = 'gray'
        dtype   = np.uint8
        pix_shift = 0
        shape = [-1, height, width]
    elif width==960 and height==720: # Webcam
        pix_fmt = 'rgb48le'
        dtype   = np.uint16
        pix_shift = 8
        shape = [-1, height, width, 3]
    else:
        raise NotImplementedError('Unknown format')

    out, = ffmpeg.input(filename).output('pipe:',
                                         format='rawvideo',
                                         pix_fmt=pix_fmt).run(capture_stdout=True)[0]

    video = np.right_shift(np.frombuffer(out, dtype).reshape(shape),
                           pix_shift)

    # We now try to load the time base.
    if fn_time is not None:
        if not os.path.isfile(fn_time):
            logger.warning('Cannot load the time file %s. '%fn_time + \
                           'Proceeding without time base.')

        time = process_time_file(fn_time)
    else:
        time = np.arange(video.shape[0])

    if video.ndim == 4:
        video = np.mean(video, axis = -1) # Averaging the color to grayscale.

    output = { 'nf': video.shape[0],
               'nx': width,
               'ny': height,
               'frames': video,
               'tframes': time
              }

    return output

def get_filenames(self, shot: int, camera_name: str):
    """
    From the shotnumber, this function returns both the filename of the video
    and the filename of the XML containing the time data.

    Pablo Oyola - poyola@us.es
    Balazs Tal  - balazs.tal@ipp.mpg.de

    :param shot: shotnumber to get the actual video filenames.
    """

    datadir = '/afs/ipp/u/augd/rawfiles/VRT/%2i/S%5i/' % (shot/1000, shot,)
    filename_video = os.path.join(datadir, 'S%5i_%s.mp4') % (shot, camera_name,)
    if shot < 40396:
        filename_time = os.path.join(datadir, 'Prot',
                                     'FrameProt',
                                     '%s_FrameProt.xml' % camera_name)
    else:
        filename_time = os.path.join(datadir,
                                     'S%5i_%s.meta.xml' % (shot, camera_name))

    return filename_video, filename_time


class beam_cameras(BVO):
    """
    This is a class allowing to handle the iHIBP beam cameras and get from
    them the displacement of the beam line.

    Pablo Oyola - poyola@us.es
    """
    def __init__(self, shot: int, camera: str='top', remove_bg: bool=True):
        """
        Initializes the object that handles the camera(s) that watch over
        the beam in the iHIBP diagnostic during shots and analizes the
        trajectory.

        Pablo Oyola - poyola@us.es

        :param shot: shot to get the beam cameras.
        :param camera: camera to look at the beam.
        :param remove_bg: whether to remove internally the background.
        """

        if camera not in ('top', 'side'):
            raise errors.DatabaseError('Camera %s is not in the database'%camera)

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

        fn, ft, self.properties = get_filenames(shot=shot, camera=[])
        if not os.path.isfile(fn):
            raise errors.DatabaseError('Cannot find video for shot #%05d'%shot)

        dummy = read_mp4_ffmpeg(fn, ft)

        self.timebase = dummy['tframes']
        self.exp_dat = xr.Dataset()
        nt, nx, ny = dummy['frames'].shape
        px = np.arange(nx)
        py = np.arange(ny)

        self.exp_dat['frames'] = \
            xr.DataArray(dummy['frames'], dims=('t', 'px', 'py'),
                         coords={'t': dummy['tframes'].squeeze(),
                                 'px': px,
                                 'py': py})
        self.exp_dat['frames'] = self.exp_dat['frames'].transpose('px',
                                                                  'py',
                                                                  't')
        self.type_of_file = '.mp4'

        ## Geometry of the diagnostic head used to record the video
        self.geometryID = None
        ## Camera calibration to relate the scintillator and the camera sensor,
        # each diagnostic will read its camera calibration from its database
        self.CameraCalibration = None
        ## Scintillator plate:
        self.scintillator = None

        # We now load the calibration for the cameras.
        with open('./Data/Calibrations/iHIBP/beam_watchdog.yml', 'rt') as fid:
            data = yaml_load(fid)

        # Getting the interesting camera data.
        self.camcal = data[camera.lower()]

        # We now remove the initial noise by substracting either the average
        # frame from beginning to 0.0 seconds (shot starts) or just
        # removing the first frame.
        if remove_bg:
            self.subtract_noise(t1=self.exp_dat.t[0],
                                t2=0.0, flag_copy=True)
        else:
            self.subtract_noise(frame=self.exp_dat.isel(t=0),
                                flag_copy=True)

    def prepare_rois(self):
        """

        """
        pass

    def get_beam_line(self, time: float=None, pix_avg: int=5):
        """
        Computes the beam line by fitting it to a Gaussian function.

        Pablo Oyola - poyola@us.es

        :param time: time range to get the beam line.
        :param pix_avg: pixels to average for making a smoother fitting process.
        """

        # Checking that the time input is prepared.
        if time is None:
            time = self.timebase

        time = np.atleast_1d(time, dtype=float)

        if np.any(time > self.timebase.max()):
            raise ValueError('Some of the time points are higher than ' + \
                             'the time base!')

        if np.any(time < self.timebase.min()):
            raise ValueError('Some of the time points are lower ' + \
                             'than the time base!')

        if time.size == 2:
            t0 = np.abs(self.timebase - time.min()).argmin()
            t1 = np.abs(self.timebase - time.max()).argmin()

            time = self.timebase[t0:t1]

        # We make a rolling average of the frames to


