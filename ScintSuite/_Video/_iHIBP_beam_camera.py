"""
iHIBP-AUG geometry.

This library allows for the calculation of the beam deflection coming from the
stray magnetic field using the video cameras for that.

Balazs Tal - balazs.tal@ipp.mpg.de

There are a set of 2/3 cameras looking at the beam and are used to characterize
the beam properties, such as sizes, divergence, shift and direction. It is
observed that during shots the beam moves and produces that scintillator image
changes significantly.

The two main cameras observing the beam are:
    - 'top', which looks at the beam from the top side and gives us hints on the
    beam motion on the horizontal/toroidal direction.
    - 'side', which looks at the beam from the left side and gives us hints on
    the beam motion in the vertical/poloidal direction.

The object in this file finds and reads the camera files and load them,
including the calibration performed of the Field-of-View done by Balazs Tal.

To compute the beam line direction 2 or more Region-of-Interestest (ROIs) in
both vertical and horizontal direction. If the camera is the 'top' camera, the
data insided each ROI for each time, is averaged in the VERTICAL direction,
leading to a beam shape in 1D. That beam shape is typically have a Gaussian-ish
shape, thus a Gaussian fit is used to get the beam center (center of the
Gaussian) and the width is related to the beam size.

If several ROIs are combined then:
    - from several beam centers, we get the beam direction.
    - from several beam sizes, we can estimate the divergence of the beam.

A running example would be:
    import Lib
    vid = ScintSuite.vid.ihibp_beam_camera(40749, camera='top')
    vid.make_beam_line(time = [-2.0, 20.0], graphic_bar=True)
    vid.plot_frame(t = 5.0, beamline=True, calibrated=True)

adapted into the suite by:
Pablo Oyola - poyola@us.es
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as et
from ScintSuite._Video import BVO
import ScintSuite.errors as errors
import matplotlib.patches as patches
import lmfit
from tqdm import tqdm

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




def process_time_file(filename_time: str):
    """
    Loads the time base from an XML file for the AUG implementation.

    Balazs Tal - balazs.tal@ipp.mpg.de

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

    adapted by: Pablo Oyola - poyola@us.es

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

    out = ffmpeg.input(filename).output('pipe:',
                                         format='rawvideo',
                                         pix_fmt=pix_fmt).run(capture_stdout=True,
                                                              capture_stderr=True)[0]

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

def get_filenames(shot: int, camera_name: str):
    """
    From the shotnumber, this function returns both the filename of the video
    and the filename of the XML containing the time data.

    Pablo Oyola - poyola@us.es
    Balazs Tal  - balazs.tal@ipp.mpg.de

    :param shot: shotnumber to get the actual video filenames.
    """

    datadir = '/afs/ipp-garching.mpg.de/home/a/augd/rawfiles/LIV/%2i/%5i/' % (shot/1000, shot,)
    filename_video = os.path.join(datadir, '%5i_cam_%s.mp4'% (shot, camera_name,))
    filename_time = os.path.join(datadir, '%5i_cam_%s.xml' % (shot, camera_name,))
    return filename_video, filename_time


class beam_camera(BVO):
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

        fn, ft = get_filenames(shot=shot, camera_name='ihibp_'+camera)
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
            data = yaml_load(fid).get_data()

        # Getting the interesting camera data.
        self.camcal = data[camera.lower()]

        self.camcal['roi'] = np.array(self.camcal['roi'])

        # We now remove the initial noise by substracting either the average
        # frame from beginning to 0.0 seconds (shot starts) or just
        # removing the first frame.
        if remove_bg:
            self.subtract_noise(t1=self.exp_dat.t[0].values,
                                t2=0.0, flag_copy=True)
        else:
            self.subtract_noise(frame=self.exp_dat.isel(t=0),
                                flag_copy=True)

        # Setting to None the internal line data.
        self.line = None

    def plot_frame(self, rois: bool=True, beam_line: bool=False,
                   calibrated: bool=False, **kwargs):
        """
        Plots a given frame of the video.

        This function overloads the parent one in order to add the beam line
        and the ROIs.

        Pablo Oyola - poyola@us.es

        :param rois: plot the rois. True by default.
        :param beamline: plot the beam line. False by default.
        :param kwargs: keyword arguments to pass down to the parent class'
        function plot_frame. Use arguments t or frame_number to get the
        corresponding time point.
        """

        colors = ['r', 'y', 'g', 'm', 'c']

        extent = [0, self.exp_dat.py.size, 0, self.exp_dat.px.size]
        extent = np.array(extent, dtype=float)
        if calibrated:
            factor = self.camcal['diff'] / 10.0
            units = 'cm'
            extent /= factor
        else:
            factor = 1.0
            units  = 'pixel'



        # Plots the frame using the base class.
        ax = super().plot_frame(**kwargs, extent=extent)

        ax.set_xlabel('X [%s]'%units)
        ax.set_ylabel('Y [%s]'%units)

        if rois:
            if self.camcal['roi'] is None:
                logger.warning('There are not any ROIs declared!')

            for iroi in range(self.camcal['roi'].shape[0]):
                x0, y0, x1, y1 = self.camcal['roi'][iroi] / factor
                width = x1 - x0
                height = y1 - y0

                # Create a Rectangle patch
                rect = patches.Rectangle((x0, y0), width, height,
                                         linewidth=0, edgecolor=colors[iroi],
                                         facecolor=colors[iroi],
                                         alpha=0.50,
                                         label='ROI#%d'%iroi)

                # Add the patch to the Axes
                ax.add_patch(rect)
        if beam_line:
            if self.line is None:
                logger.warning('The line has not been precomputed!')

            if kwargs['t'] is not None:
                idx = np.abs(self.line.t - kwargs['t']).argmin().values
            else:
                time = self.getTime(kwargs['frame_number'])
                idx = np.abs(self.line.t - time).argmin()

            # Plotting the beam line.
            if(self.camcal['orientation'] == 'vertical'):
                ax.plot(self.line.line[:, idx]/factor, self.line.x/factor,
                        color='b', label='Fitted line', zorder=100)
                ax.plot(self.line.line_up[:, idx]/factor, self.line.x/factor,
                        color='m', label='1st edge', zorder=100)
                ax.plot(self.line.line_dw[:, idx]/factor, self.line.x/factor,
                        color='m', label='2nd edge', zorder=100)

                beta = self.line.beta_pix[idx].values*180.0/np.pi
                ax.text(0.05, 0.10, '$\\beta_{pix}$ = ' +\
                        str(round(float(beta), 2)) + (' º'),
                        horizontalalignment='left',
                        color='w', verticalalignment='bottom',
                        transform=ax.transAxes)


                alpha = self.line.div_pix[idx].values*180.0/np.pi
                ax.text(0.05, 0.05, '$\\alpha$ = ' +\
                        str(round(float(alpha), 2)) + (' º'),
                        horizontalalignment='left',
                        color='w', verticalalignment='bottom',
                        transform=ax.transAxes)
            else:
                ax.plot(self.line.x, self.line.line[:, idx],
                        color='b', label='Fitted line', zorder=100)

        return

    def get_roi_fit(self, time: float=None, graphic_bar: bool=False):
        """
        Computes the beam line by fitting it to a Gaussian function.

        Balazs Tal - balazs.tal@ipp.mpg.de

        adapted by:
        Pablo Oyola - poyola@us.es

        :param time: time range to get the beam line.
        :param graphic_bar: plot a graphic bar to track the progress.
        """

        # Checking that the time input is prepared.
        if time is None:
            time = self.timebase

        time = np.atleast_1d(time)

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
        elif time.size == 1:
            t0 = np.abs(self.timebase - time[0]).argmin()
            time = np.array((self.timebase[t0],))

        # Number of ROIs
        nroi = self.camcal['roi'].shape[0]

        # Preparing the results array.
        amp    = np.zeros((nroi, time.size))
        fwhm   = np.zeros((nroi, time.size))
        center = np.zeros((nroi, time.size))
        signal2noise = np.zeros((nroi, time.size))

        # Looping over the ROIs.
        for iroi in range(nroi):
            xroi = np.arange(self.camcal['roi'][iroi][1],
                             self.camcal['roi'][iroi][3] + 1)
            yroi = np.arange(self.camcal['roi'][iroi][0],
                             self.camcal['roi'][iroi][2] + 1)
            data = self.exp_dat.isel(px=xroi, py=yroi)

            # Looping in time.
            for ii, itime in tqdm(enumerate(time), disable=not graphic_bar,
                                  desc='Parsing ROI #%d'%iroi,
                                  total=time.size):
                if self.camcal['orientation'] == 'vertical':
                    frame = data.sel(t = itime).mean(dim = 'px').frames.values.astype('float64')
                    x = yroi
                else:
                    frame = data.sel(t = itime).mean(dim = 'py').frames.values.astype('float64')
                    x = xroi

                # Fitting it to a Gaussian.
                fitter = lmfit.models.GaussianModel()
                pars = fitter.guess(frame, x=x)
                res = fitter.fit(frame, pars, x=x)

                amp[iroi, ii]    = res.params['amplitude'].value
                fwhm[iroi, ii]   = res.params['fwhm'].value
                center[iroi, ii] = res.params['center'].value

                # We need now to check whether the amplitude to background is
                # non-negligible or whether the signal-to-noise is too low.
                non_xroi = np.arange(0, self.camcal['roi'][iroi][1])
                non_yroi = np.arange(0, self.camcal['roi'][iroi][0])
                data2 = self.exp_dat.frames.isel(px=non_xroi, py=non_yroi).sel(t=itime).mean()
                s2n = amp[iroi, ii] / data2.values
                # print(f't = {itime}, signal2noise = {s2n}')
                signal2noise[iroi, ii] = s2n


        return signal2noise, fwhm, center, time

    def make_beam_line(self, time: float=None, sigma: float=1.0,
                       graphic_bar: bool=False):
        """
        This function will build the line for all the points in the camera and
        store it internally.

        Balazs Tal - balazs.tal@ipp.mpg.de

        adapted by:
        Pablo Oyola - poyola@us.es

        :param time: time (range) to evaluate the beam line direction and
        divergence.
        :param sigma: how many sigma's to include to compute the limits of the
        beam.
        :param graphic_bar: send down to the get_roi_fit to visualize the
        ROI fitting.
        """

        s2n, self.line_fwhm, self.line_center, time = \
            self.get_roi_fit(time=time, graphic_bar=graphic_bar)

        self.line = xr.Dataset()

        slopes = np.zeros((self.line_fwhm.shape[1], ))
        interc = np.zeros((self.line_fwhm.shape[1], ))
        slopes1 = np.zeros((self.line_fwhm.shape[1], ))
        interc1 = np.zeros((self.line_fwhm.shape[1], ))
        slopes2 = np.zeros((self.line_fwhm.shape[1], ))
        interc2 = np.zeros((self.line_fwhm.shape[1], ))
        x = np.zeros((self.camcal['roi'].shape[0], ))


        for iroi in range(self.camcal['roi'].shape[0]):
            xroi = np.arange(self.camcal['roi'][iroi][1],
                             self.camcal['roi'][iroi][3] + 1)
            yroi = np.arange(self.camcal['roi'][iroi][0],
                             self.camcal['roi'][iroi][2] + 1)

            # From the ROIs we get the center location where they are evaluated.
            if self.camcal['orientation'] == 'vertical':
                x[iroi] = xroi.mean()
            else:
                x[iroi] = yroi.mean()

        # From this fit, we compute for all the time points the line passing
        # by all the ROIs center.
        for ii, itime in enumerate(time):
            # Fitting the central line.
            slopes[ii], interc[ii] = np.polyfit(x,
                                                self.line_center[:, ii],
                                                deg=1).tolist()

            # Fitting a line to the edge profile of the line.
            y_up = self.line_center[:, ii] + sigma * self.line_fwhm[:, ii]
            slopes1[ii], interc1[ii] = np.polyfit(x, y_up, deg=1).tolist()

            y_dw = self.line_center[:, ii] - sigma * self.line_fwhm[:, ii]
            slopes2[ii], interc2[ii] = np.polyfit(x, y_dw, deg=1).tolist()



        self.line['slopes'] = xr.DataArray(slopes, dims=('t',),
                                          coords=(time,))

        self.line['interc'] = xr.DataArray(interc, dims=('t',),
                                           coords=(time,))


        # Building up the lines.
        px0 = self.exp_dat.px.values.min()
        px1 = self.exp_dat.px.values.max()
        py0 = self.exp_dat.py.values.min()
        py1 = self.exp_dat.py.values.max()

        if self.camcal['orientation'] == 'vertical':
            limits = np.array((px0, px1))
        else:
            limits = np.array((py0, py1))

        # We build the central line.
        central_line = self.line.slopes.values[:, None] * limits[None, :] + \
                       self.line.interc.values[:, None]

        upperlim = slopes1[:, None] * limits[None, :] + \
                   interc1[:, None]
        lowerlim = slopes2[:, None] * limits[None, :] + \
                   interc2[:, None]

        coords = (limits, time)
        dims   = ('x', 't')

        self.line['line'] = xr.DataArray(central_line.T,
                                         coords=coords,
                                         dims=dims)
        self.line['line_up'] = xr.DataArray(upperlim.T,
                                            coords=coords,
                                            dims=dims)
        self.line['line_dw'] = xr.DataArray(lowerlim.T,
                                            coords=coords,
                                            dims=dims)


        # Computing the deflection angle in pixel coordinates.
        if self.camcal['orientation'] == 'vertical':
            norm = np.sqrt(self.line.slopes**2 + 1)
            u_dir_x = 1.0/norm
            u_dir_y = self.line.slopes/norm
            u_dir_pix = np.array([u_dir_x, u_dir_y])

            coords = [np.array(('x', 'y')), time]
            dims   = ['pixcoord', 't']
            attrs  = { 'desc': 'Direction in pixel coordinates',
                       'units': 'pixel',
                     }

            self.line['u_dir_pix'] = xr.DataArray(u_dir_pix, coords=coords,
                                                  dims=dims, attrs=attrs)

            coords = [time]
            dims   = ['t']
            attrs  = { 'desc':  'Angle with the Xpixels axis',
                       'units': 'rad',
                     }

            angle = np.arccos(1.0/norm)
            self.line['beta_pix'] = xr.DataArray(angle, coords=coords,
                                                 dims=dims, attrs=attrs)

            # We compute now the beam divergence.
            norm1 = np.sqrt(slopes1**2 + 1)
            u_dir_x1 = 1.0/norm1
            u_dir_y1 = slopes1/norm
            u1 = np.array([u_dir_x1, u_dir_y1])

            norm2 = np.sqrt(slopes2**2 + 1)
            u_dir_x2 = 1.0/norm2
            u_dir_y2 = slopes2/norm
            u2 = np.array([u_dir_x2, u_dir_y2])

            alpha = np.arccos(np.sqrt(u1[0, :] * u2[0, :] + \
                                      u1[1, :] * u2[1, :])) /2.0
            coords = [time]
            dims   = ['t']
            attrs  = { 'desc':  'Divergence from top camera',
                       'units': 'rad',
                     }

            angle = np.arccos(1.0/norm)
            self.line['div_pix'] = xr.DataArray(alpha, coords=coords,
                                                 dims=dims, attrs=attrs)

            # Getting x_shift.
            attrs = { 'desc': 'Beam shift along the horizontal direction',
                      'units': 'pix',
                    }
            self.line['x_shift'] = xr.DataArray(interc, coords=[time,],
                                                dims=['t',],
                                                attrs=attrs)

        else:
            raise NotImplementedError('Working on it, sorry')






