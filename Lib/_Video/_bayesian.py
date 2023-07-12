"""
Bayesian noise removal library

Pablo Oyola - poyola@us.es
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import xarray as xr
import logging
from Lib._Video._BasicVideoObject import BVO
from Lib._Video._VRTVideoObject import VRTVideo
from tqdm import tqdm
from typing import Union
from Lib._TimeTrace import TimeTrace

logger = logging.getLogger('ScintSuite.Video.Bayes')

try:
    from numba import njit, prange, jit
except ModuleNotFoundError:
    logger.warning('Cannot import numba. Using pure python routines')
    from Lib.decorators import false_njit as njit
    jit = njit
    prange = range


@njit('f8[:,:](f8, f8, i4, f8[:, :], f8[:])',
      parallel=True, nogil=True, cache=True)
def fast_interp1(xmin: float, dx: float, nx: int, z: float, xq: float):
    """
    Numba-accelerated 1D linear interpolation on a regular grid.

    Pablo Oyola - poyola@us.es

    :param xmin: minimum value of the x-grid, which is assumed regular.
    :param dx: step between points in a regular grid.
    :param nx: number of points along the grid.
    :param z: variable to be interpolated. It can have the form (nfields,  nx)
    so several fields can be interpolated at once.
    :param xq: array with elements to be evaluated.
    """

    assert nx == z.shape[0], 'First axis must be grid axis.'

    nxq = xq.shape[0]
    nfields = z.shape[-1]
    zq = np.zeros((nfields, nxq))

    for ii in prange(nxq):
        ia = int(max(0, min(nx - 2, (xq[ii] - xmin)/dx)))
        ia1 = ia + 1

        # Evaluating the weights.
        ax1 = min(1.0, max(0.0, (xq[ii] - (xmin + dx*ia))/dx))
        ax = 1.0 - ax1

        a00 = ax
        a10 = ax1

        for jj in range(nfields):
            zq[jj, ii] = z[ia,  jj]  * a00 +  z[ia1, jj]  * a10
    return zq

def decision_tanh(prob: float, x0: float, dx0: float):
    """
    Decision function using a tanh function.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  prob: array with the probability value that must range between [0, 1]
    :param  x0: probability threshold, point at which the probability of being
    noise would be 50%.
    :param  dx0: probability slope. Slope to transition from noise to signal.
    """

    assert (x0 > 0.0) and (x0 < 1.0), 'Probability threshold must be in [0, 1]'
    assert (dx0 > 0.0) and (dx0 < 1.0), 'Probability slope must be in [0, 1]'

    f = 0.5*(1.0 + np.tanh((prob-x0)/dx0))

    return f

def decision_sigmoid(prob: float, x0: float, dx0: float):
    """
    Decision function using a sigmoid function.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  prob: array with the probability value that must range between [0, 1]
    :param  x0: probability threshold, point at which the probability of being
    noise would be 50%.
    :param  dx0: probability slope. Slope to transition from noise to signal.
    """

    assert (x0 > 0.0) and (x0 < 1.0), 'Probability threshold must be in [0, 1]'
    assert (dx0 > 0.0) and (dx0 < 1.0), 'Probability slope must be in [0, 1]'

    f = 1.0/ (1.0 + np.exp(-(prob - x0)/dx0))

    return f

def decision(which: str, *args, **kwargs):
    """
    Wrapper to all the decision functions.

    Pablo Oyola - poyola@us.es

    :param which: string to choose among the different decision functions.
    """

    if which.lower() == 'tanh':
        return decision_tanh(*args, **kwargs)
    elif which.lower() == 'sigmoid':
        return decision_sigmoid(*args, **kwargs)
    else:
        raise NotImplementedError('Function %s not implemented' % which)


@njit('f8(f8[:], f8[:])', nogil=True, cache=True)
def __pearsonr(x: float, y: float):
    """
    Numba adapted calculation of the linear Pearson correlation coefficient.

    Pablo Oyola - poyola@us.es
    """

    assert len(x) == len(y), 'Sizes do not match!'
    xmean = np.mean(x)
    ymean = np.mean(y)
    xstd  = np.std(x)
    ystd  = np.std(y)

    denom = xstd * ystd

    corr = np.mean((x-xmean) * (y-ymean))
    if denom == 0.0:
        corr = 0.0
    else:
        corr /= denom

    return corr

@njit('f8[:, :](f8[:, :], f8[:], i4, i4[:])', parallel=True,
      cache=True, nogil=True)
def __fast_corr(signal0: float, signal1: float, window: int, mask: int):
    """
    Numba-accelerated correlation calculation.

    Pablo Oyola - poyola@us.es

    :param signal0: signal(s) to correlate. A 2D array is expected with
    sizes [ntime, nfields].
    :param signal1: signal to correlate. A 1D array is expected with size
    [ntime], that can be broadcasted with signal0.
    :param window: size of the window to average the correlation coefficient.
    """
    assert signal0.shape[0] == signal1.shape[0], \
        'Signals must have the first axis equal.'

    # Starting the calculation of the correlation.
    corr = np.zeros_like(signal0)
    for ii in range(signal0.shape[0]):
        idx0 = int(max(0, ii - (window - 1)//2))
        idx1 = idx0 + window
        if idx1 > signal0.size:
            idx1 = int(min(signal0.size, ii + (window - 1)//2))
            idx0 = idx1 - window

        for jj in prange(signal0.shape[1]):
            if mask[jj] == 1:
                corr[ii, jj] = __pearsonr(signal0[idx0:idx1, jj],
                                          signal1[idx0:idx1])
            else:
                corr[ii, jj] = -1000.0
    return corr

def window_corr(signal0: float, signal1: float, time0: float=None,
                time1:float=None, window: int=21, mask: bool=None):
    """
    Evaluates a time dependent correlation coeffient useful to determine how
    the correlation between two signals evolves in time.

    Pablo Oyola - poyola@us.es

    :param signal0: first signal to correlate. Time must be the first axis.
    :param signal1: second signal to correlate. Time must be the only axis.
    :param time0: when provided, the code assumes that the signal0 and signal1
    do not have the same time spacing and time1 is also required. A linear
    interpolation using the most restrictive time limits is used.
    :param time1: same as time0.
    :param window: size of the window to average.
    :param mask: mask to parse the video. If None, all the signal0 is parsed.
    """

    if window < 5:
        raise ValueError('The window size must be larger than 4.')

    if mask is not None:
        if not np.all(signal0.shape[1:] == mask.shape):
            raise ValueError('The mask is not consistent with the image.')
        tmp = np.zeros_like(mask, dtype='int32')
        tmp[mask] = 1
        mask = tmp
    else:
        mask = np.ones(signal0.shape[1:], dtype='int32')

    if time0 is not None:
        assert time1 is not None, 'Time1 must be provided!'

        t0 = max(time0.min(), time1.min())
        t1 = min(time0.max(), time1.max())

        new_time = np.linspace(t0, t1, min(len(time0), len(time1)))
        signal0_tmp = np.reshape(signal0, [signal0.shape[0], -1])

        tmin = time0.min()
        dt   = time0[1] - time0[0]
        nt   = len(time0)
        data0 = fast_interp1(tmin, dt, nt, signal0_tmp, new_time).T

        tmin = time1.min()
        dt   = time1[1] - time1[0]
        nt   = len(time1)
        data1 = fast_interp1(tmin, dt, nt, signal1[:, None], new_time).squeeze()

    elif time1 is not None:
        assert time0 is not None, 'Time0 must be provided!'

        t0 = max(time0.min(), time1.min())
        t1 = min(time0.max(), time1.max())

        new_time = np.linspace(t0, t1, min(len(time0), len(time1)))
        signal0_tmp = np.reshape(signal0, [signal0.shape[0], -1])
        tmin = time0.min()
        dt   = time0[1] - time0[0]
        nt   = len(time0)
        data0 = fast_interp1(tmin, dt, nt, signal0_tmp, new_time).T

        tmin = time1.min()
        dt   = time1[1] - time1[0]
        nt   = len(time1)
        data1 = fast_interp1(tmin, dt, nt, signal1[:, None], new_time).squeeze()
    elif (time0 is None) and (time1 is None):
        assert len(signal0) == len(signal1), \
            'If time bases are not provided, then ' + \
            'signals must be the same size'

        data0 = signal0
        data1 = signal1
        new_time = np.arange(data0.size)

    # Shapes of the data.
    shapes = [new_time.size, *signal0.shape[1:]]
    mask = mask.flatten()

    # We only do the correlation (which is the heavy-computation) using
    # numba, so we keep this as flexible as possible.
    corr = __fast_corr(data0, data1, window, mask)

    data0 = np.reshape(data0, shapes)
    corr = np.reshape(corr, shapes)

    return new_time, corr, data0


class noise_element:
    """
    Base class to allocate the noise element. Do not use this class directly,
    but implement its main members in child-classes.
    """
    def __init__(self):
        """
        Initializes the class with no data.

        Pablo Oyola - poyola@us.es
        """

        self._noise = None

    def __getitem__(self, idx):
        """
        Access to the class data using the slicing operators [...].

        Pablo Oyola - poyola@us.es
        """

        return self._noise[idx]

    def get_noise(self, frames: float, time: float=None):
        """
        Returns the noise level as a probability provided a set of input frames.
        If time is not provided the frames are considered to be starting at
        t = t_0, being t_0 the first time where the frames are provided.

        0 -> No noise.
        1 -> Noise.

        Pablo Oyola - poyola@us.es

        :param frames: frames to deduce from the noise level. The last axis must
        correspond to the time axis.
        :param time: time to correct the frames. Unused in the low level.
        """

        return np.zeros_like(frames)

class monitor_noise(noise_element):
    """
    Handles the statistics in case the noise is considered to follow a
    time-evolving signal.

    Pablo Oyola - poyola@us.es
    """
    def __init__(self, monitor: float, ref: float=None, sigma: float=5.0):
        """
        Initializes a class object containing the methods to control the noise
        reduction provided a reference and a time evolving monitor, that can be
        or not position depedent.

        Pablo Oyola - poyola@us.es

        :param ref: reference to apply the monitor. When not provided, ones are
        used as reference. When provided, the value is normalized to the maximum.
        :param monitor: time-evolving noise monitor that is used to weight how
        the noise is evolving. If a position dependent monitor is given, the
        time axis is considered the last axis.
        """

        if monitor.ndim == 1:
            self.pos_dep = False
            self.monitor = monitor / monitor.max()
        else:
            self.pos_dep = True
            self.monitor = monitor / monitor.max(axis=-1)

        # Checking the input frame.
        if ref is None:
            if self.pos_dep:
                self.ref = np.ones(self.monitor.shape[:-1])
            else:
                self.ref = np.array((1.0,), dtype=float)
        else:
            if self.pos_dep:
                if np.any(ref.shape != self.monitor.shape[:-1]):
                    raise ValueError('Reference frame is not the' + \
                                     'same shape as the monitor.')
            self.ref = ref.copy()

        self.sigma = self.set_sigma(sigma)

        # Generating the time dependent noise.
        self._noise = self.ref[..., None] * self.monitor

    def set_sigma(self, sigma: float):
        """
        Changes internally the value of the sigma to be applied when computing
        the noise-probability.

        Pablo Oyola - poyola@us.es

        :param sigma:
        """
        sigma = np.atleast_1d(sigma)
        if np.any(sigma <= 0.0):
            raise ValueError('The weighting sigma must be a positive value')

        if sigma.size > 1:
            if sigma.ndim == 1: # Time-dependent sigma.
                if (sigma.shape[0] != self.monitor.shape[-1]):
                    raise ValueError('The time-dependent sigma must ' + \
                                     'broadcast with the monitor time shape')
            else:
                if np.any(sigma.shape != self.monitor.shape):
                    raise ValueError('The timespatial-dependent sigma must ' + \
                                     'broadcast with the monitor time shape')
        return sigma

    def get_noise(self, frames: float, itime: int=None, sigma: float=None):
        """
        Returns the probability of a frame being noise using a reference frame
        and a monitor.

        Pablo Oyola - poyola@us.es

        :param frames: frames to apply the monitor evolution.
        :param itime: time indices to apply the calibration. Must broadcast
        with the input frames and must be within the index limits of the
        noise internally computed. If None, the first n time values are taken.
        :param sigma: change the sigma used. If None, the internally-set sigma
        is used instead.
        """

        # Checking the sigma input.
        if sigma is not None:
            sig = self.set_sigma(sigma)
        else:
            sig = self.sigma

        # Checking the time input.
        if itime is None:
            itime = np.arange(frames.size)

        if itime[0] < 0:
            raise ValueError('The input time must be a positive integer value.')
        elif itime[-1] > self._noise.size:
            raise ValueError('The input time must be lower than the noise size')

        # Getting the Gaussian distance between the frame and the noise.
        delta = np.maximum(0.0, frames[..., itime] - self._noise[..., itime])

        return np.exp(- delta / sig[..., itime])

class vrt_noise(VRTVideo, monitor_noise):
    """
    Using a real camera object, this class makes the internal calculation to
    generate the noise monitor.

    Pablo Oyola - poyola@us.es
    """
    def __init__(self, camera: str, shot: int, t_roi: int=None,
                 refframe: float=None):
        """
        Initializes using a given VRT camera to create a camera correlation.

        Pablo Oyola - poyola@us.es

        :param camera: camera name to load.
        :param shot: shot number to read the VRT camera.
        :param t_roi: time point to select a particular ROI.
        :param refframe: reference frame to use. If None, default in
        monitor_noise is used.
        """

        # Initializing the camera video.
        VRTVideo.__init__(self, camera=camera, shot=shot)

        # From the camera video, we get the time trace.
        if t_roi is not None:
            tt = self.getTimeTrace(t=t_roi)[0]
        else:
            mask = np.ones(self.exp_dat.frames.shape[:-1], dtype=bool)
            tt = self.getTimeTrace(mask=mask)[0]

        monitor_noise.__init__(self, monitor=tt.mean_of_roi.values,
                               ref=refframe,
                               sigma=tt.std_of_roi.values)

        # We replace the video data with the noise data.
        # self.exp_dat['frames'].values = self._noise

class corr_monitor_noise(noise_element):
    """
    This noise monitor computes the noise and applies it when a time-evolving
    monitor is given and the user inputs the time evolving frames.

    Correlation between the noise monitor and the time trace of each pixel is
    computed and then a probability is derived assuming that the closest the
    correlation coefficient/p-value is to 1 (or 0 for p-value), the most-likely
    the pixel is noise in the corresponding time point.
    """
    def __init__(self, monitor: float, time_monitor: float=None,
                 use_pvalue: bool=False, use_pearson_r2: bool=False):
        """
        Initializes the class provided the time-evolving monitor and the
        corresponding time basis. If monitor is an dataaray., the timebasis will
        be extracted from the dataArray.

        Pablo Oyola - poyola@us.es

        :param monitor: time-evolving noise monitor. If it is DataArray, the
        time basis will be retrieved from the internal values.
        :param time_monitor: if the time is not an attribute of monitor, then
        it must be explicitly provided.
        :param use_pvalue: use the p-value instead of the correlation
        coefficient. False by default.
        :param use_pearson_r2: use the Pearson coeffiecient for the correlation
        test instead of the Spearman's. Pearson is better suited for the linear
        correlations.
        """

        # Checking the time input.
        self.timebase = None
        self.multi_monitor = False

        # Checking the noise monitor.
        if isinstance(monitor, xr.DataArray):
            self._noise = monitor.values.squeeze()
        else:
            self._noise = monitor.squeeze()

        # Checking the size of the monitor.
        if self._noise.ndim > 1:
            self.multi_monitor = True
            self._noise /= self._noise.max(axis=-1) # Normalize the noise to 1.
        else:
            self._noise /= self._noise.max()


        if isinstance(monitor, xr.DataArray):
            try:
                self.timebase = monitor.t
            except:
                logger.info('Time is not in the DataArray')

        if (self.timebase is None) and (time_monitor is None):
            raise ValueError('Time basis must be provided!')
        else:
            self.timebase = np.atleast_1d(time_monitor)
            if self.timebase.size != self._noise.shape[-1]:
                raise ValueError('Input timebase and monitor last' + \
                                 'axis must broadcast')

        # Saving the values internally.
        self.use_pvalue     = use_pvalue
        self.use_pearson_r2 = use_pearson_r2

    def get_noise(self, frames: float, time: float=None, window: int=21,
                  mask: bool=None):
        """
        Return the probability of pixel or set of pixels to be noise according
        to the correlation with the internal monitor.

        Pablo Oyola - poyola@us.es

        :param frames: set of frames to correlate with the noise. Must have the
        following shape (pixels_x, pixels_y, time). If this is an xarray, the
        timebase will be retrieved from the xarray, if existing.
        :param time: time array. Ignored if the frames inputs has an attribute
        called 't'.
        """

        if self.multi_monitor:
            raise NotImplementedError('Multi monitor correlation calculation' +
                                      ' not yet implemented')

        if isinstance(frames, xr.DataArray):
            frames0 = frames.values
            try:
                timebase = frames.t.values
            except:
                logger.info('The input xarray does not have the time attribute!')

        else:
            frames0 = frames
            timebase = np.atleast_1d(time)
            if timebase.size != frames.shape[-1]:
                raise ValueError('The frame and the timebase must be consistent')

        # Moving the time axis to the first place.
        data = np.moveaxis(frames0, source=-1, destination=0)


        # Computing the time-correlation between the pixels and the internal
        # noise monitor.
        time, corr, signal0 = window_corr(signal0=data,
                                          signal1=self._noise.squeeze(),
                                          time0=timebase, time1=self.timebase,
                                          window=window, mask=mask)

        # We need to transform the correlation coefficient into a probability.
        # We use for that a decision smooth function.
        # prob = decision('tanh', corr, x0=0.95, dx0=0.05)


        # Moving the time axis to the first place.
        prob = np.moveaxis(corr, source=0, destination=-1)

        return time, prob, signal0

class vrt_noise_corr(VRTVideo, corr_monitor_noise):
    """
    Using a real camera object, this class makes the internal calculation to
    generate the noise monitor.

    Pablo Oyola - poyola@us.es
    """
    def __init__(self, camera: str, shot: int, t_roi: int=None):
        """
        Initializes using a given VRT camera to create a camera correlation.

        Pablo Oyola - poyola@us.es

        :param camera: camera name to load.
        :param shot: shot number to read the VRT camera.
        :param t_roi: time point to select a particular ROI.
        """

        # Initializing the camera video.
        VRTVideo.__init__(self, camera=camera, shot=shot)

        # From the camera video, we get the time trace.
        if t_roi is not None:
            tt = self.getTimeTrace(t=t_roi)[0]
        else:
            mask = np.ones(self.exp_dat.frames.shape[:-1], dtype=bool)
            tt = self.getTimeTrace(mask=mask)[0]

        corr_monitor_noise.__init__(self, tt.mean_of_roi.values,
                                    tt.mean_of_roi.t.values)

class bayes_noise:
    """
    Support to remove noise from images using the Bayes approach provided
    difference sources to estimate noise.

    Pablo Oyola - poyola@us.es
    """
    def __init__(self, video: BVO):

        # Inserting internally the video.
        self.video = video

        # If a video has attributes a shot file, we save it for later.
        try:
            self.shot = self.video.shot
        except:
            logger.info('Input video has no shot info.' + \
                        'It will have to be provided by the user.')

        # Getting the reference to the timebase.
        self.timebase = self.video.exp_dat.t.values
        self.npix     = [self.video.exp_dat.frames.px.values.size,
                         self.video.exp_dat.frames.py.values.size]

        # Creating empty variables useful for later.
        self.noise_monitor = dict()
        self.cur_tt_idx = 0

    def add_camera(self, camera: str, shot: int=None, method: str = 'corr',
                   t_roi: float=None):
        """"
        Appends a camera monitor for noise removal. This allows either to choose
        between two methods:
            1. corr: use windowed correlation coefficients to establish a
                     correlation between the camera evolution and the video
                     pixels. The more correlated a pixel and the monitor counts,
                     the larger is the probability of being a noise is assumed.
            2. copy: the input noise monitor is used to evolve a reference frame.
                     Not yet implemented. Use the correlation.

        Pablo Oyola - poyola@us.es

        :param camera: camera name.
        :param shot: if the shot was retrieved from the initial video, this will
        be ignored. Otherwise, it becomes a mandatory argument.
        :param method: method to apply the correlation. Either 'corr' or 'copy'.
        :param t_roi: time point to select a ROI in the camera. If None, all the
        camera will be used instead.
        """

        if method.lower() != 'corr':
            raise NotImplementedError('The method %s is not' + \
                                      '(yet?) implemented!'%method)

        # Checking whether the shot number has been provided.
        if 'shot' not in self.__dict__:
            if shot is None:
                raise Exception('A shotnumber is required!')
        else:
            shot = self.shot

        # Creating the object to handle the data.
        obj = vrt_noise_corr(camera=camera, shot=shot, t_roi=t_roi)

        name = '%s_correlation'%camera
        self.noise_monitor[name] = obj

    def add_trace(self, timetrace: Union[TimeTrace, float, xr.DataArray],
                  time: float=None, method: str='corr'):
        """"
        Uses a time trace to generate the noise removal:
            1. corr: use windowed correlation coefficients to establish a
                     correlation between the camera evolution and the video
                     pixels. The more correlated a pixel and the monitor counts,
                     the larger is the probability of being a noise is assumed.
            2. copy: the input noise monitor is used to evolve a reference frame.
                     Not yet implemented. Use the correlation.

        Pablo Oyola - poyola@us.es

        :param camera: camera name.
        :param shot: if the shot was retrieved from the initial video, this will
        be ignored. Otherwise, it becomes a mandatory argument.
        :param method: method to apply the correlation. Either 'corr' or 'copy'.
        :param t_roi: time point to select a ROI in the camera. If None, all the
        camera will be used instead.
        """

        if method.lower() != 'corr':
            raise NotImplementedError('The method %s is not' + \
                                      '(yet?) implemented!'%method)
        # Special case for a time trace
        if isinstance(timetrace, TimeTrace):
            timetrace0 = timetrace.mean_of_roi
        else:
            timetrace0 = timetrace

        # Creating the object to handle the data.
        obj = corr_monitor_noise(monitor=timetrace0, time=time)

        name = 'TimeTrace_corr_%d' % self.cur_tt_idx
        self.cur_tt_idx += 1
        self.noise_monitor[name] = obj

    def from_video(self, t0: float, method: str='corr'):
        """
        Initializes a new noise monitor starting from a timetrace of a given
        piece of the video itself.

        Pablo Oyola - poyola@us.es
        """
        # If the time trace is not provided, we give the user the opportunity
        # to choose a part of the scintillator to use its timetrace as a
        # signal shape for the noise substraction.
        timetrace = self.video.getTimeTrace(t=t0)[0]

        # We first rescale the monitor to the range [0, 1]
        monitor = timetrace['mean_of_roi'].values \
            - timetrace['mean_of_roi'].values.min()
        monitor /= monitor.max()

        # In case the monitor is not evaluated at the same points:
        tmin = timetrace['t'].values[0]
        dt   = timetrace['t'].values[1] - timetrace['t'].values[0]
        nt   = timetrace['t'].values.size
        monitor = fast_interp1(tmin, dt, nt, monitor[None, :].T,
                               self.video.exp_dat.t.values).squeeze()
        monitor = np.maximum(0.0, monitor)

    def rm_noise(self, only_apply: str=None):
        """
        Using the internal noise handlers, we use a Bayesian approach to
        remove the noise.

        Pablo Oyola - poyola@us.es

        :param frames: frames to remove the noise to.
        :param time: if the input frames is not xarray.DataArray, the
        time input must explicitly be provided.
        :param only_apply: list with the noise filters to be applied.
        """

        frames = self.video.exp_dat.frames

        time, image = self.remove_noise(frames=frames, only_apply=only_apply)

        self.helper = (time, image)

        # We now clone the video object.
        new_vid = BVO(empty=True)
        for ii in self.video.__dict__:
            new_vid.__dict__[ii] = self.video.__dict__[ii]

        new_vid.exp_dat['frames'] = \
            xr.DataArray(image, dims=('px', 'py', 't'),
                         coords={'px': np.arange(image.shape[0]),
                                 'py': np.arange(image.shape[1]),
                                 't': time.squeeze()})
        new_vid.type_of_file = 'parsed'

        return new_vid

    def remove_noise(self, frames: float, time: float=None,
                     only_apply: str=None):
        """
        Using the internal noise handlers, we use a Bayesian approach to
        remove the noise.

        Pablo Oyola - poyola@us.es

        :param frames: frames to remove the noise to.
        :param time: if the input frames is not xarray.DataArray, the
        time input must explicitly be provided.
        :param only_apply: list with the noise filters to be applied.
        """

        # Checking that all the filters are available.

        if only_apply is None:
            only_apply = list(self.noise_monitor.keys())
        else:
            if isinstance(only_apply, str):
                only_apply = (only_apply,)

            # Loop to see whether there are some filters that are not available
            # in the object.
            for ii in only_apply:
                if ii not in self.noise_monitor:
                    raise ValueError('Noise monitor %s is not in the list!'%ii)

        # Checking the time input.
        if isinstance(frames, xr.DataArray):
            try:
                timebase = frames.t.values
            except:
                logger.info('Time basis not found in the data array')
                if time is None:
                    raise ValueError('Time basis must be provided')
                timebase = time

            frames0 = frames.values
        else:
            frames0 = frames
            if time is None:
                raise ValueError('Time basis must be provided')

            timebase = time

        if timebase.shape[0] != frames0.shape[-1]:
            raise ValueError('Frames and time must share the same 1st axis shape')

        # Check whether the video object has some scintillator-based mask
        try:
            mask = self.video.scint_mask
        except:
            logger.info('Scintillator mask not found. Computing full video.')
            mask = None

        # Applying filter
        factor = np.ones_like(frames0)
        for ii in only_apply:
            tmp = self.noise_monitor[ii]
            logger.debug(f'Applying filter {ii}')

            time, prob, s0 = tmp.get_noise(frames0, timebase, mask=mask)

        return time, prob, s0

