"""
iHIBP video object.

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""
from Lib.LibVideo._BasicVideoObject import BVO
import xml.etree.ElementTree as et
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import gamma, gammainc
import Lib.errors as errors
import Lib.LibPaths as p
from Lib.LibMachine import machine
import Lib.LibMap.Calibration as libcal
from Lib.LibMap.Scintillator import Scintillator
import Lib.LibData.AUG.DiagParam as params
import Lib.LibTimeTraces as sstt
pa = p.Path(machine)
del p

def decision(prob: float, x0: float, dx0: float):
    """
    Decision function choosing whether a pixel is noise or not.
    """

    assert (x0 > 0.0) and (x0 < 1.0), 'Probability threshold must be in [0, 1]'
    assert (dx0 > 0.0) and (dx0 < 1.0), 'Probability dependence must be in [0, 1]'

    f = 0.5*(1.0 + np.tanh((prob-x0)/dx0))

    return f


# --- Auxiliar routines to find the path towards the camera files
def guessiHIBPfilename(shot: int):
    """
    Guess the filename of a video

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shot: shot number

    @return f: the name of the file/folder
    """
    base_dir  = params.iHIBPext[0]['path'](shot)
    extension = params.iHIBPext[0]['extension'](shot)

    f = None
    if shot < 99999:  # PCO camera, stored in AFS
        name = 'S%05d_HIBP.%s'%(shot, extension)
        f = os.path.join(base_dir, name)

    return f

def ihibp_get_time_basis(shot: int):
    """"
    The iHIBP videos are defined in terms of time relative to the beginning of
    the recording trigger, and we need time relative to the shot trigger.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shot: shot to get the timing.
    """
    fn = params.iHIBPext[0]['path_times'](shot)

    if not os.path.isfile(fn):
        raise FileNotFoundError('Cannot find the path to access the frames'+\
                                ' timing: %s'%fn)

    # Opening the time configuration file.
    root = et.parse(fn).getroot()

    # From the root we get the properties.
    properties = dict()
    for ikey in root.keys():
        properties[ikey] = root.get(ikey)
        try:
            if properties[ikey][0:2] == '0x':
                properties[ikey] = int(properties[ikey], base=16)
            else:
                properties[ikey] = float(properties[ikey])
                if float(int(properties[ikey])) == float(properties[ikey]):
                    properties[ikey] = int(properties[ikey])
        except:
            pass

    # Now we read the rest of the entries.
    nframs = len(root)
    timestamp = np.zeros((nframs,), dtype=float)
    framenum  = np.zeros((nframs,), dtype=int)
    for ii, iele in enumerate(root):
        tmp = iele.attrib
        framenum[ii] = int(tmp['frameNumber'])
        timestamp[ii] = float(int(tmp['time'], base=16))

    # By substracting thte TS06 time (shot trigger) we get shot-relative
    # timing.
    timestamp -= properties['ts6Time']

    # The time so calculated is given in nanoseconds:
    timestamp = timestamp * 1e-9 # -> to seconds
    return timestamp, framenum, properties

class iHIBPvideo(BVO):
    """
    Basic video object for the iHIBP camera and data handling.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """
    def __init__(self, shot: int, calib: libcal.CalParams=None,
                 scobj: Scintillator=None, signal_threshold: float=5.0):
        """
        Initializes the object with the video data if found.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param shot: pulse number to read video data.
        """

        # --- Try to load the shotfile
        fn = guessiHIBPfilename(shot=shot)
        if not os.path.isfile(fn):
            raise errors.DatabaseError('Cannot find video for shot #%05d'%shot)
        # We initialize the parent class with the iHIBP video.
        super().__init__(file=fn, shot=shot)

        # Getting the time correction.
        self.timecal, self.cam_prop, _ = ihibp_get_time_basis(shot=shot)
        self.exp_dat['tframes'] = self.timecal
        self.timebase = self.timecal


        # Let's check whether the user provided the calibration parameters.
        if calib is None:
            print('Retrieving the calibration parameters for iHIBP')
            caldb = libcal.CalibrationDatabase(pa.ihibp_calibration_db)
            self.calib = caldb.get_calibration(shot=shot,
                                               camera='SLOWCAMERA',
                                               cal_type='PIX',
                                               diag_ID=1)
        else:
            self.calib = calib


        # --- Checking if the scintillator plate is provided:
        if scobj is None:
            print('Getting standard scintillator plate for iHIBP')
            fn_sc = pa.ihibp_scint_plate
            self.scintillator = Scintillator(file=fn_sc, format='FILDSIM')
        else:
            self.scintillator = scobj

        # Updating the calibration in the scintillator.
        self.scintillator.calculate_pixel_coordinates(self.calib)

        # --- Apply now the background noise substraction.
        self.subtract_noise(t1=-1.0, t2=0.0, flag_copy=True)

        # --- i-HIBP scintillator distorted.
        self.scint_path = np.array(self.scintillator.get_path_pix()).T[1:, ...]
        mask = sstt.roipoly(path = self.scint_path)
        self.scint_mask = mask.getMask(self.exp_dat['frames'][..., 0])

        # --- Getting which is the first illuminated frame:
        tt = sstt.TimeTrace(self, self.scint_mask)
        flags = tt.time_base > self.exp_dat['t2_noise']
        time = tt.time_base[flags]
        self.dsignal_dt = tt.mean_of_roi[flags]

        t0_idx = np.where(self.dsignal_dt > signal_threshold)[0][0]
        self.t0     = time[t0_idx]
        print('Using t0 = %.3f as the reference frame'%self.t0)
        self.frame0 = self.exp_dat['frames'][..., self.getFrameIndex(t=self.t0)]

    def plot_frame(self, plotScintillatorPlate: bool=True, **kwargs):
        """"
        This function wraps the parent plot_frame function to plot the frame
        along with the scintillator plate.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @params kwargs: same arguments than BVO.plot_frame.
        """

        ax = super().plot_frame(**kwargs)

        if plotScintillatorPlate:
            self.scintillator.plot_pix(ax=ax, plt_par={'color': 'w'})

        return ax

    def set_background_monitor(self, timetrace: sstt.TimeTrace=None):
        """
        This function sets in the class which is going to be the time-dependence
        of the noise caused by the background light emission.

        If the user does not provide a timetrace, this function will automatically
        fall back to the standard signal. The standard signal is just asking
        the user for a small region of the scintillator where there is no signal
        and then use the timetrace of the noise evolution in that piece to
        track the noise evolution.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param timetrace: timetrace to be used as a monitor for the background
        light. If None, we will fall back to the standard monitor that is just
        taking a small piece of the scintillator.
        @param signal_threshold: minimum change in the counts number to be
        considered the first illuminated frame.
        """

        # If the time trace is not provided, we give the user the opportunity
        # to choose a part of the scintillator to use its timetrace as a
        # signal shape for the noise substraction.
        if timetrace is None:
            timetrace = self.getTimeTrace(t=self.t0)

        # We first rescale the monitor to the range [0, 1]
        monitor = timetrace.mean_of_roi - timetrace.mean_of_roi.min()
        monitor /= monitor.max()

        # In case the monitor is not evaluated at the same points:
        monitor = interp1d(timetrace.time_base, monitor, kind='linear',
                           bounds_error=False, fill_value=0.0)(self.timebase)
        monitor = np.maximum(0.0, monitor)

        # Then we scale monitor using the frame timetrace value at the initial
        # frame point.
        t0_idx = self.getFrameIndex(t=self.t0)
        scale  = 1.0 / monitor[t0_idx]

        self.frame_noise = self.frame0[:, :, None] * monitor[None, None, :]
        self.frame_noise *= scale
        self.monitor = monitor

    def substract_noise(self, x0: float=0.5, dx0: float=0.01):
        """
        Substract time-dependent background noise.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        if 'frame_noise' not in self.__dict__:
            raise Exception('A particular time-dependence'+\
                            ' of the noise is required')

        self.exp_dat['original_frames'] = self.exp_dat['frames']
        # Getting the Gaussian distance between the frame and the noise.
        self.dframe = np.maximum(0.0,self.exp_dat['frames'] - self.frame_noise)
        # self.exp_dat['frames'] =  self.dframe

        # Now, we consider that the error in the noise estimation is the max
        # between the 1% of the noise nominal value or the sigma of the
        # estimation.
        stdroi = np.ones_like(self.dframe)*np.mean(self.frame_noise, axis=(0, 1))
        self.sigma_noise = np.maximum(5.0, stdroi)

        # The probability of a given pixel signal to be noise is:
        self.prob = np.exp(-self.dframe/self.sigma_noise)

        # We will now use a decision function to decide whether a pixel is noise
        # or not.
        # mask = np.ones_like(self.dframe, dtype=bool)*self.scint_mask[..., None]
        # self.prob = np.ma.MaskedArray(self.prob, mask = ~mask)
        frames = self.exp_dat['frames']*(1.0 - decision(self.prob, x0, dx0))
        self.exp_dat['frames'] = frames

    def getTimeTrace(self, t: float = None, mask=None):
        """
        Calculate the timeTrace of the video.
        This overloads the parent function to include the possibility that if
        neither mask nor time are provided, the scintillator mask is used
        instead.


        Pablo Oyola - pablo.oyola@ipp.mpg.de

        adapted from the BasicVideoObject (BVO) from:
        Jose Rueda Rueda: jrrueda@us.es

        @param t: time of the frame to be plotted for the selection of the roi
        @param mask: bolean mask of the ROI

        @returns timetrace: a timetrace object
        """

        if (t is None) and (mask is None):
            mask = self.scint_mask

        return super().getTimeTrace(t=t, mask=mask)


