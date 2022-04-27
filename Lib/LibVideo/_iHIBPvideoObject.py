"""
iHIBP video object.

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""
from Lib.LibVideo._BasicVideoObject import BVO
import xml.etree.ElementTree as et
import os
import numpy as np
import matplotlib.pyplot as plt
import Lib.errors as errors
import Lib.LibPaths as p
from Lib.LibMachine import machine
import Lib.LibMap.Calibration as libcal
from Lib.LibMap.Scintillator import Scintillator
import Lib.LibData.AUG.DiagParam as params
import struct
pa = p.Path(machine)
del p


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
                 scobj: Scintillator=None):
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
