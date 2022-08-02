"""Diagnostics and parameters of ASDEX Upgrade"""

import numpy as np
import Lib.errors as errors
from math import pi
import os
# -----------------------------------------------------------------------------
# --- AUG parameters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In AUG shots numbers are written with 5 numbers 00001

## Field and current direction
## @todo> This is hardcored here, at the end there are only 2 weeks of reverse
# field experiments in  the whole year, but if necesary, we could include here
# some kind of method to check the sign calling the AUG database
Bt_sign = -1   # +1 Indicates the positive phi direction (counterclockwise)
It_sign = 1  # -1 Indicates the negative phi direction (clockwise)
IB_sign = Bt_sign * It_sign

num_of_gyrotrons = 8  # Number of gyrotrons available in AUG.

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
_fild1 = {'path': '/p/IPP/AUG/rawfiles/FIT/',  # Path for the video files
          'camera': 'PHANTOM',  # Type of used camera
          'extension': lambda shot:\
          '_v710.cin' if shot < 41202 else '_ID9404.cin',  # Extension of the video
          'label': 'FILD1',  # Label for the diagnostic, for FILD6 (rFILD)
          'diag': 'FHC',  # name of the diagnostic for the fast channel
          'channel': 'FILD3_',  # prefix of the name of each channel (shotfile)
          'nch': 20}  # Number of fast channels

_fild2 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD2/',
          'extension': lambda shot: '', 'label': 'FILD2', 'diag': 'FHA',
          'channel': 'FIPM_',
          'nch': 20, 'camera': 'CCD'}

_fild3 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD3/',
          'extension': lambda shot: '', 'label': 'FILD3', 'diag': 'xxx',
          'channel': 'xxxxx',
          'nch': 99, 'camera': 'CCD'}

# FILD4 coil position from CAD. Coil dimensions in catholic units.
# Parking position from FARO measurements
_fild4 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD4/',
          'extension': lambda shot: '', 'label': 'FILD4', 'diag': 'FHD',
          'channel': 'Chan-',
          'nch': 32, 'camera': 'CCD', 'coil': {'R_coil': 2.2252,
                                               'Z_coil': -0.3960,
                                               'l': 0.115, 'A': 0.00554,
                                               'N': 250,
                                               'theta_parking': -21.4/180*pi,
                                               'R_parking': 2.0824,
                                               'Z_parking': -0.437}}

_fild5 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD5/',
          'extension': lambda shot: '', 'label': 'FILD5', 'diag': 'FHE',
          'channel': 'Chan-',
          'nch': 64, 'camera': 'CCD'}

FILD = (_fild1, _fild2, _fild3, _fild4, _fild5)


# -----------------------------------------------------------------------------
# --- IHIBP PARAMETERS
# -----------------------------------------------------------------------------
IHIBP_scintillator_X = np.array((0.0, 6.6))  # [cm]
IHIBP_scintillator_Y = np.array((-17.0, 0.0))  # [cm]

iHIBP = {'port_center': [0.687, -3.454, 0.03], 'sector': 13,
         'beta_std': 4.117, 'theta_std': 0.0, 'source_radius': 7.0e-3}


def _iHIBP1_path(shot: int = 42000):
    """
    Contain hardcored path of were iHIBP camera data is stored.
    """

    shot_str = '%05d' % shot
    shot_path = os.path.join(shot_str[0:2], 'S%05d' % shot)

    if shot < 99999:
        path = os.path.join('/afs/ipp/home/a/augd/rawfiles/VRT', shot_path)
    else:
        raise errors.NotValidInput('Wrong shot number?')
    return path


def _iHIBP1_timepath(shot: int = 42000):
    """
    Contain hardcored path of were iHIBP camera data is stored.
    """

    shot_str = '%05d' % shot
    shot_path = os.path.join(shot_str[0:2], 'S%05d' % shot)

    if shot < 40395:
        path = os.path.join('/afs/ipp/home/a/augd/rawfiles/VRT', shot_path,
                            'Prot', 'FrameProt', 'HIBP_FrameProt.xml')
    elif shot < 99999:
        name = 'S%s_HIBP.meta.xml' % shot_str
        path = os.path.join('/afs/ipp/home/a/augd/rawfiles/VRT', shot_path,
                            name)
    else:
        raise errors.NotValidInput('Wrong shot number?')
    return path


def _iHIBP1_extension(shot: int = 42000):
    """
    Contain hardcored extensions of were iHIBP camera data.
    """
    if shot < 99999:
        ext = 'mp4'
    else:
        raise errors.NotValidInput('Wrong shot number?')
    return ext


_ihibp1 = {
    'path': _iHIBP1_path,  # Path for the video files
    # Extension of the video file, none if png.
    'extension': _iHIBP1_extension,
    'path_times': _iHIBP1_timepath
}

iHIBPext = (_ihibp1,)
# -----------------------------------------------------------------------------
# --- INPA
# -----------------------------------------------------------------------------


def _INPA1_path(shot=42000):
    """
    Contain hardcored paths of were INPA data is stored

    Last update: 09/02/2022
    """
    if shot < 40260:
        path = '/afs/ipp-garching.mpg.de/home/f/fild/INPA1'
    elif shot < 41202:
        path = '/afs/ipp-garching.mpg.de/home/a/augd/rawfiles/INP'
    elif shot >= 41202:
        path = '/p/IPP/AUG/rawfiles/FIT/'
    else:
        raise errors.NotValidInput('Wrong shot number?')
    return path


def _INPA1_extension(shot=42000):
    """
    Contain hardcored extensions of were INPA data is stored

    Last update: 09/02/2022
    """
    if shot < 41202:
        ext = ''
    elif shot >= 41202:
        ext = '_ID24167.cin'
    else:
        raise errors.NotValidInput('Wrong shot number?')
    return ext


_inpa1 = {
    'path': _INPA1_path,  # Path for the video files
    'extension': _INPA1_extension,  # Extension of the video file, none if png
    'label': '',  # Label for the diagnostic, for FILD6 (rFILD)
    'diag': 'NPI',  # name of the diagnostic for the fast channel
    'channel': 'PMT_C',  # prefix of the name of each channel (shotfile)
    'nch': 8  # Number of fast channels
}

INPA = (_inpa1,)

# -----------------------------------------------------------------------------
# --- Magnetics data.
# -----------------------------------------------------------------------------
mag_coils_grp2coilName = {
    'C07': ['C07', np.arange(1, 32)],
    'C09': ['C07', np.arange(1, 32)],
    'B-31_5_11': ['B31', np.arange(5, 1)],
    'B-31_32_27': ['C07', np.arange(32, 38)]
}

mag_coils_phase_B31 = (1, 2, 3, 12, 13, 14)
