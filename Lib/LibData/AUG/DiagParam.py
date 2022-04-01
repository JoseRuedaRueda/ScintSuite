"""Diagnostics and parameters of ASDEX Upgrade"""

import numpy as np
import Lib.errors as errors
import warnings
from math import pi as pi
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

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
_fild1 = {'path': '/p/IPP/AUG/rawfiles/FIT/',  # Path for the video files
          'camera': 'PHANTOM',  # Type of used camera
          'extension': '_v710.cin',  # Extension of the video file, none if png
          'label': 'FILD1',  # Label for the diagnostic, for FILD6 (rFILD)
          'diag': 'FHC',  # name of the diagnostic for the fast channel
          'channel': 'FILD3_',  # prefix of the name of each channel (shotfile)
          'nch': 20}  # Number of fast channels

_fild2 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD2/',
          'extension': '', 'label': 'FILD2', 'diag': 'FHA', 'channel': 'FIPM_',
          'nch': 20, 'camera': 'CCD'}

_fild3 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD3/',
          'extension': '', 'label': 'FILD3', 'diag': 'xxx', 'channel': 'xxxxx',
          'nch': 99, 'camera': 'CCD'}

# FILD4 coil position from CAD. Coil dimensions in catholic units.
# Parking position from FARO measurements
_fild4 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD4/',
          'extension': '', 'label': 'FILD4', 'diag': 'FHD', 'channel': 'Chan-',
          'nch': 32, 'camera': 'CCD', 'coil': {'R_coil': 2.2252,
                                               'Z_coil': -0.3960,
                                               'l': 0.115, 'A': 0.00554,
                                               'N': 250,
                                               'theta_parking': -21.4/180*pi,
                                               'R_parking': 2.0824,
                                               'Z_parking': -0.437}}

_fild5 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD5/',
          'extension': '', 'label': 'FILD5', 'diag': 'FHE', 'channel': 'Chan-',
          'nch': 64, 'camera': 'CCD'}

FILD = (_fild1, _fild2, _fild3, _fild4, _fild5)


# -----------------------------------------------------------------------------
# --- IHIBP PARAMETERS
# -----------------------------------------------------------------------------
IHIBP_scintillator_X = np.array((0.0, 6.6))  # [cm]
IHIBP_scintillator_Y = np.array((-17.0, 0.0))  # [cm]

iHIBP = {'port_center': [0.687, -3.454, 0.03], 'sector': 13,
         'beta_std': 4.117, 'theta_std': 0.0, 'source_radius': 7.0e-3}


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
    elif shot < 99999:
        path = '/afs/ipp-garching.mpg.de/home/a/augd/rawfiles/INP'
    else:
        raise errors.NotValidInput('Wrong shot number?')
    return path


def _INPA1_extension(shot=42000):
    """
    Contain hardcored extensions of were INPA data is stored

    Last update: 09/02/2022
    """
    if shot < 99999:
        ext = ''
    else:
        raise errors.NotValidInput('Wrong shot number?')
    return ext


_inpa1 = {
    'path': _INPA1_path,  # Path for the video files
    'extension': _INPA1_extension,  # Extension of the video file, none if png
    'label': '',  # Label for the diagnostic, for FILD6 (rFILD)
    'diag': '',  # name of the diagnostic for the fast channel
    'channel': '',  # prefix of the name of each channel (shotfile)
    'nch': None  # Number of fast channels
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
