"""Diagnostics and parameters of ASDEX Upgrade"""

import numpy as np
import warnings
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
fild1 = {'path': '/p/IPP/AUG/rawfiles/FIT/',  # Path for the video files
         'camera': 'PHANTOM',  # Type of used camera
         'extension': '_v710.cin',  # Extension of the video file, none for png
         'label': 'FILD1',  # Label for the diagnostic, for FILD6 (rFILD)
         'diag': 'FHC',  # name of the diagnostic for the fast channel
         'channel': 'FILD3_',  # prefix of the name of each channel (shotfile)
         'nch': 20}  # Number of fast channels

fild2 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD2/',
         'extension': '', 'label': 'FILD2', 'diag': 'FHA', 'channel': 'FIPM_',
         'nch': 20, 'camera': 'CCD'}

fild3 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD3/',
         'extension': '', 'label': 'FILD3', 'diag': 'xxx', 'channel': 'xxxxx',
         'nch': 99, 'camera': 'CCD'}

fild4 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD4/',
         'extension': '', 'label': 'FILD4', 'diag': 'FHD', 'channel': 'Chan-',
         'nch': 32, 'camera': 'CCD'}

fild5 = {'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD5/',
         'extension': '', 'label': 'FILD5', 'diag': 'FHE', 'channel': 'Chan-',
         'nch': 64, 'camera': 'CCD'}

FILD = (fild1, fild2, fild3, fild4, fild5)

# -----------------------------------------------------------------------------
# --- IHIBP PARAMETERS
# -----------------------------------------------------------------------------
IHIBP_scintillator_X = np.array((0.0, 6.6))  # [cm]
IHIBP_scintillator_Y = np.array((-17.0, 0.0))  # [cm]

iHIBP = {'port_center': [0.687, -3.454, 0.03], 'sector': 13,
         'beta_std': 4.117, 'theta_std': 0.0, 'source_radius': 7.0e-3}

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
