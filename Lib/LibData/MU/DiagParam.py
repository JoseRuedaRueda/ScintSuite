"""Diagnostics and parameters of MAST Upgrade"""

import numpy as np
import warnings
# -----------------------------------------------------------------------------
# --- MU parameters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In AUG shots numbers are written with 5 numbers 00001

## Field and current direction
Bt_sign = -1   # +1 Indicates the positive phi direction (counterclockwise)
It_sign = +1  # -1 Indicates the negative phi direction (clockwise)
IB_sign = Bt_sign * It_sign

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
# All values except for beta, are extracted from the paper:
# J. Ayllon-Guerola et al. 2019 JINST14 C10032
# betas are taken to be -12.0 for AUG
# fild 5 alpha extracted from FARO measurements
fild1 = {'path': '/p/IPP/AUG/rawfiles/FIT/',  # Path for the video files
         'extension': '_v710.cin'}  # Extension of the video file, none for png


fild2 = {'alpha': 0.0, 'beta': -12.0, 'sector': 3, 'r': 2.180,
         'z': 0.3, 'phi_tor': 57.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD2/',
         'extension': '', 'label': 'FILD2', 'diag': 'FHA', 'channel': 'FIPM_',
         'nch': 20, 'camera': 'CCD'}


FILD = (fild1)
