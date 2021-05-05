"""Diagnostics and parameters of ASDEX Upgrade"""

import numpy as np

# -----------------------------------------------------------------------------
# --- AUG parameters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In AUG shots numbers are written with 5 numbers 00001

## Field and current direction
## @todo> This is hardcored here, at the end there are only 2 weeks of reverse
# field experiments in  the whole year, but if necesary, we could include here
# some kind of method to check the sign calling the AUG database
Bt_sign = 1   # +1 Indicates the positive phi direction (counterclockwise)
It_sign = -1  # -1 Indicates the negative phi direction (clockwise)
IB_sign = Bt_sign * It_sign

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
# All values except for beta, are extracted from the paper:
# J. Ayllon-Guerola et al. 2019 JINST14 C10032
# betas are taken to be -12.0 for AUG
# fild 5 alpha extracted from FARO measurements
fild1 = {'alpha': 0.0,   # Alpha angle [deg], see paper
         'beta': -12.0,  # beta angle [deg], see FILDSIM doc
         'sector': 8,    # The sector where FILD is located
         'r': 2.180,     # Radial position [m]
         'z': 0.3,       # Z position [m]
         'phi_tor': 169.75,  # Toroidal position, [deg]
         'path': '/p/IPP/AUG/rawfiles/FIT/',  # Path for the video files
         'camera': 'PHANTOM',  # Type of used camera
         'extension': '_v710.cin',  # Extension of the video file, none for png
         'label': 'FILD1',  # Label for the diagnostic, for FILD6 (rFILD)
         'diag': 'FHC',  # name of the diagnostic for the fast channel
         'channel': 'FILD3_',  # prefix of the name of each channel (shotfile)
         'nch': 20}  # Number of fast channels

fild2 = {'alpha': 0.0, 'beta': -12.0, 'sector': 3, 'r': 2.180,
         'z': 0.3, 'phi_tor': 57.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD2/',
         'extension': '', 'label': 'FILD2', 'diag': 'FHA', 'channel': 'FIPM_',
         'nch': 20, 'camera': 'CCD'}

fild3 = {'alpha': 72.0, 'beta': -12.0, 'sector': 13, 'r': 1.975,
         'z': 0.765, 'phi_tor': 282.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD3/',
         'extension': '', 'label': 'FILD3', 'diag': 'xxx', 'channel': 'xxxxx',
         'nch': 99, 'camera': 'CCD'}

fild4 = {'alpha': 0.0, 'beta': -12.0, 'sector': 8, 'r': 2.035,
         'z': -0.462, 'phi_tor': 169.75,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD4/',
         'extension': '', 'label': 'FILD4', 'diag': 'FHD', 'channel': 'Chan-',
         'nch': 32, 'camera': 'CCD'}

fild5 = {'alpha': -41.7, 'beta': -12.0, 'sector': 7, 'r': 1.772,
         'z': -0.798, 'phi_tor': 147.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD5/',
         'extension': '', 'label': 'FILD5', 'diag': 'FHE', 'channel': 'Chan-',
         'nch': 64, 'camera': 'CCD'}

fild6 = {'alpha': 0.0, 'beta': 171.3, 'sector': 8, 'r': 2.180,
         'z': 0.3, 'phi_tor': 169.75,
         'path': '/p/IPP/AUG/rawfiles/FIT/',
         'extension': '_v710.cin', 'label': 'RFILD',
         'diag': 'FHC', 'channel': 'FILD3_', 'nch': 20, 'camera': 'CCD'}

FILD = (fild1, fild2, fild3, fild4, fild5, fild6)
## FILD diag names:
# fast-channels:
fild_diag = ['FHC', 'FHA', 'XXX', 'FHD', 'FHE', 'FHC']
fild_signals = ['FILD3_', 'FIPM_', 'XXX', 'Chan-', 'Chan-', 'FILD3_']
fild_number_of_channels = [20, 20, 99, 32, 64, 20]

# -----------------------------------------------------------------------------
# --- IHIBP PARAMETERS
# -----------------------------------------------------------------------------
IHIBP_scintillator_X = np.array((0.0, 6.6))  # [cm]
IHIBP_scintillator_Y = np.array((-17.0, 0.0))  # [cm]


# -----------------------------------------------------------------------------
# --- Magnetics data.
# -----------------------------------------------------------------------------
mag_coils_grp2coilName = {
    'C07': ['C07', np.arange(1, 32)],
    'C09': ['C07', np.arange(1, 32)],
    'B-31_5_11': ['B31', np.arange(5, 1)],
    'B-31_32_27': ['C07', np.arange(32, 38)]
}
