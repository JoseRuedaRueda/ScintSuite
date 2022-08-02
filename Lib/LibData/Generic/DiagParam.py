"""
Dummy diagnostics and paramters.

Just a gross and partial copy of the AUG library, reduced to allow you to load
the suite at your laptop
"""

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
Bt_sign = -1   # +1 Indicates the positive phi direction (counterclockwise)
It_sign = 1  # -1 Indicates the negative phi direction (clockwise)
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

FILD = (fild1)

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
