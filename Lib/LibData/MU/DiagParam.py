"""Diagnostics and parameters of MAST Upgrade"""

# -----------------------------------------------------------------------------
# --- MU parameters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In MU shots numbers are written with 5 numbers 00001

## Field and current direction
Bt_sign = -1   # +1 Indicates the positive phi direction (counterclockwise)
It_sign = +1  # -1 Indicates the negative phi direction (clockwise)
IB_sign = Bt_sign * It_sign

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
# All values except for beta, are extracted from XXXXXXXXXXX:
#
firstShot = 46515
fild1 = {'adqfreq': lambda shot:\
    23 if shot < firstShot else 100,  # Extension of the video
    't_trig': lambda shot:\
    -2.5 if shot < firstShot else -1.0,
    'extension': lambda shot: '' if shot < firstShot else '.nc', 
    'label': 'FILD1', 'path': lambda shot:\
    # '/home/jrivero/FILD_MASTu_data/FILD_CCD' if shot < firstShot else '/fild-data/XIMEAshotfiles'}
    # '/home/jrivero/FILD_MASTu_data/FILD_CCD' if shot < firstShot else '/FILD1_remote_store'}
    '/home/jrivero/FILD_MASTu_data/FILD_CCD' if shot < firstShot else '/home/jqw5960/mastu/experiments/MU02'}
# fild1 = {'adqfreq': 23, 't_trig': -2.5,
#          'extension': lambda shot: '', 'label': 'FILD1', 'camera': 'CCD',
#          'path': ,}

FILD = (fild1,)
