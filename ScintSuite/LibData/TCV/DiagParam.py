"""Diagnostics and parameters of TCV"""

# -----------------------------------------------------------------------------
# --- TCV parameters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In TCV shots numbers are written with 5 numbers 00001

## Field and current direction
Bt_sign = -1   # +1 Indicates the positive phi direction (counterclockwise)
It_sign = +1  # -1 Indicates the negative phi direction (clockwise)
IB_sign = Bt_sign * It_sign

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
# All values except for beta, are extracted from XXXXXXXXXXX:
#
fild1 = {'adqfreq': 23, 't_trig': -2.5,
         'extension': lambda shot: '', 'label': 'FILD1', 'camera': 'CCD',
         'path': '/home/jrivero/FILD_MASTu_data/FILD_CCD',}

FILD = (fild1,)
