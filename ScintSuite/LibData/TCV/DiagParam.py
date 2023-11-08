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
#modified by J. Poley, 7/11/2023
fild1 = {'adqfreq': 1000, 't_trig': 0.,
         'extension': lambda shot: '', 'label': 'FILD', 'camera': 'CCD',
         'path': '/videodata/pcfild002/data/fild002',}

FILD = (fild1,)
