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
firstShotXIMEA = 46515
firstShotPrefix = 48578
def prefix(shot:int):
    if shot < firstShotPrefix:
        pref = ''
    elif (shot > firstShotPrefix) and (shot<100000):
        pref = 'xfx0'
    else:
        pref = 'xfx'
    return pref
fild1 = {'adqfreq': lambda shot: 23 if shot < firstShotXIMEA else 500,  
    't_trig': lambda shot: -2.5 if shot < firstShotXIMEA else -1.0,
    'extension': lambda shot: '' if shot < firstShotXIMEA else '.nc', # Extension of the video
    'prefix': prefix, # Prefix of the file name
    'label': 'FILD1', 
    'path': lambda shot:\
    # '/home/jrivero/FILD_MASTu_data/FILD_CCD' if shot < firstShot else '/fild-data/XIMEAshotfiles'}
    # '/home/jrivero/FILD_MASTu_data/FILD_CCD' if shot < firstShot else '/FILD1_remote_store'}
    '/home/jrivero/FILD_MASTu_data/FILD_CCD' if shot < firstShotXIMEA else '/home/jqw5960/mastu/experiments/SHOTFILES_XIMEA'}

FILD = (fild1,)
