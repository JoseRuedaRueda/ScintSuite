"""Routines to load data from the tokamak database"""

from Lib._Machine import machine
if machine == 'AUG':
    from Lib.LibData.AUG import *
elif machine == 'MU':
    from Lib.LibData.MU import *
else:
    from Lib.LibData.Generic import *
