"""Routines to load data from the tokamak database"""

from ScintSuite._Machine import machine
if machine == 'AUG':
    from ScintSuite.LibData.AUG import *
elif machine == 'MU':
    from ScintSuite.LibData.MU import *
else:
    from ScintSuite.LibData.Generic import *
