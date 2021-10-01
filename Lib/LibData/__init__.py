"""Routines to load data from the tokamak database"""

from Lib.LibMachine import machine
if machine == 'AUG':
    from Lib.LibData.AUG import *
