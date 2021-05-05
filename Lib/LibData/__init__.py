"""Contains routine to load data from the tokamak database"""

from LibMachine import machine
if machine == 'AUG':
    from AUG import *
