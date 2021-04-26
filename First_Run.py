"""
Test the installed folders and dependencies

jose rueda: jrrueda@us.es

This script should check if youhave installed all modules which are necesary
to run the suite and, if you say yes, install the ones missing.
pip installations seems a bit more complicated, you must run the command
yourself. Sorry

If git is not found, call 'module load git'
"""
from paths_suite import paths_of_the_suite
import os
# -----------------------------------------------------------------------------
# --- Check folders:
# -----------------------------------------------------------------------------
paths_of_the_suite()
import LibMachine as m
import LibPaths as p
machine = m.machine
print('We are in: ', machine)
paths = p.Path(machine)
# Test FILDSIM:
if not os.path.isdir(paths.FILDSIM):
    print('FILDSIM code not found, revise LibPaths.py')
if not os.path.isdir(paths.INPASIM):
    print('INPASIM code not found, revise LibPaths.py')
if not os.path.isdir(paths.Results):
    print('Results directory not found, creating it')
    os.system('mkdir ' + paths.Results)
if not os.path.isdir(paths.FILDStrikeMapsRemap):
    print('StrikeMaps for remapping not found, revise LibPaths.py')
if not os.path.isdir(paths.FILDStrikeMapsTomography):
    print('StrikeMaps for remapping not found, revise LibPaths.py')
if not os.path.isdir(paths.ScintSuite):
    print('The name of the Suite directory is not correct in LibPaths.py')
    print('Most likely due to wrong, git clone, just change the folder name:')
    print('Call it: ScintillatorSuite')
if not os.path.isdir(paths.tracker):
    print('iHIMSIM tracker not found!')
if not os.path.isdir(os.path.join(paths.ScintSuite, 'MyRoutines')):
    os.system('mkdir ' + os.path.join(paths.ScintSuite, 'MyRoutines'))

# -----------------------------------------------------------------------------
# --- Check necesary modules:
# -----------------------------------------------------------------------------
try:
    from roipoly import RoiPoly
except ImportError:
    print('roipoly not found, you won"t be able to define ROIs')
    x = int(input('Install it?: (1 = yes) '))
    if x == 1:
        instructions = \
            'cd \n' +\
            'git clone https://github.com/georgedeath/roipoly.py.git \n' +\
            'cd roipoly.py \n' +\
            'git checkout bugfix-spyder'
        os.system(instructions)

try:
    import cv2
except ImportError:
    print('There would be no support for the mp4 videos, open cv not found')
    print('To install it:')
    print('Just execute "pip install opencv-python" in this python console')

try:
    import lmfit
except ImportError:
    print('lmfit not found, you cannot calculate resolutions')
    print('To install it:')
    print('Just execute "pip install lmfit" in this python console')

try:
    import f90nml
except ImportError:
    print('f90nml not found, you cannot remap')
    print('To install it:')
    print('Just execute "pip install f90nml" in this python console')
