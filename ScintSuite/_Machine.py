"""Detect in which machine are we (AUG, SMART...)"""
import os
import sys
import logging
import ScintSuite.errors as errors
logger = logging.getLogger('ScintSuite.Machine')

# ------------------------------------------------------------------------------
# %% Check the machine
# ------------------------------------------------------------------------------
detectedMachines = []
# ---- Check if we are in AUG
if os.path.isdir('/shares/departments/AUG'):
    # We could be in AUG
    try:
        import aug_sfutils
        detectedMachines.append('AUG')
        machine = 'AUG'
    except ModuleNotFoundError:
        pass
# ---- Check if we are at D3D
if os.path.isdir('/fusion/projects/xpsi'):
    # We are in D3D
    detectedMachines.append('D3D')
    machine = 'D3D'
    # Add the path to the s3-pure library, this is to download FILD data
sys.path.append('/fusion/usc/src/s3-pure')
# ---- Check if we are in MU
if os.path.isdir('/common/uda-scratch') or os.path.isdir('/home/muadmin/package'):
    detectedMachines.append('MU')
    machine = 'MU'
else:
    try:
        import pyEquilibrium
        detectedMachines.append('MU')
        machine = 'MU'
    except ModuleNotFoundError:
        pass
# ---- Check that we only have one possitive
if len(detectedMachines) == 1:
    #  Best case, we found ourselves
    logger.info('Detected machine: %s' % detectedMachines[0])
elif len(detectedMachines) == 0:
    machine = 'Generic'
    text = 'Not recognised tokamak enviroment, no database available'
    logger.warning('23: %s' % text)
if len(detectedMachines) > 1:
    logger.error('We have detected more than one machine. Wehere are you?')
    logger.error('Detected machines %s' % detectedMachines)
    raise errors.NotMachineSelected(text)

