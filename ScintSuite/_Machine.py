"""Detect in which machine are we (AUG, SMART...)"""
import os
import logging
logger = logging.getLogger('ScintSuite.Machine')

# ------------------------------------------------------------------------------
# %% Check the machine
# ------------------------------------------------------------------------------
AUG = False
MU = False
# ---- Check if we are in AUG
try:
    import aug_sfutils
    AUG = True
    machine = 'AUG'
except ModuleNotFoundError:
    pass
# ---- Check if we are in MU
if os.path.isdir('/common/uda-scratch') or os.path.isdir('/home/muadmin/package'):
    MU = True
    machine = 'MU'
else:
    try:
        import pyEquilibrium
        MU = True
        machine = 'MU'
    except ModuleNotFoundError:
        pass
# ---- Check that we only have one possitive
if AUG and MU:
    text = 'Both AUG and MU are True, this is not possible'
    logger.error('%s' % text)
    raise ValueError(text)

elif not (AUG or MU):
    machine = 'Generic'
    text = 'Not recognised tokamak enviroment, no database available'
    logger.warning('23: %s' % text)
