"""Detect in which machine are we (AUG, SMART...)"""
import os
import logging
logger = logging.getLogger('ScintSuite.Machine')

if os.path.isdir('/common/uda-scratch'):
    machine = 'MU'
elif os.path.isdir('/afs/ipp/aug/ads-diags/common/python/lib'):
    machine = 'AUG'
else:
    machine = 'Generic'
    text = 'Not recognised tokamak enviroment, no database available'
    logger.warning('23: %s' % text)
