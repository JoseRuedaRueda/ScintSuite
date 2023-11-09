"""Detect in which machine are we (AUG, SMART...)"""
import os
import logging
logger = logging.getLogger('ScintSuite.Machine')

try:
    import aug_sfutils

    journal = aug_sfutils.JOURNAL()
except:
    if os.path.isdir('/common/uda-scratch'):
        machine = 'MU'
    elif os.path.isdir('/home/muadmin/package'):
        machine = 'MU'
    elif os.path.isdir('/afs/ipp/aug/ads-diags/common/python/lib'):
        try:
            import aug_sfutils
            machine = 'AUG'
        except ModuleNotFoundError:
            machine = 'Generic'
    else:
        machine = 'Generic'
        text = 'Not recognised tokamak enviroment, no database available'
        logger.warning('23: %s' % text)
else:
    machine = 'AUG'
    text = 'Recognised AUG tokamak enviroment, database available'
    logger.info('%s' % text)