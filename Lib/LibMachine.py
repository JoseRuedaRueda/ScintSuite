"""Detect in which machine are we (AUG, SMART...)"""
import os
if os.path.isdir('/afs/ipp/aug/ads-diags/common/python/lib'):
    machine = 'AUG'
else:
    machine = 'Generic'
    print('Not recognised machine')
    print('Assume that your are using your personal computer')
    print('Database call will not work')
