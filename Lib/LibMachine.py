"""Detect in which machine are we (AUG, SMART...)"""
import os
if os.path.isdir('/afs/ipp-garching.mpg.de'):
    machine = 'AUG'
