"""Paths to the different folders and codes"""
import os
from Lib.LibMachine import machine

class Path:
    """Paths of the different codes and folders"""

    ## @todo: include here a proper machine dependent path to png files
    def __init__(self, machine=machine):
        """Initialise the class"""
        home = os.getenv("HOME")
        self.FILDSIM = os.path.join(home, 'FILDSIM/')
        self.INPASIM = os.path.join(home, 'INPASIM/')
        self.Results = os.path.join(home, 'ScintSuite/Results')
        self.ScintSuite = os.path.join(home, 'ScintSuite/')
        self.FILDStrikeMapsRemap = os.path.join(self.ScintSuite, 'Data',
                                                'StrikeMaps', 'FILD', 'Remap')
        self.FILDStrikeMapsTomography = \
            os.path.join(self.ScintSuite, 'Data',
                         'StrikeMaps', 'FILD', 'Tomography')
        self.tracker = os.path.join(home, 'ihibpsim', 'bin', 'tracker.go')
        self.ihibpsim_strline_database = os.path.join(self.ScintSuite,
                                                      'Data',
                                                      'StrikeMaps',
                                                      'iHIBP')
        if machine == 'AUG':
            self.iHIBP_videos = '/afs/ipp/home/a/augd/rawfiles/VRT/'
            self.FILDStrikeMapsRemap += '/AUG/'
            self.FILDStrikeMapsTomography += '/AUG/'
            self.bcoils_phase_corr = os.path.join(self.ScintSuite,
                                                  'Data',
                                                  'Magnetics',
                                                  'balloning_correction.dat')
