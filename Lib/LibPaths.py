"""Paths to the different folders and codes"""
import os
from Lib.LibMachine import machine
from Lib.LibSideFunctions import update_case_insensitive
import f90nml


class Path:
    """
    Paths of the different codes and folders

    It will try to look for the paths in the file Data/MyData/Paths.txt

    Please do not change this class to avoid merge conflics
    """

    def __init__(self, machine=machine):
        """Initialise the class"""
        home = os.getenv("HOME")
        self.FILDSIM = os.path.join(home, 'FILDSIM/')
        self.SINPA = os.path.join(home, 'SINPA/')
        self.ScintSuite = os.path.join(home, 'ScintSuite/')
        self.Results = os.path.join(self.ScintSuite, 'Results')
        self.FILDStrikeMapsRemap = os.path.join(self.ScintSuite, 'Data',
                                                'StrikeMaps', 'FILD', 'Remap')
        self.FILDStrikeMapsTomography = \
            os.path.join(self.ScintSuite, 'Data',
                         'StrikeMaps', 'FILD', 'Tomography')
        self.FILDPositionDatabase = ''
        self.INPALogbook = ''

        self.iHIBPsim = os.path.join(home, 'ihibpsim')
        self.ihibp_bins = os.path.join(self.iHIBPsim, 'bin')
        self.ihibp_res = os.path.join(self.iHIBPsim, 'sims')
        self.ihibp_repo = os.path.join(self.iHIBPsim, 'repo')
        self.tracker = os.path.join(self.ihibp_bins, 'tracker.go')

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
                                                  'Magnetics')
            self.FILD4_trajectories = '/afs/ipp/home/j/javih/FILD4/'
            self.fonts = [
                '/usr/share/fonts/truetype',
                '/usr/share/fonts/truetype/ms-corefonts',
                '/usr/share/texmf/fonts/opentype/public/lm',
                '/usr/share/texmf/fonts/opentype/public/lm-math'
            ]
        else:  # Generic case, assume you have linux :-)
            self.fonts = [
                '/usr/share/fonts/truetype',
                '/usr/share/fonts/opentype',
            ]
        # Load the custom paths
        file = os.path.join(self.ScintSuite, 'Data', 'MyData', 'Paths.txt')
        nml = f90nml.read(file)
        update_case_insensitive(self.__dict__, nml['paths'])
