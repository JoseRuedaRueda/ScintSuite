"""Paths to the different folders and codes"""
import os
from ScintSuite._Machine import machine
from ScintSuite._SideFunctions import update_case_insensitive
import f90nml

from ScintSuite import home

class Path:
    """
    Paths of the different codes and folders

    It will try to look for the paths in the file Data/MyData/Paths.txt

    Please do not change this class to avoid merge conflics
    """

    def __init__(self, machine=machine):
        """Initialise the class"""
        # home = os.getenv("HOME")
        home_dir_user = os.getenv("HOME")
        self.FILDSIM = os.path.join(home_dir_user, 'FILDSIM/')
        self.SINPA = os.path.join(home_dir_user, 'SINPA/')
        self.FIDASIM4 = os.path.join(home_dir_user, 'FIDASIM4/')
        self.ScintSuite = home
        self.Results = os.path.join(self.ScintSuite, 'Results')
        self.FILDStrikeMapsRemap = os.path.join(self.ScintSuite, 'Data',
                                                'StrikeMaps', 'FILD', 'Remap')
        self.FILDPositionDatabase = ''
        self.INPALogbook = ''
        self.StrikeMapDatabase = None
        if machine == 'AUG':
            self.FILDStrikeMapsRemap += '/AUG/'
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
