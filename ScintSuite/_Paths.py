"""Paths to the different folders and codes"""
import os
from ScintSuite._Machine import machine
from ScintSuite._SideFunctions import update_case_insensitive
import yaml

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
        self.Results = {
            'default': os.path.join(self.ScintSuite, 'Results')
        }
        self.StrikeMapDatabase = {
            'FILD': os.path.join(self.ScintSuite, 'Data',
                                                'StrikeMaps', 'FILD', 'Remap')
        }
        self.ScintPlates = os.path.join(self.ScintSuite, 'Data', 'Plates')
        self.FILDPositionDatabase = ''
        self.INPALogbook = ''
        if machine == 'AUG':
            self.StrikeMapDatabase['FILD'] += '/AUG/'
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
        filename = os.path.join(self.ScintSuite, 'Settings.yml')
        with open(filename, 'r') as stream:
            try:
                settings = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                raise Exception('Error reading the settings file')
        nml = settings['UserPaths']
        update_case_insensitive(self.__dict__, nml)
    
    def __getitem__(self, key):
        """
        Get the path of the object
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"Path {key} not found in Path class")
    
    def __setitem__(self, key, value):
        """
        Set the path of the object
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            raise KeyError(f"Path {key} not found in Path class")
    def __contains__(self, key):
        """
        Check if the path is in the object
        """
        return key in self.__dict__
