"""Paths to the different folders and codes"""
import os


class Path:
    """Paths of the different codes and folders"""

    ## @todo: include here a proper machine dependent path to png files
    def __init__(self, machine='AUG'):
        """Initialise the class"""
        home = os.getenv("HOME")
        self.FILDSIM = os.path.join(home, 'FILDSIM/')
        self.INPASIM = os.path.join(home, 'INPASIM/')
        self.Results = os.path.join(home, 'ScintSuite/Results')
        self.StrikeMaps = os.path.join(home, 'FILD_Strike_maps2/')
        self.ScintSuite = os.path.join(home, 'ScintSuite/')
        self.tracker = os.path.join(home, 'iHIBPsim', 'bin/')
        if machine == 'AUG':
            self.CinFiles = '/p/IPP/AUG/rawfiles/FIT/'
            self.PngFiles = '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/'
            self.iHIBP_videos = '/afs/ipp/home/a/augd/rawfiles/VRT/'
