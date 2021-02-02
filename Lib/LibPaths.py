"""Paths to the different folders and codes"""
import os


class Path:
    """Paths of the different codes and folders"""

    ## @todo: include here a proper machine dependent path to png files
    def __init__(self, machine='AUG'):
        """Initialise the class"""
        self.FILDSIM = os.path.join(os.getenv("HOME"), 'FILDSIM/')
        self.INPASIM = os.path.join(os.getenv("HOME"), 'INPASIM/')
        self.StrikeMaps = os.path.join(os.getenv("HOME"), 'FILD_Strike_maps2/')
        self.ScintSuite = os.path.join(os.getenv("HOME"), 'ScintSuite/')
        self.tracker = os.path.join(os.getenv("HOME"), 'iHIBPsim', 'bin/')
        if machine == 'AUG':
            self.CinFiles = '/p/IPP/AUG/rawfiles/FIT/'
            self.PngFiles = '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/'
            self.iHIBP_videos = '/afs/ipp/home/a/augd/rawfiles/VRT/'
