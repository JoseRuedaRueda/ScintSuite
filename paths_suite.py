"""@package paths
Contains the path to initialise the suite
"""

import os
import sys


def paths_of_the_suite(machine='AUG'):
    """
    Add to the path all the necesary forlders for the suite to run.

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

    @param machine: Machine where we are working
    """

    # --- Section 0: Name of the auxiliar folders (located at home directory)
    ROIPOLY = 'roipoly.py-bugfix-spyder/roipoly'
    HOME_DIR = os.getenv("HOME")
    SUITE_DIR = os.getcwd()
    # LIB_DIR = 'Lib'
    EXAMPLE_DIR = 'Examples'
    AUG_Python = '/afs/ipp/aug/ads-diags/common/python/lib'

    # --- Section 1: Add folders to path
    sys.path.extend([os.path.join(HOME_DIR, ROIPOLY),
                     # os.path.join(SUITE_DIR, LIB_DIR),
                     os.path.join(SUITE_DIR, EXAMPLE_DIR)])

    if machine == 'AUG':
        sys.path.extend([AUG_Python])


class p:
    """Paths of the different codes"""
    def __init__(self, machine='AUG'):
        """Initialise the class"""
        self.machine = machine
        self.FILDSIM = os.path.join(os.getenv("HOME"), 'FILDSIM/')
        self.INPASIM = os.path.join(os.getenv("HOME"), 'INPASIM/')
        self.CinFiles = '/p/IPP/AUG/rawfiles/FIT/'
        self.StrikeMaps = os.path.join(os.getenv("HOME"), 'FILD_Strike_maps/')


if __name__ == "__main__":
    paths_of_the_suite()
    pa = p(machine='AUG')
