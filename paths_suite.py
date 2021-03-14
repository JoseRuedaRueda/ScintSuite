"""Contains the path to initialise the suite"""

import os
import sys


def paths_of_the_suite(machine='AUG'):
    """
    Add to the path all the necessary folders for the suite to run.

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

    @param machine: Machine where we are working
    """
    # --- Section 0: Name of the auxiliary folders (located at home directory)
    ROIPOLY = 'roipoly.py'
    HOME_DIR = os.getenv("HOME")
    SUITE_DIR = os.getcwd()
    LIB_DIR = 'Lib'
    LIB_iHIBP = 'Lib/iHIBP'
    # -- AUG folders:
    AUG_Python = '/afs/ipp/aug/ads-diags/common/python/lib'

    # --- Section 1: Add folders to path
    sys.path.extend([os.path.join(HOME_DIR, ROIPOLY),
                     os.path.join(SUITE_DIR, LIB_iHIBP),
                     os.path.join(SUITE_DIR, LIB_DIR)])

    if machine == 'AUG':
        sys.path.extend([AUG_Python])
        cluster = os.getenv('HOST')
        if cluster[:4] != 'toki':
            print('We are not in toki')
            print('The suite has not been tested outside toki')
            print('Things can go wrong!')


if __name__ == "__main__":
    paths_of_the_suite()
