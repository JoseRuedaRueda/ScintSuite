"""Contains the path to initialize the suite"""

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
    Suite_LIBs = {
        'Base': SUITE_DIR,
        # 'LIB_DIR': 'Lib',
        # 'LIB_iHIBP': 'Lib/iHIBP',
        # 'LIB_INPA': 'Lib/INPA',
        # 'LIB_INPASIM': 'Lib/INPA/INPASIM',
        # 'LIB_FILDSIM': 'Lib/LibFILDSIM',
        # 'LIB_OPTICS': 'Lib/Optics',
        # 'LIB_GUIs': 'Lib/GUIs',
        # 'LIB_DATA': 'Lib/LibData'
    }

    # -- Machine dependent folders:
    Machine_libs = {
        'AUG': {
            'AUG_Python': '/afs/ipp/aug/ads-diags/common/python/lib',
            # 'Suite_AUG': os.path.join(SUITE_DIR, 'Lib/LibData/AUG')
        },
        'MU': {
            # 'MU_Python': '/usr/local/modules/default/python',
            'MU_Python': '/usr/local/depot/Python-3.7/lib'
        }
    }

    # --- Section 1: Add folders to path
    # Extra python modules:
    sys.path.extend([os.path.join(HOME_DIR, ROIPOLY)])
    # Suite directories:
    for lib in Suite_LIBs.keys():
        sys.path.extend([os.path.join(SUITE_DIR, Suite_LIBs[lib])])
    # Machine dependent paths:
    if machine in Machine_libs:
        for lib in Machine_libs[machine].keys():
            sys.path.extend([os.path.join(SUITE_DIR, Machine_libs[machine][lib])])

    # Check the cluster where we are working
    if machine == 'AUG':
        cluster = os.getenv('HOST')
        if cluster[:4] != 'toki':
            print('We are not in toki')
            print('The suite has not been tested outside toki')
            print('Things can go wrong!')
    elif machine == 'MU':
        print('You are in MU')
        print('This environment is still being tested')
    else:
        print('Not recognised tokamak environment')
        print('We assume you are in your personal PC')
        print('You will not have access to databases')


if __name__ == "__main__":

    if os.path.isdir('/common/uda-scratch'):
        machine = 'MU'
    elif os.path.isdir('/afs/ipp-garching.mpg.de'):
        machine = 'AUG'
    else:
        machine = 'Generic'
        print('Not recognised machine')
        print('Assume that your are using your personal computer')
        print('Database call will not work')

    paths_of_the_suite(machine)
