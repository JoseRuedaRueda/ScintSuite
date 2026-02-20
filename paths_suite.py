"""Contains the path to initialize the suite"""

import os
import sys
from ScintSuite._Machine import machine


def envPathExtend():
    """
    Add to the path all the necessary folders for the suite to run.

    Jose Rueda Rueda: jrrueda@us.es

    @param machine: Machine where we are working
    """
    # --- Section 0: Name of the auxiliary folders (located at home directory)
    SUITE_DIR = os.getcwd()
    Suite_LIBs = {
        'Base': SUITE_DIR,
        
    }

    # -- Machine dependent folders:
    Machine_libs = {
        'AUG': {
            'Giovanni': '/afs/ipp/home/g/git/python/fbm/',
        },
        'MU': {
            # 'MU_Python': '/usr/local/modules/default/python',
            'MU_Python': '/usr/local/depot/Python-3.7/lib'
        },
        'D3D': {
        }
    }

    # --- Section 1: Add folders to path
    # Suite directories:
    for lib in Suite_LIBs.keys():
        sys.path.extend([os.path.join(SUITE_DIR, Suite_LIBs[lib])])
    # Machine dependent paths:
    if machine in Machine_libs:
        for lib in Machine_libs[machine].keys():
            sys.path.extend([Machine_libs[machine][lib],])

    # Check the cluster where we are working
    if machine == 'AUG':
        cluster = os.getenv('HOST')
        if cluster[:4] != 'toki':
            print('We are in %s' % cluster)
            print('We are not in toki')
            print('The suite has not been tested outside toki')
            print('Things can go wrong!')
    elif machine == 'MU':
        print('You are in MU')


if __name__ == "__main__":

    envPathExtend()
