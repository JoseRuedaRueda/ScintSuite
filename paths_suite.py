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
    SUITE_DIR = os.getcwd()
    Suite_LIBs = {
        'Base': SUITE_DIR,
    }

    # -- Machine dependent folders:
    Machine_libs = {
        'AUG': {
            'AUG_Python': '/afs/ipp/aug/ads-diags/common/python/lib',
            # 'Suite_AUG': os.path.join(SUITE_DIR, 'Lib/LibData/AUG')
        }
    }

    # --- Section 1: Add folders to path
    # Extra python modules:

    # Suite directories:
    for lib in Suite_LIBs.keys():
        sys.path.extend([os.path.join(SUITE_DIR, Suite_LIBs[lib])])
    # Machine dependent paths:
    if machine in Machine_libs:
        for lib in Machine_libs[machine].keys():
            sys.path.extend([os.path.join(SUITE_DIR,
                             Machine_libs[machine][lib])])

    # Check the cluster where we are working
    if os.path.isdir('/afs/ipp/aug/ads-diags/common/python/lib'):
        machine = 'AUG'
        cluster = os.getenv('HOST')
        if cluster[:4] != 'toki':
            print('We are not in toki')
            print('The suite has not been tested outside toki')
            print('Things can go wrong!')
    else:
        machine = 'Generic'
        print('Not recognised machine')
        print('Assume that your are using your personal computer')
        print('Database call will not work')


if __name__ == "__main__":

    paths_of_the_suite()
