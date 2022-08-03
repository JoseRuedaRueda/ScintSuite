"""Just the version of the suite, to label outputs"""
import numpy as np
version = '1.0.1'
codename = 'San Jacobo'


def exportVersion(filename):
    """
    Save the version of the suite into a file
    """
    v = version.split('.')
    v1 = int(v[0])
    v2 = int(v[1])
    v3 = int(v[2])
    with open(filename, 'w') as f:
        f.write('Version ID1: %i\n'%v1)
        f.write('Version ID2: %i\n'%v2)
        f.write('Version ID3: %i\n'%v3)
        f.write('Codename: %s\n'%codename)


def readVersion(filename):
    with open(filename, 'r') as f:
        v = np.zeros(3, dtype='int')
        for i in range(3):
            v[i] = int(f.readline().split(':')[-1])
    return v
