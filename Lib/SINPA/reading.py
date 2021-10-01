"""
Read data from SINPA

Introduced in version 6.0.0

Jose Rueda Rueda: jrrueda@us.es
"""
import numpy as np
import f90nml
import Lib.LibParameters as sspar


def read_strike_points(filename, verbose=False):
    """
    Read the strike points from a SINPA simulation

    Jose Rueda: jrrueda@us.es

    @param filename: filename of the file
    @param verbose: flag to print information on the file
    """
    fid = open(filename, 'rb')
    data = {
        'versionID1': np.fromfile(fid, 'int32', 1)[0],
        'versionID2': np.fromfile(fid, 'int32', 1)[0],
    }
    if data['versionID1'] < 1:
        data['runID'] = np.fromfile(fid, 'S50', 1)[:]
        data['ngyr'] = np.fromfile(fid, 'int32', 1)[0]
        data['Gyroradius'] = np.fromfile(fid, 'float64', data['ngyr'])
        data['nalpha'] = np.fromfile(fid, 'int32', 1)[0]
        data['Alphas'] = np.fromfile(fid, 'float64', data['nalpha'])
        data['ncolumns'] = np.fromfile(fid, 'int32', 1)[0]
        data['counters'] = np.zeros((data['nalpha'], data['ngyr']), np.int)
        data['data'] = np.empty((data['nalpha'], data['ngyr']),
                                dtype=np.ndarray)
        for ig in range(data['ngyr']):
            for ia in range(data['nalpha']):
                data['counters'][ia, ig] = np.fromfile(fid, 'int32', 1)[0]
                data['data'][ia, ig] = \
                    np.reshape(np.fromfile(fid, 'float64',
                                           data['ncolumns']
                                           * data['counters'][ia, ig]),
                               (data['counters'][ia, ig], data['ncolumns']),
                               order='F')
        return data


def read_orbits(filename, verbose):
    """
    Read the orbits from an orbit file

    Jose Rueda: jrrueda@us.es
    """
    fid = open(filename, 'rb')
    fid.seek(-4, sspar.SEEK_END)
    nOrbits = np.fromfile(fid, 'int32', 1)[0]
    fid.seek(0, sspar.SEEK_BOF)
    orbits = {
        'versionID1': np.fromfile(fid, 'int32', 1)[0],
        'versionID2': np.fromfile(fid, 'int32', 1)[0],
        'runID': np.fromfile(fid, 'S50', 1)[:],
        'counters': np.zeros(nOrbits, np.int),
        'data': np.empty(nOrbits, dtype=np.ndarray)
    }
    for i in range(nOrbits):
        orbits['counters'][i] = np.fromfile(fid, 'int32', 1)[0]
        print(orbits['counters'][i])
        orbits['data'][i] = \
            np.reshape(np.fromfile(fid, 'float64', orbits['counters'][i] * 3),
                       (orbits['counters'][i], 3), order='F')
    return orbits


def read_namelist(filename, verbose=False):
    """
    Read a FILDSIM namelist

    Jose Rueda: jrrueda@us.es

    just a wrapper for the f90nml capabilities, copied from the FILDSIM library

    @param filename: full path to the filename to read
    @param verbose: Flag to print the nml in the console

    @return nml: dictionary with all the parameters of the FILDSIM run
    """
    nml = f90nml.read(filename)
    if verbose:
        print(nml)
    return nml
