"""
Routines to read strike map data from FILDSIM/SINPA.

Jose Rueda Rueda

None of these methods is expected to be called directly, is the StrikeMap class
will call these guys and read the map

Public methods:

Private methods:
    - _readSmapSINPA: read strike map generated with SINPA
    - _readSmapFILDSIM: read strike map generated with FILDSIM

Introduced in version 0.10.0
"""
import logging
import numpy as np
import Lib.errors as errors
from copy import deepcopy
from Lib._basicVariable import BasicVariable

logger = logging.getLogger('ScintSuite.StrikeMaps')
# --- Order of the variables written in the strike map file.
# If the there is no entrance for version X is because there were no changes in
# the file structure
FILDSIMorder = {
    0: {  # ID0 of the code version
        'name': ['gyroradius', 'pitch', 'x3', 'x1', 'x2', 'avgIniGyrophase',
                 'n_strike_points', 'collimator_factor', 'avgIncidentAngle'],
        'units': ['cm', 'degree', 'cm', 'cm', 'cm', 'rad', '', '', 'degree'],
    },
}
SINPAfildOrder = {
    0: {  # ID0 of the code version
        'name': ['gyroradius', 'pitch', 'x3', 'x1', 'x2', 'avgIniGyrophase',
                 'n_strike_points', 'collimator_factor', 'avgIncidentAngle'],
        'units': ['cm', 'degree', 'm', 'm', 'm', 'rad', '', '', 'degree'],
    },
}
SINPAinpaOrder = {
    0: {  # ID0 of the code version
        'name': ['gyroradius', 'alpha', 'x3', 'x1', 'x2', 'avgIniGyrophase',
                 'n_strike_points', 'collimator_factor', 'avgIncidentAngle',
                 'x0', 'y0', 'z0', 'd0'],
        'units': ['cm', 'rad', 'm', 'm', 'm', 'rad', '', '', 'degree', 'm',
                  'm', 'm', 'm'],
    },
}


def _readSmapSINPA(filename: str):
    """
    Read strike map data from a SINPA file.

    Jose Rueda: jrrueda

    :param  filename: name of the file to read

    :return header: dictionary with the metadata of the strike map
    :return data: dictionary with the strike map data
    """
    # read the header of the file. In the second line of the strike map, there
    # is the version, it is a bit weird the way I try to identify the version
    # number, but is the workaround I found to be backwards compatible
    # Same to distiguish between FILD and INPA, for this we look for
    # 'pitch-angle' in the header
    header = {
        'file': filename,
        'code': 'SINPA',
    }
    with open(filename, 'r') as f:
        f.readline()  # Header with the date
        v_str = f.readline().split('#')[0].strip()
        if len(v_str) == 3:  # SINPA version > 0.3, 2 number saved
            header['versionID1'] = int(v_str[0])
            header['versionID2'] = int(v_str[2])
        else:   # SINPA version <= 0.3, just one number saved
            header['versionID1'] = 0
            header['versionID2'] = int(v_str[0])
        name_line = f.readline()
        if 'pitch-angle' in name_line.lower():
            header['diagnostic'] = 'FILD'
            order = SINPAfildOrder
        elif 'alpha' in name_line.lower():
            header['diagnostic'] = 'INPA'
            order = SINPAinpaOrder
            # Variables to be used in the remap/analysis
        else:
            raise Exception('Not recognised diagnostic in the Smap')
    # Find the proper header for the map
    found_header = False
    id_version = header['versionID1']
    while not found_header:
        try:
            header['variables'] = deepcopy(order[id_version])
            found_header = True
        except KeyError:
            id_version -= 1
        # if the id_version is already -1, just stop, something
        # went wrong
        if id_version < 0:
            raise Exception('Not undestood SINPA version')
    # read the data
    dummy = np.loadtxt(filename, skiprows=3)
    # check that the data has the proper number of colums
    nrow, ncol = dummy.shape
    if ncol != len(header['variables']['name']):
        raise Exception('Wrong number of columns in the file')
    # Take only rows where markers arrived
    ix = np.where(np.array(header['variables']['name']) == 'x3')
    ind = ~np.isnan(dummy[:, ix]).squeeze()
    # Get the matrix which translate between points in the list with the
    # Save the data in the output dictionary:
    data = dict.fromkeys(header['variables']['name'])
    counter = 0
    for key in data.keys():
        data[key] = BasicVariable(
            name=key, units=header['variables']['units'][counter],
            data=dummy[ind, counter]
            )
        counter += 1
    # Get the unique values of the gyroradius
    header['unique_gyroradius'] = np.unique(data['gyroradius'].data)
    header['ngyroradius'] = header['unique_gyroradius'].size
    # if we have INPA, proceed with alpha and the R0 value
    if header['diagnostic'] == 'INPA':
        # Save the unique alpha values
        header['unique_alpha'] = np.unique(data['alpha'].data)
        header['nalpha'] = header['unique_alpha'].size
        # Get the radial CX position as a matrix as well as the collimator
        data['R0'] = BasicVariable(
            name='R0',
            units='m',
            data=np.sqrt(data['x0'].data**2 + data['y0'].data**2)
        )
        R0 = np.full((header['nalpha'], header['ngyroradius']), np.nan)
        z0 = np.full((header['nalpha'], header['ngyroradius']), np.nan)
        y0 = np.full((header['nalpha'], header['ngyroradius']), np.nan)
        x0 = np.full((header['nalpha'], header['ngyroradius']), np.nan)
        coll = np.zeros((header['nalpha'], header['ngyroradius']))
        R0_indeces = np.full((header['nalpha'], header['ngyroradius']), np.nan,
                             dtype=int)
        for ip in range(header['nalpha']):
            for ir in range(header['ngyroradius']):
                # By definition, flags can only have one True
                flags = \
                    (data['gyroradius'].data == header['unique_gyroradius'][ir]) \
                    * (data['alpha'].data == header['unique_alpha'][ip])
                if np.sum(flags) > 0:
                    R0[ip, ir] = data['R0'].data[flags]
                    z0[ip, ir] = data['z0'].data[flags]
                    y0[ip, ir] = data['y0'].data[flags]
                    x0[ip, ir] = data['x0'].data[flags]
                    coll[ip, ir] = data['collimator_factor'].data[flags]
                    R0_indeces[ip, ir] = np.where(flags)[0]

        header['unique_R0'] = np.nanmean(R0, axis=1)
        header['unique_z0'] = np.nanmean(z0, axis=1)
        header['unique_y0'] = np.nanmean(y0, axis=1)
        header['unique_x0'] = np.nanmean(x0, axis=1)
        header['nR0'] = header['unique_R0'].size
        # We do the mean because due to MC markers, each rl can have minor
        # diferences in the R0 value.
        # Now write this unique value at all places
        for ip in range(header['nalpha']):
            for ir in range(header['ngyroradius']):
                try:
                    data['R0'].data[R0_indeces[ip, ir]] = header['unique_R0'][ip]
                    data['z0'].data[R0_indeces[ip, ir]] = header['unique_z0'][ip]
                    data['y0'].data[R0_indeces[ip, ir]] = header['unique_y0'][ip]
                    data['x0'].data[R0_indeces[ip, ir]] = header['unique_x0'][ip]
                except IndexError:
                    pass
        # And add R0 to the header information
        header['variables']['name'].append('R0')
        header['variables']['units'].append('m')
        # Add te collimator factor as a variable
        data['collimator_factor_matrix'] = BasicVariable(
            name='Collimator Factor Matrix',
            units='%',
            data=coll,
        )
        # Save the variables used for the spawning of the markers
        header['MC_variables'] = (
            BasicVariable(name='R0', units='m', data=header['unique_R0']),
            BasicVariable(name='gyroradius', units='cm',
                          data=header['unique_gyroradius']),
            )
        #
        header['shape'] = (header['nalpha'], header['ngyroradius'])
    elif header['diagnostic'] == 'FILD':  # for FILD just the pitch
        header['unique_pitch'] = np.unique(data['pitch'].data)
        header['npitch'] = header['unique_pitch'].size
        coll = np.zeros((header['npitch'], header['ngyroradius']))
        for ip in range(header['npitch']):
            for ir in range(header['ngyroradius']):
                # By definition, flags can only have one True
                flags = \
                    (data['gyroradius'].data == header['unique_gyroradius'][ir]) \
                    * (data['pitch'].data == header['unique_pitch'][ip])
                if np.sum(flags) > 0:
                    coll[ip, ir] = data['collimator_factor'].data[flags]
        data['collimator_factor_matrix'] = BasicVariable(
            name='Collimator Factor Matrix',
            units='%',
            data=coll,
        )
        header['MC_variables'] = (
            BasicVariable(name='pitch', units='$\\degree$',
                          data=header['unique_pitch']),
            BasicVariable(name='gyroradius', units='cm',
                          data=header['unique_gyroradius']),
            )
        header['shape'] = (header['npitch'], header['ngyroradius'])
    else:
        raise errors.NotImplementedError('To be done')
    return header, data


def _readSmapFILDSIM(filename: str):
    """
    Read strike map data from a FILDSIM file.

    Jose Rueda: jrrueda

    :param  filename: name of the file to read

    :return header: dictionary with the metadata of the strike map
    :return data: dictionary with the strike map data
    """
    # read the header of the file. In the second line of the strike map, there
    # is the version, it is a bit weird the way I try to identify the version
    # number, but is the workaround I found to be backwards compatible
    # Same to distiguish between FILD and INPA, for this we look for
    # 'pitch-angle' in the header
    header = {
        'file': filename,
        'code': 'FILDSIM',
        'versionID1': 0,   # FILDSIM has no information on the file version
        'versionID2': 0,
        'diagnostic': 'FILD',
        'variables': deepcopy(FILDSIMorder[0]),
    }
    # read the data
    dummy = np.loadtxt(filename, skiprows=3)
    # check that the data has the proper number of colums
    nrow, ncol = dummy.shape
    if ncol != len(header['variables']['name']):
        raise Exception('Wrong number of columns in the file')
    # Take only rows where markers arrived
    ix = np.where(np.array(header['variables']['name']) == 'x1')
    ind = ~np.isnan(dummy[:, ix]).squeeze()
    # Save the data in the output dictionary:
    data = dict.fromkeys(header['variables']['name'])
    counter = 0
    for key in data.keys():
        data[key] = BasicVariable(
            name=key, units=header['variables']['units'][counter],
            data=dummy[ind, counter]
            )
        counter += 1
    header['unique_pitch'] = np.unique(data['pitch'].data)
    header['npitch'] = header['unique_pitch'].size
    # Get the unique values of gyroradius and pitch/alpha
    header['unique_gyroradius'] = np.unique(data['gyroradius'].data)
    header['ngyroradius'] = header['unique_gyroradius'].size
    # Get the shape, which is the grid size of launchig the markers
    header['shape'] = (header['npitch'], header['ngyroradius'])
    # Get the collimator factor as a matrix
    coll = np.zeros((header['npitch'], header['ngyroradius']))
    for ip in range(header['npitch']):
        for ir in range(header['ngyroradius']):
            # By definition, flags can only have one True
            flags = \
                (data['gyroradius'].data == header['unique_gyroradius'][ir]) \
                * (data['pitch'].data == header['unique_pitch'][ip])
            if np.sum(flags) > 0:
                coll[ip, ir] = data['collimator_factor'].data[flags]
    data['collimator_factor_matrix'] = BasicVariable(
        name='Collimator Factor Matrix',
        units='%',
        data=coll,
    )
    # Set the MC variables
    header['MC_variables'] = (
        BasicVariable(name='pitch', units='$\\degree$',
                      data=header['unique_pitch']),
        BasicVariable(name='gyroradius', units='cm',
                      data=header['unique_gyroradius']),
        )
    return header, data


def readSmap(filename, code: str = None):
    """
    Read a strike map from file.

    Jose Rueda Rueda: jrrueda@us.es

    Just a wrapper to reading routines.

    :param  filename: file to load
    :param  code: name of the used code, if None, the code will be guessed from
        the extension name
    """
    if code is None:
        if filename.endswith('.dat'):
            code = 'FILDSIM'
        elif filename.endswith('.map'):
            code = 'SINPA'
        else:
            logger.error(filename)
            msg = 'File name not code standard, you need to give the code'
            raise errors.NotValidInput(msg)
    if code == 'SINPA':
        header, data = _readSmapSINPA(filename)
    elif code == 'FILDSIM':
        header, data = _readSmapFILDSIM(filename)
    elif code == 'iHIBPsim':
        raise errors.NotImplementedError('Not yet incldued')
    else:
        raise errors.NotValidInput('Not understood code')

    return header, data
