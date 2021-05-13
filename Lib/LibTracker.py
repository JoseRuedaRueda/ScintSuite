"""Contains the methods and classes to interact with iHIBPsim tracker"""

import warnings
import numpy as np
try:
    import f90nml
except ImportError:
    warnings.warn('You cannot read FILDSIM namelist nor remap',
                  category=UserWarning)


def write_tracker_namelist(user_nml, file, overwrite=True):
    """
    Write fortran namelist

    jose rueda: jrrueda@us.es

    just a wrapper for the f90nml file writter

    To see the meaning of all parameters, look at the nicely written iHIBPsim
    documentation

    @param user_nml: namelist containing the desired fields.
    @param file: full path towards the desired file
    @param overwrite: flag to overwrite the namelist (if exist)

    f90nml format adopted in version 0.4.10
    """
    # Default namelist:
    nml = {
        'FIELD_FILES': {
            'Bfield_name': '',
            'Efield_name': '',
            'Efield_on': False,
            'equ_file': '',
        },
        'PROFILES': {
            'Te_name': '',
            'ne_name': '',
            'n0_name': '',
            'Zeff': 1.0
        },
        'TABLES': {
            'beamAttenuationModule': False,
            'elec_name': '',
            'CX_name': '',
        },
        'INTEGRATION': {
            'dt': 1.0e-9,
            'max_step': 20000000,
            'file_out': 'out.bin',
        },
        'ORBITS_CONF': {
            'save_orbits': False,
            'num_orbits': 1.0,
            'file_orbits': 'orbit.bin',
            'dt_orbit': 1.0e-8,
        },
        'DEPOSITION': {
            'markerNumber': 1,
            'depos_file': 'markers.bin',
            'verbose': True
        },
        'SCINTILLATOR': {
            'triangle_file': 'scintillator.dat'
        }
    }
    # Update the fields, if we just use nml.update(user_nml), if user_nml has
    # the block 'ORBITS_CONF', but inside it just the field 'save_orbits',
    # because the user only wants to update that field, it will fail, as all
    # the block 'ORBITS_CONF' would be replaced by one with just that field, so
    # we need to perform the comparison one by one
    for key in nml.keys():
        if key in user_nml:
            nml[key].update(user_nml[key])
    f90nml.write(nml, file, force=overwrite, sort=True)
    print('Namelist written: ', file)
    return


def write_markers(filename: str, markers: dict):
    """
    Write the information of the markers to be followed by the tracker

    Jose Rueda Rueda: jrrueda@us.es

    @para filename: name of the file to be written
    @param markers: dictionary containing all the info of the markers
    """
    n = len(markers['R'])
    ID = np.arange(0, n, dtype=np.float64) + 1.0

    with open(filename, 'w') as fid:
        # Write header with grid size information:
        # np.array([11], dtype=np.int32).tofile(fid)
        # np.array([n], dtype=mnp.int32).tofile(fid)
        dummy = np.vstack((ID.flatten(), markers['R'].flatten(),
                           markers['z'].flatten(), markers['phi'].flatten(),
                           markers['vR'].flatten(), markers['vt'].flatten(),
                           markers['vz'].flatten(), markers['m'].flatten(),
                           markers['q'].flatten(), markers['logw'].flatten(),
                           markers['t'].flatten()))
        dummy.T.tofile(fid)

        # ver = np.array(version.split('.'), dtype=np.int32)
        # ver.tofile(fid)


def cart2pol(r, v=None):
    """
    Transform from Cartesian coordinates to cylindrical (polar)

    @param r: position vector [x, y, z]
    @param v: If present the output will be the velocity
    @return out: if v is none [R,z,phi]. If v is not none [vR, vz, vphi]
    """
    phi = np.arctan2(r[1], r[0])

    if v is None:
        R = np.sqrt(r[0]**2 + r[1]**2)
        return np.array([R, r[2], phi])
    else:
        vr = v[0] * np.cos(phi) + v[1] * np.sin(phi)
        vphi = - v[0] * np.sin(phi) + v[1] * np.cos(phi)
        return np.array([vr, v[2], vphi])
