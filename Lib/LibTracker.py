"""Contains the methods and classes to interact with iHIBPsim tracker"""

import numpy as np
import f90nml


# -----------------------------------------------------------------------------
# --- Tracker inputs
# -----------------------------------------------------------------------------
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


def write_markers(markers: dict, filename: str):
    """
    Write the information of the markers to be followed by the tracker

    Jose Rueda Rueda: jrrueda@us.es

    @para filename: name of the file to be written
    @param markers: dictionary containing all the info of the markers
    """
    n = len(markers['R'])
    ID = np.arange(0, n, dtype=np.float64) + 1.0
    # Just a check for legacy compatibility vt and vphi, the name was changed
    if 'vt' in markers.keys():
        markers['vphi'] = markers['vt']

    with open(filename, 'w') as fid:
        # Write header with grid size information:
        # np.array([11], dtype=np.int32).tofile(fid)
        # np.array([n], dtype=mnp.int32).tofile(fid)
        dummy = np.vstack((ID.flatten(), markers['R'].flatten(),
                           markers['z'].flatten(), markers['phi'].flatten(),
                           markers['vR'].flatten(), markers['vphi'].flatten(),
                           markers['vz'].flatten(), markers['m'].flatten(),
                           markers['q'].flatten(), markers['logw'].flatten(),
                           markers['t'].flatten()))
        dummy.T.tofile(fid)


# -----------------------------------------------------------------------------
# --- Tracker outputs
# -----------------------------------------------------------------------------
def load_deposition(filename: str):
    """
    Load the deposition file created for iHIBPsim

    Jose Rueda Rueda: jrrueda@us.es

    @param filename: Name of the orbit file
    """
    # --- Open the file and load the data
    with open(filename, 'r') as f:
        A = np.fromfile(f, np.float64)
    nmarkers = int(A.size / 11)
    Mar = A.reshape((nmarkers, 11))
    del A

    output = {
        'data': Mar,
        'ID': Mar[:, 0],
        'R': Mar[:, 1],
        'z':  Mar[:, 2],
        'phi':  Mar[:, 3],
        'time': Mar[:, 10],
        'm': Mar[:, 7],
        'q': Mar[:, 8],
        'vR':  Mar[:, 4],
        'vt':  Mar[:, 5],
        'vz':  Mar[:, 6],
        'w':  np.exp(Mar[:, 9])
    }
    return output


# -----------------------------------------------------------------------------
# --- legacy compatibility
# -----------------------------------------------------------------------------
def write_namelist_legacy_(name_of_namelist_file: str,
                           Bfield_name: str,
                           Efield_on: str = '.FALSE.',
                           Efield_name: str = '',
                           equ_file: str = '', Te_name: str = '',
                           ne_name: str = '', n0_name: str = '',
                           Zeff: float = 1.0,
                           beamAtt: str = '.FALSE.',
                           elec_name: str = '',
                           CX_name: str = '',
                           dt: float = 5.0e-11,
                           max_step: int = 1000000000,
                           save_orbits: str = '.TRUE.', Nmarkers: int = 250,
                           depos_file: str = '', triangle_file: str = '',
                           file_out: str = 'output.bin',
                           file_orbits: str = 'orbits.bin',
                           num_orbits: float = 1.0,
                           dt_orbit: float = 1.0e-8):
    """
    Write the namelist for the particle traker

    Jose Rueda Rueda: jrrueda@us.es

    To see the meaning of all parameters, look at the nicely written iHIBPsim
    documentation
    """
    with open(name_of_namelist_file, 'w') as f:
        # -------------------------------------------------------- FIELDS_FILES
        print("&FIELD_FILES", file=f, sep='')
        # Magnetic field file
        print("Bfield_name = '", Bfield_name, "',", file=f, sep='')
        # Use or not electric field.
        print("Efield_on = ", Efield_on, ",", file=f, sep='')
        # Electric field file
        print("Efield_name = '", Efield_name, "',", file=f, sep='')
        # Equilibrium file: if not provided, then the simulation is
        # constrained by the Bfield size.
        print("equ_file = '", equ_file, "',", file=f, sep='')
        print("/", file=f, sep='')
        print("", file=f, sep='')
        # ------------------------------------------------------------ PROFILES
        print("&PROFILES", file=f, sep='')
        # Path for the Te profile.
        print("Te_name = '", Te_name, "',", file=f, sep='')
        # Path for the electron density profile.
        print("ne_name = '", ne_name, "',", file=f, sep='')
        # Ion density profile. If this is not provided, electron is used
        print("n0_name = '", n0_name, "',", file=f, sep='')
        # Modifies the input n0 by this factor. Set to 1.0d0 by default
        print("Zeff = ", Zeff, ",", file=f, sep='')
        print('/', file=f, sep='')
        print("", file=f, sep='')
        # -------------------------------------------------------------- TABLES
        print('&TABLES', file=f, sep='')
        # Activate or not the beam  module to compute the weight evolution.
        print("beamAttenuationModule = ", beamAtt, ",", file=f, sep='')
        # Electron impact ionization reaction rates.
        print("elec_name = '", elec_name, "',", file=f, sep='')
        # Charge-exchange reaction rates.
        print("CX_name = '", CX_name, "',", file=f, sep='')
        print('/', file=f, sep='')
        print("", file=f, sep='')
        # --------------------------------------------------------- INTEGRATION
        print('&INTEGRATION', file=f, sep='')
        # Time step for integration.
        print("dt = ", dt, ",", file=f, sep='')
        # Maximum number of steps for the simulation.
        print("max_step = ", max_step, ",", file=f, sep='')
        # File where the final points of the particles are stored.
        print("file_out = '", file_out, "',", file=f, sep='')
        print('/', file=f, sep='')
        print("", file=f, sep='')
        # --------------------------------------------------------- ORBITS_CONF
        print('&ORBITS_CONF', file=f, sep='')
        # States if this module is active.
        print("save_orbits = ", save_orbits, ",", file=f, sep='')
        # Value in range [0, 1] stating percentaje of particles to store.
        print("num_orbits = ", num_orbits, ",", file=f, sep='')
        # Time step for the orbit-recording.
        print("dt_orbit = ", dt_orbit, ",", file=f, sep='')
        # File to store the orbits
        print("file_orbits = '", file_orbits, "',", file=f, sep='')
        print('/', file=f, sep='')
        print("", file=f, sep='')
        # ---------------------------------------------------------- DEPOSITION
        print('&DEPOSITION', file=f, sep='')
        # Number of markers in the deposition file
        print("markerNumber = ", Nmarkers, ",", file=f, sep='')
        # Initial position of the markers to follow
        print("depos_file = '", depos_file, "',", file=f, sep='')
        print('/', file=f, sep='')
        print("", file=f, sep='')
        # ------------------------------------------------- SCINTILLATOR / WALL
        print('&SCINTILLATOR', file=f, sep='')
        # File containing the triangles describing the wall/scintillator.
        print("triangle_file = '", triangle_file, "',", file=f, sep='')
        print('/', file=f, sep='')
        print("", file=f, sep='')
        # ---------------------------------------------------------------------
    print('Namelist written: ', name_of_namelist_file)


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
