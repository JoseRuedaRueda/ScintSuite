"""Contains the methods and classes to interact with iHIBPsim tracker"""

import numpy as np
from version_suite import version
from LibMachine import machine
import LibPlotting as ssplt
import LibParameters as sspar
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as intp
if machine == 'AUG':
    import LibDataAUG as ssdat


def prepare_B_field(shot: int, time: float, diag: str = 'EQH',
                    Rmin: float = 1.0, Rmax: float = 2.2, nR: int = 120,
                    zmin: float = -1.0, zmax: float = 1.0, nz: int = 200,
                    phimin: float = 0., phimax: float = 2.*np.pi,
                    nphi: int = 64):
    """
    Load the magnetic field to launch an iHIBPsim simulation

    Jose Rueda: jrrueda@us.es

    @param    shot: Shot number
    @param    time: time [s]
    @param    diag: The diagnostic name where to load the field
    @param    Rmin: Rmin of the grid, in m
    @param    Rmax: Rmax of the grid, in m
    @param    nR: number of points in the R direction
    @param    zmin: zmin of the grid
    @param    zmax: zmax of the grid
    @param    nz: Number of points in the z direction
    @param    phimin: Still not used, ignore it
    @param    phimax: Still not used, ignore it
    @param    nphi: Still not used, ignore it

    @return:  Dictionary containing:
             -#'nR':
             -#'nz':
             -#'nphi':
             -#'nt':
             -#'Rmin':
             -#'Rmax':
             -#'zmin':
             -#'zmax':
             -#'phimin': Ignore
             -#'phimax': Ignore
             -#'tmin': Ignore
             -#'tmax': Ignore
             -#'fr': Field radial component[nr,nz]
             -#'fz': Field z component[nr,nz]
             -#'ft': Field t component[nr,nz]
             -#'R': R grid in R
             -#'z': z grid in z
             -#'phi': Ignore
    """
    ## todo: implement the 3D part
    R = np.linspace(Rmin, Rmax, num=nR)
    z = np.linspace(zmin, zmax, num=nz)
    phi = np.linspace(phimin, phimax, num=nphi)
    RR, zz = np.meshgrid(R, z)
    grid_shape = RR.shape
    br, bz, bt, bp = ssdat.get_mag_field(shot, RR.flatten(), zz.flatten(),
                                         diag=diag, time=time)
    Br = np.reshape(br, grid_shape).T
    Bz = np.reshape(bz, grid_shape).T
    Bt = np.reshape(bt, grid_shape).T

    field = {'nR': np.array([len(R)], dtype=np.int32),
             'nz': np.array([len(z)], dtype=np.int32),
             'nphi': np.array([1], dtype=np.int32),
             'nt': np.array([1], dtype=np.int32),
             'Rmin': np.array([Rmin], dtype=np.float64),
             'Rmax': np.array([Rmax], dtype=np.float64),
             'zmin': np.array([zmin], dtype=np.float64),
             'zmax': np.array([zmax], dtype=np.float64),
             'phimin': np.array([phimin], dtype=np.float64),
             'phimax': np.array([phimax], dtype=np.float64),
             'tmin': np.array([time - 0.005], dtype=np.float64),
             'tmax': np.array([time + 0.005], dtype=np.float64),
             'fr': Br.astype(np.float64),
             'fz': Bz.astype(np.float64),
             'ft': Bt.astype(np.float64),
             'R': R,
             'z': z,
             'phi': phi}
    return field


def write_field(filename: str, field: dict):
    """
    Write field (electric or magnetic)

    Jose Rueda: jrrueda@us.es

    @param filename: name of the file where to write
    @param field: dictionary created by the load_field routine of this library
    """
    with open(filename, 'w') as fid:
        # Write header with grid size information:
        field['nR'].tofile(fid)
        field['nz'].tofile(fid)
        field['nphi'].tofile(fid)
        field['nt'].tofile(fid)

        # Write grid ends:
        field['Rmin'].tofile(fid)
        field['Rmax'].tofile(fid)
        field['zmin'].tofile(fid)
        field['zmax'].tofile(fid)
        field['phimin'].tofile(fid)
        field['phimax'].tofile(fid)
        field['tmin'].tofile(fid)
        field['tmax'].tofile(fid)

        # Write fields
        field['fr'].T.tofile(fid)
        field['ft'].T.tofile(fid)
        field['fz'].T.tofile(fid)

        ver = np.array(version.split('.'), dtype=np.int32)
        ver.tofile(fid)


def load_field(filename, dimensions: int = 2, verbose: bool = True):
    """
    Load the field used in the tracker

    Jose Rueda: jrrueda@us.es

    Note: Only 2D is supported right now

    @todo Pablo, please, include 3d and 4d options

    @param filename: full path to the file with the field
    @param dimensions: if 2, field will be interpreted just as B[nr,nz],
    if 3, include phi dimension and if 4, includes time.
    """
    field = {'nR': None, 'nz': None, 'nphi': None, 'nt': None, 'Rmin': None,
             'Rmax': None, 'zmin': None, 'zmax': None, 'phimin': None,
             'phimax': None, 'tmin': None, 'tmax': None, 'fr': None,
             'fz': None, 'ft': None, 'R': None, 'z': None, 'phi': None}
    with open(filename, 'r') as fid:
        # Read header with grid size information:
        field['nR'] = np.fromfile(fid, 'int32', 1)
        field['nz'] = np.fromfile(fid, 'int32', 1)
        field['nphi'] = np.fromfile(fid, 'int32', 1)
        field['nt'] = np.fromfile(fid, 'int32', 1)

        # Read grid ends:
        field['Rmin'] = np.fromfile(fid, 'float64', 1)
        field['Rmax'] = np.fromfile(fid, 'float64', 1)
        field['zmin'] = np.fromfile(fid, 'float64', 1)
        field['zmax'] = np.fromfile(fid, 'float64', 1)
        field['phimin'] = np.fromfile(fid, 'float64', 1)
        field['phimax'] = np.fromfile(fid, 'float64', 1)
        field['tmin'] = np.fromfile(fid, 'float64', 1)
        field['tmax'] = np.fromfile(fid, 'float64', 1)

        # Generate grids
        field['R'] = np.linspace(field['Rmin'], field['Rmax'],
                                 num=int(field['nR'][0]))
        field['z'] = np.linspace(field['zmin'], field['zmax'],
                                 num=int(field['nz'][0]))
        field['phi'] = \
            np.linspace(field['phimin'], field['phimax'],
                        num=int(field['nphi'][0]))

        # Read fields
        if dimensions == 2:
            field['fr'] = \
                np.reshape(np.fromfile(fid, 'float64',
                                       int(field['nR'][0]*field['nz'][0])),
                                      (field['nz'][0], field['nR'][0])).T
            field['ft'] = \
                np.reshape(np.fromfile(fid, 'float64',
                                       int(field['nR'][0]*field['nz'][0])),
                                      (field['nz'][0], field['nR'][0])).T
            field['fz'] = \
                np.reshape(np.fromfile(fid, 'float64',
                                       int(field['nR'][0]*field['nz'][0])),
                                      (field['nz'][0], field['nR'][0])).T
        else:
            raise Exception('Sorry, Option not implemented, talk to Pablo')

        try:
            ver = np.fromfile(fid, 'int32', 3)
        except IOError:
            ver = None

    if verbose:
        print('--------------------------------------------------------------')
        if ver is not None:
            print('Done with version ', str(ver[0]), '.', str(ver[1]), '.',
                  str(ver[2]))
        print('Rmin: ', field['Rmin'])
        print('Rmax: ', field['Rmax'])
        print('nR: ', field['nR'])
        print('-')
        print('zmin: ', field['zmin'])
        print('zmax: ', field['zmax'])
        print('nz: ', field['nz'])
        print('-')
        print('phimin: ', field['phimin'])
        print('phimax: ', field['phimax'])
        print('nphi: ', field['nphi'])
        print('--------------------------------------------------------------')
    return field


def write_tracker_namelist(name_of_namelist_file: str,
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
        ## todo : talk with pablo about Zeff
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

        ver = np.array(version.split('.'), dtype=np.int32)
        ver.tofile(fid)


def load_orbits(filename: str, counter: int = 1, full_info: bool = True):
    """
    Load the orbit files created by the iHIBPsim tracker

    Jose Rueda Rueda: jrrueda@us.es

    @param filename: Name of the orbit file
    @param counter: number of orbits to load
    @param full_info: If false, only the trajectory will be loaded, if true,
    also the velocity, weight, charge and mass will be loaded
    @return output: list with all the dictionaries with the orbit information:
        -# 'R', 'z', 'phi': for the position
        -# 'vR', 'vz', 'vt': for the velocity
        -# 'q', 'm': charge and mass
        -# 'logw': logarithmic weight
        -# 'time': time of each point
        -# 'ID': ID of the markers
    """
    # Load the whole file:
    # Future message for Pablo, this is a bit too much if the output file has
    # a couple of Gb, maybe we could save things in the file as:
    #       id
    #       number of saved points
    #       points
    # In this way we do not repeat the mass or the id so we saved space and we
    # just need to have in memory one orbit, and we can use fseek...
    # In any case, the charge can  change because your code include CX
    # reactions, but not the mass or the ID so I will save in the output just
    # one mass and one ID
    # --- Open the file and load the data
    with open(filename, 'r') as f:
        A = np.fromfile(f, np.float64)
    nmarkers = int(A.size / 11)
    Mar = A.reshape((nmarkers, 11))
    del A
    # --- See what markers were saved
    saved_ID = np.unique(Mar[:, 0])
    # --- Initialise the list
    # dum1 = {'ID': None, 'R': None, 'z': None, 'phi': None, 'vR': None,
    #         'vz': None, 'vt': None, 'q': None, 'm': None, 'logw': None,
    #         't': None}
    # # @todo Some problems due to 'copies' can appear, look for another way of
    # implementing this
    output = []
    kk = 0
    for id in saved_ID:
        flags = Mar[:, 0] == id
        output.append({})
        output[kk]['ID'] = id
        output[kk]['R'] = Mar[flags, 1]
        output[kk]['z'] = Mar[flags, 2]
        output[kk]['phi'] = Mar[flags, 3]
        output[kk]['time'] = Mar[flags, 10]
        output[kk]['m'] = Mar[flags, 7][0]
        output[kk]['q'] = Mar[flags, 8]
        if full_info:
            output[kk]['vR'] = Mar[flags, 4]
            output[kk]['vt'] = Mar[flags, 5]
            output[kk]['vz'] = Mar[flags, 6]
            output[kk]['logw'] = Mar[flags, 9]
        if kk == counter - 1:
            return output
        kk += 1
    # If we reached this point, all the orbits we wanted could not be loaded:
    print('Not all the desired orbits could be loaded. Check file ??')
    return output[:kk]


def orbit_energy(orbit):
    """
    Get the energy of the marker in each point

    Jose Rueda: jrrueda@us.es

    @todo: mission for Pablo, this should be just a method in the future
    'orbit' method

    @param orbit: the orbit loaded by the load orbit method. Notice, the
    full_info flag mut be on!
    @return orbit: orbit dict but with an extra 'field' the energy in keV
    """
    n = len(orbit)
    for kk in range(n):
        # orbit[kk]['E'] = 0.5 * orbit['m'] * \
        #     np.sqrt(orbit['vR']**2 + orbit['vz']**2 + orbit['vt']**2) * \
        #     sspar.mp / sspar.c**2
        ## todo: this is hardcored, but there is a bug in iHIBPsim
        orbit[kk]['E'] =  \
            (orbit[kk]['vR']**2 + orbit[kk]['vz']**2 +
             orbit[kk]['vt']**2) * sspar.mp / sspar.c**2
    return orbit


def orbit_pitch(orbit, file: str = None, shot: int = None, IpBt: int = -1,
                dimensions: int = 2, order: int = 1, limit: bool = True,
                limit_step: int = 12000):
    """
    Calculate the pitch of the particle in each point of the orbit

    Jose Rueda: ruejo@ipp.mpg.de

    Note: In principle you can execute this just giving the path to the file
    with the magnetic field used in the simulation or, giving the shot. In this
    second case, the field from the database will be loaded. This option is
    still not implementes but is leaved here as a future option, I still do not
    have clear how this will be implemented in the GUI. Maybe the easier thing
    will be to leaver the B file created and include a button saying: clean B
    files...

    @todo: mission for Pablo, this should be just a method in the future
    'orbit' method
    @todo: implement pitch calculation when the field is given as 3 or 4d

    @param orbit: the orbit loaded by the load orbit method. Notice, the
    full_info flag mut be on!
    @param IpBt: sign of the current againt the toroidal field. the pitch will
    be defined as: pitch = IpBt * vpar(respect to b)/v
    @param dimensions: dimensions of the field
    @param interp_options: options for the scipy.interpolate.interp2d (or3).
    By default: linear interpolation, to be consistent with the tracker, and
    values outside the interpolation domain will be taken as zero
    @return orbit: orbit dict but with an extra 'field' the pitch
    """
    # if 'kind' not in interp_options:
    #     interp_options['kind'] = 'linear'
    # if 'fill_value' not in interp_options:
    #     interp_options['fill_value'] = 0.0
    # Check inputs:
    if (file is None) and (shot is None):
        raise Exception('Hello?? No file nor shot?!?')
    # Load the field
    if file is not None:
        field = load_field(file, dimensions=dimensions, verbose=False)
        print('Field loaded!')
    # Perform the calculation
    if dimensions == 2:
        RR, zz = np.meshgrid(field['R'], field['z'])
        BR_int = intp(field['R'], field['z'], field['fr'], kx=order, ky=order)
        Bz_int = intp(field['R'], field['z'], field['fz'], kx=order, ky=order)
        Bt_int = intp(field['R'], field['z'], field['ft'], kx=order, ky=order)
        for k in range(len(orbit)):
            if (limit is True) and (len(orbit[k]) > limit_step):
                raise Exception('Hola Pablo te has pasao de puntos')

            bR = BR_int.ev(orbit[k]['R'], orbit[k]['z'])
            bz = Bz_int.ev(orbit[k]['R'], orbit[k]['z'])
            bt = Bt_int.ev(orbit[k]['R'], orbit[k]['z'])
            orbit[k]['pitch'] = IpBt * ((orbit[k]['vR'] * bR +
                                         orbit[k]['vz'] * bz +
                                         orbit[k]['vt'] * bt) /
                                        np.sqrt(bR**2 + bz**2 + bt**2) /
                                        np.sqrt(orbit[k]['vR']**2 +
                                                orbit[k]['vz']**2 +
                                                orbit[k]['vt']**2))
            print('Orbit: ', k, ' of ', len(orbit) - 1, 'done')
    else:
        raise Exception('Option not implemented, talk to Pablo')

    return orbit


def orbit_mu(orbit, file: str = None, shot: int = None, IpBt: int = -1,
             dimensions: int = 2, order: int = 1, limit: bool = True,
             limit_step: int = 12000):
    """
    Calculate the pitch of the particle in each point of the orbit

    Jose Rueda: ruejo@ipp.mpg.de

    Note: In principle you can execute this just giving the path to the file
    with the magnetic field used in the simulation or, giving the shot. In this
    second case, the field from the database will be loaded. This option is
    still not implemented but is leaved here as a future option, I still do not
    have clear how this will be implemented in the GUI. Maybe the easier thing
    will be to leaver the B file created and include a button saying: clean B
    files...

    @todo: mission for Pablo, this should be just a method in the future
    'orbit' method
    @todo: implement pitch calculation when the field is given as 3 or 4d

    @param orbit: the orbit loaded by the load orbit method. Notice, the
    full_info flag mut be on!
    @param dimensions: dimensions of the field
    @param interp_options: options for the scipy.interpolate.interp2d (or3).
    By default: linear interpolation, to be consistent with the tracker, and
    values outside the interpolation domain will be taken as zero
    @return orbit: orbit dict but with an extra 'field' the pitch
    """
    if (file is None) and (shot is None):
        raise Exception('Hello?? No file nor shot?!?')
    # Load the field
    if file is not None:
        field = load_field(file, dimensions=dimensions, verbose=False)
        print('Field loaded!')
    # Perform the calculation
    if dimensions == 2:
        RR, zz = np.meshgrid(field['R'], field['z'])
        BR_int = intp(field['R'], field['z'], field['fr'], kx=order, ky=order)
        Bz_int = intp(field['R'], field['z'], field['fz'], kx=order, ky=order)
        Bt_int = intp(field['R'], field['z'], field['ft'], kx=order, ky=order)
        for k in range(len(orbit)):
            if (limit is True) and (len(orbit[k]) > limit_step):
                raise Exception('Hola Pablo te has pasao de puntos')

            bR = BR_int.ev(orbit[k]['R'], orbit[k]['z'])
            bz = Bz_int.ev(orbit[k]['R'], orbit[k]['z'])
            bt = Bt_int.ev(orbit[k]['R'], orbit[k]['z'])
            b = np.sqrt(bR**2 + bz**2 + bt**2)
            v = np.sqrt(orbit[k]['vR']**2 + orbit[k]['vz']**2 +
                        orbit[k]['vt']**2)
            p = (orbit[k]['vR'] * bR + orbit[k]['vz'] * bz
                 + orbit[k]['vt'] * bt) / b / v
            orbit[k]['mu'] = orbit[k]['m'] * (1 - p**2) * v**2 / 2.0 / b \
                * sspar.mp_kg
            print('Orbit: ', k, ' of ', len(orbit) - 1, 'done')
    else:
        raise Exception('Option not implemented, talk to Pablo')

    return orbit


def orbit_p_phi(orbit, file: str = None, shot: int = None, order: int = 1,
                limit: bool = True,
                limit_step: int = 12000):
    """
    Calculate the pitch of the particle in each point of the orbit

    Jose Rueda: ruejo@ipp.mpg.de

    Note: In principle you can execute this just giving the path to the file
    with the magnetic field used in the simulation or, giving the shot. In this
    second case, the field from the database will be loaded. This option is
    still not implemented but is leaved here as a future option, I still do not
    have clear how this will be implemented in the GUI. Maybe the easier thing
    will be to leaver the B file created and include a button saying: clean B
    files...

    @todo: mission for Pablo, this should be just a method in the future
    'orbit' method
    @todo: implement reading the equilibrium file of the tracker

    @param orbit: the orbit loaded by the load orbit method. Notice, the
    full_info flag mut be on!
    @param dimensions: dimensions of the field
    @param interp_options: options for the scipy.interpolate.interp2d (or3).
    By default: linear interpolation, to be consistent with the tracker, and
    values outside the interpolation domain will be taken as zero
    @return orbit: orbit dict but with an extra 'field' the pitch
    """
    if (file is None) and (shot is None):
        raise Exception('Hello?? No file nor shot?!?')
    # Load the field
    if file is not None:
        raise Exception('To be done')
        # To be implemented once I learn the structure of Pablo's equilibrium
        # field = load_equilibrium(file, dimensions=dimensions, verbose=False)
        # print('Equilibrium loaded loaded!')
    else:
        equ = ssdat.meq.equ_map(shot, diag='EQH')
        equ.read_pfm()
        i = np.argmin(np.abs(equ.t_eq - orbit[0]['time'][0]))
        PFM = equ.pfm[:, :, i].squeeze()
        PFM_int = intp(equ.Rmesh, equ.Zmesh, PFM, kx=order, ky=order)
        equ.Close()
    # Perform the calculation
    for k in range(len(orbit)):
        if (limit is True) and (len(orbit[k]) > limit_step):
            raise Exception('Hola Pablo te has pasao de puntos')
        orbit[k]['Pphi'] = sspar.mp_kg * orbit[k]['m'] * orbit[k]['vt'] *\
            orbit[k]['R'] - orbit[k]['q'] * sspar.ec * \
            PFM_int.ev(orbit[k]['R'], orbit[k]['z'])
        print('Orbit: ', k, ' of ', len(orbit) - 1, 'done')

    return orbit


def plot_orbit(orbit, view: str = '2D', ax_options: dict = {}, ax=None,
               line_options: dict = {}, shaded3d_options: dict = {},
               imin: int = 0, imax: int = None):
    """
    Plot the orbit

    Jose Rueda: jrrueda@us.es

    @param orbit: Orbit dictionary created by 'load_orbits', not the full list,
    just the one you want to plot
    @param view: '2D' to plot, (R,z), (x,y). '3D' to plot the 3D orbit
    @param ax_options: options for the function axis_beauty
    @param line_options: options for the line plot (markers, colors and so on)
    @param ax: axes where to plot, if none, new ones will be created. Note,
    if the '2D' mode is used, ax should be a list of axes, the first one for
    the Rz projection
    @param shaded3d_options: dictionary with the options for the plotting of
    the 3d vessel
    """
    # --- Initialise the plotting parameters
    ax_options['ratio'] = 'equal'
    # The ratio must be always equal, otherwise is terrorism
    if 'fontsize' not in ax_options:
        ax_options['fontsize'] = 16
    if 'grid' not in ax_options:
        ax_options['grid'] = 'both'
    if 'linewidth' not in line_options:
        line_options['linewidth'] = 2
    # --- Get cartesian coordinates:
    x = orbit['R'] * np.cos(orbit['phi'])
    y = orbit['R'] * np.sin(orbit['phi'])
    if imax is None:
        imax = len(x) - 1
    if imax > len(x):
        imax = len(x) - 1
    if view == '2D':
        # Open the figure
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        # Plot the Rz, projection
        ax[0].plot(orbit['R'][imin:imax], orbit['z'][imin:imax],
                   label='ID: ' + str(orbit['ID']), **line_options)
        ax_options['xlabel'] = 'R [m]'
        ax_options['ylabel'] = 'z [m]'
        ssplt.plot_vessel(ax=ax[0])
        ax[0] = ssplt.axis_beauty(ax[0], ax_options)
        # Plot the xy projection

        ax[1].plot(x[imin:imax], y[imin:imax],
                   label='ID: ' + str(orbit['ID']), **line_options)
        # plot the initial and final points in a different color
        ax[1].plot(x[imax], y[imax], 'o', color='r')
        ax[1].plot(x[imin], y[imin], 'o', color='g')
        ax_options['xlabel'] = 'x [m]'
        ax_options['ylabel'] = 'y [m]'
        ssplt.plot_vessel(projection='toroidal', ax=ax[1])
        ax[1] = ssplt.axis_beauty(ax[1], ax_options)
        plt.tight_layout()
    else:
        # Open the figure
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        # Plot the orbit
        ax.plot(x[imin:imax], y[imin:imax],
                orbit['z'][imin:imax], **line_options)
        ssplt.plot_vessel(ax=ax, projection='3D', params3d=shaded3d_options)
        ax_options['xlabel'] = 'x [m]'
        ax_options['ylabel'] = 'y [m]'
        ax_options['zlabel'] = 'z [m]'
        # ax = ssplt.axis_beauty(ax, ax_options)
    return ax


def plot_orbit_time_evolution(orbit, id_to_plot=[0], LN: float = 1.0,
                              grid: bool = True, FS: float = 14.0):
    """
    Plot the orbit parameters (energy, pitch, mu, Pphi)

    Jose Rueda Rueda: Jose Rueda Rueda

    NOTE: This is just to have a quick look, is NOT ready-to-publish format

    @param orbit: array or orbits loaded by load_orbits
    @param plot_options: font name and so on
    @param LN: Line width to plot
    @param FS: FontSize
    @param grid: If true, activate the grid
    @param id_to_plot: orbits to be plotted, list, default [0]
    """
    # Open the figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
    # Plot the constants
    for i in id_to_plot:
        # Plot the energy
        ax1.plot(orbit[i]['time'], orbit[i]['E'], linewidth=LN,
                 label='Orbit ' + str(orbit[i]['ID']))
        # Plot the pitch
        ax2.plot(orbit[i]['time'], orbit[i]['pitch'], linewidth=LN,
                 label='Orbit ' + str(orbit[i]['ID']))
        # Plot the magnetic moment
        ax3.plot(orbit[i]['time'], orbit[i]['mu'], linewidth=LN,
                 label='Orbit ' + str(orbit[i]['ID']))
        # Plot the Canonical momentum
        ax4.plot(orbit[i]['time'], orbit[i]['Pphi'], linewidth=LN,
                 label='Orbit ' + str(orbit[i]['ID']))
    plt.legend()
    if grid:
        ax1.grid(True, which='minor', linestyle=':')
        ax1.minorticks_on()
        ax1.grid(True, which='major')

        ax2.grid(True, which='minor', linestyle=':')
        ax2.minorticks_on()
        ax2.grid(True, which='major')

        ax3.grid(True, which='minor', linestyle=':')
        ax3.minorticks_on()
        ax3.grid(True, which='major')

        ax4.grid(True, which='minor', linestyle=':')
        ax4.minorticks_on()
        ax4.grid(True, which='major')

    ax1.set_ylabel('E [keV]', fontsize=FS)
    ax2.set_ylabel('$\\lambda$', fontsize=FS)
    ax3.set_ylabel('$\\mu [J/T]$', fontsize=FS)
    ax4.set_ylabel('$P_\\phi$', fontsize=FS)
    plt.xlabel('Time [s]')
    return


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
