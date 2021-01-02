"""Contains the methods and classes to interact with iHIBPsim tracker"""

import numpy as np
from version_suite import version
from LibMachine import machine
import LibPlotting as ssplt
import matplotlib.pyplot as plt
if machine == 'AUG':
    import LibDataAUG as ssdat


def write_fields(filename: str, field: dict):
    """
    Write fields (electric or magnetics)

    Jose Rueda: jose.rueda@ipp.mpg.de

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


def prepare_B_field(shot: int, time: float, diag: str = 'EQH',
                    Rmin: float = 1.0, Rmax: float = 2.2, nR: int = 120,
                    zmin: float = -1.0, zmax: float = 1.0, nz: int = 200,
                    phimin: float = 0., phimax: float = 2.*np.pi,
                    nphi: int = 64):
    """
    Load the magnetic field to launch an iHIBPsim simulation

    Jose Rueda: jose.rueda@ipp.mpg.de

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

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

    To see the meaning of all parameters, look at the nicelly written iHIBPsim
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

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

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

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

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
    # Future message for pablo, this is a bit too match if the output file has
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
    dum1 = {'ID': None, 'R': None, 'z': None, 'phi': None, 'vR': None,
            'vz': None, 'vt': None, 'q': None, 'm': None, 'logw': None,
            't': None}
    output = [dum1]*counter
    kk = 0
    for id in saved_ID:
        flags = Mar[:, 0] == id
        output[kk]['ID'] = id
        output[kk]['R'] = Mar[flags, 1]
        output[kk]['z'] = Mar[flags, 2]
        output[kk]['phi'] = Mar[flags, 3]
        output[kk]['time'] = Mar[flags, 10]
        output[kk]['m'] = Mar[flags, 7][0]
        output[kk]['q'] = Mar[flags, 8]
        if full_info:
            output[kk]['vr'] = Mar[flags, 4]
            output[kk]['vz'] = Mar[flags, 5]
            output[kk]['vt'] = Mar[flags, 6]
            output[kk]['logw'] = Mar[flags, 9]
        if kk == counter - 1:
            return output
        kk += 1
    # If we reached this point, all the orbits we wanted could not be loaded:
    print('Not all the desired orbits could be loaded. Check file ??')
    return output


def plot_orbit(orbit, view: str = '2D', ax_options: dict = {}, ax=None,
               line_options: dict = {}):
    """
    Plot the orbit

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param orbit: Orbit dictionary created by 'load_orbits', not the full list,
    just the one you want to plot
    @param view: '2D' to plot, (R,z), (x,y). '3D' to plot the 3D orbit
    @param ax_options: options for the function axis_beauty
    @param line_options: options fot the line plot (markers, colors and so on)
    @param ax: axes where to plot, if none, new ones will be created. Note,
    if the '2D' mode is used, ax should be a list of axes, the first one for
    the Rz projection
    @todo implement plotting toroidal vessel
    """
    # --- Initialise the plotting parameters
    if 'fontsize' not in ax_options:
        ax_options['fontsize'] = 16
    if 'grid' not in ax_options:
        ax_options['grid'] = 'both'
    if 'linewidth' not in line_options:
        line_options['linewidth'] = 2

    if view == '2D':
        # Open the figure
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        # Plot the Rz, projection
        ax[0].plot(orbit['R'], orbit['z'], label='ID: ' + str(orbit['ID']),
                   **line_options)
        ax_options['xlabel'] = 'R [m]'
        ax_options['ylabel'] = 'z [m]'
        ssplt.plot_vessel(ax[0])
        ax[0] = ssdat.axis_beauty(ax[0], ax_options)
        # Plot the xy projection
        x = orbit['R'] * np.cos(orbit['phi'])
        y = orbit['R'] * np.sin(orbit['phi'])
        ax[1].plot(x, y, label='ID: ' + str(orbit['ID']), **line_options)
        ax_options['xlabel'] = 'x [m]'
        ax_options['ylabel'] = 'y [m]'
        # plot_vessel(ax[1])
        ax[1] = ssplt.axis_beauty(ax[1], ax_options)
    return ax
