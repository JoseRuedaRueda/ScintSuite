"""Routines to interact with FILDSIM"""
import os
import numpy as np
import math as ma
import Lib.LibParameters as ssp
from Lib.LibMachine import machine
from Lib.LibPaths import Path
import Lib.LibData as ssdat
import f90nml
paths = Path(machine)


# -----------------------------------------------------------------------------
# --- FILD Orientation
# -----------------------------------------------------------------------------
def calculate_fild_orientation(Br, Bz, Bt, alpha, beta, verbose=False):
    """
    Routine to calculate the magnetic field orientation with respect to FILD

    Important note, the original routine written by Joaquin in IDL received as
    input bt, br, bz... notice the different order or the inputs,
    to be consistent with the rest of this suite

    @param br: Magnetic field in the r direction
    @param bz: Magnetic field in the z direction
    @param bt: Magnetic field in the toroidal direction
    @param alpha: Poloidal orientation of FILD. Given in deg
    @param beta: Pitch orientation of FILD, given in deg

    @return theta: Euler angle to use as input in fildsim.f90 given in deg
    @return phi: Euler angle to use as input in fildsim.f90 given in deg
    """
    # In AUG the magnetic field orientation is counter current.
    # FILDSIM.f90 works with the co-current reference
    bt = ssdat.IB_sign * Bt
    br = ssdat.IB_sign * Br
    bz = ssdat.IB_sign * Bz

    # Transform to radians
    alpha = alpha * np.pi / 180.0
    beta = beta * np.pi / 180.0

    # Refer the orientation of the BField to the orientation of FILD
    # Alpha measured from R (positive getting away from axis) to Z (positive)
    bt1 = bt
    br1 = br * np.cos(alpha) - bz * np.sin(alpha)
    bz1 = br * np.sin(alpha) + bz * np.cos(alpha)

    # Now we rotate to the pitch orientation of FILD
    # Now we rotate in the toroidal z-plane
    br2 = br1
    bt2 = bt1 * np.cos(beta) - bz1 * np.sin(beta)
    bz2 = bt1 * np.sin(beta) + bz1 * np.cos(beta)

    # NOW BT2,BZ2 and BR2 are the coordinates referred to the local FILD
    # reference.
    # According to FILDSIM.f90
    # BT2 ---> Y component
    # BR2 ---> X component
    # BZ2 ---> Z component
    # THEN theta and phi are:
    # PHI --> Euler Angle measured from y(positive) to x(positive)
    # THETA --> Euler Angle measured from x(positive) to z(negative)

    phi = ma.atan2(br2, bt2) * 180.0 / np.pi
    theta = ma.atan2(-bz2, bt2) * 180.0 / np.pi
    if verbose:
        print('Bt, Bz, Br and B: ', bt, bz, br, ma.sqrt(bt**2 + bz**2 + br**2))
        print('FILD orientation is (alpha,beta)= ', alpha * 180.0 / np.pi,
              beta * 180.0 / np.pi)
        print('Alpha rotation: ', bt1, bz1, br1, ma.sqrt(bt1**2 + bz1**2
                                                         + br1**2))
        print('Bx By Bz and B in FILDSIM are: ', br2, bt2, bz2, ma.sqrt(
            bt2**2 + bz2**2 + br2**2))
        print('Euler angles are (phi,theta): ', phi, theta)

    return phi, theta


# -----------------------------------------------------------------------------
# --- RUN FILDSIM
# -----------------------------------------------------------------------------
def write_namelist(nml, p=os.path.join(paths.FILDSIM, 'cfg_files'),
                   overwrite=True):
    """
    Write fortran namelist

    jose rueda: jrrueda@us.es

    just a wrapper for the f90nml file writter

    @param p: full path towards the desired folder to store the namelist
    @param nml: namelist containing the desired fields.
    @param overwrite: flag to overwrite the namelist (if exist)

    @return file: The path to the written file
    """
    file = os.path.join(p, nml['config']['runid'] + '.cfg')
    f90nml.write(nml, file, force=overwrite)
    return file


def read_namelist(filename):
    """
    Read a FILDSIM namelist

    Jose Rueda: jrrueda@us.es

    just a wrapper for the f90nml capabilities

    @param filename: full path to the filename to read

    @return nml: dictionary with all the parameters of the FILDSIM run
    """
    return f90nml.read(filename)


def run_FILDSIM(namelist, queue=False):
    """
    Execute a FILDSIM simulation

    Jose Rueda and Anton J. van Vuuren

    @param namelist: full path to the namelist
    """

    if not queue:
        FILDSIM = os.path.join(paths.FILDSIM, 'bin', 'fildsim.exe')
        # namelist = ' ' + run_ID + '.cfg'
        os.system(FILDSIM + ' ' + namelist)
    else:
        '''
        write batch file to submit
        '''
        nml = read_namelist(namelist)
        f = open(nml['config']['result_dir']+'/Submit.sh', 'w')
        f.write('#!/bin/bash -l \n')
        f.write('#SBATCH -J FILDSIM_%s      #Job name \n' %(nml['config']['runid']))
        f.write('#SBATCH -o ./%x.%j.out        #stdout (%x=jobname, %j=jobid) \n')
        f.write('#SBATCH -e ./%x.%j.err        #stderr (%x=jobname, %j=jobid) \n')
        f.write('#SBATCH -D ./                 #Initial working directory \n')
        f.write('#SBATCH --partition=s.tok     #Queue/Partition \n')
        f.write('#SBATCH --qos=s.tok.short \n')
        f.write('#SBATCH --nodes=1             #Total number of nodes \n')
        f.write('#SBATCH --ntasks-per-node=1   #MPI tasks per node \n')
        f.write('#SBATCH --cpus-per-task=1     #CPUs per task for OpenMP \n')
        f.write('#SBATCH --mem 5GB           #Set mem./node requirement (default: 63000 MB, max: 190GB) \n')
        f.write('#SBATCH --time=03:59:00       #Wall clock limit \n')
        f.write('## \n')
        f.write('#SBATCH --mail-type=end       #Send mail, e.g. for begin/end/fail/none \n')
        f.write('#SBATCH --mail-user=%s@ipp.mpg.de  #Mail address \n' %(os.getenv("USER")))

        f.write('# Run the program: \n')
        FILDSIM = os.path.join(paths.FILDSIM, 'bin', 'fildsim.exe')
        f.write(FILDSIM + ' ' + namelist)
        f.close()

        os.system('sbatch '+ nml['config']['result_dir'] + '/Submit.sh')
    return


def guess_strike_map_name_FILD(phi: float, theta: float, machine: str = 'AUG',
                               decimals: int = 1):
    """
    Give the name of the strike-map file

    Jose Rueda Rueda: jrrueda@us.es

    Files are supposed to be named as given in the NamingSM_FILD.py file.
    The data base is composed by strike maps calculated each 0.1 degree

    @param phi: phi angle as defined in FILDSIM
    @param theta: theta angle as defined in FILDSIM
    @param machine: 3 characters identifying the machine
    @param decimals: number of decimal numbers to round the angles

    @return name: the name of the strike map file
    """
    # Taken from one of Juanfran files :-)
    p = round(phi, ndigits=decimals)
    t = round(theta, ndigits=decimals)
    if p < 0:
        if t < 0:
            name = machine +\
                "_map_{0:010.5f}_{1:010.5f}_strike_map.dat".format(p, t)
        else:
            name = machine +\
                "_map_{0:010.5f}_{1:09.5f}_strike_map.dat".format(p, t)
    else:
        if t < 0:
            name = machine +\
                "_map_{0:09.5f}_{1:010.5f}_strike_map.dat".format(p, t)
        else:
            name = machine +\
                "_map_{0:09.5f}_{1:09.5f}_strike_map.dat".format(p, t)
    return name


def find_strike_map(rfild: float, zfild: float,
                    phi: float, theta: float, strike_path: str,
                    machine: str = 'AUG',
                    FILDSIM_options={}, clean: bool = True,
                    decimals: int = 1):
    """
    Find the proper strike map. If not there, create it

    Jose Rueda Rueda: jrrueda@us.es

    @param    rfild: radial position of FILD (in m)
    @param    zfild: Z position of FILD (in m)
    @param    phi: phi angle as defined in FILDSIM
    @param    theta: beta angle as defined in FILDSIM
    @param    strike_path: path of the folder with the strike maps
    @param    FILDSIM_path: path of the folder with FILDSIM
    @param    machine: string identifying the machine. Defaults to 'AUG'.
    @param    FILDSIM_options: FILDSIM namelist options
    @param    clean: True: eliminate the strike_points.dat when calling FILDSIM
    @param    decimals: Number of decimals for theta and phi angles

    @return   name:  name of the strikemap to load

    @raises   Exception: If FILDSIM is call but the file is not created.
    """
    # Find the name of the strike map
    name = guess_strike_map_name_FILD(phi, theta, machine=machine,
                                      decimals=decimals)
    # See if the strike map exist
    if os.path.isfile(os.path.join(strike_path, name)):
        return name
    # If do not exist, create it
    # load reference namelist
    # Reference namelist
    nml = f90nml.read(os.path.join(strike_path, 'parameters.cfg'))
    # If a FILDSIM naelist was given, overwrite reference parameters with the
    # desired by the user, else set at least the proper geometry directory
    if FILDSIM_options is not None:
        # Set the geometry directory
        if 'plate_setup_cfg' in FILDSIM_options:
            if 'geometry_dir' not in FILDSIM_options['plate_setup_cfg']:
                FILDSIM_options['plate_setup_cfg']['geometry_dir'] = \
                    os.path.join(paths.FILDSIM, 'geometry/')
        else:
            nml['plate_setup_cfg']['geometry_dir'] = \
                os.path.join(paths.FILDSIM, 'geometry/')
        # set the rest of user defined options
        for block in FILDSIM_options.keys():
            nml[block].update(FILDSIM_options[block])
    else:
        nml['plate_setup_cfg']['geometry_dir'] = \
            os.path.join(paths.FILDSIM, 'geometry/')

    # set namelist name, theta and phi
    nml['config']['runid'] = name[:-15]
    nml['config']['result_dir'] = strike_path
    nml['input_parameters']['theta'] = round(theta, ndigits=decimals)
    nml['input_parameters']['phi'] = round(phi, ndigits=decimals)
    conf_file = write_namelist(nml)
    # run the FILDSIM simulation
    bin_file = os.path.join(paths.FILDSIM, 'bin', 'fildsim.exe')
    os.system(bin_file + ' ' + conf_file)

    if clean:
        strike_points_name = name[:-15] + '_strike_points.dat'
        os.system('rm ' + os.path.join(strike_path, strike_points_name))

    if os.path.isfile(os.path.join(strike_path, name)):
        return name
    # If we reach this point, something went wrong
    a = 'FILDSIM simulation has been done but the strike map can be found'
    raise Exception(a)


# -----------------------------------------------------------------------------
# --- Read plates
# -----------------------------------------------------------------------------
def read_plate(filename):
    """
    Read FILDSIM plate

    jose rueda: jrrueda@us.es,
    based on a pice of code writtan by ajvv

    @param filename: full path to the plate to read

    @return plate: dictionary with:
        -# 'name': name of the palte,
        -# 'N_vertices': number of vertices
        -# 'vertices': vertices coordinates
    """
    f = open(filename, 'r')
    f.readline()  # dummy line
    name = (str(f.readline()).split('='))[1]
    N_vertices = int((str(f.readline()).split('='))[1])
    vertices = np.zeros((N_vertices, 3))

    for n in range(N_vertices):
        xyz = str(f.readline()).split(',')
        vertices[n, 0] = float(xyz[0])
        vertices[n, 1] = float(xyz[1])
        vertices[n, 2] = float(xyz[2])

    plate = {
        'name': name,
        'N_vertices': N_vertices,
        'vertices': vertices
    }
    f.close()
    return plate


def read_orbits(orbits_file, orbits_index_file):
    """
    Read FILDSIM orbits

    ajvv

    @param orbits_file: full path to the orbits file
    @param orbits_index_file: full path to the orbits_index file
    @return orbit: list with orbit trajectories where each tracetory
                   is given as an array with shape:
                   (number of trajectory points, 3).
                   The second index refers to the x, y and z coordinates
                   of the trajectory points.
    """
    orbits = []

    orbits_index_data = np.loadtxt(orbits_index_file)
    orbits_data = np.loadtxt(orbits_file)

    ##Todo check if there are no orbits
    if len(orbits_index_data) == 1:
        orbits_index_data = [orbits_index_data]

    ii = 0
    for i in orbits_index_data:
        orbits.append( orbits_data[ii: ii+ int(i), :] )
        ii += int(i)

    return orbits

# -----------------------------------------------------------------------------
# --- Energy definition FILDSIM
# -----------------------------------------------------------------------------
def get_energy(gyroradius, B: float, A: float = 2., Z: int = 1):
    """
    Calculate the energy given a gyroradius, FILDSIM criteria

    jose Rueda: jrrueda@us.es

    @param gyroradius: Larmor radius as taken from FILD strike map [in cm]
    @param B: Magnetic field, [in T]
    @param A: Ion mass number
    @param Z: Ion charge [in e units]

    @return E: the energy [in eV]
    """
    m = ssp.mp * A  # Mass of the ion
    E = 0.5 * (gyroradius/100.0 * Z * B)**2 / m * ssp.c ** 2
    return E


def get_gyroradius(E, B: float, A: float = 2., Z: int = 1):
    """
    Calculate the gyroradius given an energy, FILDSIM criteria

    jose Rueda: jrrueda@us.es

    @param energy: Energy [eV]
    @param B: Magnetic field, [in T]
    @param A: Ion mass number
    @param Z: Ion charge [in e units]

    @return r: Larmor radius as taken from FILD strike map [in cm]
    """
    m = ssp.mp * A  # Mass of the ion
    r = 100. * np.sqrt(2.0 * E * m / ssp.c**2) / Z / B
    return r
