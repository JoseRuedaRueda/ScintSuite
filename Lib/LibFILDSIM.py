"""Routines to iinteract with FILDSIM"""
import os
import numpy as np
import math as ma
import LibParameters as ssp
import LibDataAUG as ssdat

# @todo: include here an if to load aug librery or another one


def calculate_fild_orientation(Br, Bz, Bt, alpha, beta, verbose=False):
    """
    Routine to calculate the magnetic field orientation with tespect to FILD

    Important note, the original routine writen by joaquin in IDL received as
    input bt, br, bz... notice the diferent order or the inputs,
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
    if Bt < 0:
        bt = -Bt
        br = -Br
        bz = -Bz
    else:
        bt = Bt
        br = Br
        bz = Bz

    # Transform to radians
    alpha = alpha * np.pi / 180.0
    beta = beta * np.pi / 180.0

    # Refer the orientation of the BField wo the orientation of FILD
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

    phi = ma.atan(br2 / bt2) * 180.0 / np.pi
    theta = ma.atan(-bz2 / bt2) * 180.0 / np.pi
    if verbose:
        print('Bt, Bz, Br and B: ', bt, bz, br, ma.sqrt(bt**2 + bz**2 + br**2))
        print('FILD orientation is (alpha,beta)= ', alpha * 180.0 / np.pi,
              beta * 180.0 / np.pi)
        print('Alpha rotation: ', bt1, bz1, br1, ma.sqrt(bt1**2 + bz1**2 +
                                                         br1**2))
        print('Bx By Bz in FILDSIM are: ', bt2, bz2, br2, ma.sqrt(
            bt2**2 + bz2**2 + br2**2))
        print('Euler angles are (phi,theta): ', phi, theta)

    return phi, theta


def calculate_absolute_calibration_frame(cal_frame, exposure_time,
                                         pinhole_area, calib_exposure_time,
                                         int_photon_flux, pixel_area_covered):
    """
    Calculate absolute photon flux

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

    @param cal_frame: calibration frame [counts]
    @param exposure_time: in s
    @param pinhole_area: in m^2
    @param calib_exposure_time: in s
    @param int_photon_flux: in Photons/(s m^2)
    @param pixel_area_covered: in m^2
    @return frame: Calibrated frame [Counts * s * m^2 / photons]
    """
    frame = cal_frame * exposure_time * pinhole_area / (
                calib_exposure_time * int_photon_flux * pixel_area_covered)

    return frame


def calculate_absolute_flux(raw_frame, calibration_frame, efficiency_energy,
                            efficiency_yield, b_field, band_pass_filter,
                            interpolated_gyro, interpolated_fcol, roi,
                            pinhole_area, exposure_time, A, Z,
                            calib_exposure_time, pixel_area_covered,
                            int_photon_flux, method, ignore_fcol=False,
                            lower_limit=0.05):
    """
    Routine to estimate absolute flux of losses

    @todo this is half done
    Based on the IDL routine writen by J. Galdón (jgaldon@us.es)

    @param raw_frame: as 2D numpy array
    @param calibration_frame: as 2D numpy array
    @param efficiency_energy: [Energy given in eV]
    @param efficiency_yield:  [Gieven in photons /ion]
    @param b_field: [T]
    @param band_pass_filter:
    @param interpolated_gyro: [cm]
    @param interpolated_fcol: [Nscint/Npinhole*100]
    @param roi:
    @param pinhole_area: [m^2]
    @param exposure_time: [s]
    @param A:
    @param Z:
    @param calib_exposure_time:
    @param pixel_area_covered:
    @param int_photon_flux:
    @param method:
    @param ignore_fcol:
    @param lower_limit:
    @return:
    """
    # Check if we are ignoring the collimator factor which means that we will
    # do the calculation in the scintillator, not at the pinhole [Real
    # plasma losses are at the pinhole, not the scintillator]
    if ignore_fcol:
        interpolated_fcol = interpolated_fcol * 0 + 1.0
        print('Ignoring collimator factor',
              '--> Calculating Ion Flux at the scintillator not the pinhole!')
    else:
        # FILDSIM gives data in %, let's normalise it to unity to work
        interpolated_fcol = interpolated_fcol / 100.0
        # Remove values which are to low, just to avoid numerical problems,
        # as the collimator factors goes in the denominator of the final
        # expression
        flags = interpolated_fcol < lower_limit
        interpolated_fcol[flags] = 0.0

    # Check frames sizes and resize if necessary
    if raw_frame.shape != raw_frame.shape:
        print('Size of the data frame and calibration frame not matching!!')
        print('Artifitially resizing the calibration frame!!')
        ## todo implement the rebining of the calibration frame

    # Check if the remap was done before calling this function
    if np.max(interpolated_gyro) <= 0:
        print('Remap was not done before call this function. Returning')
        return

    # Transform gyroradius to energy
    interpolated_energy = 0.5 * (interpolated_gyro / 100.0) ** 2 * b_field **\
        2 * Z ** 2 / ssp.mp * ssp.c ** 2
    ## todo ask joaquin about this stuff of imposing efficiency larger than 1
    efficiency_frame = np.interp(interpolated_energy, efficiency_energy,
                                 efficiency_yield)
    efficiency_frame[efficiency_energy < 1.0] = 1.0


def write_namelist(p: str, runID: str = 'test', result_dir: str = './results/',
                   backtrace: str = '.false.', N_gyroradius: int = 7,
                   N_pitch: int = 7, save_orbits: int = 0,
                   verbose: str = '.False.',
                   N_ions: int = 1500, step: float = 0.01,
                   helix_length: float = 10.0,
                   gyroradius=[2., 3., 4., 5., 6., 7., 8.],
                   pitch_angle=[80., 70., 60., 50., 40., 30., 20.],
                   min_gyrophase: float = 1.0,
                   max_gyrophase: float = 1.8, start_x=[-0.025, 0.025],
                   start_y=[-0.1, 0.1], start_z=[0.0, 0.0],
                   theta=0.0, phi=0.0, geometry_dir='./geometry/',
                   N_scintillator=1, N_slits=6,
                   scintillator_files=['aug_fild1_scint.pl'],
                   slit_files=['aug_fild1_pinhole_1_v2.pl',
                               'aug_fild1_pinhole_2_v2.pl',
                               'aug_fild1_slit_1_v2.pl',
                               'aug_fild1_slit_back_v2.pl',
                               'aug_fild1_slit_lateral_1_v2.pl',
                               'aug_fild1_slit_lateral_2_v2.pl']):
    """
    Write the namellist for a FILDSIM simulation

    All parameters of the namelist have the meaning given in FILDSIM. See
    FILDSIM documentation for a full description

    @param p: path where to write the file
    """
    # check the input parameters
    ngyr = len(gyroradius)
    npitch = len(pitch_angle)
    nscint = len(scintillator_files)
    nslit = len(slit_files)
    if ngyr != N_gyroradius:
        raise Exception("Number of  Gyroradius do not match N_gyroradius")
    if npitch != N_pitch:
        raise Exception("Number of pitches does not match N_pitch")
    if nscint != N_scintillator:
        raise Exception("Number of scintillator files does not match")
    if nslit != N_slits:
        print(nslit)
        print(N_slits)
        raise Exception("Number of slit files does not match")

    name_of_file = os.path.join(p, runID + '.cfg')
    # Open the aug_fild1_pinhole_2
    f = open(name_of_file, "w")
    print("&config", file=f, sep='')
    print("runID='", runID, "',", file=f, sep='')
    print("result_dir='", result_dir, "',", file=f, sep='')
    print("backtrace=", backtrace, ",", file=f, sep='')
    print("N_gyroradius=", N_gyroradius, ",", file=f, sep='')
    print("N_pitch=", N_pitch, ",", file=f, sep='')
    print("save_orbits=", save_orbits, ",", file=f, sep='')
    print("verbose=", verbose, ",", file=f, sep='')
    print("/", file=f, sep='')

    print("", file=f, sep='')
    print("&input_parameters", file=f, sep='')
    print("N_ions=", N_ions, ",", file=f, sep='')
    print("step=", step, ",", file=f, sep='')
    print("helix_length=", helix_length, ",", file=f, sep='')
    print("gyroradius", end='=', file=f, sep='')
    for i in range(ngyr-1):
        print(gyroradius[i], end=',', file=f, sep='')
    print(gyroradius[ngyr-1], ',', file=f, sep='')
    print("pitch_angle", end='=', file=f, sep='')
    for i in range(npitch-1):
        print(pitch_angle[i], end=',', file=f, sep='')
    print(pitch_angle[npitch-1], ',', file=f, sep='')
    print('gyrophase_range=', min_gyrophase, ',', max_gyrophase, ',',
          file=f, sep='')
    print('start_x=', start_x[0], ',', start_x[1], ',', file=f, sep='')
    print('start_y=', start_y[0], ',', start_y[1], ',', file=f, sep='')
    print('start_z=', start_x[0], ',', start_x[1], ',', file=f, sep='')
    print('theta=', theta, ',', file=f, sep='')
    print('phi=', phi, ',', file=f, sep='')
    print('/', file=f, sep='')

    print("", file=f, sep='')
    print('&plate_setup_cfg', file=f, sep='')
    print("geometry_dir='", geometry_dir, "',", file=f, sep='')
    print("N_scintillator=", N_scintillator, ',', file=f, sep='')
    print('N_slits=', N_slits, ',', file=f, sep='')
    print('/', file=f, sep='')

    print("", file=f, sep='')
    print('&plate_files', file=f, sep='')
    print("scintillator_files=", end="", file=f, sep='')
    for i in range(N_scintillator-1):
        print("'", scintillator_files[i], end="',", file=f, sep='')
    print("'", scintillator_files[N_scintillator-1], "',", file=f, sep='')
    print("slit_files=", end="", file=f, sep='')
    for i in range(N_slits-1):
        print("'", slit_files[i], end="',", file=f, sep='')
    print("'", slit_files[N_slits-1], "',", file=f, sep='')
    print('/', file=f, sep='')
    f.close()


def run_FILDSIM(FILDSIM_path, run_ID):
    """
    Execute a FILDSIM simulation

    @todo Include the capability of connecting to a external machine

    @param FILDSIM_path: path to the FILDSIM code (main folder)
    @param runID: run ID, the configuration file is assumed to be runID.cfg
    """
    FILDSIM = os.path.join(path, file)
    namelist = ' ' + run_ID + '.cfg'
    os.system(FILDSIM + namelist)


def guess_strike_map_name_FILD(phi: float, theta: float, machine: str = 'AUG'):
        """
        Give the name of the strike-map file

        Jose Rueda Rueda: jose.rueda@ipp.mpg.de

        Files are supposed to be named as given in the NamingSM_FILD.py file.
        The data base is composed by strike maps calculated each 0.1 degree

        @param phi: phi angle as defined in FILDSIM
        @param theta: theta angle as defined in FILDSIM
        @param machine: 3 characters identifying the machine
        @return name: the name of the strike map file
        """
        # Set the angles with 1 decimal digit
        phi_1 = round(phi, ndigits=1)
        phi_label = str(abs(phi_1)) + '0000'
        if abs(phi_1) < 10.0:
            phi_label = '00' + phi_label
        elif abs(phi_1) < 100.0:
            phi_label = '0' + phi_label
        elif abs(phi) > 360.0:
            print('Phi is larger than 360º?!?', phi)

        if phi_1 < 0:
            phi_label = '-' + phi_label

        theta_1 = round(theta, ndigits=1)
        theta_label = str(abs(theta_1)) + '0000'
        if abs(theta_1) < 10.0:
            theta_label = '00' + theta_label
        elif abs(theta_1) < 100.0:
            theta_label = '0' + theta_label
        elif abs(theta_1) > 360.0:
            print('Theta is larger than 360º?!?', theta_label)

        if theta_1 < 0:
            theta_label = '-' + theta_label

        name = machine + '_map_' + phi_label + '_' + theta_label + \
            '_strike_map.dat'
        return name


# def find_strike_map(rfild: float, zfild: float, t: float, shot: int,
#                     alpha: float, beta: float, strike_path: str,
#                     FILDSIM_path: str, machine: str = 'AUG',
#                     FILDSIM_options={}, clean: bool = True):
#     """
#     Find the proper strike map. If not there, create it
#
#     Jose Rueda Rueda: jose.rueda@ipp.mpg.de
#
#     @param    rfild: radial position of FILD (in m)
#     @type:    float
#
#     @param    zfild: Z position of FILD (in m)
#     @type:    float
#
#     @param    t: Time point (to load the B field)
#     @type:    float
#
#     @param    shot: Shot number (to load the B field)
#     @type:    int
#
#     @param    alpha: Alpha angle as defined in FILDSIM
#     @type:    float
#
#     @param    beta: beta angle as defined in FILDSIM
#     @type:    float
#
#     @param    strike_path: path of the folder with the strike maps
#     @type:    str
#
#     @param    FILDSIM_path: path of the folder with FILDSIM
#     @type:    str
#
#     @param    machine: string identifying the machine. Defaults to 'AUG'.
#     @type:    str
#
#     @param    FILDSIM_options: FILDSIM namelist options
#     @type:    type
#
#     @param    clean: True: eliminate the strike_points.dat generated FILDSIM
#     @type:    bool
#
#     @return   name:  name of the strikemap to load
#     @rtype:   str
#
#     @raises   ExceptionName: If FILDSIM is call but the file is not created.
#     """
#     # Get the magnetic field at FILD position
#     br, bz, bt, bp = ssdat.get_mag_field(shot, rfild, zfild, time=t)
#     # Calculate the orientation of FILD respect to the magnetic field
#     phi, theta = calculate_fild_orientation(br, bz, bt, alpha, beta)
#     # Find the name of the strike map
#     name = guess_strike_map_name_FILD(phi, theta, machine=machine)
#     # See if the strike map exist
#     if os.path.isfile(os.path.join(strike_path, name)):
#         return name
#     # If do not exist, create it
#     # set namelist name
#     FILDSIM_options['runID'] = name[:-15]
#     FILDSIM_options['result_dir'] = strike_path
#     if 'geometry_dir' not in FILDSIM_options:
#         FILDSIM_options['geometry_dir'] = os.path.join(FILDSIM_path,
#                                                        '/geometry/')
#     path_conf = os.path.join(FILDSIM_path, 'cfg_files')
#     write_namelist(path_conf, **FILDSIM_options)
#     # run the FILDSIM simulation
#     conf_file = os.path.join(FILDSIM_path, 'cfg_files',
#                              FILDSIM_options['runID'] + '.cfg')
#     bin_file = os.path.join(FILDSIM_path, 'bin/fildsim.exe')
#     os.system(bin_file + ' ' + conf_file)
#
#     if clean:
#         strike_points_name = name[:-15] + '_strike_points.dat'
#         os.system('rm ' + os.path.join(strike_path, strike_points_name))
#
#     if os.path.isfile(os.path.join(strike_path, name)):
#         return name
#     # If we reach this point, somethin went wrong
#     a = 'FILDSIM simulation has been done but the strike map can be found'
#     raise Exception(a)


def find_strike_map(rfild: float, zfild: float,
                    phi: float, theta: float, strike_path: str,
                    FILDSIM_path: str, machine: str = 'AUG',
                    FILDSIM_options={}, clean: bool = True):
    """
    Find the proper strike map. If not there, create it

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

    @param    rfild: radial position of FILD (in m)
    @type:    float

    @param    zfild: Z position of FILD (in m)
    @type:    float

    @param    phi: phi angle as defined in FILDSIM
    @type:    float

    @param    theta: beta angle as defined in FILDSIM
    @type:    float

    @param    strike_path: path of the folder with the strike maps
    @type:    str

    @param    FILDSIM_path: path of the folder with FILDSIM
    @type:    str

    @param    machine: string identifying the machine. Defaults to 'AUG'.
    @type:    str

    @param    FILDSIM_options: FILDSIM namelist options
    @type:    type

    @param    clean: True: eliminate the strike_points.dat when calling FILDSIM
    @type:    bool

    @return   name:  name of the strikemap to load
    @rtype:   str

    @raises   ExceptionName: If FILDSIM is call but the file is not created.
    """
    # Find the name of the strike map
    name = guess_strike_map_name_FILD(phi, theta, machine=machine)
    # See if the strike map exist
    if os.path.isfile(os.path.join(strike_path, name)):
        return name
    # If do not exist, create it
    # set namelist name
    FILDSIM_options['runID'] = name[:-15]
    FILDSIM_options['result_dir'] = strike_path
    if 'geometry_dir' not in FILDSIM_options:
        FILDSIM_options['geometry_dir'] = \
            os.path.join(FILDSIM_path, 'geometry/')
        # print(FILDSIM_path)
        # print(FILDSIM_options['geometry_dir'])
    path_conf = os.path.join(FILDSIM_path, 'cfg_files')
    write_namelist(path_conf, **FILDSIM_options)
    # run the FILDSIM simulation
    conf_file = os.path.join(FILDSIM_path, 'cfg_files',
                             FILDSIM_options['runID'] + '.cfg')
    bin_file = os.path.join(FILDSIM_path, 'bin/fildsim.exe')
    os.system(bin_file + ' ' + conf_file)

    if clean:
        strike_points_name = name[:-15] + '_strike_points.dat'
        os.system('rm ' + os.path.join(strike_path, strike_points_name))

    if os.path.isfile(os.path.join(strike_path, name)):
        return name
    # If we reach this point, somethin went wrong
    a = 'FILDSIM simulation has been done but the strike map can be found'
    raise Exception(a)
