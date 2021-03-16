"""Routines to interact with FILDSIM"""
import os
import warnings
import numpy as np
import math as ma
import LibParameters as ssp
import LibPlotting as ssplt
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.special import erf       # error function
from tqdm import tqdm               # For waitbars
from LibMachine import machine
from LibPaths import Path
if machine == 'AUG':
    import LibDataAUG as ssdat
try:
    import f90nml
except ImportError:
    warnings.warn('You cannot read FILDSIM namelist',
                  category=UserWarning)

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

    phi = ma.atan(br2 / bt2) * 180.0 / np.pi
    theta = ma.atan(-bz2 / bt2) * 180.0 / np.pi
    if verbose:
        print('Bt, Bz, Br and B: ', bt, bz, br, ma.sqrt(bt**2 + bz**2 + br**2))
        print('FILD orientation is (alpha,beta)= ', alpha * 180.0 / np.pi,
              beta * 180.0 / np.pi)
        print('Alpha rotation: ', bt1, bz1, br1, ma.sqrt(bt1**2 + bz1**2
                                                         + br1**2))
        print('Bx By Bz in FILDSIM are: ', bt2, bz2, br2, ma.sqrt(
            bt2**2 + bz2**2 + br2**2))
        print('Euler angles are (phi,theta): ', phi, theta)

    return phi, theta


# -----------------------------------------------------------------------------
# --- WEIGHT FUNCTION AND TOMOGRAPHY PREPARATION
# -----------------------------------------------------------------------------
def calculate_photon_flux(raw_frame, calibration_frame_in, exposure_time,
                          pinhole_area, calib_exposure_time,
                          int_photon_flux, pixel_area_covered, mask=None):
    """
    Convert counts int photons/s in a FILD frame

    Jose Rueda: jrrueda@us.es

    Note: Based in an IDL routine of Joaquin Galdón.

    @param calibration_frame_in: Frame with the calibration image, in counts
    @param raw_frame: frame to be calibrated, in counts
    @param exposure_time: in s
    @param pinhole_area: in m^2
    @param calib_exposure_time: in s
    @param int_photon_flux: in Photons/(s m^2)
    @param pixel_area_covered: in m^2
    @param mask: binary mask created with the routines of the TimeTraces module

    @todo complete this documentation
    @return output: Dictionary with the fields:
        -# 'photon_flux_frame':
        -# 'photon_flux':
        -# 'calibration_frame':
        -# 'exposure_time':
        -# 'pinhole_Area':
    """
    # Check frame size
    s1 = raw_frame.shape
    s2 = calibration_frame_in.shape

    if len(s1) > 1:
        if s1 != s2:
            print('Size of data frame and calibration frame not matching!')
            print('Use mean calibration frame method!')
            raise Exception('Size not matching')
    else:
        print('Using mean calibration frame method -->')
        print('Using single (mean) value instead of 2D array')

    # get the calibration frame in Counts*s*m^2/photons
    calibration_frame_out = \
        calculate_absolute_calibration_frame(calibration_frame_in,
                                             exposure_time,
                                             pinhole_area, calib_exposure_time,
                                             int_photon_flux,
                                             pixel_area_covered)
    # Calibrate the raw frame we want to invert
    photon_flux_frame = np.ma.masked_invalid(raw_frame / calibration_frame_out)
    photon_flux = photon_flux_frame.sum()

    # Apply a mask, if needed
    if mask is not None:
        print('Using a ROI for the photon frame calculation')
        dummy = np.zeros(photon_flux_frame.shape)
        dummy[mask] = photon_flux_frame[mask]
        photon_flux_frame = dummy
        photon_flux = photon_flux_frame.sum()

    # Create the output dictionary
    output = {
        'photon_flux_frame': photon_flux_frame,
        'photon_flux': photon_flux,
        'calibration_frame': calibration_frame_out,
        'exposure_time': exposure_time,
        'pinhole_Area': pinhole_area}
    return output


def calculate_absolute_calibration_frame(cal_frame, exposure_time,
                                         pinhole_area, calib_exposure_time,
                                         int_photon_flux, pixel_area_covered):
    """
    Calculate absolute photon flux

    Jose Rueda Rueda: jrrueda@us.es

    About the units:
        -# Int_photon_flux --> Photons/(s*m^2)
        -# Pixel_area_covered --> m^2
        -# Calib_exposure_time --> s
        -# exposure_time --> s
        -# pinhole_area --> m^2
        -# Calibration frame (input) --> Counts

        -# Raw Frame --> Counts
        -# Calibration frame (output) --> Counts*s*m^2/photons
        -# Photon flux frame --> Photons/(s*m^2)

    @param cal_frame: calibration frame [counts]
    @param exposure_time: in s
    @param pinhole_area: in m^2
    @param calib_exposure_time: in s
    @param int_photon_flux: in Photons/(s m^2)
    @param pixel_area_covered: in m^2
    @return frame: Calibrated frame [Counts * s * m^2 / photons]
    """
    frame = cal_frame * exposure_time * pinhole_area / \
        (calib_exposure_time * int_photon_flux * pixel_area_covered)

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
    Based on the IDL routine written by J. Galdón (jgaldon@us.es)

    @param raw_frame: as 2D numpy array
    @param calibration_frame: as 2D numpy array
    @param efficiency_energy: [Energy given in eV]
    @param efficiency_yield:  [Given in photons /ion]
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
        # FILDSIM gives data in %, let's normalize it to unity to work
        interpolated_fcol = interpolated_fcol / 100.0
        # Remove values which are to low, just to avoid numerical problems,
        # as the collimator factors goes in the denominator of the final
        # expression
        flags = interpolated_fcol < lower_limit
        interpolated_fcol[flags] = 0.0

    # Check frames sizes and resize if necessary
    if raw_frame.shape != raw_frame.shape:
        print('Size of the data frame and calibration frame not matching!!')
        print('Artificially resizing the calibration frame!!')
        ## todo implement the re-binning of the calibration frame

    # Check if the remap was done before calling this function
    if np.max(interpolated_gyro) <= 0:
        print('Remap was not done before call this function. Returning')
        return

    # Transform gyroradius to energy
    interpolated_energy = 0.5 * (interpolated_gyro / 100.0)**2 * b_field **\
        2 * Z ** 2 / ssp.mp * ssp.c ** 2 / A
    ## todo ask Joaquin about this stuff of imposing efficiency larger than 1
    efficiency_frame = np.interp(interpolated_energy, efficiency_energy,
                                 efficiency_yield)
    efficiency_frame[efficiency_energy < 1.0] = 1.0


def build_weight_matrix(SMap, rscint, pscint, rpin, ppin,
                        efficiency: dict = {}):
    """Under development"""
    # Check the StrikeMap
    if SMap.resolution is None:
        SMap.calculate_resolutions()

    print('Calculating FILD weight matrix')
    # Parameters of the scintillator grid:
    nr_scint = len(rscint)
    np_scint = len(pscint)

    dr_scint = abs(rscint[1] - rscint[0])
    dp_scint = abs(pscint[1] - pscint[0])

    # Pinhole grid
    nr_pin = len(rpin)
    np_pin = len(ppin)

    # dr_pin = abs(rpin[1] - rpin[0])
    # dp_pin = abs(ppin[1] - ppin[0])

    # Build the collimator factor matrix, in the old IDL implementation,
    # matrices for the sigmas and skew where also build, but in this python
    # code these matrices where constructed by the fitting routine
    ngyr = len(SMap.strike_points['gyroradius'])
    npitch = len(SMap.strike_points['pitch'])
    fcol_aux = np.zeros((ngyr, npitch))
    for ir in range(ngyr):
        for ip in range(npitch):
            flags = (SMap.gyroradius == SMap.strike_points['gyroradius'][ir]) \
                * (SMap.pitch == SMap.strike_points['pitch'][ip])
            if np.sum(flags) > 0:
                fcol_aux[ir, ip] = SMap.collimator_factor[flags]

    # Interpolate the resolution and collimator factor matrices
    # - Create grid for interpolation:
    xx, yy = np.meshgrid(SMap.strike_points['gyroradius'],
                         SMap.strike_points['pitch'])
    xxx = xx.flatten()
    yyy = yy.flatten()
    # - Create grid to evaluate the interpolation
    xxx_pin, yyy_pin = np.meshgrid(rpin, ppin)
    # - sigmas:
    dummy = SMap.resolution['Gyroradius']['sigma'].T
    dummy = dummy.flatten()
    flags = np.isnan(dummy)
    x1 = xxx[~flags]
    y1 = yyy[~flags]
    z1 = dummy[~flags]
    sigmar = interpolate.griddata((x1, y1), z1, (xxx_pin, yyy_pin))
    sigmar = sigmar.T

    dummy = SMap.resolution['Pitch']['sigma'].T
    dummy = dummy.flatten()
    flags = np.isnan(dummy)
    x1 = xxx[~flags]
    y1 = yyy[~flags]
    z1 = dummy[~flags]
    sigmap = interpolate.griddata((x1, y1), z1, (xxx_pin, yyy_pin))
    sigmap = sigmap.T
    # - Collimator factor
    dummy = fcol_aux.T
    dummy = dummy.flatten()
    x1 = xxx[~flags]
    y1 = yyy[~flags]
    z1 = dummy[~flags]
    fcol = interpolate.griddata((x1, y1), z1, (xxx_pin, yyy_pin))
    fcol = fcol.T
    # - Centroids:
    dummy = SMap.resolution['Gyroradius']['center'].T
    dummy = dummy.flatten()
    flags = np.isnan(dummy)
    x1 = xxx[~flags]
    y1 = yyy[~flags]
    z1 = dummy[~flags]
    centroidr = interpolate.griddata((x1, y1), z1, (xxx_pin, yyy_pin))
    centroidr = centroidr.T

    dummy = SMap.resolution['Pitch']['center'].T
    dummy = dummy.flatten()
    flags = np.isnan(dummy)
    x1 = xxx[~flags]
    y1 = yyy[~flags]
    z1 = dummy[~flags]
    centroidp = interpolate.griddata((x1, y1), z1, (xxx_pin, yyy_pin))
    centroidp = centroidp.T
    # - Screw
    try:
        dummy = SMap.resolution['Gyroradius']['gamma'].T
        dummy = dummy.flatten()
        flags = np.isnan(dummy)
        x1 = xxx[~flags]
        y1 = yyy[~flags]
        z1 = dummy[~flags]
        gamma = interpolate.griddata((x1, y1), z1, (xxx_pin, yyy_pin))
        gamma = gamma.T
        print('Using Screw Gaussian model!')
        screw_model = True
    except KeyError:
        print('Using pure Gaussian model!')
        screw_model = False

    # Get the efficiency, if needed:
    if not bool(efficiency):
        print('No efficiency data given, skipping efficiency')
        eff = np.ones(nr_pin)
    else:
        eff = np.ones(nr_pin)
    # Build the weight matrix. We will use brute force, I am sure that there is
    # a tensor product implemented in python which does the job in a more
    # efficient way, bot for the moment, I will leave exactly as in the
    # original IDL routine
    res_matrix = np.zeros((nr_scint, np_scint, nr_pin, np_pin))
    print('Creating matrix')
    for ii in tqdm(range(nr_scint)):
        for jj in range(np_scint):
            for kk in range(nr_pin):
                for ll in range(np_pin):
                    if fcol[kk, ll] > 0:
                        if screw_model:
                            res_matrix[ii, jj, kk, ll] = \
                                eff[kk] * fcol[kk, ll] * dr_scint * dp_scint *\
                                1./(np.pi * sigmar[kk, ll] * sigmap[kk, ll]) *\
                                np.exp(-(rscint[ii] - centroidr[kk, ll])**2 /
                                       (2. * sigmar[kk, ll]**2)) * \
                                0.5 * (1 + erf(gamma[kk, ll] *
                                       (rscint[ii] - centroidr[kk, ll]) /
                                       sigmar[kk, ll] / np.sqrt(2.))) * \
                                np.exp(-(pscint[jj] - centroidp[kk, ll])**2 /
                                       (2. * sigmap[kk, ll]**2))
                        else:
                            res_matrix[ii, jj, kk, ll] = \
                                eff[kk] * fcol[kk, ll] * dr_scint * dp_scint *\
                                .5/(np.pi * sigmar[kk, ll] * sigmap[kk, ll]) *\
                                np.exp(-(rscint[ii] - centroidr[kk, ll])**2 /
                                       (2. * sigmar[kk, ll]**2)) * \
                                np.exp(-(pscint[jj] - centroidp[kk, ll])**2 /
                                       (2. * sigmap[kk, ll]**2))
                    else:
                        res_matrix[ii, jj, kk, ll] = 0.0
    res_matrix[np.isnan(res_matrix)] = 0.0
    return res_matrix


def plot_W(W4D, pr, pp, sr, sp, pp0=None, pr0=None, sp0=None, sr0=None,
           cmap=None, nlev=20):
    """
    Plot the weight function

    Jose Rueda Rueda: jrrueda@us.es

    @todo: add titles and print the used point

    @param W4D: 4-D weight function
    @param pr: array of gyroradius at the pinhole used to calculate W
    @param pp: array of pitches at the pinhole used to calculate W
    @param sr: array of gyroradius at the scintillator used to calculate W
    @param sp: array of pitches at the scintillator used to calculate W
    @param pp0: precise radius wanted at the pinhole to plot the scintillator W
    @param pr0: precise pitch wanted at the pinhole to plot the scintillator W
    @param sp0: precise radius wanted at the pinhole to plot the scintillator W
    @param sr0: precise pitch wanted at the pinhole to plot the scintillator W
    """
    # --- Color map
    if cmap is None:
        ccmap = ssplt.Gamma_II()
    # --- Potting of the scintillator weight
    # We will select a point of the pinhole and see how it seen in the
    # scintillator
    if (pp0 is not None) and (pr0 is not None):
        ip = np.argmin(abs(pp - pp0))
        ir = np.argmin(abs(pr - pr0))
        W = W4D[:, :, ir, ip]
        fig, ax = plt.subplots()
        a = ax.contourf(sp, sr, W, nlev, cmap=ccmap)
        plt.colorbar(a, ax=ax)
    if (sp0 is not None) and (sr0 is not None):
        ip = np.argmin(abs(pp - sp0))
        ir = np.argmin(abs(pr - sr0))
        W = W4D[ir, ip, :, :]
        fig, ax = plt.subplots()
        a = ax.contourf(pp, pr, W, nlev, cmap=ccmap)
        plt.colorbar(a, ax=ax)


# -----------------------------------------------------------------------------
# --- RUN FILDSIM
# -----------------------------------------------------------------------------
def write_namelist(nml, p=os.path.join(paths.FILDSIM, 'cfg_files')):
    """
    Write fortran namelist

    jose rueda: jrrueda@us.es

    just a wrapper for the f90nml file writter

    @param p: full path towards the desired file
    @param nml: namelist containing the desired fields.

    @return file: The path to the written file
    """
    file = os.path.join(p, nml['config']['runid'] + '.cfg')
    f90nml.write(nml, file)
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


def run_FILDSIM(namelist):
    """
    Execute a FILDSIM simulation

    @todo Include the capability of connecting to a external machine

    @param namelist: full path to the namelist
    """
    FILDSIM = os.path.join(paths.FILDSIM, 'bin', 'fildsim.exe')
    # namelist = ' ' + run_ID + '.cfg'
    os.system(FILDSIM + ' ' + namelist)


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
    if phi < 0:
        if theta < 0:
            name = machine +\
                "_map_{0:010.5f}_{1:010.5f}_strike_map.dat".format(p, t)
        else:
            name = machine +\
                "_map_{0:010.5f}_{1:09.5f}_strike_map.dat".format(p, t)
    else:
        if theta < 0:
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
# --- Energy definition FILDSIM
# -----------------------------------------------------------------------------
def get_energy(gyroradius, B: float, A: int = 2, Z: int = 1):
    """
    Relate the gyroradius with the associated energy (FILDSIM definition)

    @param gyroradius: Larmor radius as taken from FILD strike map [in cm]
    @param B: Magnetic field, [in T]
    @param A: Ion mass number
    @param Z: Ion charge [in e units]
    @return E: the energy [in eV]
    """
    m = ssp.mp * A  # Mass of the ion
    E = 0.5 * (gyroradius/100.0 * Z * B)**2 / m * ssp.c ** 2
    return E
