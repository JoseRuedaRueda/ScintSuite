"""@package LibFILDSIM Routines to iinteract with FILDSIM"""
import numpy as np
import math as ma
import LibParameters as ssp


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


def get_energy_FILD(gyroradius, B: float, A: int = 2, Z: int = 1):
    """
    Relate the gyroradius with the associated energy (FILDSIM definition)

    @param gyroradius: Particle Larmor radius as taken from FILD strike map (
    in cm)
    @param B: Magnetc field, in T
    @param A: Ion mass number
    @param Z: Ion charge (in e units)
    @return E: the energy in eV
    """
    m = ssp.mp * A  # Mass of the ion
    E = 0.5 * (gyroradius/100.0 * Z * B)**2 / m * ssp.c ** 2
    return E
