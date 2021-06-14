"""Module to remap the scintillator

It contains the routines to load and align the strike maps, as well as
perform the remapping. Contain the classes: Scintillator(), StrikeMap(),
CalibrationDataBase(), Calibration()
"""
# import time
import math
import datetime
import time
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interp
import Lib.LibPlotting as ssplt
import Lib.LibFILDSIM as ssFILDSIM
import Lib.LibUtilities as ssextra
from Lib.LibMachine import machine
import Lib.LibPaths as p
import Lib.LibIO as ssio
import Lib.LibData as ssdat
from tqdm import tqdm   # For waitbars
pa = p.Path(machine)
del p

try:
    import lmfit
except ImportError:
    warnings.warn('lmfit not found, you cannot calculate resolutions')


def transform_to_pixel(x, y, grid_param):
    """
    Transform from X,Y coordinates (scintillator) to pixels in the camera

    Jose Rueda Rueda: jrrueda@us.es

    @param x: Array of positions to be transformed, x coordinate
    @param y: Array of positions to be transformed, y coordinate
    @param grid_param: Object containing all the information for the
    transformation, see class CalParams()
    @return xpixel: x positions in pixels
    @return ypixel: y position in pixels
    @todo Include a model to take into account the distortion
    """
    alpha = grid_param.deg * np.pi / 180
    xpixel = (np.cos(alpha) * x - np.sin(alpha) * y) * grid_param.xscale + \
        grid_param.xshift
    ypixel = (np.sin(alpha) * x + np.cos(alpha) * y) * grid_param.yscale + \
        grid_param.yshift

    return xpixel, ypixel


# -----------------------------------------------------------------------------
# --- Align the scintillator
# -----------------------------------------------------------------------------
def get_points(fig, scintillator, plt_flag: bool = True, npoints: int = 3):
    """
    Get the points of the scintillator via ginput method

    Jose Rueda: jrrueda@us.es

    @param fig: Axis where the scintillator is drawn
    @param scintillator: Scintillator object
    @param plt_flag: flag to plot
    @param npoints: number of points in the selection
    @return index: array with the index of each point, to be located in the
    scintillator.coord_real
    """
    print('Select the points in the scintillator')
    points = fig.ginput(npoints)
    index = np.zeros(npoints, dtype=int)
    for i in range(npoints):
        relative_pos = scintillator.coord_real[:, 1:3] - points[i]
        # print(relative_pos)
        index[i] = int(
            np.argmin(relative_pos[:, 1] ** 2 + relative_pos[:, 0] ** 2))
        # print(index[i])
        if plt_flag:
            plt.plot(scintillator.coord_real[index[i], 1],
                     scintillator.coord_real[index[i], 2], 'o')
            plt.text(scintillator.coord_real[index[i], 1],
                     scintillator.coord_real[index[i], 2], str(i + 1))
    return index


def calculate_transformation_factors(scintillator, fig, plt_flag: bool = True):
    """
    Calculate the transformation factor to align the strike map

    Jose Rueda: jrrueda@us.es

    Calculate the transformation factor to align the strike map with the
    camera sensor. It ask for the user to select some points in the
    scintillator and to select the same points in the calibration file

    @param scintillator: Scintillator object
    @param fig: figure where the calibration image is drawn
    @param plt_flag: flag to plot the selected points to align the scintillator
    @return xmag: magnification factor in the x direction to align strike map
    @return mag: magnification factor in the y direction
    @return alpha: rotation angle to orientate the strike map
    @return offset: offset to place the strike map
    @todo check with a non inverted png if the minus I included to align the
    scintillator is needed or not
    """
    # Plot the scintillator
    fig_scint, ax_scint = plt.subplots()
    scintillator.plot_real(ax_scint)

    # Select the points in the scintillator
    index = get_points(fig_scint, scintillator, plt_flag)
    npoints = index.size
    print(npoints)
    # Select the corresponding points in the reference frame
    print('Select the points on the calibration frame, in the same order as '
          'before')
    points_frame = fig.ginput(npoints)
    # Define the vectors which will give us the reference
    v21_real = np.array((scintillator.coord_real[index[0], 1]
                         - scintillator.coord_real[index[1], 1],
                         scintillator.coord_real[index[0], 2]
                         - scintillator.coord_real[index[1], 2], 0))
    v23_real = np.array((scintillator.coord_real[index[2], 1]
                         - scintillator.coord_real[index[1], 1],
                         scintillator.coord_real[index[2], 2]
                         - scintillator.coord_real[index[1], 2], 0))

    v21_pix = np.array([points_frame[0][0], points_frame[0][1], 0]) - \
        np.array([points_frame[1][0], points_frame[1][1], 0])

    v23_pix = np.array([points_frame[2][0], points_frame[2][1], 0]) - \
        np.array([points_frame[1][0], points_frame[1][1], 0])

    # See if an inversion of one of the axis is needed or not.
    normal_real = np.cross(v21_real, v23_real)
    normal_pix = np.cross(v21_pix, v23_pix)

    # If the normal has opposite signs, an inversion must be done
    if normal_pix[2] * normal_real[2] < 0:
        sign = -1.0
    else:
        sign = 1.0
    # With this sign in mind, now we can proceed with the calculation of the
    # ratio and the gyration angle
    # Initialize the variables
    alpha = 0  # Rotation to be applied
    mag = 0  # Magnification
    offset = np.zeros(2)
    for i in range(npoints):
        # We will use the pair formed by each point and the following,
        # for the case of the last point in the list, just take the next one
        i2 = i + 1
        if i2 == npoints:
            i2 = 0
        # Distance in the real scintillator
        d_real = np.sqrt((scintillator.coord_real[index[i], 2]
                          - scintillator.coord_real[index[i2], 2]) ** 2
                         + (scintillator.coord_real[index[i], 1]
                            - scintillator.coord_real[index[i2], 1]) ** 2)
        # Distance in the sensor
        dummy = np.array(points_frame[i]) - np.array(points_frame[i2])
        d_pix = np.sqrt(dummy[1] ** 2 + dummy[0] ** 2)
        # Accumulate the magnification factor (we will normalise at the end)
        mag = mag + d_pix / d_real
        # Calculate the angles
        alpha_r = -  math.atan2(scintillator.coord_real[index[i], 2]
                                - scintillator.coord_real[index[i2], 2],
                                sign * scintillator.coord_real[index[i], 1]
                                - sign * scintillator.coord_real[index[i2], 1])
        # If alpha == 180, it can be also -180, atan2 fails here, check which
        # one is the case
        if int(alpha_r * 180 / np.pi) == 180:
            print('Correcting angle')
            if scintillator.coord_real[index[i2], 1] > scintillator.coord_real[
                                                        index[i], 1]:
                alpha_r = - alpha_r

        alpha_px = - math.atan2(dummy[1], dummy[0])
        alpha = alpha + (alpha_px - alpha_r)
        # Transform the coordinate to estimate the offset
        x_new = (scintillator.coord_real[index[i], 1]
                 * math.cos(alpha_px - alpha_r)
                 - scintillator.coord_real[index[i], 2]
                 * math.sin(alpha_px - alpha_r)) * d_pix / d_real * sign
        y_new = (scintillator.coord_real[index[i], 1]
                 * math.sin(alpha_px - alpha_r)
                 + scintillator.coord_real[index[i], 2]
                 * math.cos(alpha_px - alpha_r)) * d_pix / d_real
        offset = offset + np.array(points_frame[i]) - np.array((x_new, y_new))
        # print(alpha_px*180/np.pi, alpha_real*180/np.pi)
        # print((alpha_px-alpha_real)*180/np.pi)
    # Normalise magnification and angle
    mag = mag / npoints
    xmag = sign * mag
    alpha = alpha / npoints * 180 / np.pi
    offset = offset / npoints
    cal = CalParams()
    cal.xscale = xmag
    cal.yscale = mag
    cal.xshift = offset[0]
    cal.yshift = offset[1]
    cal.deg = alpha
    return cal


# -----------------------------------------------------------------------------
# --- Remap and profiles
# -----------------------------------------------------------------------------
def remap(smap, frame, x_edges=None, y_edges=None, mask=None, method='MC'):
    """
    Remap a frame

    Jose Rueda: jrrueda@us.es

    Edges are only needed if you select the centers method, if not, they will
    be 'inside' the transformation matrix already

    @param smap: StrikeMap() object with the strike map
    @param frame: the frame to be remapped
    @param x_edges: edges of the x coordinate, for FILD, pitch [º]
    @param y_edges: edges of the y coordinate, for FILD, gyroradius [cm]
    @param method: procedure for the remap
        - MC: Use the transformation matrix calculated with markers at the chip
        - centers: Consider just the center of each pixel (Old IDL method)
    """
    # --- 0: Check inputs
    if smap.grid_interp is None:
        print('Grid interpolation was not done, performing grid interpolation')
        smap.interp_grid(frame.shape)

    if method == 'MC':
        if mask is None:
            H = np.tensordot(smap.grid_interp['transformation_matrix'],
                             frame, 2)
        else:
            dummy = smap.grid_interp['transformation_matrix'].copy()
            dummy[..., mask] = 0
            dummy_frame = frame.copy()
            dummy_frame[mask] = 0
            H = np.tensordot(dummy, dummy_frame, 2)

    else:  # similar to old IDL implementation
        # --- 1: Information of the calibration
        if smap.diag == 'FILD':
            # Get the gyroradius and pitch of each pixel
            if mask is None:
                x = smap.grid_interp['pitch'].flatten()
                y = smap.grid_interp['gyroradius'].flatten()
            else:
                x = smap.grid_interp['pitch'][mask].flatten()
                y = smap.grid_interp['gyroradius'][mask].flatten()

        # --- 3: Remap (via histogram)
        if mask is None:
            z = frame.flatten()
        else:
            z = frame[mask].flatten()
        H, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges],
                                           weights=z)
        # Normalise H to counts per unit of each axis
        delta_x = xedges[1] - xedges[0]
        delta_y = yedges[1] - yedges[0]
        H /= delta_x * delta_y

    return H


def gyr_profile(remap_frame, pitch_centers, min_pitch: float,
                max_pitch: float, verbose: bool = False,
                name=None, gyr=None):
    """
    Cut the FILD signal to get a profile along gyroradius

    @author:  Jose Rueda: jrrueda@us.es

    @param    remap_frame: np.array with the remapped frame
    @param    pitch_centers: np array produced by the remap function
    @param    min_pitch: minimum pitch to include
    @param    max_pitch: Maximum pitch to include
    @param    verbose: if true, the actual pitch interval will be printed
    @param    name: if given, the profile will be exported in ASCII format
    @param    gyr: the gyroradius values, to export
    @return   profile:  the profile in gyroradius
    @raises   ExceptionName: exception if the desired pitch range is not in the
    frame
    """
    # See which cells do we need
    flags = (pitch_centers <= max_pitch) * (pitch_centers >= min_pitch)
    if np.sum(flags) == 0:
        raise Exception('No single cell satisfy the condition!')
    # The pitch centers is the centroid of the cell, but a cell include counts
    # which pitches are in [p0-dp,p0+dp], therefore, let give to the user these
    # to values
    used_pitches = pitch_centers[flags]
    delta = pitch_centers[1] - pitch_centers[0]
    min_used_pitch = used_pitches[0] - 0.5 * delta
    max_used_pitch = used_pitches[-1] + 0.5 * delta
    dummy = remap_frame[flags, :]
    profile = np.sum(dummy, axis=0)
    if verbose:
        print('The minimum pitch used is: ', min_used_pitch)
        print('The maximum pitch used is: ', max_used_pitch)
    if name is not None:
        if gyr is not None:
            date = datetime.datetime.now()
            line = 'Gyroradius profile: ' +\
                date.strftime("%d-%b-%Y (%H:%M:%S.%f)") +\
                '\n' +\
                'The minimum pitch used is: ' + str(min_used_pitch) +\
                '\n' +\
                'The maximum pitch used is: ' + str(max_used_pitch) +\
                '\n' +\
                'Gyroradius [cm]                     ' + \
                'Counts                        '
            length = len(gyr)
            np.savetxt(name, np.hstack((gyr.reshape(length, 1),
                       profile.reshape(length, 1))),
                       delimiter='   ,   ', header=line)
        else:
            raise Exception('You want to export but no gyr axis was given')
    return profile


def pitch_profile(remap_frame, gyr_centers, min_gyr: float,
                  max_gyr: float, verbose: bool = False,
                  name=None, pitch=None):
    """
    Cut the FILD signal to get a profile along pitch

    @author:  Jose Rueda: jrrueda@us.es

    @param    remap_frame: np.array with the remapped frame
    @type:    ndarray

    @param    gyr_centers: np array produced by the remap function
    @type:    ndarray

    @param    min_gyr: minimum pitch to include
    @type:    float

    @param    max_gyr: Maximum pitch to include
    @type:    float

    @param    verbose: if true, the actual pitch interval will be printed
    @type:    bool

    @param    name: Full path to the file to export the profile. if present,
    file willbe written

    @param    pitch: array of pitches used in the remapped, only used if the
    export option is activated

    @return   profile:  pitch profile of the signal

    @raises   ExceptionName: exception if the desired gyroradius range is not
    in the frame
    """
    # See which cells do we need
    flags = (gyr_centers <= max_gyr) * (gyr_centers >= min_gyr)
    if np.sum(flags) == 0:
        raise Exception('No single cell satisfy the condition!')
    # The r centers is the centroid of the cell, but a cell include counts
    # which radius are in [r0-dr,r0+dr], therefore, let give to the user these
    # to values
    used_gyr = gyr_centers[flags]
    delta = gyr_centers[1] - gyr_centers[0]
    min_used_gyr = used_gyr[0] - 0.5 * delta
    max_used_gyr = used_gyr[-1] + 0.5 * delta
    dummy = remap_frame[:, flags]
    profile = np.sum(dummy, axis=1)
    if verbose:
        print('The minimum gyroradius used is: ', min_used_gyr)
        print('The maximum gyroradius used is: ', max_used_gyr)

    if name is not None:
        if pitch is not None:
            date = datetime.datetime.now()
            line = '# Pitch profile: ' +\
                date.strftime("%d-%b-%Y (%H:%M:%S.%f)") +\
                '\n' +\
                'The minimum gyroradius used is: ' + str(min_used_gyr) +\
                '\n' +\
                'The maximum gyroradius used is: ' + str(max_used_gyr) +\
                '\n' +\
                'Pitch [º]                     ' + \
                'Counts                        '
            length = len(pitch)
            np.savetxt(name, np.hstack((pitch.reshape(length, 1),
                       profile.reshape(length, 1))),
                       delimiter='   ,   ', header=line)
        else:
            raise Exception('You want to export but no pitch was given')
    return profile


def estimate_effective_pixel_area(frame_shape, xscale: float, yscale: float,
                                  type: int = 0):
    """
    Estimate the effective area covered by a pixel

    Jose Rueda Rueda: jrrueda@us.es based on a routine of Joaquín Galdón

    If there is no distortion:
    Area_covered_by_1_pixel: A_omega=Area_scint/#pixels inside scintillator
    #pixels inside scint=L'x_scint*L'y_scint=Lx_scint*xscale*Ly_scint*yscale
    xscale and yscale are in units of : #pixels/cm
    So A_omega can be approximated by: A_omega=1/(xscale*yscale) [cm^2]

    @param frame_shape: shape of the frame
    @params yscale: the scale [#pixel/cm] of the calibration to align the map
    @params xscale: the scale [#pixel/cm] of the calibration to align the map
    @param type: 0, ignore distortion, 1 include distortion
    @return area: Matrix where each element is the area covered by that pixel
    @todo Include the model of distortion
    """
    # Initialise the matrix:
    area = np.zeros(frame_shape)

    if type == 0:
        area[:] = abs(1./(xscale*yscale)*1.e-4)  # 1e-4 to be in m^2

    return area


def remap_all_loaded_frames_FILD(video, calibration, shot, rmin: float = 1.0,
                                 rmax: float = 10.5, dr: float = 0.1,
                                 pmin: float = 15.0, pmax: float = 90.0,
                                 dp: float = 1.0, rprofmin: float = 1.5,
                                 rprofmax: float = 9.0, pprofmin: float = 20.0,
                                 pprofmax: float = 90.0, rfild: float = 2.186,
                                 zfild: float = 0.32, alpha: float = 0.0,
                                 beta: float = -12.0,
                                 fildsim_options: dict = {},
                                 method: int = 1,
                                 verbose: bool = False, mask=None,
                                 machine: str = 'AUG',
                                 decimals: int = 1,
                                 smap_folder: str = None,
                                 map=None,
                                 remap_method: str = 'centers',
                                 MC_number: int = 100):
    """
    Remap all loaded frames from a FILD video

    Jose Rueda Rueda: jrrueda@us.es
    @todo finish documentation of this function

    @param    video: Video object (see LibVideoFiles)
    @param    calibration: Calibation object (see Calibration class)
    @param    shot: shot number
    @param    rmin: minimum gyroradius to consider [cm]
    @param    rmax: maximum gyroradius to consider [cm]
    @param    dr: bin width in the radial direction
    @param    pmin: Minimum pitch to consider [º]
    @param    pmax: Maximum pitch to consider [º]
    @param    dp: bin width in pitch [º]
    @param    rprofmin: minimum gyrodarius to calculate pitch profiles [cm]
    @param    rprofmax: maximum gyroradius to calculate pitch profiles [cm]
    @param    pprofmin: minimum pitch for gyroradius profiles [º]
    @param    pprofmax: maximum pitch for gyroradius profiles [º]
    @param    rfild: Radial position of FILD [m]
    @param    zfild: height avobe the mid plane of FILD head [m]
    @param    alpha: Alpha orientation of FILD head [º]
    @param    beta: beta orientation of FILD head [º]
    @param    method: method to interpolate the strike maps, default 1: linear
    @param    decimals: Number of decimals to look for the strike map
    @param    smap_folder: folder where to look for strike maps, if none, the
    code will use the indicated by LibPaths
    @param    map: Strike map to be used, if none, we will look in the folder
    for the right strike map
    @param    mask: binary mask defining the region of the scintillator we want
    to map. If it is a string pointing to a file, the mask saved in that file
    will be loaded
    @param    remap_method: 'MC' or 'centers', the method to be used for the
    remapping of the frames (MC recomended for tomography, but it needs 3
    minutes per new strike map...)
    @param    number of MC markers for the MC remap

    @return   output: dictionary containing all the outputs:
        -# 'frames': remaped_frames [xaxis(pitch), yaxis(r), taxis]
        -# 'xaxis': pitch,
        -# 'yaxis': gyr,
        -# 'xlabel': 'Pitch', label to plot
        -# 'ylabel': '$r_l$', label to plot
        -# 'xunits': '{}^o', units of the pitch
        -# 'yunits': 'cm', units of the gyroradius
        -# 'sprofx': signal integrated in gyroradius vs time
        -# 'sprofy': signal_integrated in pitch vs time
        -# 'sprofxlabel': label for sprofx
        -# 'sprofylabel': label for sprofy
        -# 'bfield': Modulus of the field vs time [T]
        -# 'phi': phi, calculated phi angle, FILDSIM [deg]
        -# 'theta': theta, calculated theta angle FILDSIM [deg]
        -# 'theta_used': theta_used for the remap [deg]
        -# 'phi_used': phi_used for the remap [deg]
        -# 'tframes': time of the frames
        -# 'existing_smaps': array indicating which smaps where found in the
        database and which don't
    @return   opt: dictionary containing all the input parameters
    """
    # Check inputs strike map
    print('.-. . -- .- .--. .--. .. -. --.')
    if map is None:
        got_smap = False
    else:
        got_smap = True
        print('A StrikeMap was given, we will remap all frames with it')

    if smap_folder is None:
        smap_folder = pa.FILDStrikeMapsRemap

    if type(mask) is str:
        # the user gave us a saved mask, not the matrix, so load the matrix:
        file = ssio.check_open_file(mask)
        [mask] = ssio.read_variable_ncdf(file, ['mask'], human=True)
        # tranform to bool
        mask = mask.astype(np.bool)
    # Print  some info:
    if not got_smap:
        print('Looking for strikemaps in: ', smap_folder)
    # Check the tipe of remap
    if remap_method == 'centers':
        MC_number = 0  # to turn off the transformation matrix calculation
    # Get frame shape:
    nframes = len(video.exp_dat['nframes'])
    frame_shape = video.exp_dat['frames'].shape[0:2]

    # Get the time (to measure elapsed time)
    tic = time.time()

    # Get the magnetic field: In principle we should be able to do this in an
    # efficient way, but the AUG library to access magnetic field is kind of a
    # shit in python 3, so we need a work around
    if not got_smap:
        if machine == 'AUG':
            print('Opening shotfile from magnetic field')
            import map_equ as meq
            equ = meq.equ_map(shot, diag='EQH')
        br = np.zeros(nframes)
        bz = np.zeros(nframes)
        bt = np.zeros(nframes)
    b_field = np.zeros(nframes)
    # br, bz, bt, bp =\
    #     ssdat.get_mag_field(shot, rfild, zfild,
    #                         time=video.exp_dat['tframes'])
    # Get the modulus of the field
    # b_field = np.sqrt(br**2 + bz**2 + bt**2)
    # Initialise the variables:
    ngyr = int((rmax-rmin)/dr) + 1
    npit = int((pmax-pmin)/dp) + 1
    remaped_frames = np.zeros((npit, ngyr, nframes))
    signal_in_gyr = np.zeros((ngyr, nframes))
    signal_in_pit = np.zeros((npit, nframes))
    theta = np.zeros(nframes)
    phi = np.zeros(nframes)
    name_old = ' '
    name = ' '
    exist = np.zeros(nframes, np.bool)
    # --- Calculate the grid
    p_edges = pmin - dp/2 + np.arange(npit+1) * dp
    g_edges = rmin - dr/2 + np.arange(ngyr+1) * dr
    gyr = 0.5 * (g_edges[0:-1] + g_edges[1:])
    pitch = 0.5 * (p_edges[0:-1] + p_edges[1:])
    # --- Calculate the theta and phi angles
    if not got_smap:  # if no smap was given calculate the theta and phi
        print('Calculating theta and phi')
        for iframe in tqdm(range(nframes)):
            if machine == 'AUG':
                tframe = video.exp_dat['tframes'][iframe]
                br[iframe], bz[iframe], bt[iframe], bp =\
                    ssdat.get_mag_field(shot, rfild, zfild, time=tframe,
                                        equ=equ)
                b_field[iframe] = np.sqrt(br[iframe]**2 + bz[iframe]**2
                                          + bt[iframe]**2)
            phi[iframe], theta[iframe] = \
                ssFILDSIM.calculate_fild_orientation(br[iframe], bz[iframe],
                                                     bt[iframe], alpha, beta)
            name = ssFILDSIM.guess_strike_map_name_FILD(phi[iframe],
                                                        theta[iframe],
                                                        machine=machine,
                                                        decimals=decimals)
            # See if the strike map exist
            if os.path.isfile(os.path.join(smap_folder, name)):
                exist[iframe] = True
    else:
        exist = np.ones(nframes, np.bool)
    # See how many strike maps we need to calculate:
    if not got_smap:
        nnSmap = np.sum(~exist)  # Number of Smaps missing
        dummy = np.arange(nframes)     #
        existing_index = dummy[exist]  # Index of the maps we have
        non_existing_index = dummy[~exist]
        theta_used = np.round(theta, decimals=decimals)
        phi_used = np.round(phi, decimals=decimals)

        if nnSmap == 0:
            print('--. .-. . .- -')
            print('Ideal situation, not a single map needs to be calculated')
        elif nnSmap == nframes:
            print('Non a single strike map, full calculation needed')
        elif nnSmap != 0:
            print('We need to calculate, at most:', nnSmap, 'StrikeMaps')
            print('Write 1 to proceed, 0 to take the closer'
                  + '(in time) existing strikemap')
            x = int(input('Enter answer:'))
            if x == 0:
                print('We will not calculate new strike maps')
                print('Looking for the closer ones')
                for i in tqdm(range(len(non_existing_index))):
                    ii = non_existing_index[i]
                    icloser = ssextra.find_nearest_sorted(existing_index, ii)
                    theta_used[ii] = theta[icloser]
                    phi_used[ii] = phi[icloser]
    print('Remapping frames ...')
    for iframe in tqdm(range(nframes)):
        if not got_smap:
            name = ssFILDSIM.find_strike_map(rfild, zfild, phi_used[iframe],
                                             theta_used[iframe], smap_folder,
                                             machine=machine,
                                             FILDSIM_options=fildsim_options,
                                             decimals=decimals)
        # Only reload the strike map if it is needed
        if name != name_old:
            map = StrikeMap(0, os.path.join(smap_folder, name))
            map.calculate_pixel_coordinates(calibration)
            # print('Interpolating grid')
            map.interp_grid(frame_shape, plot=False, method=method,
                            MC_number=MC_number,
                            grid_params={'ymin': rmin, 'ymax': rmax,
                                         'dy': dr,
                                         'xmin': pmin, 'xmax': pmax,
                                         'dx': dp})
        name_old = name
        # remap the frames
        remaped_frames[:, :, iframe] = \
            remap(map, video.exp_dat['frames'][:, :, iframe], x_edges=p_edges,
                  y_edges=g_edges, mask=mask, method=remap_method)
        # Calculate the gyroradius and pitch profiles
        dummy = remaped_frames[:, :, iframe].squeeze()
        signal_in_gyr[:, iframe] = gyr_profile(dummy, pitch, pprofmin,
                                               pprofmax)
        signal_in_pit[:, iframe] = pitch_profile(dummy, gyr, rprofmin,
                                                 rprofmax)
    if verbose:
        toc = time.time()
        print('Whole time interval remapped in: ', toc-tic, ' s')
        print('Average time per frame: ', (toc-tic) / nframes, ' s')
    output = {'frames': remaped_frames, 'xaxis': pitch, 'yaxis': gyr,
              'xlabel': 'Pitch', 'ylabel': '$r_l$',
              'xunits': '$\\degree$', 'yunits': 'cm',
              'sprofx': signal_in_pit, 'sprofy': signal_in_gyr,
              'sprofxlabel': 'Signal integrated in r_l',
              'sprofylabel': 'Signal integrated in pitch',
              'bfield': b_field, 'phi': phi, 'theta': theta,
              'theta_used': theta_used, 'phi_used': phi_used,
              'tframes': video.exp_dat['tframes'],
              'existing_smaps': exist}
    opt = {'rmin': rmin, 'rmax': rmax, 'dr': dr, 'pmin': pmin, 'pmax': pmax,
           'dp': dp, 'rprofmin': rprofmin, 'rprofmax': rprofmax,
           'pprofmin': pprofmin, 'pprofmax': pprofmax, 'rfild': rfild,
           'zfild': zfild, 'alpha': alpha, 'beta': beta,
           'calibration': calibration, 'decimals': decimals,
           'smap_folder': smap_folder}
    return output, opt


# -----------------------------------------------------------------------------
# --- Fitting routines
# -----------------------------------------------------------------------------
def _fit_to_model_(data, bins=20, model='Gauss', normalize=True):
    """
    Make histogram of input data and fit to a model

    Jose Rueda: jrrueda@us.es

    @param bins: Can be the desired number of bins or the edges
    @param model: 'Gauss' Pure Gaussian, 'sGauss' Screw Gaussian
    """
    # --- Make the histogram of the data
    hist, edges = np.histogram(data, bins=bins)
    hist = hist.astype(np.float64)
    if normalize:
        hist /= hist.max()  # Normalise to  have the data between 0 and 1
    cent = 0.5 * (edges[1:] + edges[:-1])
    # --- Make the fit
    if model == 'Gauss':
        model = lmfit.models.GaussianModel()
        params = model.guess(hist, x=cent)
        result = model.fit(hist, params, x=cent)
        par = {'amplitude': result.params['amplitude'].value,
               'center': result.params['center'].value,
               'sigma': result.params['sigma'].value}
    elif model == 'sGauss':
        model = lmfit.models.SkewedGaussianModel()
        params = model.guess(hist, x=cent)
        result = model.fit(hist, params, x=cent)
        par = {'amplitude': result.params['amplitude'].value,
               'center': result.params['center'].value,
               'sigma': result.params['sigma'].value,
               'gamma': result.params['gamma'].value}

    return par, result


# -----------------------------------------------------------------------------
# --- Classes
# -----------------------------------------------------------------------------
class CalibrationDatabase:
    """Database of parameter to align the scintillator"""

    def __init__(self, filename: str, n_header: int = 5):
        """
        Read the calibration database, to align the strike maps

        See database page for a full documentation of each field

        @author Jose Rueda Rueda: jrrueda@us.es

        @param filename: Complete path to the file with the calibrations
        @param n_header: Number of header lines
        @return database: Dictionary containing the database information
        """
        ## Name of file with the data
        self.file = filename
        ## Header of the file
        self.header = []
        ## Dictionary with the data from the calibration. See @ref database
        ## for a full description of the meaning of each field
        self.data = {'ID': [], 'camera': [], 'shot1': [], 'shot2': [],
                     'xshift': [], 'yshift': [], 'xscale': [], 'yscale': [],
                     'deg': [], 'cal_type': [], 'diag_ID': []}
        # Open the file
        with open(filename) as f:
            for i in range(n_header):
                # Lines with description
                self.header = self.header + [f.readline()]
            # Database itself
            for line in f:
                dummy = line.split()
                self.data['ID'] = self.data['ID'] + [int(dummy[0])]
                self.data['camera'] = self.data['camera'] + [dummy[1]]
                self.data['shot1'] = self.data['shot1'] + [int(dummy[2])]
                self.data['shot2'] = self.data['shot2'] + [int(dummy[3])]
                self.data['xshift'] = self.data['xshift'] + [float(dummy[4])]
                self.data['yshift'] = self.data['yshift'] + [float(dummy[5])]
                self.data['xscale'] = self.data['xscale'] + [float(dummy[6])]
                self.data['yscale'] = self.data['yscale'] + [float(dummy[7])]
                self.data['deg'] = self.data['deg'] + [float(dummy[8])]
                self.data['cal_type'] = self.data['cal_type'] + [dummy[9]]
                self.data['diag_ID'] = self.data['diag_ID'] + [int(dummy[10])]

    def append_to_database(self, camera: str, shot1: int, shot2: int,
                           xshift: float, yshift: float, xscale: float,
                           yscale: float, deg: float, cal_type: str,
                           diag_ID: str):
        """
        Add a new entry to the database

        @param camera:
        @param shot1:
        @param shot2:
        @param xshift:
        @param yshift:
        @param xscale:
        @param yscale:
        @param deg:
        @param cal_type:
        @param diag_ID:
        """
        self.data['ID'] = self.data['ID'] + [self.data['ID'][-1] + 1]
        self.data['camera'] = self.data['camera'] + [camera]
        self.data['shot1'] = self.data['shot1'] + [shot1]
        self.data['shot2'] = self.data['shot2'] + [shot2]
        self.data['xshift'] = self.data['xshift'] + [xshift]
        self.data['yshift'] = self.data['yshift'] + [yshift]
        self.data['xscale'] = self.data['xscale'] + [xscale]
        self.data['yscale'] = self.data['yscale'] + [yscale]
        self.data['deg'] = self.data['deg'] + [deg]
        self.data['cal_type'] = self.data['cal_type'] + [cal_type]
        self.data['diag_ID'] = self.data['diag_ID'] + [diag_ID]

    def write_database_to_txt(self, file: str = ''):
        """
        Write database into a txt.

        If no name is given, the name of the loaded file will be used but a
        'new' will be added. Example: if the file from where the info has
        been loaded is 'calibration.txt' the new file would be
        'calibration_new.txt'. This is just to be save and avoid overwriting
        the original database.

        @param file: name of the file where to write the results
        @return : a file created with your information
        """
        if file == '':
            file = self.file[:-4] + '_new.txt'
        with open(file, 'w') as f:
            # Write the header
            for i in range(len(self.header)):
                f.write(self.header[i])
            # Write the database information
            for i in range(len(self.data['ID'])):
                line = str(self.data['ID'][i]) + ' ' + \
                       self.data['camera'][i] + ' ' + \
                       str(self.data['shot1'][i]) + ' ' + \
                       str(self.data['shot2'][i]) + ' ' + \
                       str(self.data['xshift'][i]) + ' ' + \
                       str(self.data['yshift'][i]) + ' ' + \
                       str(self.data['xscale'][i]) + ' ' + \
                       str(self.data['yscale'][i]) + ' ' + \
                       str(self.data['deg'][i]) + ' ' + \
                       self.data['cal_type'][i] + ' ' + \
                       str(self.data['diag_ID'][i]) + ' ' + '\n'
                f.write(line)
            print('File ' + file + ' writen')

    def get_calibration(self, shot, camera, cal_type, diag_ID):
        """
        Give the calibration parameter of a precise database entry

        @param shot: Shot number for which we want the calibration
        @param camera: Camera used
        @param cal_type: Type of calibration we want
        @param diag_ID: ID of the diagnostic we want
        @return cal: CalParams() object
        """
        flags = np.zeros(len(self.data['ID']))
        for i in range(len(self.data['ID'])):
            if (self.data['shot1'][i] <= shot) * \
                    (self.data['shot2'][i] >= shot) * \
                    (self.data['camera'][i] == camera) * \
                    (self.data['cal_type'][i] == cal_type) * \
                    (self.data['diag_ID'][i] == diag_ID):
                flags[i] = True

        n_true = sum(flags)

        if n_true == 0:
            raise Exception('No entry find in the database, revise database')
        elif n_true > 1:
            print('Several entries fulfill the condition')
            print('Possible entries:')
            print(self.data['ID'][flags])
            raise Exception()
        else:
            dummy = np.argmax(np.array(flags))
            cal = CalParams()
            cal.xscale = self.data['xscale'][dummy]
            cal.yscale = self.data['yscale'][dummy]
            cal.xshift = self.data['xshift'][dummy]
            cal.yshift = self.data['yshift'][dummy]
            cal.deg = self.data['deg'][dummy]

        return cal


class StrikeMap:
    """Class with the information of the strike map"""

    def __init__(self, flag=0, file: str = None, machine='AUG', theta=None,
                 phi=None, decimals=1):
        """
        Initialise the class

        Thera are 2 ways of selecting the smap: give the full path to the file,
        or give the theta and phi angles and the machine, so the strike map
        will be selected from the remap database

        @param flag: 0  means FILD, 1 means INPA, 2 means iHIBP (you can also
        write directly 'FILD', 'INPA', 'iHIBP')
        @param file: Full path to file with the strike map
        @param machine: machine, to look in the datbase
        @param theta: theta angle  (see FILDSIM doc)
        @param phi: phi angle (see FILDSIM doc)
        @param decimals: decimals to look in the database

        Notes: machine, theta and phi options introduced in version 0.4.14
        """
        ## Associated diagnostic
        if flag == 0 or flag == 'FILD':
            self.diag = 'FILD'
        elif flag == 2 or flag == 'iHIBP':
            self.diag = 'iHIBP'
        else:
            print('Flag: ', flag)
            raise Exception('Diagnostic not implemented')
        ## X-position, in pixels, of the strike map (common)
        self.xpixel = None
        ## Y-Position, in pixels, of the strike map (common)
        self.ypixel = None
        ## file
        if file is not None:
            self.file = file

        if flag == 0 or flag == 'FILD':
            # Read the file
            if file is None:
                smap_folder = pa.FILDStrikeMapsRemap
                dumm = ssFILDSIM.guess_strike_map_name_FILD(phi,
                                                            theta,
                                                            machine=machine,
                                                            decimals=decimals)
                file = os.path.join(smap_folder, dumm)
                self.file = file
            if not os.path.isfile(file):
                print('Strike map no fpun in the database')
            dummy = np.loadtxt(file, skiprows=3)
            # See which rows has collimator factor larger than zero (ie see for
            # which combination of energy and pitch some markers has arrived)
            ind = dummy[:, 7] > 0
            # Initialise the class
            ## Gyroradius of map points
            self.gyroradius = dummy[ind, 0]
            ## Simulated gyroradius (unique points of self.gyroradius)
            self.unique_gyroradius = np.unique(self.gyroradius)
            ## Energy of map points
            self.energy = None
            ## Pitch of map points
            self.pitch = dummy[ind, 1]
            ## Simulated pitches (unique points of self.pitch)
            self.unique_pitch = np.unique(self.pitch)
            ## x coordinates of map points (common)
            self.x = dummy[ind, 2]
            ## y coordinates of map points (common)
            self.y = dummy[ind, 3]
            ## z coordinates of map points (common)
            self.z = dummy[ind, 4]
            ## Average initial gyrophase of map markers
            self.avg_ini_gyrophase = dummy[ind, 5]
            ## Number of markers striking in this area
            self.n_strike_points = dummy[ind, 6]
            ## Collimator factor as defined in FILDSIM
            self.collimator_factor = dummy[ind, 7]
            ## Average incident angle of the FILDSIM markers
            self.avg_incident_angle = dummy[ind, 8]
            ## Translate from pixels in the camera to velocity space
            self.grid_interp = None
            ## Strike points used to calculate the map
            self.strike_points = None
            ## Resolution of FILD for each strike point
            self.resolution = None
            ## Interpolators (gyr, pitch)-> sigma_r, sigma_p, and so on
            self.intepolators = None
            ## Colimator facror as a matrix
            # This simplify a lot W calculation and forward modelling:
            self.ngyr = len(self.unique_gyroradius)
            self.npitch = len(self.unique_pitch)
            self.collimator_factor_matrix = np.zeros((self.ngyr, self.npitch))
            for ir in range(self.ngyr):
                for ip in range(self.npitch):
                    # By definition, flags can only have one True
                    flags = (self.gyroradius == self.unique_gyroradius[ir]) \
                        * (self.pitch == self.unique_pitch[ip])
                    if np.sum(flags) > 0:
                        self.collimator_factor_matrix[ir, ip] = \
                            self.collimator_factor[flags]

    def plot_real(self, ax=None,
                  marker_params: dict = {}, line_params: dict = {},
                  labels: bool = False,
                  rotation_for_gyr_label: float = 90.0,
                  rotation_for_pitch_label: float = 30.0):
        """
        Plot the strike map (x,y = dimensions in the scintillator)

        Jose Rueda: jrrueda@us.es

        @param ax: Axes where to plot
        @param markers_params: parameters for plt.plot() to plot the markers
        @param line_params: parameters for plt.plot() to plot the markers
        @param labels: flag to add the labes (gyroradius, pitch) on the plot
        """
        # Default plot parameters:
        marker_options = {
            'markersize': 6,
            'fillstyle': 'none',
            'color': 'w',
            'marker': 'o',
            'linestyle': 'none'
        }
        marker_options.update(marker_params)
        line_options = {
            'color': 'w',
            'marker': ''
        }
        line_options.update(line_params)

        if ax is None:
            fig, ax = plt.subplots()

        if self.diag == 'FILD':
            # Draw the lines of constant gyroradius (energy). These are the
            # 'horizontal' lines]
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_options)

                if (i % 2 == 0):  # add gyro radius labels
                    ax.text((self.y[flags])[0]-0.2,
                            (self.z[flags])[0], f'{float(uniq[i]):g}',
                            horizontalalignment='right',
                            verticalalignment='center')

            ax.annotate('Gyroradius (cm)',
                        xy=(min(self.y) - 0.5,
                            (max(self.z) - min(self.z))/2 + min(self.z)),
                        rotation=rotation_for_gyr_label,
                        horizontalalignment='center',
                        verticalalignment='center')

            # Draw the lines of constant pitch. 'Vertical' lines
            uniq = np.unique(self.pitch)
            n = len(uniq)
            for i in range(n):
                flags = self.pitch == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_options)

                ax.text((self.y[flags])[-1],
                        (self.z[flags])[-1] - 0.1,
                        f'{float(uniq[i]):g}',
                        horizontalalignment='center',
                        verticalalignment='top')

            ax.annotate('Pitch [$\\degree$])',
                        xy=((max(self.y) - min(self.y))/2 + min(self.y),
                            min(self.z) - 0.1),
                        rotation=rotation_for_pitch_label,
                        horizontalalignment='center',
                        verticalalignment='center')
        else:
            raise Exception('Diagnostic not implemented')

        # Plot some markers in the grid position
        ax.plot(self.y, self.z, **marker_options)
        return

    def plot_pix(self, ax=None, marker_params: dict = {},
                 line_params: dict = {}):
        """
        Plot the strike map (x,y = pixels on the camera)

        Jose Rueda: jrrueda@us.es

        @param ax: Axes where to plot
        @param marker_params: parameters for the centroid plotting
        @param line_params: parameters for the lines plotting
        @return: Strike maps over-plotted in the axis
        """
        # Default plot parameters:
        marker_options = {
            'markersize': 6,
            'fillstyle': 'none',
            'color': 'w',
            'marker': 'o',
            'linestyle': 'none'
        }
        marker_options.update(marker_params)
        line_options = {
            'color': 'w',
            'marker': ''
        }
        line_options.update(line_params)

        if ax is None:
            fig, ax = plt.subplots()

        # Draw the lines of constant gyroradius, energy, or rho (depending on
        # the particular diagnostic) [These are the 'horizontal' lines]
        if self.diag == 'FILD':
            # Lines of constant gyroradius
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_options)
            # Lines of constant pitch
            uniq = np.unique(self.pitch)
            n = len(uniq)
            for i in range(n):
                flags = self.pitch == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_options)
        else:
            raise Exception('Not implemented diagnostic')

        # Plot some markers in the grid position
        ## @todo include labels energy/pitch in the plot
        ax.plot(self.xpixel, self.ypixel, **marker_options)

    def calculate_pixel_coordinates(self, calib):
        """
        Transform the real coordinates of the map into pixels

        Jose Rueda Rueda: jrrueda@us.es

        @param calib: a CalParams() object with the calibration info
        """
        self.xpixel, self.ypixel = transform_to_pixel(self.y, self.z, calib)

    def interp_grid(self, frame_shape, method=2, plot=False, verbose=False,
                    grid_params: dict = {}, MC_number: int = 100):
        """
        Interpolate grid values on the frames

        @param smap: StrikeMap() object
        @param frame_shape: Size of the frame used for the calibration (in px)
        @param method: method to calculate the interpolation:
            - 1: griddata linear (you can also write 'linear')
            - 2: griddata cubic  (you can also write 'cubic')
        @param plot: flag to perform a quick plot to see the interpolation
        @param verbose: flag to print some info along the way
        @param grid_params: grid options for the transformationn matrix grid
        @param MC_number: Number of MC markers for the transformation matrix,
        if this number < 0, the transformation matrix will not be calculated
        """
        # --- 0: Check inputs
        if self.xpixel is None:
            raise Exception('Transform to pixel the strike map before')
        # Default grid options
        grid_options = {
            'ymin': 1.2,
            'ymax': 10.5,
            'dy': 0.1,
            'xmin': 20.0,
            'xmax': 90.0,
            'dx': 1.0
        }
        grid_options.update(grid_params)
        # --- 1: Create grid for the interpolation
        # Note, it seems transposed, but the reason is that the calibration
        # paramters were adjusted with the frame transposed (to agree with old
        # IDL implementation) therefore we have to transpose a bit almost
        # everything. Sorry for the headache
        grid_x, grid_y = np.mgrid[0:frame_shape[1], 0:frame_shape[0]]
        # --- 2: Interpolate the grid
        # Prepare the grid for te griddata method:
        dummy = np.column_stack((self.xpixel, self.ypixel))
        # Prepare the options and interpolators for later
        if method == 1 or method == 'linear':
            met = 'linear'
            interpolator = scipy_interp.LinearNDInterpolator
        elif method == 2 or method == 'cubic':
            met = 'cubic'
            interpolator = scipy_interp.CloughTocher2DInterpolator
        else:
            raise Exception('Not recognized interpolation method')
        if verbose:
            print('Using %s interpolation of the grid' % met)
        if self.diag == 'FILD':
            # Initialise the structure
            self.grid_interp = {
                'gyroradius': None,
                'pitch': None,
                'collimator_factor': None,
                'interpolators': {
                    'gyroradius': None,
                    'pitch': None,
                    'collimator_factor': None
                },
                'transformation_matrix': None
            }
            # Get gyroradius values of each pixel
            dummy2 = scipy_interp.griddata(dummy, self.gyroradius,
                                           (grid_x, grid_y), method=met,
                                           fill_value=1000)
            self.grid_interp['gyroradius'] = dummy2.copy().T
            # Get pitch values of each pixel
            dummy2 = scipy_interp.griddata(dummy, self.pitch, (grid_x, grid_y),
                                           method=met, fill_value=1000)
            self.grid_interp['pitch'] = dummy2.copy().T
            # Get collimator factor
            dummy2 = scipy_interp.griddata(dummy, self.collimator_factor,
                                           (grid_x, grid_y), method=met,
                                           fill_value=1000)
            self.grid_interp['collimator_factor'] = dummy2.copy().T
            # Calculate the interpolator
            grid = list(zip(self.xpixel, self.ypixel))
            self.grid_interp['interpolators']['gyroradius'] = \
                interpolator(grid, self.gyroradius, fill_value=1000)
            self.grid_interp['interpolators']['pitch'] = \
                interpolator(grid, self.pitch, fill_value=1000)
            self.grid_interp['interpolators']['collimator_factor'] = \
                interpolator(grid, self.collimator_factor, fill_value=1000)
            # --- Prepare the transformation matrix
            # Initialise the random number generator
            rand = np.random.default_rng()
            generator = rand.uniform
            # Prepare the edges for the r, pitch histogram
            n_gyr = int((grid_options['ymax'] - grid_options['ymin'])
                        / grid_options['dy']) + 1
            n_pitch = int((grid_options['xmax'] - grid_options['xmin'])
                          / grid_options['dx']) + 1
            pitch_edges = grid_options['xmin'] - grid_options['dx']/2 \
                + np.arange(n_pitch+1) * grid_options['dx']
            gyr_edges = grid_options['ymin'] - grid_options['dy']/2 \
                + np.arange(n_gyr+1) * grid_options['dy']
            # Initialise the transformation matrix
            transform = np.zeros((n_pitch, n_gyr,
                                  frame_shape[0], frame_shape[1]))
            # Calculate the transformation matrix
            if MC_number > 0:
                print('Calculating transformation matrix')
                for i in tqdm(range(frame_shape[0])):
                    for j in range(frame_shape[1]):
                        # Generate markers coordinates in the chip, note the
                        # first dimmension of the frame is y-pixel
                        # (IDL heritage)
                        x_markers = j + generator(size=MC_number)
                        y_markers = i + generator(size=MC_number)
                        # Calculate the r-pitch coordinates
                        r_markers = self.grid_interp['interpolators']\
                            ['gyroradius'](x_markers, y_markers)
                        p_markers = self.grid_interp['interpolators']\
                            ['pitch'](x_markers, y_markers)
                        # make the histogram in the r-pitch space
                        H, xedges, yedges = \
                            np.histogram2d(p_markers, r_markers,
                                           bins=[pitch_edges, gyr_edges])
                        transform[:, :, i, j] = H.copy()
                # Normalise the transformation matrix
                transform /= MC_number
                transform /= (grid_options['dx'] * grid_options['dy'])
                # This last normalization will be removed once we include the
                # jacobian somehow
                self.grid_interp['transformation_matrix'] = transform

        # --- Plot
        if plot:
            if self.diag == 'FILD':
                fig, axes = plt.subplots(2, 2)
                # Plot the scintillator grid
                self.plot_pix(axes[0, 0], line_params={'color': 'k'})
                # Plot the interpolated gyroradius
                c1 = axes[0, 1].imshow(self.grid_interp['gyroradius'],
                                       cmap=ssplt.Gamma_II(),
                                       vmin=0, vmax=10, origin='lower')
                fig.colorbar(c1, ax=axes[0, 1], shrink=0.9)
                # Plot the interpolated pitch
                c2 = axes[1, 0].imshow(self.grid_interp['pitch'],
                                       cmap=ssplt.Gamma_II(),
                                       vmin=0, vmax=90, origin='lower')
                fig.colorbar(c2, ax=axes[1, 0], shrink=0.9)
                # Plot the interpolated collimator factor
                c3 = axes[1, 1].imshow(self.grid_interp['collimator_factor'],
                                       cmap=ssplt.Gamma_II(),
                                       vmin=0, vmax=50, origin='lower')
                fig.colorbar(c3, ax=axes[1, 1], shrink=0.9)

    def get_energy(self, B0: float, Z: int = 1, A: int = 2):
        """Get the energy associated with each gyroradius

        Jose Rueda: jrrueda@us.es

        @param self:
        @param B0: Magnetic field [in T]
        @param Z: the charge [in e units]
        @param A: the mass number
        """
        if self.diag == 'FILD':
            self.energy = ssFILDSIM.get_energy(self.gyroradius, B0, A=A, Z=Z)
        return

    def load_strike_points(self, file=None, verbose: bool = True):
        """
        Load the strike points used to calculate the map

        Jose Rueda: ruejo@ipp.mpg.de

        @param file: File to be loaded. It should contain the strike points in
        FILDSIM format (if we are loading FILD). If none, name will be deduced
        from the self.file variable, so the strike points are supposed to be in
        the same folder than the strike map
        """
        if file is None:
            file = self.file[:-14] + 'strike_points.dat'
        if verbose:
            print('Reading strike points: ', file)
        if self.diag == 'FILD':
            self.strike_points = {}
            # Load all the data
            self.strike_points['Data'] = np.loadtxt(file, skiprows=3)
            # Check with version of FILDSIM was used
            if len(self.strike_points['Data'][0, :]) == 9:
                print('Old FILDSIM format, initial position NOT included')
                old = True
            elif len(self.strike_points['Data'][0, :]) == 12:
                print('New FILDSIM format, initial position included')
                old = False
            else:
                raise Exception('Error loading file, not recognised columns')
            # Write some help
            self.strike_points['help'] =\
                ['0: Gyroradius (cm)',
                 '1: Pitch Angle (deg)',
                 '2: Initial Gyrophase',
                 '3-5: X (cm)  Y (cm)  Z (cm)',
                 '6: Remapped gyro',
                 '7: Remapped pitch',
                 '8: Incidence angle']
            if not old:
                self.strike_points['help'].append('9-11: Xi (cm)  Yi (cm)'
                                                  + '  Zi (cm)')
            # Get the 'pinhole' gyr and pitch values:
            self.strike_points['pitch'] = \
                np.unique(self.strike_points['Data'][:, 1])
            self.strike_points['gyroradius'] = \
                np.unique(self.strike_points['Data'][:, 0])
        return

    def plot_strike_points(self, ax=None, plt_param={}):
        """
        Scatter plot of the strik points

        Note, no weighting is done, just a scatter plot, this is not a
        sofisticated ready to print figure maker but just a quick plot to see
        what is going on

        @param ax: axes where to plot, if not given, new figure will pop up
        @param plt_param: options for the matplotlib scatter function
        """
        # Open the figure if needed:
        if ax is None:
            fig, ax = plt.subplots()
        # plot
        if self.diag == 'FILD':
            ax.scatter(self.strike_points['Data'][:, 4],
                       self.strike_points['Data'][:, 5], ** plt_param)

    def calculate_resolutions(self, diag_params: dict = {},
                              min_statistics: int = 100,
                              adaptative: bool = True):
        """
        Calculate the resolution associated with each point of the map

        Jose Rueda Rueda: jrrueda@us.es

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch
        @param min_statistics: Minimum number of points for a given r p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_statistics: Minimum number of counts to perform the fit
        @param adaptative: If true, the bin width will be adapted such that the
        number of bins in a sigma of the distribution is 4. If this is the
        case, dpitch, dgyr, will no longer have an impact
        """
        if self.strike_points is None:
            raise Exception('You should load the strike points first!!')
        if self.diag == 'FILD':
            # --- Prepare options:
            diag_options = {
                'dpitch': 1.0,
                'dgyr': 0.1,
                'p_method': 'Gauss',
                'g_method': 'sGauss'
            }
            diag_options.update(diag_params)
            dpitch = diag_options['dpitch']
            dgyr = diag_options['dgyr']
            p_method = diag_options['p_method']
            g_method = diag_options['g_method']
            npitch = self.strike_points['pitch'].size
            nr = self.strike_points['gyroradius'].size
            # --- Pre-allocate variables
            npoints = np.zeros((nr, npitch))
            parameters_pitch = {'amplitude': np.zeros((nr, npitch)),
                                'center': np.zeros((nr, npitch)),
                                'sigma': np.zeros((nr, npitch)),
                                'gamma': np.zeros((nr, npitch))}
            parameters_gyr = {'amplitude': np.zeros((nr, npitch)),
                              'center': np.zeros((nr, npitch)),
                              'sigma': np.zeros((nr, npitch)),
                              'gamma': np.zeros((nr, npitch))}
            fitg = []
            fitp = []
            gyr_array = []
            pitch_array = []
            print('Calculating FILD resolutions')
            for ir in tqdm(range(nr)):
                for ip in range(npitch):
                    # --- Select the data
                    data = self.strike_points['Data'][
                        (self.strike_points['Data'][:, 0] ==
                         self.strike_points['gyroradius'][ir]) *
                        (self.strike_points['Data'][:, 1] ==
                         self.strike_points['pitch'][ip]), :]
                    npoints[ir, ip] = len(data[:, 0])

                    # --- See if there is enough points:
                    if npoints[ir, ip] < min_statistics:
                        parameters_gyr['amplitude'][ir, ip] = np.nan
                        parameters_gyr['center'][ir, ip] = np.nan
                        parameters_gyr['sigma'][ir, ip] = np.nan
                        parameters_gyr['gamma'][ir, ip] = np.nan

                        parameters_pitch['amplitude'][ir, ip] = np.nan
                        parameters_pitch['center'][ir, ip] = np.nan
                        parameters_pitch['sigma'][ir, ip] = np.nan
                        parameters_pitch['gamma'][ir, ip] = np.nan
                    else:  # If we have enough points, make the fit
                        # Prepare the bin edges according to the desired width
                        edges_pitch = \
                            np.arange(start=data[:, 7].min() - dpitch,
                                      stop=data[:, 7].max() + dpitch,
                                      step=dpitch)
                        edges_gyr = \
                            np.arange(start=data[:, 6].min() - dgyr,
                                      stop=data[:, 6].max() + dgyr,
                                      step=dgyr)
                        # --- Reduce (if needed) the bin width, we will set the
                        # bin width as 1/4 of the std, to ensure a good fitting
                        if adaptative:
                            n_bins_in_sigma = 4
                            sigma_r = np.std(data[:, 6])
                            new_dgyr = sigma_r / n_bins_in_sigma
                            edges_gyr = \
                                np.arange(start=data[:, 6].min() - new_dgyr,
                                          stop=data[:, 6].max() + new_dgyr,
                                          step=new_dgyr)
                            sigma_p = np.std(data[:, 7])
                            new_dpitch = sigma_p / n_bins_in_sigma
                            edges_pitch = \
                                np.arange(start=data[:, 7].min() - dpitch,
                                          stop=data[:, 7].max() + dpitch,
                                          step=new_dpitch)
                        # --- Proceed to fit
                        par_p, resultp = _fit_to_model_(data[:, 7],
                                                        bins=edges_pitch,
                                                        model=p_method)
                        par_g, resultg = _fit_to_model_(data[:, 6],
                                                        bins=edges_gyr,
                                                        model=g_method)
                        fitp.append(resultp)
                        fitg.append(resultg)
                        gyr_array.append(self.strike_points['gyroradius'][ir])
                        pitch_array.append(self.strike_points['pitch'][ip])
                        # --- Save the data in the matrices:
                        # pitch parameters:
                        parameters_pitch['amplitude'][ir, ip] = \
                            par_p['amplitude']
                        parameters_pitch['center'][ir, ip] = par_p['center']
                        parameters_pitch['sigma'][ir, ip] = par_p['sigma']
                        if p_method == 'Gauss':
                            parameters_pitch['gamma'][ir, ip] = np.nan
                        elif p_method == 'sGauss':
                            parameters_pitch['gamma'][ir, ip] = par_p['gamma']
                        # gyroradius parameters:
                        parameters_gyr['amplitude'][ir, ip] = \
                            par_g['amplitude']
                        parameters_gyr['center'][ir, ip] = par_g['center']
                        parameters_gyr['sigma'][ir, ip] = par_g['sigma']
                        if g_method == 'Gauss':
                            parameters_gyr['gamma'][ir, ip] = np.nan
                        elif g_method == 'sGauss':
                            parameters_gyr['gamma'][ir, ip] = par_g['gamma']

            self.resolution = {'Gyroradius': parameters_gyr,
                               'Pitch': parameters_pitch,
                               'nmarkers': npoints,
                               'fits': {
                                    'Gyroradius': fitg,
                                    'Pitch': fitp,
                                    'FILDSIM_gyroradius': np.array(gyr_array),
                                    'FILDSIM_pitch': np.array(pitch_array),
                               },
                               'gyroradius_model': g_method,
                               'pitch_model': p_method}
            # --- Prepare the interpolators:
            self.calculate_interpolators()
        return

    def calculate_interpolators(self):
        """
        Calculate the interpolators which relates gyr, pitch with the
        resolution parameters

        Jose Rueda: jrrueda@us.es
        """
        if self.diag == 'FILD':
            # --- Prepare the interpolators:
            # Prepare grid
            xx, yy = np.meshgrid(self.strike_points['gyroradius'],
                                 self.strike_points['pitch'])
            xxx = xx.flatten()
            yyy = yy.flatten()
            self.interpolators = {'pitch': {}, 'gyroradius': {}}
            for i in self.resolution['Gyroradius'].keys():
                dummy = self.resolution['Gyroradius'][i].T
                dummy = dummy.flatten()
                flags = np.isnan(dummy)
                x1 = xxx[~flags]
                y1 = yyy[~flags]
                z1 = dummy[~flags]
                if np.sum(~flags) > 4:
                    self.interpolators['gyroradius'][i] = \
                        scipy_interp.LinearNDInterpolator(
                            np.vstack((x1, y1)).T,
                            z1)
            for i in self.resolution['Pitch'].keys():
                dummy = self.resolution['Pitch'][i].T
                dummy = dummy.flatten()
                flags = np.isnan(dummy)
                x1 = xxx[~flags]
                y1 = yyy[~flags]
                z1 = dummy[~flags]
                if np.sum(~flags) > 4:
                    self.interpolators['pitch'][i] = \
                        scipy_interp.LinearNDInterpolator(
                            np.vstack((x1, y1)).T,
                            z1)
            # Collimator factor
            dummy = self.collimator_factor_matrix.T
            dummy = dummy.flatten()
            flags = np.isnan(dummy)
            x1 = xxx[~flags]
            y1 = yyy[~flags]
            z1 = dummy[~flags]
            if np.sum(~flags) > 4:
                self.interpolators['collimator_factor'] = \
                    scipy_interp.LinearNDInterpolator(np.vstack((x1, y1)).T,
                                                      z1)
            # positions:
            YMATRIX = np.zeros((self.npitch, self.ngyr))
            ZMATRIX = np.zeros((self.npitch, self.ngyr))
            for ir in range(self.ngyr):
                for ip in range(self.npitch):
                    flags = (self.gyroradius == self.unique_gyroradius[ir]) \
                        * (self.pitch == self.unique_pitch[ip])
                    if np.sum(flags) > 0:
                        # By definition, flags can only have one True
                        # yes, x is smap.y... FILDSIM notation
                        YMATRIX[ip, ir] = self.y[flags]
                        ZMATRIX[ip, ir] = self.z[flags]
            self.interpolators['x'] = \
                scipy_interp.LinearNDInterpolator(np.vstack((xxx.flatten(),
                                                             yyy.flatten())).T,
                                                  YMATRIX.flatten())
            self.interpolators['y'] = \
                scipy_interp.LinearNDInterpolator(np.vstack((xxx.flatten(),
                                                             yyy.flatten())).T,
                                                  ZMATRIX.flatten())
        return

    def plot_resolutions(self, ax_param: dict = {}, cMap=None, nlev: int = 20):
        """
        Plot the resolutions

        Jose Rueda: jrrueda@us.es

        @todo: Implement label size in colorbar

        @param ax_param: parameters for the axis beauty function. Note, labels
        of the color axis are hard-cored, if you want custom axis labels you
        would need to draw the plot on your own
        @param cMap: is None, Gamma_II will be used
        @param nlev: number of levels for the contour
        """
        # --- Initialise the settings:
        if cMap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = cMap
        ax_options = {
            'xlabel': '$\\lambda [\\degree]$',
            'ylabel': '$r_l [cm]$'
        }
        ax_options.update(ax_param)

        # --- Open the figure and prepare the map:
        fig, ax = plt.subplots(1, 2, figsize=(12, 10),
                               facecolor='w', edgecolor='k')

        if self.diag == 'FILD':
            # Plot the gyroradius resolution
            a1 = ax[0].contourf(self.strike_points['pitch'],
                                self.strike_points['gyroradius'],
                                self.resolution['Gyroradius']['sigma'],
                                levels=nlev, cmap=cmap)
            fig.colorbar(a1, ax=ax[0], label='$\\sigma_r [cm]$')
            ax[0] = ssplt.axis_beauty(ax[0], ax_param)
            # plot the pitch resolution
            a = ax[1].contourf(self.strike_points['pitch'],
                               self.strike_points['gyroradius'],
                               self.resolution['Pitch']['sigma'],
                               levels=nlev, cmap=cmap)
            fig.colorbar(a, ax=ax[1], label='$\\sigma_\\lambda$')
            ax[1] = ssplt.axis_beauty(ax[1], ax_options)
            plt.tight_layout()
            return

    def plot_collimator_factor(self, ax_param: dict = {}, cMap=None,
                               nlev: int = 20):
        """
        Plot the collimator factor

        Jose Rueda: jrrueda@us.es

        @todo: Implement label size in colorbar

        @param ax_param: parameters for the axis beauty function. Note, labels
        of the color axis are hard-cored, if you want custom axis labels you
        would need to draw the plot on your own
        @param cMap: is None, Gamma_II will be used
        @param nlev: number of levels for the contour
        """
        # --- Initialise the settings:
        if cMap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = cMap
        ax_options = {
            'xlabel': '$\\lambda [\\degree]$',
            'ylabel': '$r_l [cm]$'
        }
        ax_options.update(ax_param)

        # --- Open the figure and prepare the map:
        fig, ax = plt.subplots(1, 1, figsize=(6, 10),
                               facecolor='w', edgecolor='k')

        if self.diag == 'FILD':
            # Plot the gyroradius resolution
            a1 = ax.contourf(self.strike_points['pitch'],
                             self.strike_points['gyroradius'],
                             self.collimator_factor_matrix,
                             levels=nlev, cmap=cmap)
            fig.colorbar(a1, ax=ax, label='Collimating factor')
            ax = ssplt.axis_beauty(ax, ax_options)

            plt.tight_layout()
        return

    def sanity_check_resolutions(self):
        """
        Plot basic quantities of the resolution calculation as a test

        Jose Rueda: jrrueda@us.es

        Designed to quickly see some figures of merit of the resolution
        calculation, ie, compare the centroids of the fits with the actual
        values the particles were iniciated in FILDSIM
        """
        if self.diag == 'FILD':
            axis_param = {'grid': 'both', 'ratio': 'equal'}
            # Centroids comparison:
            cen_g = []
            cen_p = []
            fild_g = []
            # Arange centroids by pitch (gyroradius)
            for p in np.unique(self.resolution['fits']['FILDSIM_pitch']):
                dummy = []
                dummy_FILDSIM = []
                print(p)
                nfits = len(self.resolution['fits']['FILDSIM_gyroradius'])
                for i in range(nfits):
                    if self.resolution['fits']['FILDSIM_pitch'][i] == p:
                        dummy.append(self.resolution['fits']['Gyroradius'][i]\
                                     .params['center'].value)
                        dummy_FILDSIM.append(self.resolution['fits']\
                                             ['FILDSIM_gyroradius'][i])

                cen_g.append(dummy.copy())
                fild_g.append(dummy_FILDSIM.copy())
            for i in range(len(self.resolution['fits']['FILDSIM_pitch'])):
                cen_p.append(self.resolution['fits']['Pitch'][i]\
                             .params['center'].value)
            figc, axc = plt.subplots(1, 2)
            for i in range(len(fild_g)):
                label_plot = \
                    str(np.unique(self.resolution['fits']['FILDSIM_pitch'])[i])
                axc[0].plot(fild_g[i], cen_g[i], 'o', label=label_plot)
            axc[0].set_xlabel('FILDSIM')
            axc[0].legend()
            axc[0] = ssplt.axis_beauty(axc[0], axis_param)
            axc[1].plot(self.resolution['fits']['FILDSIM_pitch'],
                        cen_p, 'o')
            axc[1].set_xlabel('FILDSIM')
            axc[1] = ssplt.axis_beauty(axc[1], axis_param)

    def plot_pitch_histograms(self, diag_params: dict = {},
                              adaptative: bool = True,
                              min_statistics=100,
                              gyroradius=3,
                              plot_fit=True,
                              axarr=None, dpi=100, alpha=0.5):
        """
        Calculate the resolution associated with each point of the map

        Ajvv

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch
        @param min_statistics: Minimum number of points for a given r p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_statistics: Minimum number of counts to perform the fit
        @param adaptative: If true, the bin width will be adapted such that the
        number of bins in a sigma of the distribution is 4. If this is the
        case, dpitch, dgyr, will no longer have an impact
        """
        if self.strike_points is None:
            raise Exception('You should load the strike points first!!')
        if self.diag == 'FILD':
            # --- Prepare options:
            diag_options = {
                'dpitch': 1.0,
                'dgyr': 0.1,
                'p_method': 'Gauss',
                'g_method': 'sGauss'
            }
            diag_options.update(diag_params)
            dpitch = diag_options['dpitch']
            p_method = diag_options['p_method']

            npitch = self.strike_points['pitch'].size
            ir = np.argmin(abs(self.strike_points['gyroradius'] - gyroradius))

            for ip in range(npitch):
                # --- Select the data
                data = self.strike_points['Data'][
                    (self.strike_points['Data'][:, 0] ==
                     self.strike_points['gyroradius'][ir]) *
                    (self.strike_points['Data'][:, 1] ==
                     self.strike_points['pitch'][ip]), :]

                if len(data[:, 0]) < min_statistics:
                    continue
                # Prepare the bin edges according to the desired width
                edges_pitch = \
                    np.arange(start=data[:, 7].min() - dpitch,
                              stop=data[:, 7].max() + dpitch,
                              step=dpitch)

                # --- Reduce (if needed) the bin width, we will set the
                # bin width as 1/4 of the std, to ensure a good fitting
                if adaptative:
                    n_bins_in_sigma = 4
                    sigma_p = np.std(data[:, 7])
                    new_dpitch = sigma_p / n_bins_in_sigma
                    edges_pitch = \
                        np.arange(start=data[:, 7].min() - dpitch,
                                  stop=data[:, 7].max() + dpitch,
                                  step=new_dpitch)
                # --- Proceed to fit
                par_p, resultp = _fit_to_model_(data[:, 7],
                                                bins=edges_pitch,
                                                model=p_method,
                                                normalize=False)

                if axarr is None:
                    fig, axarr = plt.subplots(nrows=1, ncols=1,
                                              figsize=(6, 10),
                                              facecolor='w', edgecolor='k',
                                              dpi=dpi)
                    ax_pitch = axarr  # topdown view, should see pinhole surfac
                    ax_pitch.set_xlabel('Pitch [$\\degree$]')
                    ax_pitch.set_ylabel('Counts')
                    ax_pitch.set_title(
                        'Pitch resolution at gyroradius '
                        + str(self.strike_points['gyroradius'][ir])+' cm')

                    created_ax = True

                cent = 0.5 * (edges_pitch[1:] + edges_pitch[:-1])
                fit_line = ax_pitch.plot(cent, resultp.best_fit,
                                         label='_nolegend_')
                label_plot = \
                    f"{float(self.strike_points['pitch'][ip]):g}"\
                    + '$\\degree$'
                ax_pitch.hist(data[:, 7], bins=edges_pitch, alpha=alpha,
                              label=label_plot, color=fit_line[0].get_color())

        ax_pitch.legend(loc='best')

        if created_ax:
            fig.tight_layout()
            fig.show()

        return

    def plot_gyroradius_histograms(self, diag_params: dict = {},
                                   adaptative: bool = True,
                                   min_statistics=100,
                                   pitch=30,
                                   plot_fit=True,
                                   axarr=None, dpi=100, alpha=0.5):
        """
        Calculate the resolution associated with each point of the map

        Ajvv

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch
        @param min_statistics: Minimum number of points for a given r p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_statistics: Minimum number of counts to perform the fit
        @param adaptative: If true, the bin width will be adapted such that the
        number of bins in a sigma of the distribution is 4. If this is the
        case, dpitch, dgyr, will no longer have an impact
        """
        if self.strike_points is None:
            raise Exception('You should load the strike points first!!')
        if self.diag == 'FILD':
            # --- Prepare options:
            diag_options = {
                'dpitch': 1.0,
                'dgyr': 0.1,
                'p_method': 'Gauss',
                'g_method': 'sGauss'
            }
            diag_options.update(diag_params)
            dgyr = diag_options['dgyr']
            g_method = diag_options['g_method']

            nr = self.strike_points['gyroradius'].size

            ip = np.argmin(abs(self.strike_points['pitch'] - pitch))

            for ir in range(nr):
                # --- Select the data
                data = self.strike_points['Data'][
                    (self.strike_points['Data'][:, 0] ==
                     self.strike_points['gyroradius'][ir]) *
                    (self.strike_points['Data'][:, 1] ==
                     self.strike_points['pitch'][ip]), :]

                if len(data[:, 0]) < min_statistics:
                    continue
                # Prepare the bin edges according to the desired width
                edges_gyr = \
                    np.arange(start=data[:, 6].min() - dgyr,
                              stop=data[:, 6].max() + dgyr,
                              step=dgyr)
                # --- Reduce (if needed) the bin width, we will set the
                # bin width as 1/4 of the std, to ensure a good fitting
                if adaptative:
                    n_bins_in_sigma = 4
                    sigma_r = np.std(data[:, 6])
                    new_dgyr = sigma_r / n_bins_in_sigma
                    edges_gyr = \
                        np.arange(start=data[:, 6].min() - new_dgyr,
                                  stop=data[:, 6].max() + new_dgyr,
                                  step=new_dgyr)

                # --- Proceed to fit

                par_g, resultg = _fit_to_model_(data[:, 6],
                                                bins=edges_gyr,
                                                model=g_method,
                                                normalize=False)
                if axarr is None:
                    fig, axarr = \
                        plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                     facecolor='w', edgecolor='k', dpi=dpi)
                    ax_gyroradius = axarr
                    ax_gyroradius.set_xlabel('Gyroradius [cm]')
                    ax_gyroradius.set_ylabel('Counts')
                    title_plot = 'Gyroradius resolution at pitch '\
                        + str(self.strike_points['pitch'][ip]) + '$\\degree$'
                    ax_gyroradius.set_title(title_plot)

                    created_ax = True

                cent = 0.5 * (edges_gyr[1:] + edges_gyr[:-1])
                fit_line = ax_gyroradius.plot(cent, resultg.best_fit,
                                              label='_nolegend_')
                label_plot = \
                    f"{float(self.strike_points['gyroradius'][ir]):g}" + '[cm]'
                ax_gyroradius.hist(data[:, 6], bins=edges_gyr,
                                   alpha=alpha, label=label_plot,
                                   color=fit_line[0].get_color())

        ax_gyroradius.legend(loc='best')

        if created_ax:
            fig.tight_layout()
            fig.show()

        return


class CalParams:
    """
    Information to relate points in the camera sensor the scintillator

    In a future, it will contain the correction of the optical distortion and
    all the methods necessary to correct it.
    """

    def __init__(self):
        """Initialize the class"""
        # To transform the from real coordinates to pixel (see
        # transform_to_pixel())
        ## pixel/cm in the x direction
        self.xscale = 0.0
        ## pixel/cm in the y direction
        self.yscale = 0
        ## Offset to align 0,0 of the sensor with the scintillator
        self.xshift = 0
        ## Offset to align 0,0 of the sensor with the scintillator
        self.yshift = 0
        ## Rotation angle to transform from the sensor to the scintillator
        self.deg = 0.0

    def print(self):
        """Print calibration"""
        print('xscale: ', self.xscale)
        print('yscale: ', self.yscale)
        print('xshift: ', self.xshift)
        print('yshift: ', self.yshift)
        print('deg: ', self.deg)


class Scintillator:
    """
    Class with the scintillator information.

    Note, the notation is given by FILDSIM, and it is a bit misleading,
    in FILDSIM x,y,z axis are defined, the scintillator lies in a plane of
    constant x, so the only variables to play with are y,z. However, x,
    y are always used to refer to x horizontal and vertical direction in the
    camera sensor. We have to live with this. Just ignore the x coordinates
    of the scintillator data and work with y,z as they were x,y
    """

    def __init__(self, file: str, material: str = 'TG-green'):
        """
        Initialize the class

        @param    file: Path to the file with the scintillator geometry
        @param    material: Defaults to 'TG-green'
        """
        ## Material used in the scintillator plate
        self.material = material
        # Read the file
        with open(file) as f:
            # Dummy line with description
            f.readline()
            # Line with the scintillator name
            dummy = f.readline()
            ## Name of the scintillator plate given in the simulation
            self.name = dummy[5:-1]
            # Line with the number of vertices
            dummy = f.readline()
            ## Number of vertices
            self.n_vertices = int(dummy[11:-1])
            # Skip the data with the vertices and the normal vector
            for i in range(self.n_vertices + 3):
                f.readline()
            ## Units in which the scintillator data is loaded:
            dummy = f.readline()
            self.orig_units = dummy[:-1]

        ## Coordinates of the vertex of the scintillator (X,Y,Z). In cm
        self.coord_real = np.loadtxt(file, skiprows=3, delimiter=',',
                                     max_rows=self.n_vertices)
        ## Normal vector
        self.normal_vector = np.loadtxt(file, skiprows=4 + self.n_vertices,
                                        delimiter=',', max_rows=1)
        ## Coordinates of the vertex of the scintillator in pixels
        self.xpixel = None
        self.ypixel = None
        # We want the coordinates in cm, if 'cm' is not the unit, apply the
        # corresponding transformation. (Void it is interpreter as cm)
        factors = {'cm': 1., 'm': 100., 'mm': 0.1, 'inch': 2.54}
        if self.orig_units in factors:
            self.coord_real = self.coord_real * factors[self.orig_units]
        else:
            print('Not recognised unit, possible wrong format file!!!')
            print('Maybe you are using and old FILDSIM file, so do not panic')
            return

    def plot_pix(self, ax=None, plt_par: dict = {}):
        """
        Plot the scintillator, in pixels, in the axes ax

        @param ax: axes where to plot
        @param plt_par: dictionary with the parameters to plot
        @return: Nothing, just update the plot
        """
        if 'color' not in plt_par:
            plt_par['color'] = 'r'
        if 'markerstyle' not in plt_par:
            plt_par['marker'] = ''
        if 'linestyle' not in plt_par:
            plt_par['linestyle'] = '--'
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.xpixel, self.ypixel, **plt_par)

    def plot_real(self, ax=None, plt_par: dict = {}):
        """
        Plot the scintillator, in cm, in the axes ax

        @param ax: axes where to plot
        @param plt_par: dictionary with the parameters to plot
        @return: Nothing, just update the plot
        """
        if 'color' not in plt_par:
            plt_par['color'] = 'r'
        if 'markerstyle' not in plt_par:
            plt_par['marker'] = ''
        if 'linestyle' not in plt_par:
            plt_par['linestyle'] = '--'
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.coord_real[:, 1], self.coord_real[:, 2], **plt_par)

    def calculate_pixel_coordinates(self, calib):
        """
        Transform the real coordinates of the map into pixels

        Jose Rueda Rueda: jrrueda@us.es

        @param calib: a CalParams() object with the calibration info
        @return: Nothing, just update the plot
        """
        dummyx = self.coord_real[:, 1]
        dummyy = self.coord_real[:, 2]

        self.xpixel, self.ypixel = transform_to_pixel(dummyx, dummyy, calib)
        return
