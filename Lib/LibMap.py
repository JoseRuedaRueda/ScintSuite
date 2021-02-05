"""Module to remap the scintillator

It contains the routines to load and aling the strike maps, as well as
perform the remapping
"""
# import time
import math
import datetime
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interp
import LibPlotting as ssplt
import LibFILDSIM as ssFILDSIM
from LibMachine import machine
import LibPaths as p
from tqdm import tqdm   # For waitbars
pa = p.Path(machine)
del p
if machine == 'AUG':
    import LibDataAUG as ssdat

try:
    import lmfit
except ModuleNotFoundError:
    print('lmfit not found, you cannot calculate resolutions')


def transform_to_pixel(x, y, grid_param):
    """
    Transform from X,Y coordinates (scintillator) to pixels in the camera

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

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

    Jose Rueda: jose.rueda@ipp.mpg.de

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

    Jose Rueda: jose.rueda@ipp.mpg.de

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
    v21_real = np.array((scintillator.coord_real[index[0], 1] -
                         scintillator.coord_real[index[1], 1],
                         scintillator.coord_real[index[0], 2] -
                         scintillator.coord_real[index[1], 2], 0))
    v23_real = np.array((scintillator.coord_real[index[2], 1] -
                         scintillator.coord_real[index[1], 1],
                         scintillator.coord_real[index[2], 2] -
                         scintillator.coord_real[index[1], 2], 0))

    v21_pix = np.array([points_frame[0][0], points_frame[0][1], 0]) - \
        np.array([points_frame[1][0], points_frame[1][1], 0])

    v23_pix = np.array([points_frame[2][0], points_frame[2][1], 0]) - \
        np.array([points_frame[1][0], points_frame[1][1], 0])

    # See if an inversion of one of the axis is needed or not.
    normal_real = np.cross(v21_real, v23_real)
    normal_pix = np.cross(v21_pix, v23_pix)

    # If the normals has opposite signs, an inversion must be done
    if normal_pix[2] * normal_real[2] < 0:
        sign = -1.0
    else:
        sign = 1.0
    # With this sign in mind, now we can proceed with the calculation of the
    # ratio and the gyration angle
    # Initialise the variables
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
        d_real = np.sqrt((scintillator.coord_real[index[i], 2] -
                          scintillator.coord_real[index[i2], 2]) ** 2 +
                         (scintillator.coord_real[index[i], 1] -
                          scintillator.coord_real[index[i2], 1]) ** 2)
        # Distance in the sensor
        dummy = np.array(points_frame[i]) - np.array(points_frame[i2])
        d_pix = np.sqrt(dummy[1] ** 2 + dummy[0] ** 2)
        # Cumulate the magnification factor (we will normalise at the end)
        mag = mag + d_pix / d_real
        # Calculate the angles
        alpha_r = -  math.atan2(scintillator.coord_real[index[i], 2] -
                                scintillator.coord_real[index[i2], 2],
                                sign * scintillator.coord_real[index[i], 1] -
                                sign * scintillator.coord_real[index[i2], 1])
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
        x_new = (scintillator.coord_real[index[i], 1] *
                 math.cos(alpha_px - alpha_r) -
                 scintillator.coord_real[index[i], 2] *
                 math.sin(alpha_px - alpha_r)) * d_pix / d_real * sign
        y_new = (scintillator.coord_real[index[i], 1] *
                 math.sin(alpha_px - alpha_r) +
                 scintillator.coord_real[index[i], 2] *
                 math.cos(alpha_px - alpha_r)) * d_pix / d_real
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
def remap(smap, frame, x_min=20.0, x_max=80.0, delta_x=1,
          y_min=1.5, y_max=10.0, delta_y=0.2):
    """
    Remap a frame

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param smap: StrikeMap() object with the strike map
    @param frame: the frame to be remapped
    @param x_min: Minimum value of the x coordinate, for FILD, pitch [º]
    @param x_max: Maximum value of the x coordinate
    @param delta_x: Spacing for the x coordinate
    @param y_min: Minimim value of the y coordinate, for FILD, gyroradius [cm]
    @param y_max: Maximum value of the y coordinate
    @param delta_x: Spacing of the y coordinate
    """
    # --- 0: Check inputs
    if smap.gyr_interp is None:
        raise Exception('Interpolate strike map before!!!')

    # --- 1: Edges of the histogram
    x_edges = np.arange(start=x_min, stop=x_max, step=delta_x)
    y_edges = np.arange(start=y_min, stop=y_max, step=delta_y)

    # --- 2: Information of the calibration
    if smap.diag == 'FILD':
        x = smap.pit_interp.flatten()   # pitch associated to each pixel
        y = smap.gyr_interp.flatten()   # gyroradius associated to each pixel

    # --- 3: Remap (via histogram)
    z = frame.flatten()
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges],
                                       weights=z)
    # Normalise H to counts per unit of each axis
    H /= delta_x * delta_y

    # --- 4: Calculate the centroids of the bins, for later plotting
    x_cen = 0.5 * (x_edges[0:-1] + x_edges[1:])
    y_cen = 0.5 * (y_edges[0:-1] + y_edges[1:])

    return H, x_cen, y_cen


def gyr_profile(remap_frame, pitch_centers, min_pitch: float,
                max_pitch: float, verbose: bool = False,
                name=None, gyr=None):
    """
    Cut the FILD signal to get a profile along gyroradius

    @author:  Jose Rueda: jose.rueda@ipp.mpg.de

    @param    remap_frame: np.array with the remapped frame
    @type:    ndarray

    @param    pitch_centers: np array produced by the remap function
    @type:    ndarray

    @param    min_pitch: minimum pitch to include
    @type:    float

    @param    max_pitch: Maximum pitch to include
    @type:    float

    @param    verbose: if true, the actual pitch interval will be printed
    @type:    bool

    @return   profile:  the profile in gyroradius

    @raises   ExceptionName: exception if the desired pitch range is not in the
    frame
    """
    # See which cells do we need
    flags = (pitch_centers < max_pitch) * (pitch_centers > min_pitch)
    if np.sum(flags) == 0:
        raise Exception('No single cell satisfy the condition!')
    # The pitch centers is the centroid of the cell, but a cell include counts
    # which pitchs are in [p0-dp, p0+dp], therefore, let give to the user these
    # to values
    used_pitches = pitch_centers[flags]
    delta = pitch_centers[1] - pitch_centers[0]
    min_used_pitch = used_pitches[0] - 0.5 * delta
    max_used_pitch = used_pitches[-1] + 0.5 * delta
    dummy = remap_frame[flags, :]
    profile = np.sum(dummy, axis=0)
    if verbose:
        print('The minimum pitch used is: ', min_used_pitch)
        print('The Maximum pitch used is: ', max_used_pitch)
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
            raise Exception('You want to export but no pitch was given')
    return profile


def pitch_profile(remap_frame, gyr_centers, min_gyr: float,
                  max_gyr: float, verbose: bool = False,
                  name=None, pitch=None):
    """
    Cut the FILD signal to get a profile along pitch

    @author:  Jose Rueda: jose.rueda@ipp.mpg.de

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
    flags = (gyr_centers < max_gyr) * (gyr_centers > min_gyr)
    if np.sum(flags) == 0:
        raise Exception('No single cell satisfy the condition!')
    # The pitch centers is the centroid of the cell, but a cell include counts
    # which pitchs are in [p0-dp, p0+dp], therefore, let give to the user these
    # to values
    used_gyr = gyr_centers[flags]
    delta = gyr_centers[1] - gyr_centers[0]
    min_used_gyr = used_gyr[0] - 0.5 * delta
    max_used_gyr = used_gyr[-1] + 0.5 * delta
    dummy = remap_frame[:, flags]
    profile = np.sum(dummy, axis=1)
    if verbose:
        print('The minimum gyroradius used is: ', min_used_gyr)
        print('The Maximum gyroradius used is: ', max_used_gyr)

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
    Estimate the efective area covered by a pixel

    Jose Rueda Rueda: ruejo@ipp.mpg.de based on a routine of Joaquín Galdón

    If there is no distortion:
    Area_covered_by_1_pixel: A_omega=Area_scint/#pixels inside scintillator
    #pixels inside scint=L'x_scint*L'y_scint=Lx_scint*xscale*Ly_scint*yscale
    xscale and yscale are in units of : #pixels/cm
    So A_omega can be approximated by: A_omega=1/(xscale*yscale) [cm^2]

    @param frame_shape: shape of the frame
    @params yscale: the scale [#pixel/cm] of the calibration to align the map
    @params xscale: the scale [#pixel/cm] of the calibration to align the map
    @param type: 0, ignore distortion, 1 include distortion
    @return area: Matrix where each element is the areacavered by that pixel
    @todo Include the model of distortion
    """
    # Initialise the matrix:
    area = np.zeros(frame_shape)

    if type == 0:
        area[:] = abs(1./(xscale*yscale)*1.e-4)  # 1e-4 to be in m^2

    return area


# This seems repeated with the FILDSIM module. Solve repetition!!!
# def FILD_calculate_photon_flux(raw_frame, calibration_frame,
#                                pinhole_area: float, exposure_time: float,
#                               calib_exposure_time: float, pixel_area_covered,
#                                int_photon_flux, mask=None):
#     """
#     Convert a FILD frame into photon/s
#
#     Jose Rueda: ruejo@ipp.mpg.de  based on an IDL routine of Joaquin Galdón
#
#     About the units:
#         -# photon_flux: Photons/(s*m^2)
#         -# Area covered by the pixel: m^2
#         -# Calibration exposure time: s
#         -# Exposure time: s
#         -# pinhole area: m^2
#         -# Calibration frame [input]: counts
#         -# Calibration frame [output]: counts*s*m^2/photons
#         -# Photon flux frame: Photons/(s*m^2)
#     @todo include docstring of the inputs
#     """
#     # Chec frame shapes
#     # --- Section 0: Check the inputs
#     s1 = raw_frame.shape
#     s2 = calibration_frame.shape
#
#     if calibration_frame.size == 1:
#         print('Using mean calibration frame method:')
#         print('So Using single (mean) value instead of a 2D array')
#     else:
#         if (s1[0] != s2[0]) or (s1[1] != s2[0]):
#             print('Size of data frame and calibration frame not matching!!')
#             raise Exception('Use mean calibration frame method!!')
#     # --- Section 1: calibrate the frames
#     cal_frame_cal = \
#         FILD_absolute_calibration_frame(calibration_frame, exposure_time,
#                                         pinhole_area, calib_exposure_time,
#                                         int_photon_flux, pixel_area_covered)
#     photon_flux_frame = raw_frame / cal_frame_cal
#     photon_flux = np.nansum(photon_flux_frame, dtype=np.float)
#
#     if mask is not None:
#         print('USING ROI METHOD FOR PHOTON FRAME CALCULATION')
#         dummy_frame = photon_flux_frame * 0.0
#         dummy_frame[mask] = photon_flux_frame[mask]
#
#         photon_flux_frame = dummy_frame
#         photon_flux = np.nansum(photon_flux_frame, dtype=np.float)
#
#     return {'photon_flux_frame': photon_flux_frame,
#             'photon_flux': photon_flux, 'calibration_frame': cal_frame_cal,
#             'exposure_time': exposure_time, 'pinhole_area': pinhole_area}


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
                                 verbose: bool = False):
    """
    Remap all loaded frames from a FILD video

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de
    @todo finish documentation of this function

    @param    video: Video object (see LibVideoFiles)
    @param    calibration: Calibation object (see Calibration class)
    @param    shot: shot number
    @param    rmin: minimum gyroradius to consider [cm]
    @type:    float
    @param    rmax: maximum gyroradius to consider [cm]
    @type:    float

    @param    dr: Description of parameter `dr`. Defaults to 0.1.
    @type:    float

    @param    pmin: Description of parameter `pmin`. Defaults to 15.0.
    @type:    float

    @param    pmax: Description of parameter `pmax`. Defaults to 90.0.
    @type:    float

    @param    dp: Description of parameter `dp`. Defaults to 1.0.
    @type:    float

    @param    rprofmin: Description of parameter `rprofmin`. Defaults to 1.0.
    @type:    float

    @param    rprofmax: Description of parameter `rprofmax`. Defaults to 4.7.
    @type:    float

    @param    pprofmin: Description of parameter `pprofmin`. Defaults to 20.0.
    @type:    float

    @param    pprofmax: Description of parameter `pprofmax`. Defaults to 90.0.
    @type:    float

    @param    rfild: Description of parameter `rfild`. Defaults to 2.186.
    @type:    float

    @param    zfild: Description of parameter `zfild`. Defaults to 0.32.
    @type:    float

    @param    alpha: Description of parameter `alpha`. Defaults to 0.0.
    @type:    float

    @param    beta: Description of parameter `beta`. Defaults to -12.0.
    @type:    float

    @param    method: ethod to interpolate the strike maps, default 1: linear

    @return:  Description of returned object.
    @rtype:   type

    @raises   ExceptionName: Why the exception is raised.
    """
    # Print just some info:
    print('Looking for strikemaps in: ', pa.StrikeMaps)
    # Get frame shape:
    nframes = len(video.exp_dat['nframes'])
    frame_shape = video.exp_dat['frames'].shape[0:2]
    # Get the time (to measure elapsed time)
    tic = time.time()
    # Get the magnetic field: In principle we should be able to do this in an
    # efficient way, but the AUG library to acces magnetic field is kind of a
    # shit in python 3, so we need a work around
    if machine == 'AUG':
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
    # Get the dimension of the gyr and pitch profiles. Depending on the exact
    # values, the ends can enter or not... To look for the dimession I just
    # create a dummy vector and avoid 'complicated logic'
    dum = np.arange(start=rmin, stop=rmax, step=dr)
    ngyr = len(dum) - 1
    dum = np.arange(start=pmin, stop=pmax, step=dp)
    npit = len(dum) - 1
    remaped_frames = np.zeros((npit, ngyr, nframes))
    signal_in_gyr = np.zeros((ngyr, nframes))
    signal_in_pit = np.zeros((npit, nframes))
    # b_field = np.zeros(nframes)
    theta = np.zeros(nframes)
    phi = np.zeros(nframes)
    name_old = ' '
    print('Remapping frames')
    for iframe in tqdm(range(nframes)):
        if machine == 'AUG':
            tframe = video.exp_dat['tframes'][iframe]
            br[iframe], bz[iframe], bt[iframe], bp =\
                ssdat.get_mag_field(shot, rfild, zfild, time=tframe, equ=equ)
            b_field[iframe] = np.sqrt(br[iframe]**2 + bz[iframe]**2 +
                                      bt[iframe]**2)
        phi[iframe], theta[iframe] = \
            ssFILDSIM.calculate_fild_orientation(br[iframe], bz[iframe],
                                                 bt[iframe], alpha, beta)
        name = ssFILDSIM.find_strike_map(rfild, zfild, phi[iframe],
                                         theta[iframe], pa.StrikeMaps,
                                         pa.FILDSIM,
                                         FILDSIM_options=fildsim_options)
        # Only reload the strike map if it is needed
        if name != name_old:
            map = StrikeMap(0, os.path.join(pa.StrikeMaps, name))
            map.calculate_pixel_coordinates(calibration)
            # print('Interpolating grid')
            map.interp_grid(frame_shape, plot=False, method=method)
        name_old = name
        remaped_frames[:, :, iframe], pitch, gyr = \
            remap(map, video.exp_dat['frames'][:, :, iframe], x_min=pmin,
                  x_max=pmax, delta_x=dp, y_min=rmin, y_max=rmax, delta_y=dr)
        # Calculate the gyroradius and pitch profiles
        dummy = remaped_frames[:, :, iframe].squeeze()
        signal_in_gyr[:, iframe] = gyr_profile(dummy, pitch, pprofmin,
                                               pprofmax)
        signal_in_pit[:, iframe] = pitch_profile(dummy, gyr, rprofmin,
                                                 rprofmax)
        if verbose:
            print('### Frame:', iframe + 1, 'of', nframes, 'remapped')
            toc = time.time()
            print('Whole time interval remaped in: ', toc-tic, ' s')
            print('Average time per frame: ', (toc-tic) / nframes, ' s')
    output = {'frames': remaped_frames, 'xaxis': pitch, 'yaxis': gyr,
              'xlabel': 'Pitch', 'ylabel': '$r_l$',
              'xunits': '{}^o', 'yunits': 'cm',
              'sprofx': signal_in_pit, 'sprofy': signal_in_gyr,
              'sprofxlabel': 'Signal integrated in r_l',
              'sprofylabel': 'Signal integrated in pitch',
              'bfield': b_field, 'phi': phi, 'theta': theta,
              'tframes': video.exp_dat['tframes']}
    opt = {'rmin': rmin, 'rmax': rmax, 'dr': dr, 'pmin': pmin, 'pmax': pmax,
           'dp': dp, 'rprofmin': rprofmin, 'rprofmax': rprofmax,
           'pprofmin': pprofmin, 'pprofmax': pprofmax, 'rfild': rfild,
           'zfild': zfild, 'alpha': alpha, 'beta': beta}
    return output, opt


# -----------------------------------------------------------------------------
# --- Fitting routines
# -----------------------------------------------------------------------------
def _fit_to_model_(data, bins=20, model='Gauss'):
    """
    Make histogram of input data and fit to a model

    Jose Rueda: jrrueda@us.es

    @param bins: Can be the desired number of bins or the edges
    @param model: 'Gauss' Pure Gaussian, 'sGauss' Screw Gaussian
    """
    # --- Make the histogram of the data
    hist, edges = np.histogram(data, bins=bins)
    hist = hist.astype(np.float64)
    hist /= hist.max()  # Normalise to  have de data between 0 and 1
    cent = 0.5 * (edges[1:] + edges[:-1])
    # --- Make the fit
    if model == 'Gauss':
        model = lmfit.models.GaussianModel()
        params = model.guess(hist, x=cent)
        result = model.fit(hist, params, x=cent)
        par = {'amplitude': result.params['amplitude'].value,
               'center': result.params['center'].value,
               'sigma': result.params['sigma'].value}
    if model == 'sGauss':
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

        @author Jose Rueda Rueda: jose.rueda@ipp.mpg.de

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
        @return : a file created inth your information
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
            print('No entry find in the database, revise database')
            return
        elif n_true > 1:
            print('Several entries fulfull the condition')
            print('Possible entries:')
            print(self.data['ID'][flags])
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

    def __init__(self, flag=0, file: str = None):
        """
        Initialise the class

        @param flag: 0  means fild, 1 means INPA, 2 means iHIBP (you can also
        write directly 'FILD', 'INPA', 'iHIBP')
        @param file: Full path to file with the strike map
        @todo Eliminate flag and extract info from file name??
        """
        ## Associated diagnostic
        if flag == 0 or flag == 'FILD':
            self.diag = 'FILD'
        elif flag == 2 or flag == 'iHIBP':
            self.diag = 'iHIBP'
        else:
            print('Flag: ', flag)
            raise Exception('Diagnostic not implemented')
        ## X-position, in pixles, of the strike map (commond)
        self.xpixel = None
        ## Y-Position, in pixels, of the strike map (commond)
        self.ypixel = None

        if flag == 0 or flag == 'FILD':
            # Read the file
            dummy = np.loadtxt(file, skiprows=3)
            # See which rows has collimator factor larger than zero (ie see for
            # which combination of energy and pitch some markers has arrived)
            ind = dummy[:, 7] > 0
            # Initialise the class
            ## Gyroradius of map points
            self.gyroradius = dummy[ind, 0]
            ## Energy of map points
            self.energy = None
            ## Pitch of map points
            self.pitch = dummy[ind, 1]
            ## x coordinates of map points (commond)
            self.x = dummy[ind, 2]
            ## y coordinates of map points (commond)
            self.y = dummy[ind, 3]
            ## z coordinates of map points (commond)
            self.z = dummy[ind, 4]
            ## Average initial gyrophase of map markers
            self.avg_ini_gyrophase = dummy[ind, 5]
            ## Number of markers striking in this area
            self.n_strike_points = dummy[ind, 6]
            ## Collimator factor as defined in FILDSIM
            self.collimator_factor = dummy[ind, 7]
            ## Average incident angle of the FILDSIM markers
            self.avg_incident_angle = dummy[ind, 8]
            ## Translate from pixels in the camera to gyroradius
            self.gyr_interp = None
            ## Translate from pixels in the camera to pitch
            self.pit_interp = None
            ## Translate from pixels in the camera to collimator factor
            self.col_interp = None
            ## Strike points used to calculate the map
            self.strike_points = None
            ## Resolution of FILD for each strike point
            self.resolution = None

    def plot_real(self, ax=None,
                  plt_param: dict = {}, line_param: dict = {}):
        """
        Plot the strike map (x,y = dimensions in the scintillator)

        Jose Rueda: jose.rueda@ipp.mpg.de

        @param ax: Axes where to plot
        @param plt_param: Parameters for plot beauty, example: markersize=6
        @return: Strike maps over-plotted in the axis
        """
        # Define some standard parameters:
        if plt_param is None:
            plt_param = {}
        if 'markersize' not in plt_param:
            plt_param['markersize'] = 6
        if 'fillstyle' not in plt_param:
            plt_param['fillstyle'] = 'none'
        if 'color' not in plt_param:
            plt_param['color'] = 'w'
        if 'marker' not in plt_param:
            plt_param['marker'] = 'o'
        if 'linestyle' not in plt_param:
            plt_param['linestyle'] = 'none'

        if 'color' not in line_param:
            line_param['color'] = 'w'
        if 'markerstyle' not in line_param:
            line_param['marker'] = ''

        if ax is None:
            fig, ax = plt.subplots()

        # Draw the lines of constant gyroradius, energy, or rho (depending on
        # the particular diagnostic) [These are the 'horizontal' lines]
        if hasattr(self, 'gyroradius'):
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_param)
        else:
            return
            ## @todo: talk with Pablo about his strike maps and his coordinates

        # Draw the lines of constant pitch (depending of the diagnostic,
        # in INPA would be constant radius [these are the vertical lines]
        if hasattr(self, 'pitch'):
            uniq = np.unique(self.pitch)
            n = len(uniq)
            for i in range(n):
                flags = self.pitch == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_param)
        else:
            return
            ## @todo: change == by a < tol??

        # Plot some markers in the grid position
        ## @todo include labels energy/pitch in the plot
        ax.plot(self.y, self.z, **plt_param)

    def plot_pix(self, ax=None, plt_param: dict = {},
                 line_param: dict = {}):
        """
        Plot the strike map (x,y = pixels on the camera)

        Jose Rueda: jose.rueda@ipp.mpg.de

        @param ax: Axes where to plot
        @param plt_param: Parameters for plot beauty, example: markersize=6
        @return: Strike maps over-plotted in the axis
        """
        # Define some standard parameters:
        if plt_param is None:
            plt_param = {}
        if 'markersize' not in plt_param:
            plt_param['markersize'] = 6
        if 'fillstyle' not in plt_param:
            plt_param['fillstyle'] = 'none'
        if 'color' not in plt_param:
            plt_param['color'] = 'w'
        if 'marker' not in plt_param:
            plt_param['marker'] = 'o'
        if 'linestyle' not in plt_param:
            plt_param['linestyle'] = 'none'

        if 'color' not in line_param:
            line_param['color'] = 'w'
        if 'markerstyle' not in line_param:
            line_param['marker'] = ''

        if ax is None:
            fig, ax = plt.subplots()

        # Draw the lines of constant gyroradius, energy, or rho (depending on
        # the particular diagnostic) [These are the 'horizontal' lines]
        if hasattr(self, 'gyroradius'):
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_param)
        else:
            return
            ## @todo: talk with Pablo about his strike maps and his coordinates

        # Draw the lines of constant pitch (depending of the diagnostic,
        # in INPA would be constant radius [these are the vertical lines]
        if hasattr(self, 'pitch'):
            uniq = np.unique(self.pitch)
            n = len(uniq)
            for i in range(n):
                flags = self.pitch == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_param)
        else:
            return
            ## @todo: change == by a < tol??

        # Plot some markers in the grid position
        ## @todo include labels energy/pitch in the plot
        ax.plot(self.xpixel, self.ypixel, **plt_param)

    def calculate_pixel_coordinates(self, calib):
        """
        Transform the real coordinates of the map into pixels

        Jose Rueda Rueda: jose.rueda@ipp.mpg.de

        @param calib: a CalParams() object with the calibration info
        """
        self.xpixel, self.ypixel = transform_to_pixel(self.y, self.z, calib)

    def interp_grid(self, frame_shape, method=2, plot=False):
        """
        Interpolate grid values on the frames

        Error codes:
            - 0: Scintillator map pixel position not calculated before
            - 1: Interpolating method not recognised

        @param smap: StrikeMap() object
        @param frame_shape: Size of the frame used for the calibration (in px)
        @param method: method to calculate the interpolation:
            - 0: griddata nearest (not recomended)
            - 1: griddata linear
            - 2: griddata cubic
        """
        # --- 0: Check inputs
        if self.xpixel is None:
            raise Exception('Transform to pixel the strike map before')
        # --- 1: Create grid for the interpolation
        grid_x, grid_y = np.mgrid[0:frame_shape[1], 0:frame_shape[0]]
        # --- 2: Interpolate the grid
        dummy = np.column_stack((self.xpixel, self.ypixel))
        if method == 0:
            met = 'nearest'
        elif method == 1:
            met = 'linear'
        elif method == 2:
            met = 'cubic'
        else:
            raise Exception('Not recognised interpolation method')
        if self.diag == 'FILD':
            self.gyr_interp = scipy_interp.griddata(dummy,
                                                    self.gyroradius,
                                                    (grid_x, grid_y),
                                                    method=met,
                                                    fill_value=1000)
            self.gyr_interp = self.gyr_interp.transpose()
            self.pit_interp = scipy_interp.griddata(dummy,
                                                    self.pitch,
                                                    (grid_x, grid_y),
                                                    method=met,
                                                    fill_value=1000)
            self.pit_interp = self.pit_interp.transpose()
            self.col_interp = scipy_interp.griddata(dummy,
                                                    self.collimator_factor,
                                                    (grid_x, grid_y),
                                                    method=met,
                                                    fill_value=1000)
            self.col_interp = self.col_interp.transpose()

        # --- Plot
        if plot:
            if self.diag == 'FILD':
                fig, axes = plt.subplots(2, 2)
                # Plot the scintillator grid
                self.plot_pix(axes[0, 0], line_param={'color': 'k'})
                axes[0, 0].set_xlim(0, frame_shape[0])
                axes[0, 0].set_ylim(0, frame_shape[1])
                # Plot the interpolated gyroradius
                c1 = axes[0, 1].contourf(grid_x, grid_y,
                                         self.gyr_interp.transpose(),
                                         cmap=ssplt.Gamma_II())
                axes[0, 1].set_xlim(0, frame_shape[0])
                axes[0, 1].set_ylim(0, frame_shape[1])
                fig.colorbar(c1, ax=axes[0, 1], shrink=0.9)
                # Plot the interpolated gyroradius
                c2 = axes[1, 0].contourf(grid_x, grid_y,
                                         self.pit_interp.transpose(),
                                         cmap=ssplt.Gamma_II())
                axes[1, 0].set_xlim(0, frame_shape[0])
                axes[1, 0].set_ylim(0, frame_shape[1])
                fig.colorbar(c2, ax=axes[1, 0], shrink=0.9)
                # Plot the interpolated gyroradius
                c3 = axes[1, 1].contourf(grid_x, grid_y,
                                         self.col_interp.transpose(),
                                         cmap=ssplt.Gamma_II())
                axes[1, 1].set_xlim(0, frame_shape[0])
                axes[1, 1].set_ylim(0, frame_shape[1])
                fig.colorbar(c3, ax=axes[1, 1], shrink=0.9)

    def get_energy(self, B0: float, Z: int = 1, A: int = 2):
        """Get the energy associated with each gyroradius

        Jose Rueda: jose.rueda@ipp.mpg.de

        @param self:
        @param B0: Magnetic field [in T]
        @param Z: the charge [in e units]
        @param A: the mass number
        """
        if self.diag == 'FILD':
            self.energy = ssFILDSIM.get_energy(self.gyroradius, B0, A=A, Z=Z)
        return

    def load_strike_points(self, file, verbose: bool = True):
        """
        Load the strike points used to calculate the map

        Jose Rueda: ruejo@ipp.mpg.de

        @param file: File to be loaded. It should contai the strike points in
        FILDSIM format (if we are loading FILD)
        """
        if verbose:
            print('Reading strike points')
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
                self.strike_points['help'].append('9-11: Xi (cm)  Yi (cm)' +
                                                  '  Zi (cm)')
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

    def calculate_resolutions(self, diag_options={'dpitch': 1.0, 'dgyr': 0.1,
                              'p_method': 'Gauss', 'g_method': 'sGauss'},
                              min_statistics: int = 100,
                              adaptative: bool = True):
        """
        Calculate the resolution associated with each point of the map

        Jose Rueda Rueda

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch
        @param min_statistics: Minimum number of points for a given r p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_n_bins: Minimum number of bins for the fitting, if the
        selected bin_width is such that the are less bins, the bin width will
        automatically ajusted
        """
        if self.strike_points is None:
            raise Exception('You should load the strike points first!!')
        if self.diag == 'FILD':
            # --- Prepare options:
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
                        # --- Save the data in the matrices:
                        if p_method == 'Gauss':
                            parameters_pitch['amplitude'][ir, ip] = \
                                par_p['amplitude']
                            parameters_pitch['center'][ir, ip] = \
                                par_p['center']
                            parameters_pitch['sigma'][ir, ip] = par_p['sigma']
                            parameters_pitch['gamma'][ir, ip] = np.nan
                        elif p_method == 'sGauss':
                            parameters_pitch['amplitude'][ir, ip] = \
                                par_p['amplitude']
                            parameters_pitch['center'][ir, ip] = \
                                par_p['center']
                            parameters_pitch['sigma'][ir, ip] = par_p['sigma']
                            parameters_pitch['gamma'][ir, ip] = par_p['gamma']

                        if g_method == 'Gauss':
                            parameters_gyr['amplitude'][ir, ip] = \
                                par_g['amplitude']
                            parameters_gyr['center'][ir, ip] = par_g['center']
                            parameters_gyr['sigma'][ir, ip] = par_g['sigma']
                            parameters_gyr['gamma'][ir, ip] = np.nan
                        elif g_method == 'sGauss':
                            parameters_gyr['amplitude'][ir, ip] = \
                                par_g['amplitude']
                            parameters_gyr['center'][ir, ip] = par_g['center']
                            parameters_gyr['sigma'][ir, ip] = par_g['sigma']
                            parameters_gyr['gamma'][ir, ip] = par_g['gamma']

        self.resolution = {'Gyroradius': parameters_gyr,
                           'Pitch': parameters_pitch,
                           'nmarkers': npoints,
                           'fit_Gyroradius': fitg,
                           'fit_Pitch': fitp}
        return

    def plot_resolutions(self, ax_param: dict = {}, cMap=None, nlev: int = 20):
        """
        Plot the resolutions

        Jose Rueda: ruejo@ipp.mpg.de

        @todo: Implement label size in colorbar

        @param ax_param: parameters for the axis beauty function. Note, labels
        of the color axis are hard-cored, if you want custom axis labels you
        would need to draw the plot on your own
        @param cMap: is None, Gamma_II will be used
        @param nlev: number of levels for the contour
        """
        # Open the figure and prepare the map:
        fig, ax = plt.subplots(1, 2)

        if cMap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = cMap
        if 'fontsize' not in ax_param:
            ax_param['fontsize'] = 14
            # cFS = 14
        # else:
            # cFS = ax_param['fontsize']
        if 'xlabel' not in ax_param:
            ax_param['xlabel'] = '$\\lambda [{}^o]$'
        if 'ylabel' not in ax_param:
            ax_param['ylabel'] = '$r_l [cm]$'

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
            ax[1] = ssplt.axis_beauty(ax[1], ax_param)
            plt.tight_layout()
            return


class CalParams:
    """
    Information to relate points in the camera sensor the scintillator

    In a future, it will contains the correction of the optical distortion and
    all the methods necesary to correct it.
    """

    def __init__(self):
        """Initialize of the class"""
        # To transform the from real coordinates to pixel (see
        # transform_to_pixel())
        ## pixel/cm in the x direction
        self.xscale = 0
        ## pixel/cm in the y direction
        self.yscale = 0
        ## Offset to align 0,0 of the sensor with the scintillator
        self.xshift = 0
        ## Offset to align 0,0 of the sensor with the scintillator
        self.yshift = 0
        ## Rotation angle to transform from the sensor to the scintillator
        self.deg = 0

    def print(self):
        """Print calibration"""
        print('xcale: ', self.xscale)
        print('ycale: ', self.yscale)
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
        factors = {'cm': 1, 'm': 0.01, 'mm': 0.1, 'inch': 2.54}
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

        Jose Rueda Rueda: jose.rueda@ipp.mpg.de

        @param calib: a CalParams() object with the calibration info
        @return: Nothing, just update the plot
        """
        dummyx = self.coord_real[:, 1]
        dummyy = self.coord_real[:, 2]

        self.xpixel, self.ypixel = transform_to_pixel(dummyx, dummyy, calib)
