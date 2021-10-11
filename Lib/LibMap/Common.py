"""
Common auxiliar routines for the FILD and INPA mapping.

Jose Rueda Rueda - jrrueda@us.es

Introduced in version 0.6.0

"""
import numpy as np
import warnings
import datetime
try:
    import lmfit
except ImportError:
    warnings.warn('lmfit not found, you cannot calculate resolutions')


# ------------------------------------------------------------------------------
# --- Scintillator to pixels
# ------------------------------------------------------------------------------
# @ToDo: Need to include the case of distortion
def transform_to_pixel(x, y, grid_param):
    """
    Transform from X,Y coordinates (scintillator) to pixels in the camera.

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


# ------------------------------------------------------------------------------
# --- Fitting functions
# ------------------------------------------------------------------------------
def _fit_to_model_(data, bins=20, model='Gauss', normalize=True):
    """
    Make histogram of input data and fit to a model.

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
# --- Remap and profiles
# -----------------------------------------------------------------------------
def remap(smap, frame, x_edges=None, y_edges=None, mask=None, method='MC'):
    """
    Remap a frame.

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
    Cut the FILD signal to get a profile along gyroradius.

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
    Cut the FILD signal to get a profile along pitch.

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
    Estimate the effective area covered by a pixel.

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
