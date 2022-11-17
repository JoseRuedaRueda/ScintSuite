"""
Common auxiliary routines for the FILD and INPA mapping.

Jose Rueda Rueda - jrrueda@us.es

Introduced in version 0.6.0
"""
import logging
import datetime
import numpy as np
from Lib.decorators import deprecated
logger = logging.getLogger('ScintSuite.MappingCommon')
try:
    import lmfit
except (ImportError, ModuleNotFoundError):
    logger.wargning('10: You cannot calculate resolutions')
__all__ = ['transform_to_pixel', 'XYtoPixel', '_fit_to_model_',
           'remap', 'gyr_profile', 'pitch_profile',
           'estimate_effective_pixel_area']


# -----------------------------------------------------------------------------
# --- Scintillator to pixels
# -----------------------------------------------------------------------------
def transform_to_pixel(x: np.ndarray, y: np.ndarray, cal):
    """
    Transform from X,Y coordinates (scintillator) to pixels in the camera.

    Jose Rueda Rueda: jrrueda@us.es
    Hannah Lindl: hannah.lindl@ipp.mpg.de

    :param  x: np.array of positions to be transformed, x coordinate
    :param  y: np.array of positions to be transformed, y coordinate
    :param  cal: Object containing all the information for the
    transformation, see class CalParams()

    :return xpixel: x positions in pixels
    :return ypixel: y position in pixels
    """
    eps = 1e-7  # Level for the distortion coefficient to be considered as
    #             zero (see below)
    # Perform the undistorted transformation
    alpha = cal.deg * np.pi / 180
    xpixel = (np.cos(alpha) * x - np.sin(alpha) * y) * cal.xscale + \
        cal.xshift
    ypixel = (np.sin(alpha) * x + np.cos(alpha) * y) * cal.yscale + \
        cal.yshift
    # Apply the distortion
    if abs(cal.c1) > eps:
        xp = xpixel - cal.xcenter
        yp = ypixel - cal.ycenter
        rp = np.sqrt(xp**2 + yp**2)
        if ('type' not in cal.__dict__) or (cal.type != 'non-poly'):
            # Get the r array respect to the optical axis
            D = cal.c1 * rp
            xpixel = (1 + D) * xp + cal.xcenter
            ypixel = (1 + D) * yp + cal.ycenter
            if cal.c1 < 0:
                rlim = -0.5 / cal.c1
                rcamera_limit = (1 + cal.c1 * rlim) * rlim
                flags = rp > rlim
                xpixel[flags] = rcamera_limit*xp[flags]/rp[flags] + cal.xcenter
                ypixel[flags] = rcamera_limit*yp[flags]/rp[flags] + cal.ycenter
        else:
            print('non-poly method is used')
            d = (1-np.sqrt(1-4*cal.c1*rp**2))/(2*cal.c1*rp)
            xpixel = xp*(1+cal.c1*d**2) + cal.xcenter
            ypixel = yp*(1+cal.c1*d**2) + cal.ycenter
    return xpixel, ypixel


class XYtoPixel:
    """
    Parent class for object with both coordinates in real and camera space

    For example for Scintillator and strike maps, which contain information of
    their coordinates in the real space and of their coordinates in the camera
    (pixel) space

    It is not intended to be initialised directly by the user. The StrikeMap, or
    scintillator objects will do it. Pease use those child classes.

    Jose Rueda: jrrueda@us.es

    Introduced in version 0.8.2
    """

    def __init__(self):
        """
        Initialise the object

        Notice that this is just a parent class, each child (scintillator or
        strike map), will fill its contents as needed
        """
        ## Coordinates of the vertex of the scintillator (X,Y,Z).
        self._coord_real = {
            'x1': None,
            'x2': None,
            'x3': None
        }
        ## Normal vector to the plane containing the element
        self._normal_vector = None
        ## Coordinates of the vertex of the scintillator in pixels
        self._coord_pix = {
            'x': None,
            'y': None
        }
        self.CameraCalibration = None

    def calculate_pixel_coordinates(self, cal):
        """
        Transfom real coordinates into pixel.

        Just a wrapper to the function transform_to_pixel

        Jose Rueda Rueda: jrrueda@us.es

        :param  cal: camera calibration to apply

        :return: nothing, just fill the _coord_pix dictionary
        """
        self._coord_pix['x'], self._coord_pix['y'] = \
            transform_to_pixel(self._coord_real['x1'], self._coord_real['x2'],
                               cal)
        self.CameraCalibration = cal


def estimate_effective_pixel_area(frame_shape, xscale: float, yscale: float,
                                  type: int = 0):
    """
    Estimate the effective area covered by a pixel.

    Jose Rueda Rueda: jrrueda@us.es based on a routine of Joaquín Galdón

    If there is no distortion:
    Area_covered_by_1_pixel: A_omega=Area_scint/#pixels inside scintillator
    #pixels inside scint=L'x_scint*L'y_scint=Lx_scint*xscale*Ly_scint*yscale
    xscale and yscale are in units of : #pixels/cm
    So A_omega can be approximated by: A_omega=1/(xscale*yscale) [cm^2 or m^2]

    :param  frame_shape: shape of the frame
    :param s yscale: the scale [#pixel/cm] of the calibration to align the map
    :param s xscale: the scale [#pixel/cm] of the calibration to align the map
    :param  type: 0, ignore distortion, 1 include distortion (not done)
    :return area: Matrix where each element is the area covered by that pixel
    @todo Include the model of distortion
    @todo now that the default calibrations are in m^-1, we should remove the
    1e-4, I am not a fan of including an extra optional argument...
    """
    # Initialise the matrix:
    area = np.zeros(frame_shape)

    if type == 0:
        area[:] = abs(1./(xscale*yscale)*1.e-4)  # 1e-4 to be in m^2

    return area

# -----------------------------------------------------------------------------
# --- Fitting functions
# -----------------------------------------------------------------------------
def _fit_to_model_(data, bins: int = 20, model: str = 'Gauss',
                   normalize: bool = True,
                   confidence_level: float = 0.9544997,
                   uncertainties: bool = True):
    """
    Make histogram of input data and fit to a model.

    Jose Rueda: jrrueda@us.es

    :param  bins: Can be the desired number of bins or the edges
    :param  model: 'Gauss' Pure Gaussian, 'sGauss' Screw Gaussian
    :param  normalize: flag to normalise the number of counts in the bins
        between 0, 1; to improve fit performance
    :param  confidence_level: confidence level for the uncertainty determination
    :param  uncertainties: flag to calcualte the uncertainties of the fit

    :return par: Dictionary containing the fit parameters
    :return results: the lmfit model object with the results
    :return normalization: The used normalization for the histogram
    :return unc_output: The width of the confidence interval. Notice that this
        is half the average of the upper and lower limits, so symmetric
        confidence interval is assumed. If you need non symstric conficende
        intervals you would need to run conf_interval of the fit on your own
    """
    # --- Make the histogram of the data
    hist, edges = np.histogram(data, bins=bins)
    hist = hist.astype(np.float64)
    if normalize:
        normalization = hist.max()
        hist /= normalization  # Normalise to  have the data between 0 and 1
    else:
        normalization = 1.0
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
    # --- Get the uncertainty
    if uncertainties:
        uncertainty = lmfit.conf_interval(result, result,
                                          sigmas=[confidence_level])
        unc_output = par.copy()
        for key in unc_output.keys():
            unc_output[key] =\
                0.5 * abs(uncertainty[key][0][1] - uncertainty[key][2][1])
                # TODO: Isn't this conflicting with the confidence interval above?
    else:
        unc_output = par.copy()
        for key in unc_output.keys():
            unc_output[key] = np.nan

    return par, result, normalization, unc_output


# -----------------------------------------------------------------------------
# --- Remap and profiles
# -----------------------------------------------------------------------------
def remap(smap, frame, x_edges=None, y_edges=None, mask=None, method='MC'):
    """
    Remap a frame.

    Jose Rueda: jrrueda@us.es

    Edges are only needed if you select the centers method, if not, they will
    be 'inside' the transformation matrix already

    :param  smap: StrikeMap() object with the strike map
    :param  frame: the frame to be remapped
    :param  x_edges: edges of the x coordinate, for FILD, pitch [º]
    :param  y_edges: edges of the y coordinate, for FILD, gyroradius [cm]
    :param  method: procedure for the remap
        - MC: Use the transformation matrix calculated with markers at the chip
        - centers: Consider just the center of each pixel (Old IDL method)
    """
    # --- 0: Check inputs
    if smap._grid_interp is None:
        text = '27: Interpolators not present in the strike map'\
               +    ', calculating them with default settings'
        logger.warning(text)
        if method.lower() == 'mc':
            # Deduce the needed grid. This is for the case we need to calculate
            # a traslation matrix for this new strike map, using as reference
            # an old grid.
            # @TODO: check this becuase some times fails in the edge
            dx = x_edges[1] - x_edges[0]
            dy = y_edges[1] - y_edges[0]
            xmin = x_edges[0] - dx/2
            xmax = x_edges[-1] + dx/2
            ymin = y_edges[0] - dy/2
            ymax = y_edges[-1] + dy/2
            grid_params = {
                'ymin': ymin,
                'ymax': ymax,
                'dy': dy,
                'xmin': xmin,
                'xmax': xmax,
                'dx': dx
            }
            smap.interp_grid(frame.shape, MC_number=300,
                             grid_params=grid_params)
        else:
            smap.interp_grid(frame.shape, MC_number=0)

    if method.lower() == 'mc':  # Monte Carlo approach to the remap
        # Get the name of the transformation matrix to use
        name = smap._to_remap[0].name + '_' + smap._to_remap[1].name
        if mask is None:
            H = np.tensordot(smap._grid_interp['transformation_matrix'][name],
                             frame, 2)
        else:
            # Set to zero everything outside the mask
            dummy_frame = frame.copy()
            dummy_frame[~mask] = 0
            # Perform the tensor product as before
            H = np.tensordot(smap._grid_interp['transformation_matrix'][name],
                             dummy_frame, 2)

    else:  # similar to old IDL implementation, faster but noisy
        namex = smap._to_remap[0].name
        namey = smap._to_remap[1].name
        # --- 1: Information of the calibration
        # Get the phase variables at each pixel
        x = smap._grid_interp[namex].flatten()
        y = smap._grid_interp[namey].flatten()

        # --- 2: Remap (via histogram)
        if mask is None:
            z = frame.flatten()
        else:
            z = frame.copy()
            z[~mask] = 0
            z = z.flatten()
        H, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges],
                                           weights=z)
        # Normalise H to counts per unit of each axis
        delta_x = xedges[1] - xedges[0]
        delta_y = yedges[1] - yedges[0]
        H /= delta_x * delta_y

    return H


@deprecated('Deprecated! Please use integrate_remap from the video object')
def gyr_profile(remap_frame, pitch_centers, min_pitch: float,
                max_pitch: float, verbose: bool = False,
                name=None, gyr=None):
    """
    Cut the FILD signal to get a profile along gyroradius.

    DEPRECATED!!!! Please use the integrate_remap method from the video object

    @author:  Jose Rueda: jrrueda@us.es

    :param     remap_frame: np.array with the remapped frame
    :param     pitch_centers: np array produced by the remap function
    :param     min_pitch: minimum pitch to include
    :param     max_pitch: Maximum pitch to include
    :param     verbose: if true, the actual pitch interval will be printed
    :param     name: if given, the profile will be exported in ASCII format
    :param     gyr: the gyroradius values, to export
    :return   profile:  the profile in gyroradius
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


@deprecated('Deprecated! Please use integrate_remap from the video object')
def pitch_profile(remap_frame, gyr_centers, min_gyr: float,
                  max_gyr: float, verbose: bool = False,
                  name=None, pitch=None):
    """
    Cut the FILD signal to get a profile along pitch.

    DEPRECATED!!!! Please use the integrate_remap method from the video object


    @author:  Jose Rueda: jrrueda@us.es

    :param     remap_frame: np.array with the remapped frame
    @type:    ndarray

    :param     gyr_centers: np array produced by the remap function
    @type:    ndarray

    :param     min_gyr: minimum pitch to include
    @type:    float

    :param     max_gyr: Maximum pitch to include
    @type:    float

    :param     verbose: if true, the actual pitch interval will be printed
    @type:    bool

    :param     name: Full path to the file to export the profile. if present,
    file willbe written

    :param     pitch: array of pitches used in the remapped, only used if the
    export option is activated

    :return   profile:  pitch profile of the signal

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
