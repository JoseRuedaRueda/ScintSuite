"""
Common auxiliary routines for the FILD and INPA mapping.

Jose Rueda Rueda - jrrueda@us.es

Introduced in version 0.6.0
"""
import logging
import datetime
import numpy as np
import ScintSuite._CustomFitModels as models
import math
from ScintSuite.decorators import deprecated
from ScintSuite._Mapping._Calibration import CalParams
from scipy.interpolate import griddata
from lmfit import Parameters
from lmfit import Model
from scipy import special
logger = logging.getLogger('ScintSuite.MappingCommon')
try:
    import lmfit
except (ImportError, ModuleNotFoundError):
    logger.warning('10: You cannot calculate resolutions')
__all__ = ['transform_to_pixel', 'XYtoPixel', '_fit_to_model_',
           'remap', 'gyr_profile', 'pitch_profile',
           'estimate_effective_pixel_area']


# -----------------------------------------------------------------------------
# ---- Real position to pixels
# -----------------------------------------------------------------------------
def transform_to_pixel(x: np.ndarray, y: np.ndarray, cal: CalParams):
    """
    Transform from X,Y coordinates (scintillator) to pixels in the camera.
    
    

    Parameters
    ----------
    x : numpy.ndarray
        Array of positions to be transformed, x coordinate.
    y : numpy.ndarray
        Array of positions to be transformed, y coordinate.
    cal : CalParams
        Object containing all the information for the transformation.

    Returns
    -------
    xpixel : numpy.ndarray
        x positions in pixels.
    ypixel : numpy.ndarray
        y positions in pixels.

    Examples
    --------
    >>> cal = ss.mapping.CalParams()
    >>> cal.xscale = cal.yscale = 27.0
    >>> cal.xshift = cal.yshift = 0.0
    >>> cal.deg = 25.0
    >>> x = np.array([35, 45, 22, 105])
    >>> y = np.array([15, 35, 27, 106])
    >>> xp, yp = ss.mapping.transform_to_pixel(x, y, cal)
    
    :Authors:
        Jose Rueda - jruedaru@uci.edu
    """
    if cal.type.lower() != 'homomorphism' and cal.type.lower() != 'homomorphic':
        eps = 1e-7  # Level for the distortion coefficient to be considered as
        #             zero (see below)
        # Perform the undistorted transformation
        alpha = cal.deg * np.pi / 180
        if cal.type.lower() != 'scint':
            xpixel = (np.cos(alpha) * x - np.sin(alpha) * y) * cal.xscale + \
                cal.xshift
            ypixel = (np.sin(alpha) * x + np.cos(alpha) * y) * cal.yscale + \
                cal.yshift
        else:
            xpixel = (np.cos(alpha) * x * cal.xscale 
                      - np.sin(alpha) * y * cal.yscale) + \
                cal.xshift
            ypixel = (np.sin(alpha) * x * cal.xscale 
                      + np.cos(alpha) * y * cal.yscale ) + \
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
                if cal.c1 > 0.0:
                    rp_limit = 1.0/2.0/np.sqrt(cal.c1)
                    rp_copy = rp.copy()
                    flags = rp_copy >= rp_limit
                    rp_copy[flags] = rp_copy
                else:
                    # When k < 0, there is no way the determinant goes to negative.
                    rp_copy = rp.copy()
                d = (1-np.sqrt(1-4*cal.c1*rp**2))/(2*cal.c1*rp)
                xpixel = xp*(1+cal.c1*d**2) + cal.xcenter
                ypixel = yp*(1+cal.c1*d**2) + cal.ycenter
    else:
        logger.debug('Applying homomorphism transformation')
        coords = np.array([x,y,np.ones_like(x)])
        coords_pixel = cal.H @ coords
        xpixel = coords_pixel[0, :].flatten()
        ypixel = coords_pixel[1, :].flatten()
    return xpixel, ypixel


class XYtoPixel:
    """
    Parent class for object with both coordinates in real and camera space.

    For example for Scintillator and strike-maps, which contain information of
    their coordinates in the real space and of their coordinates in the camera
    (pixel) space.

    It is not intended to be initialised directly by the user. The StrikeMap, or
    scintillator objects will do it. Pease use those child classes.

    Jose Rueda: jrrueda@us.es

    Introduced in version 0.8.2


    Attributes
    ----------
    _coord_real : dict
        Coordinates of the vertex of the scintillator (X,Y,Z).
    _normal_vector : numpy.ndarray or None
        Normal vector to the plane containing the element.
    _coord_pix : dict
        Coordinates of the vertex of the scintillator in pixels.
    CameraCalibration : CalParams or None
        Camera calibration object.
        
        
    :Authors:
    Jose Rueda - jruedaru@uci.edu    
    """

    def __init__(self):
        """
        Initialise the object.

        Notice that this is just a parent class, each child (scintillator or
        strike map), will fill its contents as needed.
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

    def calculate_pixel_coordinates(self, cal) -> None:
        """
        Transform real coordinates into pixel coordinates.

        Parameters
        ----------
        cal : CalParams
            Camera calibration to apply.

        Returns
        -------
        None
        """
        self._coord_pix['x'], self._coord_pix['y'] = \
            transform_to_pixel(self._coord_real['x1'], self._coord_real['x2'],
                               cal)
        self.CameraCalibration = cal


def estimate_effective_pixel_area(frame_shape, xscale: float, yscale: float,
                                  type: int = 0):
    """
    Estimate the effective area covered by a pixel.

    Parameters
    ----------
    frame_shape : tuple
        Shape of the frame.
    xscale : float
        The scale [#pixel/cm] of the calibration to align the map.
    yscale : float
        The scale [#pixel/cm] of the calibration to align the map.
    type : int, optional
        0 to ignore distortion, 1 to include distortion (not implemented).

    Returns
    -------
    area : numpy.ndarray
        Matrix where each element is the area covered by that pixel.

    Notes
    -----
    Now that the default calibrations are in m^-1, the 1e-4 factor is used to convert to m^2.
    
        
    :Authors:
        Jose Rueda - jruedaru@uci.edu
    """
    # Initialise the matrix:
    area = np.zeros(frame_shape)

    if type == 0:
        area[:] = abs(1./(xscale*yscale)*1.e-4)  # 1e-4 to be in m^2

    return area


# -----------------------------------------------------------------------------
# ---- Fitting functions
# -----------------------------------------------------------------------------
def _fit_to_model_(data, bins: int = 20, model: str = 'Gauss',
                   normalize: bool = True,
                   confidence_level: float = 0.9544997,
                   uncertainties: bool = True):
    """
    Make histogram of input data and fit to a model.

    Parameters
    ----------
    data : array-like
        Input data to be histogrammed and fitted.
    bins : int or array-like, optional
        Desired number of bins or the edges.
    model : str, optional
        Model to fit ('Gauss', 'sGauss', 'raisedCosine', 'WignerSemicircle').
    normalize : bool, optional
        Flag to normalise the number of counts in the bins between 0 and 1.
    confidence_level : float, optional
        Confidence level for the uncertainty determination.
    uncertainties : bool, optional
        Flag to calculate the uncertainties of the fit.

    Returns
    -------
    par : dict
        Dictionary containing the fit parameters.
    result : lmfit.model.ModelResult
        The lmfit model object with the results.
    normalization : float
        The used normalization for the histogram.
    unc_output : dict
        The width of the confidence interval for each parameter.
        
        
        
    :Authors:
        Jose Rueda - jruedaru@uci.edu    
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
    if model.lower() == 'gauss' or model.lower() == 'gaussian':
        model = lmfit.models.GaussianModel()
        params = model.guess(hist, x=cent)
    elif model.lower() == 'sgauss' or model.lower() == 'skewedgaussian':
        model = lmfit.models.SkewedGaussianModel()
        params = model.guess(hist, x=cent)
    elif model.lower() == 'raisedcosine': #reaised cosine model for the energy
        model = models.RaisedCosine()
        # estimate the parameters using a quick fit to the skewed Gaussian
        params = model.make_params()
        SG = lmfit.models.SkewedGaussianModel()
        paramsSG = SG.guess(hist, x=cent)
        result = SG.fit(hist, paramsSG, x=cent)
        params['sigma'] = result.params['sigma']
        params['amplitude'] = result.params['amplitude']
        params['center'] = result.params['center']
        params['beta'].set(value=result.params['gamma']/result.params['sigma'], min=0.0,)
    elif model.lower() == 'wignersemicircle': #wigner semicircle model
        model = models.WignerSemicircle()
        # estimate the parameters using a quick fit to a Gaussian
        Gauss = lmfit.models.GaussianModel()
        paramsGauss = Gauss.guess(hist, x=cent)
        result = Gauss.fit(hist, paramsGauss, x=cent)
        params = model.make_params()
        params['amplitude'] = result.params['amplitude']
        params['center'] = result.params['center']
        params['sigma'] = result.params['sigma']        
    else:
        mods = ['Gauss', 'sGauss', 'raisedCosine', 'WignerSemicircle']
        logger.error('Only models: ' + ', '.join(mods) + ' are implemented')
        raise ValueError('Model not implemented')

    # Fit
    result = model.fit(hist, params, x=cent)
    par = {}

    for key in model.param_names:
        par[key] = result.params[key].value
  
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
# ---- Remap and profiles
# -----------------------------------------------------------------------------
def remap(smap, frame, x_edges=None, y_edges=None, mask=None, method='MC'):
    """
    Remap a frame.

    Parameters
    ----------
    smap : StrikeMap
        StrikeMap object with the strike map.
    frame : numpy.ndarray
        The frame to be remapped.
    x_edges : array-like, optional
        Edges of the x coordinate, for FILD, pitch [º].
    y_edges : array-like, optional
        Edges of the y coordinate, for FILD, gyroradius [cm].
    mask : numpy.ndarray, optional
        Mask to apply to the frame.
    method : str, optional
        Procedure for the remap ('MC', 'centers', 'griddata', 'forward_warping_simple', 'forward_warping_advanced').

    Returns
    -------
    H : numpy.ndarray
        Remapped frame.

    Notes
    -----
    Edges are only needed if you select the centers method, if not, they will be inside the transformation matrix already.
    The different modules and video objects will call this method internally.
    
        
    :Authors:
        Jose Rueda - jruedaru@uci.edu
        Antonin J. van Vuuren - anton.jansenvanvuuren@epfl.ch
    """
    # --- 0: Check inputs
    if smap._grid_interp is None:
        text = '27: Interpolators not present in the strike map'\
               + ', calculating them with default settings'
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
    if mask is not None and method.lower() == 'griddata':
        raise NotImplementedError('Sorry, not implemented')
    if method.lower() == 'mc':  # Monte Carlo approach to the remap
        # Get the name of the transformation matrix to use
        name = smap._to_remap[0].name + '_' + smap._to_remap[1].name
        if mask is None:
            H = np.tensordot(smap._grid_interp['transformation_matrix'][name],
                             frame.astype(float), 2)
        else:
            # Set to zero everything outside the mask
            dummy_frame = frame.copy()
            dummy_frame[~mask] = 0
            # Perform the tensor product as before
            H = np.tensordot(smap._grid_interp['transformation_matrix'][name],
                             dummy_frame, 2)
    elif method.lower() == 'griddata': # grid data interpolation
        raise NotImplementedError("This method was deprecated in ScintSuite 1.4.0")
        logger.warning('This method does not conserve the signal integral. Avoid it')
        namex = smap._to_remap[0].name
        namey = smap._to_remap[1].name
        # --- 1: Information of the calibration
        # Get the phase variables at each pixel
        x = smap._grid_interp[namex].flatten()
        y = smap._grid_interp[namey].flatten()

        idx_isnotnan = ~np.isnan(x)  #added by AJVV
        x = x[idx_isnotnan]
        y = y[idx_isnotnan]

        dummy = np.column_stack((x, y))
        # --- 2: Remap
        xcenter = 0.5 * (x_edges[1:] + x_edges[:-1])
        ycenter = 0.5 * (y_edges[1:] + y_edges[:-1])
        XX, YY = np.meshgrid(xcenter, ycenter, indexing='ij')
        if mask is None:
            z = frame.flatten().astype(float)
        else:
            z = frame.copy().astype(float)
            z[~mask] = 0
            z = z.flatten()

        z = z[idx_isnotnan]

        H = griddata(dummy, z, (XX.flatten(), YY.flatten()),
                     method='linear', fill_value=0).reshape(XX.shape)
        # Normalise H to counts per unit of each axis
        delta_x = xcenter[1] - xcenter[0]
        delta_y = ycenter[1] - ycenter[0]
        H /= delta_x * delta_y
        
    elif method.lower() == 'forward_warping_simple': # should produce smoother histogram
        '''
        #Use the grid iterpolators to translate each pixel to a phase-space value.
        #Spread the counts of a given pixel proportionally to the four closest phase space coordinates.

        #Note current implemetnation ignores points mapped to the edge of the phase-space grid
        #The assumption is the strike map usually fully covers the signal of interest
        #and this was easier to implement.

        @TODO: vectorize final step of population the phase-space image array "H".
        '''
        #phase space coordinates
        namex = smap._to_remap[0].name
        namey = smap._to_remap[1].name
        x = smap._grid_interp[namex].flatten()
        y = smap._grid_interp[namey].flatten()
        #Remove nans (since interpolation fill value was set to nan, this could be different in other branches)
        idx_isnotnan = ~np.isnan(x)
        x = x[idx_isnotnan]
        y = y[idx_isnotnan]
        z = frame.copy().astype(float)
        z = z.flatten()
        z = z[idx_isnotnan]

        xcenter = 0.5 * (x_edges[1:] + x_edges[:-1])
        ycenter = 0.5 * (y_edges[1:] + y_edges[:-1])
        delta_x = xcenter[1] - xcenter[0]
        delta_y = ycenter[1] - ycenter[0]

        #Build phase space image, similar to histogram method
        H = np.zeros((len(xcenter), len(ycenter)))

        #for ip in np.arange(x.shape[0]):  ##Left here from the original implementation (since vectorized)
        x_ip = x
        y_ip = y

        #Find where each pixel values' phase space values would fit in the defined phase space grid.
        x_index = np.searchsorted( xcenter, x_ip, side = 'right')
        y_index = np.searchsorted( ycenter, y_ip, side = 'right')

        ###Now remove edge cases
        idx_x_left_edge = np.where(x_index==0)[0]
        idx_x_right_edge = np.where(x_index==len(xcenter))[0]
        x_index[idx_x_left_edge]=1
        x_index[idx_x_right_edge]=len(xcenter)-1

        idx_y_left_edge = np.where(y_index==0)[0]
        idx_y_right_edge = np.where(y_index==len(ycenter))[0]
        y_index[idx_y_left_edge]=1
        y_index[idx_y_right_edge]=len(ycenter)-1

        z[idx_x_left_edge] = 0
        z[idx_x_right_edge] = 0
        z[idx_y_left_edge] = 0
        z[idx_y_right_edge] = 0
        ### ed of edge case removal

        ## Calculate the distance for a remaped value to the cloest grid points
        # Determine the vertices of the grid cell
        x0, y0 = xcenter[x_index-1], ycenter[y_index-1]
        x1, y1 = xcenter[x_index], ycenter[y_index]
            
        # Calculate the distances from the point to the sides of the cell
        dx0 = x_ip - x0
        dx1 = x1 - x_ip
        dy0 = y_ip - y0
        dy1 = y1 - y_ip
            
        # Calculate the area of the rectangles formed by these distances
        # This is done in order to proportionally spread the counts 
        area_total = delta_x * delta_y
        w_bottom_left = (dx1 * dy1) / area_total
        w_bottom_right = (dx0 * dy1) / area_total
        w_top_left = (dx1 * dy0) / area_total
        w_top_right = (dx0 * dy0) / area_total

        # Iterate over all remaped values and populate the phase space grid
        # This could propably be vectorised to be faster.
        for ip in np.arange(x.shape[0]):
            H[x_index[ip]-1, y_index[ip]-1] +=  z[ip] * w_bottom_left[ip]
            H[x_index[ip] , y_index[ip]-1] += z[ip] * w_bottom_right[ip]
            H[x_index[ip] , y_index[ip] ] += z[ip] * w_top_right[ip]
            H[x_index[ip]-1, y_index[ip] ] += z[ip] * w_top_left[ip]

        H /= delta_x * delta_y

    elif method.lower() == 'forward_warping_advanced': # should produce smoother histogram
        '''
        ##This method mixes forward and backwards mapping.
        The idea is to sample forward (as in the "simple" implmentation) , but then to also sample backwards.
        Meaning sample for each phase-space grid coordinate a point in the pixel space.
        The advantage, theoretically, is that each phase-space grid point will necessarily be assigned a value from pixel space.

        In practice however I found that he simple implemetation is sufficient, but I leave this implemenation
        in case there are cases where it works better. I leave this to the user.

        This implemetnation does not correctly treat edge cases. 
        '''
        # Get the phase variables names
        namex = smap._to_remap[0].name
        namey = smap._to_remap[1].name

        #Define the forward mapping function. Based on the smap interpolators
        def f(xs):

            ### xs of shape (n_points, 2)

            fx = smap._grid_interp['interpolators'][namex]
            fy = smap._grid_interp['interpolators'][namey]

            ys = np.column_stack((
                                    fx(xs[:, 0], xs[:, 1]),
                                    fy(xs[:, 0], xs[:, 1])
                                 ))

            return ys  ### ys of shape (n_points, 2)

        #Now also define the backward mapping function. Essentially the inverse function of "f".
        def f_inverse(ys):
            import scipy.interpolate as scipy_interp
            interpolator = scipy_interp.LinearNDInterpolator
            
            inverse_grid = list(zip(smap._data[namex].data,
                            smap._data[namey].data))
            
            _grid_interp_x = interpolator(inverse_grid, smap._coord_pix['x'],
                                 fill_value=np.nan #1000.0  #AJVV
                                 )

            _grid_interp_y = interpolator(inverse_grid, smap._coord_pix['y'],
                                 fill_value=np.nan #1000.0  #AJVV
                                 )

            xs = np.column_stack((
                                    _grid_interp_x(ys[:, 0], ys[:, 1]),
                                    _grid_interp_y(ys[:, 0], ys[:, 1])
                                 ))
            
            return xs

        #Map all pixel values ("xs") forward to phase space coordinates "ys"
        pix_grid_x, pix_grid_y = np.mgrid[0:frame.shape[1], 0:frame.shape[0]]
        xs = np.column_stack((pix_grid_x.flatten(), pix_grid_y.flatten()))
        ys = f(xs)

        #Remove nans (since the interpolator fill value is nan)
        idx_isnotnan = ~np.isnan(ys)
        xs = xs[idx_isnotnan[:, 0], :]
        ys = ys[idx_isnotnan[:, 0], :]

        #Define the phase space grid. This is based on user inputs for grid cell sizes.
        xcenter = 0.5 * (x_edges[1:] + x_edges[:-1])
        ycenter = 0.5 * (y_edges[1:] + y_edges[:-1])
        XX, YY = np.meshgrid(xcenter, ycenter, indexing='ij')

        # "ys_" is all phase space grid points to be mapped backwards to pixel space. 
        # in other words sample the phase-space grid ("ys_") in pixel space
        ys_ = np.column_stack((XX.flatten(), YY.flatten()))
        xs_ = f_inverse(ys_) ## "x_s" is the backwards mapped phase space points
        idx_isnotnan = ~np.isnan(xs_)
        #remove nans
        xs_ = xs_[idx_isnotnan[:, 0], :]
        ys_ = ys_[idx_isnotnan[:, 0], :]
        
        '''
        This section might be needed to deal with edge cases.
        z = frame.copy().astype(float)
        z = z.flatten()
        z = z[idx_isnotnan[:, 0]]
        '''

        ## "H" will be the mapped phase-space image (from the pixel image)
        H = np.zeros(XX.shape)
        delta_x = x_edges[1] - x_edges[0]
        delta_y = y_edges[1] - y_edges[0]

        #Combine the forward and backward samples
        ys = np.concatenate((ys, ys_))
        xs = np.concatenate((xs, xs_))
        xs = np.floor(xs).astype('int32')
        
        '''
        A very important difference from plain forward mapping is that since we sample the pixel space twice,
        once going forwad where each pixel is sampled once,
        and again going backwards, where the a given pixel could be sampled multiple times.
        Hence when we build the phase space image, to maintain the pixel image weight, we need to consider
        how many times a pixel is sampled.

        Todo: deal with edge mapped values.
              Vectorize final step of building 2D grid space "image"
        '''
        ###
        ###Determine how many times a pixel is sampled
        ###
        dtype = np.dtype((np.void, xs.dtype.itemsize * xs.shape[1]))
        structured_arr = np.ascontiguousarray(xs).view(dtype)

        # Use numpy.unique to find unique rows and their counts
        unique, counts = np.unique(structured_arr, return_counts=True)

        # Convert unique rows back to original shape
        unique = unique.view(xs.dtype).reshape(-1, xs.shape[1])

        # Combine unique rows with their counts
        result = np.column_stack((unique, counts))
        ##the first columns of result are the uniqe pixel pairs, 
        # and the third column give the number of time that pair is samples

        ##Hence create a weighted frame adjusted by the number of times a pixel is sampled
        frame_weight_corrected = np.zeros(np.shape(frame))
        frame_weight_corrected[result[:, 1], result[:, 0]] = frame[result[:, 1], result[:, 0]] /result[:, 2]
        ### end of determing times a pixel is sampled


        ##Now start with building the phase space image
        #for ip in np.arange(x.shape[0]): ##(from old implimentation)
        x_ip = ys[:, 0]
        y_ip = ys[:, 1]

        x_index = np.searchsorted( xcenter, x_ip, side = 'right')
        y_index = np.searchsorted( ycenter, y_ip, side = 'right')

        idx_x_left_edge = np.where(x_index==0)[0]
        idx_x_right_edge = np.where(x_index==len(xcenter))[0]
        x_index[idx_x_left_edge]=1
        x_index[idx_x_right_edge]=len(xcenter)-1

        idx_y_left_edge = np.where(y_index==0)[0]
        idx_y_right_edge = np.where(y_index==len(ycenter))[0]
        y_index[idx_y_left_edge]=1
        y_index[idx_y_right_edge]=len(ycenter)-1

        '''
        This implenetation needs to be adapted to correctly deal with edge sample cases
        z[idx_x_left_edge] = 0
        z[idx_x_right_edge] = 0
        z[idx_y_left_edge] = 0
        z[idx_y_right_edge] = 0
        '''
        ## Calculate the distance for a remaped value to the cloest grid points
        # Determine the vertices of the grid cell
        x0, y0 = xcenter[x_index-1], ycenter[y_index-1]
        x1, y1 = xcenter[x_index], ycenter[y_index]
            
        # Calculate the distances from the point to the sides of the cell
        dx0 = x_ip - x0
        dx1 = x1 - x_ip
        dy0 = y_ip - y0
        dy1 = y1 - y_ip
            
        # Calculate the area of the rectangles formed by these distances
        area_total = delta_x * delta_y
        w_bottom_left = (dx1 * dy1) / area_total
        w_bottom_right = (dx0 * dy1) / area_total
        w_top_left = (dx1 * dy0) / area_total
        w_top_right = (dx0 * dy0) / area_total

        for ip in np.arange(ys.shape[0]):
            H[x_index[ip]-1, y_index[ip]-1] +=  frame_weight_corrected[xs[ip, 1], xs[ip, 0]]  * w_bottom_left[ip]
            H[x_index[ip] , y_index[ip]-1] += frame_weight_corrected[xs[ip, 1], xs[ip, 0]]  * w_bottom_right[ip]
            H[x_index[ip] , y_index[ip] ] += frame_weight_corrected[xs[ip, 1], xs[ip, 0]]  * w_top_right[ip]
            H[x_index[ip]-1, y_index[ip] ] += frame_weight_corrected[xs[ip, 1], xs[ip, 0]]  * w_top_left[ip]

        H /= delta_x * delta_y

    else:  # similar to old IDL implementation, faster but noisy
        namex = smap._to_remap[0].name
        namey = smap._to_remap[1].name
        # --- 1: Information of the calibration
        # Get the phase variables at each pixel
        x = smap._grid_interp[namex].flatten()
        y = smap._grid_interp[namey].flatten()

        # --- 2: Remap (via histogram)
        if mask is None:
            z = frame.flatten().astype(float)
        else:
            z = frame.copy().astype(float)
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

    Parameters
    ----------
    remap_frame : numpy.ndarray
        Remapped frame.
    pitch_centers : numpy.ndarray
        Array produced by the remap function.
    min_pitch : float
        Minimum pitch to include.
    max_pitch : float
        Maximum pitch to include.
    verbose : bool, optional
        If True, print the actual pitch interval.
    name : str, optional
        If given, the profile will be exported in ASCII format.
    gyr : numpy.ndarray, optional
        The gyroradius values, to export.

    Returns
    -------
    profile : numpy.ndarray
        The profile in gyroradius.

    Raises
    ------
    Exception
        If the desired pitch range is not in the frame.
        
    :Authors:
        Jose Rueda - jruedaru@uci.edu
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

    Parameters
    ----------
    remap_frame : numpy.ndarray
        Remapped frame.
    gyr_centers : numpy.ndarray
        Array produced by the remap function.
    min_gyr : float
        Minimum gyroradius to include.
    max_gyr : float
        Maximum gyroradius to include.
    verbose : bool, optional
        If True, print the actual pitch interval.
    name : str, optional
        Full path to the file to export the profile. If present, file will be written.
    pitch : numpy.ndarray, optional
        Array of pitches used in the remapped, only used if the export option is activated.

    Returns
    -------
    profile : numpy.ndarray
        Pitch profile of the signal.

    Raises
    ------
    Exception
        If the desired gyroradius range is not in the frame.
        
    :Authors:
        Jose Rueda - jruedaru@uci.edu    
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
