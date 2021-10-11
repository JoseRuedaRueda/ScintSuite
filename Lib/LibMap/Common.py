"""
Common objects for the FILD and INPA mapping

Jose Rueda Rueda - jrrueda@us.es

Introduced in version 0.6.0

"""
import numpy as np
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


# ------------------------------------------------------------------------------
# --- Fitting functions
# ------------------------------------------------------------------------------
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
