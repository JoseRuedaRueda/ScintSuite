"""
Side auxiliar functions

Jose Rueda: jrrueda@us.es

Introduced in version 0.6.0
"""

import numpy as np
import xarray as xr
from scipy.signal import detrend as scipy_detrend
from typing import Optional, Union

# -----------------------------------------------------------------------------
# --- Dictionaries
# -----------------------------------------------------------------------------
def update_case_insensitive(a, b):
    """
    Update a dictionary avoiding problems due to case sensitivity

    Jose Rueda Rueda: jrrueda@us.es

    Note: This is a non-perfectly efficient workaround. Please do not use it
    routinely inside heavy loops. It will only change in a the fields contained
    in b, it will not create new fields in a

    Please, Pablo, do not kill me for this extremelly uneficient way of doing
    this

    :param s a: Main dictionary
    :param s b: Dictionary with the extra information to include in a
    """
    keys_a_lower = [key.lower() for key in a.keys()]
    keys_a = [key for key in a.keys()]
    keys_b_lower = [key.lower() for key in b.keys()]
    keys_b = [key for key in b.keys()]

    for k in keys_b_lower:
        if k in keys_a_lower:
            for i in range(len(keys_a_lower)):
                if keys_a_lower[i] == k:
                    keya = keys_a[i]
            for i in range(len(keys_b_lower)):
                if keys_b_lower[i] == k:
                    keyb = keys_b[i]
            a[keya] = b[keyb]


# -----------------------------------------------------------------------------
# --- Fitting
# -----------------------------------------------------------------------------
def line_fit_3D(x, y, z):
    """
    Fit a line to a 3D point cloud

    Jose Rueda: jrrueda@us.es

    extracted from:
    https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d

    Note, a regression is performed with OLS, so do not give a huge amount of
    points or your computer will be destroyed

    :param  x: array of x coordinates
    :param  y: array of y coordinates
    :param  z: array of z coordinates

    :return out: dict containing:
        -# v: the director vector
        -# p: the mean point of the dataset (point of the line)
    """
    data = np.concatenate((x[:, np.newaxis],
                           y[:, np.newaxis],
                           z[:, np.newaxis]),
                          axis=1)

    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = data.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.
    out = {
        'v': vv[0],
        'p': datamean
    }
    return out


# -----------------------------------------------------------------------------
# --- Grids
# -----------------------------------------------------------------------------
def createGrid(xmin: float, xmax: float, dx: float, ymin: float, ymax: float,
               dy: float):
    """
    Create the grid following the criteria stablished in the remap functions
    
    :param  ...

    @Todo: finish the documentation
    """
    nx, xedges = createGrid1D(xmin, xmax, dx)
    ny, yedges = createGrid1D(ymin, ymax, dy)

    return nx, ny, xedges, yedges


def createGrid1D(xmin: float, xmax: float, dx: float):
    """
    Create the grid following the criteria stablished in the remap functions

    :param  xmin: minimum grid value (center)
    :param  xmax: maximum grid value (center)
    :param  dx: grid spacing
    """
    nx = int((xmax - xmin) / dx) + 1
    xedges = xmin - dx/2 + np.arange(nx + 1) * dx

    return nx, xedges


# ------------------------------------------------------------------------------
# --- Filters
# ------------------------------------------------------------------------------
def running_mean(x, N):
    """

    Extracted from
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    :param x:
    :param N:
    :return:
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def gkern(l=int(4.5*6)+1, sig=4.5):
    """
    Create gaussian kernel with side length `l` and a sigma of `sig`

    Extracted from:
    https://stackoverflow.com/questions/29731726/
    how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def smooth(y, box_pts: int, mode: str = 'same', axis: int = -1):
    """Smooth signal by convolving with a box kernel.

    Args:
      y: ndarray or xarray.DataArray to smooth.
      box_pts: positive integer length of box filter.
      mode: convolution mode passed to np.convolve ('full','same','valid').
      axis: axis along which to apply smoothing (default last axis).

    Returns:
      Smoothed array of same type as input (xarray DataArray preserves coords/dims).
    """
    if box_pts is None:
        raise ValueError('box_pts must be provided')
    box_pts = int(box_pts)
    if box_pts < 1:
        raise ValueError('box_pts must be >= 1')
    if box_pts == 1:
        return y.copy() if isinstance(y, np.ndarray) else y.copy(deep=True)

    box = np.ones(box_pts, dtype=float) / box_pts

    is_xr = isinstance(y, xr.DataArray)
    arr = y.values if is_xr else np.asarray(y)

    if arr.ndim == 1:
        out = np.convolve(arr, box, mode=mode)
    else:
        out = np.apply_along_axis(lambda m: np.convolve(m, box, mode=mode), axis, arr)

    if is_xr:
        # Try to restore original shape (np.convolve 'same' returns same length)
        return xr.DataArray(out, dims=y.dims, coords=y.coords)
    return out

def detrend(x: xr.DataArray, type: str='linear',
            detrendSizeInterval: Optional[float]=0.001) -> xr.DataArray:
    """
    Detrend a signal

    :param x: signal to detrend, it must have a time dimmension with name 't' and units in the same of the detrendSizeInterval
    :param type: type of detrending, 'linear' or 'constant'
    :param detrendSizeInterval: size of the interval to detrend, in units of the time dimension of the signal.
    
    :return: detrended signal
    """
    if 't' not in x.dims:
        raise ValueError("The input signal must have a time dimension named 't'")
    # Get the detrned intertval size
    dt =(x['t'][1] - x['t'][0]).values
    npoints = int(detrendSizeInterval / dt)
    if npoints < 1:
        raise ValueError("The detrendSizeInterval must be larger than the time "
                         "interval of the signal")
    else:
        bp = np.arange(0, x.t.size, npoints)
    axis = x.dims.index('t')
    return xr.DataArray(
        scipy_detrend(x.values, type=type, bp=bp, axis=axis),
        dims=x.dims,
        coords=x.coords
    )


