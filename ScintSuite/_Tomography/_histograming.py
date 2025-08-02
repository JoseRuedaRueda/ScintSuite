"""
Routines to create histograms
"""
import numpy as np
import numba as nb
from scipy.sparse import coo_array
from ScintSuite._Tomography._martix_collapse import collapse_array3D

# ------------------------------------------------------------------------------
# --- Fast 3D histogram
# ------------------------------------------------------------------------------
@nb.njit(cache=True, nogil=True)
def _fast_hist3d(x: float, y: float, z: float,
                 xgrid: float, ygrid: float, zgrid: float,
                 f: float, ):
    """
    Generates the 3D weighted histogram of the input data.

    Modified from the MEGA library, originally written by P. Oyola
    
    returns: the 3D matrix as a 1D sparse array, following the indexing criteria of:
    out[iz + nz * (iy + ny * ix)] = f[ix, iy, iz]
    
    this returns a coo array, ideal to construct sparse matrices. Do not use it directy to multiply stuff!
    """
    
    xmin = np.min(xgrid)
    ymin = np.min(ygrid)
    zmin = np.min(zgrid)

    nx = len(xgrid)
    ny = len(ygrid)
    nz = len(zgrid)

    # Computing the spacing.
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]

    fout = np.zeros((nx, ny, nz,))

    for ii, val in enumerate(f):
        # ia = int(min(nx-2, max(0, np.floor((x[ii] - xmin)/dx))))
        # ja = int(min(ny-2, max(0, np.floor((y[ii] - ymin)/dy))))
        # ka = int(min(nz-2, max(0, np.floor((z[ii] - zmin)/dz))))

        ia = int((x[ii] - xmin)/dx)
        if ia < 0 | ia >= nx-1:
            continue
        ja = int((y[ii] - ymin)/dy)
        if ja < 0 | ja >= ny-1:
            continue
        ka = int((z[ii] - zmin)/dz)
        if ka < 0 | ka >= nz-1:
            continue
        ia1 = ia+1
        ja1 = ja+1
        ka1 = ka+1

        ax1 = min(1.0, max(0.0, (x[ii] - xgrid[ia])/dx))
        ax = 1.0 - ax1
        ay1 = min(1.0, max(0.0, (y[ii] - ygrid[ja])/dy))
        ay = 1.0 - ay1
        az1 = min(1.0, max(0.0, (z[ii] - zgrid[ka])/dz))
        az = 1.0 - az1

        a000 = ax  * ay * az
        a100 = ax1 * ay * az
        a010 = ax  * ay1 * az
        a110 = ax1 * ay1 * az
        a001 = ax  * ay * az1
        a101 = ax1 * ay * az1
        a011 = ax  * ay1 * az1
        a111 = ax1 * ay1 * az1

        fout[ia, ja, ka]    += a000 * val 
        fout[ia1, ja, ka]   += a100 * val 
        fout[ia, ja1, ka]   += a010 * val 
        fout[ia1, ja1, ka]  += a110 * val
        fout[ia, ja, ka1]   += a001 * val
        fout[ia1, ja, ka1]  += a101 * val
        fout[ia, ja1, ka1]  += a011 * val
        fout[ia1, ja1, ka1] += a111 * val

    return fout


def fast_hist3d_sparse(x: float, y: float, z: float,
                 xgrid: float, ygrid: float, zgrid: float,
                 f: float, ):
    """
    Generates the 3D weighted histogram of the input data, returning a 1D sparse array.
    
    :param x: x coordinates of the data
    :param y: y coordinates of the data
    :param z: z coordinates of the data
    :param xgrid: x grid for the histogram
    :param ygrid: y grid for the histogram
    :param zgrid: z grid for the histogram
    :param f: weights for the histogram
    
    :return: 1D sparse array with the histogram data
    :Example:
    >>> x = np.random.rand(1000)
    >>> y = np.random.rand(1000)
    >>> z = np.random.rand(1000)
    >>> xgrid = np.linspace(0, 1, 10)
    >>> ygrid = np.linspace(0, 1, 10)
    >>> zgrid = np.linspace(0, 1, 10)
    >>> f = np.random.rand(1000)
    >>> hist = fast_hist3d_sparse(x, y, z, xgrid, ygrid, zgrid, f)
    >>> hist.shape  # (1000,)
    """
    W = _fast_hist3d(x, y, z, xgrid, ygrid, zgrid, f)

    return coo_array(collapse_array3D(W))       