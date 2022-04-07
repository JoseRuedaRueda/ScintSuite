"""
Utilities library for the iHIBPsim code.

Contains generic routines to be used across the synthetic diagnostic.
"""

import numpy as np
import numba as nb

#-----------------------------------------------
#  Histogram utilities.
# -----------------------------------------------
@nb.njit(cache=True, nogil=True)
def _fast_hist1d(x: float, f: float, w: float, nx:int=31):
    """
    Generates the 1D weighted histogram of the input data.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """

    xmin = np.min(x)
    xmax = np.max(x)

    xout = np.linspace(xmin, xmax, nx)

    # Computing the spacing.
    dx = xout[1] - xout[0]

    fout = np.zeros((nx, ))
    norm = np.zeros((nx, ))
    Nout = np.zeros((nx, ))

    for ii, val in enumerate(f):
        ia = int(min(nx-2, max(0, np.floor((x[ii] - xmin)/dx))))
        ia1 = ia+1

        ax = min(1.0, max(0.0, (x[ii] - x[ia])/dx))
        ax1 = 1.0 - ax

        fout[ia]   += ax  * val * w[ii]
        fout[ia1]  += ax1 * val * w[ii]

        norm[ia]   += ax * w[ii]
        norm[ia1]  += ax1* w[ii]

        Nout[ia]   += ax
        Nout[ia1]  += ax1

    return xout, fout, Nout, norm



def hist1d(x: float, weight: float=None,
           bins: int=None, vmin: float=None, vmax: float=None,
           density: bool=False, retall: bool=False, norm_count: bool=False):
    """
    Computes the 1D weighted histogram using the cloud in cell approach with a
    linear interpolant that acts, effectively, as smoothing of the KDE.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param x: values of the coordinate to analyze.
    @param weight: weights for each of the input values. If None, weights are
    assumed to be 1, i.e., unweighted histogram.
    @param bins: number of bins to use to discretize the X-axis. By default, 51
    @param vmin: minimum value of the x-axis. If None, the smallest value of
    x will be used instead.
    @param vmax: maximum value of the x-axis. If None, the largest value of
    x will be used instead.
    """
    x = np.array(x)

    if weight is not None:
        w = np.array(weight).copy()
    else:
        w = np.ones(x.shape)

    if vmin is None:
        vmin = (np.nanmin(x),)
    if vmax is None:
        vmax = (np.nanmax(x),)

    if bins is None:
        bins = np.array((51,), dtype=int)
    else:
        bins = np.array(bins)
        if bins.size == 1:
            bins = np.ones((1,), dtype=int)*bins
        else:
            if bins[0] is None:
                bins[0] = int(51)


    flags = (x >= vmin[0]*0.99) & (x <= vmax[0]*1.01)
    xcopy = x[flags]
    wcopy = w[flags]

    xgrid, histW, histN, _ = _fast_hist1d(x=xcopy, f=np.ones(xcopy.shape),
                                          w=wcopy, nx=bins[0])

    Norm = np.sum(histW)

    if density:
        histN /= Norm

    if retall:
        return xgrid, histW, histN
    else:
        return xgrid, histW

@nb.njit(cache=True, nogil=True)
def _fast_hist2d(x: float, y: float, f: float, w: float,
                 nx:int=31, ny:int=51):
    """
    Generates the 3D weighted histogram of the input data.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    xout = np.linspace(xmin, xmax, nx)
    yout = np.linspace(ymin, ymax, ny)

    # Computing the spacing.
    dx = xout[1] - xout[0]
    dy = yout[1] - yout[0]

    fout = np.zeros((nx, ny,))
    norm = np.zeros((nx, ny,))
    Nout = np.zeros((nx, ny,))

    for ii, val in enumerate(f):
        ia = int(min(nx-2, max(0, np.floor((x[ii] - xmin)/dx))))
        ja = int(min(ny-2, max(0, np.floor((y[ii] - ymin)/dy))))

        ia1 = ia+1
        ja1 = ja+1

        ax = min(1.0, max(0.0, (x[ii] - x[ia])/dx))
        ax1 = 1.0 - ax
        ay = min(1.0, max(0.0, (y[ii] - y[ja])/dy))
        ay1 = 1.0 - ay

        a00 = ax  * ay
        a10 = ax1 * ay
        a01 = ax  * ay1
        a11 = ax1 * ay1

        fout[ia, ja]   += a00 * val * w[ii]
        fout[ia1, ja]  += a10 * val * w[ii]
        fout[ia, ja1]  += a01 * val * w[ii]
        fout[ia1, ja1] += a11 * val * w[ii]

        norm[ia, ja]   += a00 * w[ii]
        norm[ia1, ja]  += a10 * w[ii]
        norm[ia, ja1]  += a01 * w[ii]
        norm[ia1, ja1] += a11 * w[ii]

        Nout[ia, ja]   += a00
        Nout[ia1, ja]  += a10
        Nout[ia, ja1]  += a01
        Nout[ia1, ja1] += a11

    return xout, yout, fout, Nout, norm

def hist2d(x: float, y: float, weight: float=None,
           bins: int=None, vmin: float=None, vmax: float=None,
           density: bool=False, retall: bool=False, norm_count: bool=False):
    """
    Computes the 2D weighted histogram using the cloud in cell approach with a
    linear interpolant that acts, effectively, as smoothing of the KDE.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param x: values of the 1st coordinate to analyze.
    @param x: values of the 2nd coordinate to analyze.
    @param weight: weights for each of the input values. If None, weights are
    assumed to be 1, i.e., unweighted histogram.
    @param bins: number of bins to use to discretize the X-axis. By default, 51
    @param vmin: minimum value of the x-axis. If None, the smallest value of
    x will be used instead.
    @param vmax: maximum value of the x-axis. If None, the largest value of
    x will be used instead.
    """
    x = np.array(x)
    y = np.array(y)

    if weight is not None:
        w = np.array(weight).copy()
    else:
        w = np.ones(x.shape)

    if vmin is None:
        vmin = (np.nanmin(x), np.nanmin(y))
    if vmax is None:
        vmax = (np.nanmax(x), np.nanmax(y))

    if bins is None:
        bins = np.array((51, 21), dtype=int)
    else:
        bins = np.array(bins)
        if bins.size == 1:
            bins = np.ones((2,), dtype=int)*bins
        else:
            if bins[0] is None:
                bins[0] = int(51)
            if bins[1] is None:
                bins[1] = int(21)


    flags = (x < vmin[0]) | (x > vmax[0]) | (y < vmin[1]) | (y > vmax[1])
    w[flags] = 0.0

    xgrid, ygrid, histW, histN, _ = _fast_hist2d(x=x, y=y, f=np.ones(x.shape),
                                                 w=w, nx=bins[0], ny=bins[1])

    Norm = np.sum(histW)

    if density:
        histN /= Norm

    if retall:
        return xgrid, ygrid, histW, histN
    else:
        return xgrid, ygrid, histW


@nb.njit(cache=True, nogil=True)
def _fast_hist3d(x: float, y: float, z: float, f: float, w: float,
                 nx:int=31, ny:int=51, nz:int=21):
    """
    Generates the 3D weighted histogram of the input data.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)

    xout = np.linspace(xmin, xmax, nx)
    yout = np.linspace(ymin, ymax, ny)
    zout = np.linspace(zmin, zmax, nz)

    # Computing the spacing.
    dx = xout[1] - xout[0]
    dy = yout[1] - yout[0]
    dz = zout[1] - zout[0]

    fout = np.zeros((nx, ny, nz,))
    norm = np.zeros((nx, ny, nz,))
    Nout = np.zeros((nx, ny, nz,))

    for ii, val in enumerate(f):
        ia = int(min(nx-2, max(0, np.floor((x[ii] - xmin)/dx))))
        ja = int(min(ny-2, max(0, np.floor((y[ii] - ymin)/dy))))
        ka = int(min(nz-2, max(0, np.floor((z[ii] - zmin)/dz))))

        ia1 = ia+1
        ja1 = ja+1
        ka1 = ka+1

        ax = min(1.0, max(0.0, (x[ii] - x[ia])/dx))
        ax1 = 1.0 - ax
        ay = min(1.0, max(0.0, (y[ii] - y[ja])/dy))
        ay1 = 1.0 - ay
        az = min(1.0, max(0.0, (z[ii] - z[ka])/dz))
        az1 = 1.0 - az

        a000 = ax  * ay * az
        a100 = ax1 * ay * az
        a010 = ax  * ay1 * az
        a110 = ax1 * ay1 * az
        a001 = ax  * ay * az1
        a101 = ax1 * ay * az1
        a011 = ax  * ay1 * az1
        a111 = ax1 * ay1 * az1

        fout[ia, ja, ka]    += a000 * val * w[ii]
        fout[ia1, ja, ka]   += a100 * val * w[ii]
        fout[ia, ja1, ka]   += a010 * val * w[ii]
        fout[ia1, ja1, ka]  += a110 * val * w[ii]
        fout[ia, ja, ka1]   += a001 * val * w[ii]
        fout[ia1, ja, ka1]  += a101 * val * w[ii]
        fout[ia, ja1, ka1]  += a011 * val * w[ii]
        fout[ia1, ja1, ka1] += a111 * val * w[ii]


        norm[ia, ja, ka]    += a000 * w[ii]
        norm[ia1, ja, ka]   += a100 * w[ii]
        norm[ia, ja1, ka]   += a010 * w[ii]
        norm[ia1, ja1, ka]  += a110 * w[ii]
        norm[ia, ja, ka1]   += a001 * w[ii]
        norm[ia1, ja, ka1]  += a101 * w[ii]
        norm[ia, ja1, ka1]  += a011 * w[ii]
        norm[ia1, ja1, ka1] += a111 * w[ii]

        Nout[ia, ja, ka]    += a000
        Nout[ia1, ja, ka]   += a100
        Nout[ia, ja1, ka]   += a010
        Nout[ia1, ja1, ka]  += a110
        Nout[ia, ja, ka1]   += a001
        Nout[ia1, ja, ka1]  += a101
        Nout[ia, ja1, ka1]  += a011
        Nout[ia1, ja1, ka1] += a111

    return xout, yout, zout, fout, Nout, norm

def hist3d(x: float, y: float, z: float, weight: float=None,
           bins: int=None, vmin: float=None, vmax: float=None,
           density: bool=False, retall: bool=False, norm_count: bool=False):
    """
    Computes the 2D weighted histogram using the cloud in cell approach with a
    linear interpolant that acts, effectively, as smoothing of the KDE.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param x: values of the 1st coordinate to analyze.
    @param x: values of the 2nd coordinate to analyze.
    @param weight: weights for each of the input values. If None, weights are
    assumed to be 1, i.e., unweighted histogram.
    @param bins: number of bins to use to discretize the X-axis. By default, 51
    @param vmin: minimum value of the x-axis. If None, the smallest value of
    x will be used instead.
    @param vmax: maximum value of the x-axis. If None, the largest value of
    x will be used instead.
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    if weight is not None:
        w = np.array(weight).copy()
    else:
        w = np.ones(x.shape)

    if vmin is None:
        vmin = (np.nanmin(x), np.nanmin(y), np.nanmin(z))
    if vmax is None:
        vmax = (np.nanmax(x), np.nanmax(y), np.nanmax(z))

    if bins is None:
        bins = np.array((51, 21, 10), dtype=int)
    else:
        bins = np.array(bins)
        if bins.size == 1:
            bins = np.ones((3,), dtype=int)*bins
        else:
            if bins[0] is None:
                bins[0] = int(51)
            if bins[1] is None:
                bins[1] = int(21)
            if bins[2] is None:
                bins[2] = int(10)


    flags = (x < vmin[0]) | (x > vmax[0]) | (y < vmin[1]) | (y > vmax[1]) | \
            (z < vmin[2]) | (z > vmax[2])
    w[flags] = 0.0

    xgrid, ygrid, zgrid, histW, histN, _ = _fast_hist3d(x=x, y=y, z=z,
                                                        f=np.ones(x.shape),
                                                        w=w, nx=bins[0],
                                                        ny=bins[1],
                                                        nz=bins[2])

    Norm = np.sum(histW)

    if density:
        histN /= Norm

    if retall:
        return xgrid, ygrid, zgrid, histW, histN
    else:
        return xgrid, ygrid, zgrid, histW