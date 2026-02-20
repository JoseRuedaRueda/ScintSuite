"""
Methods to collapse matrices into 2 and 1D

Pablo Oyola: poyola@us.es and Jose Rueda: jrrueda@us.es
"""
import logging
import numpy as np

logger = logging.getLogger('ScintSuite.Tomography')
try:
    from numba import njit, prange
except:
    from ScintSuite.decorators import false_njit as njit
    prange = range
    logger.warning('10: Matrix collapses will be pretty slow')

# ------------------------------------------------------------------------------
# --- Matrix handling
# ------------------------------------------------------------------------------
@njit(nogil=True, parallel=True)
def collapse_array5D(W: float):
    """
    Auxiliary function to reshape the 5D array (W) into 2D

    Jose Rueda: jrrueda@us.es

    :param  W: 5D matrix

    :return W2D: Condensed 2D matrix

    :Example:
    >>> W = np.random.rand(3, 4, 5, 6, 7)
    >>> W2D = collapse_array5D(W)
    >>> W2D.shape  #(12, 210)
    """
    nx = W.shape[0]
    ny = W.shape[1]
    nalpha = W.shape[2]
    nbeta = W.shape[3]
    ngamma = W.shape[4]
    nxy = nx * ny
    ngreek = nalpha * nbeta * ngamma
    W2D = np.zeros((nxy, ngreek))
    for igamma in prange(ngamma):
        for ibeta in range(nbeta):
            for ialpha in range(nalpha):
                for iy in range(ny):
                    for ix in range(nx):
                        W2D[iy + ny * ix,
                            igamma + ngamma * (ibeta + nbeta * ialpha)] = \
                            W[ix, iy, ialpha, ibeta, igamma]
    return W2D


@njit(nogil=True, parallel=True)
def restore_array5D(W: float, nx: int, ny: int, nalpha: int, nbeta: int, ngamma: int):
    """
    Auxiliary function to reshape the 2D array (W) into 5D

    Jose Rueda:
    
    :param  W: 2D matrix
    :param  nx: Number of elements in the x direction
    :param  ny: Number of elements in the y direction
    :param  nalpha: Number of elements in the alpha direction
    :param  nbeta: Number of elements in the beta direction
    :param  ngamma: Number of elements in the gamma direction
    
    :return W5D: restored 5D matrix
    
    :Example:
    >>> W = np.random.rand(12, 210)
    >>> W5D = restore_array5D(W, 3, 4, 5, 6, 7)
    >>> W5D.shape  #(3, 4, 5, 6, 7)
    """
    W5D = np.zeros((nx, ny, nalpha, nbeta, ngamma))
    for igamma in prange(ngamma):
        for ibeta in range(nbeta):
            for ialpha in range(nalpha):
                for iy in range(ny):
                    for ix in range(nx):
                        W5D[ix, iy, ialpha, ibeta, igamma] = \
                            W[iy + ny * ix, igamma + ngamma * (ibeta + nbeta * ialpha)]
    return W5D


@njit(nogil=True, parallel=True)
def collapse_array4D(W: float):
    """
    Auxiliary function to reshape the 4D array (W) into 2D

    Pablo Oyola: poyola@us.es and Jose Rueda: jrrueda@us.es

    :param  W: 4D matrix

    :return W2D: Condensed 2D matrix

    :Example:
    >>> W = np.random.rand(3, 4, 5, 6)
    >>> W2D = collapse_array4D(W)
    >>> W2D.shape  #(12, 30)
    """
    ns = W.shape[0] * W.shape[1]
    npin = W.shape[2] * W.shape[3]
    W2D = np.zeros((ns, npin))
    for iy2 in prange(W.shape[3]):
        for ix2 in range(W.shape[2]):
            for iy in range(W.shape[1]):
                for ix in range(W.shape[0]):
                    W2D[iy + W.shape[1] * ix,
                        iy2 + W.shape[3] * ix2] = \
                        W[ix, iy, ix2, iy2]

    # for irp in prange(W.shape[3]):
    #     for ipp in range(W.shape[2]):
    #         for irs in range(W.shape[1]):
    #             for ips in range(W.shape[0]):
    #                 W2D[irs * W.shape[0] + ips,
    #                     irp * W.shape[2] + ipp] = \
    #                     W[ips, irs, ipp, irp]

    return W2D


@njit(nogil=True, parallel=True)
def restore_array4D(W: float, n0: int, n1: int, n2: int, n3: int):
    """
    Auxiliary function to reshape the 2D array (W) into 4D

    Pablo Oyola: poyola@us.es and Jose Rueda: jrrueda@us.es

    :param  W: 2D matrix

    :return W4D: restored 4D matrix

    :Example:
    >>> W = np.random.rand(12, 30)
    >>> W4D = restore_array4D(W, 3, 4, 5, 6)
    >>> W4D.shape  #(3, 4, 5, 6)
    """
    W4D = np.zeros((n0, n1, n2, n3))
    for irp in prange(n3):
        for ipp in range(n2):
            for irs in range(n1):
                for ips in range(n0):
                    W4D[ips, irs, ipp, irp] = \
                        W[irs + n1 * ips, irp + n3 * ipp]

    return W4D


@njit(nogil=True, parallel=True)
def collapse_array3D(W: float):
    """
    Auxiliary function to reshape the 3D array (W) into 1D

    Jose Rueda: jruedaru@uci.edu
    
    :param  W: 3D matrix
    
    :Example:
    >>> W = np.random.rand(3, 4, 5)
    >>> f = collapse_array3D(W)
    >>> f.shape  #(60)
    """
    nx = W.shape[0]
    ny = W.shape[1]
    nz = W.shape[2]
    n = nx * ny * nz
    f = np.zeros(n)
    for iz in prange(nz):
        for iy in range(ny):
            for ix in range(nx):
                f[iz + nz * (iy + ny * ix)] = W[ix, iy, iz]
    return f

def index3Dto1D(ix: int, iy: int, iz: int, ny: int, nz: int) -> int:
    """
    Convert 3D indices to 1D index

    Jose Rueda:
    :param ix: x index
    :param iy: y index
    :param iz: z index
    :param nx: number of elements in the x direction
    :param ny: number of elements in the y direction
    :return: 1D index
    :Example:
    >>> index3Dto1D(1, 2, 3, 4, 5) 
    # returns iz + nz * (iy + ny * ix)
    # returns 3 + 5 * (2 + 4 * 1) = 3 + 10 = 13  
    """
    return iz + nz * (iy + ny * ix)


@njit(nogil=True, parallel=True)
def restore_array3D(s1d: float, nx: int, ny:int, nz:int):
    """
    Auxiliary function to reshape the 1D array (F) into 3D

    Jose Rueda: jrrueda@us.es

    :param  s: 1D matrix

    :return s2D: Condensed 3D matrix

    :Example:
    >>> s1d = np.random.rand(60)
    >>> s2D = restore_array2D(s1d, 3, 4, 5)
    >>> s2D.shape  #(3, 4, 5)
    """
    f3d = np.zeros((nx, ny, nz))
    for iz in prange(nz):
        for iy in range(ny):
            for ix in range(nx):
                f3d[ix, iy, iz] = s1d[iz +nz*(iy + ny * ix)]
    return f3d


@njit(nogil=True, parallel=True)
def collapse_array2D(s: float):
    """
    Auxiliary function to reshape the 2D array (signal) into 1D

    Pablo Oyola: poyola@us.es and Jose Rueda: jrrueda@us.es

    :param  s: 2D matrix

    :return s1d: Condensed 1D matrix

    :Example:
    >>> s = np.random.rand(3, 4)
    >>> s1d = collapse_array2D(s)
    >>> s1d.shape  #(12,)
    """
    ns = s.size
    s1d = np.zeros(ns)
    for irs in prange(s.shape[1]):
        for ips in range(s.shape[0]):
            s1d[irs + s.shape[1] * ips] = s[ips, irs]
    return s1d


@njit(nogil=True, parallel=True)
def restore_array2D(s1d: float, nx: int, ny:int):
    """
    Auxiliary function to reshape the 1D array (S) into 2D

    Pablo Oyola: poyola@us.es and Jose Rueda: jrrueda@us.es

    :param  s: 1D matrix

    :return s2D: Condensed 2D matrix

    :Example:
    >>> s1d = np.random.rand(12)
    >>> s2D = restore_array2D(s1d, 3, 4)
    >>> s2D.shape  #(3, 4)
    """
    s2D = np.zeros((nx, ny))
    for iy in prange(ny):
        for ix in range(nx):
            s2D[ix, iy] = s1d[iy + ny * ix]
    return s2D