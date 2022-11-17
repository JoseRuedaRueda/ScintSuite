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
    logger.wargning('10: You cannot use tomography')

# ------------------------------------------------------------------------------
# --- Matrix handling
# ------------------------------------------------------------------------------
@njit(nogil=True, parallel=True)
def collapse_array4D(W: float):
    """
    Auxiliary function to reshape the 4D array (W) into 2D

    Pablo Oyola: poyola@us.es and Jose Rueda: jrrueda@us.es

    :param  W: 4D matrix

    :return W2D: Condensed 2D matrix
    """
    ns = W.shape[0] * W.shape[1]
    npin = W.shape[2] * W.shape[3]
    W2D = np.zeros((ns, npin))
    for irp in prange(W.shape[3]):
        for ipp in range(W.shape[2]):
            for irs in range(W.shape[1]):
                for ips in range(W.shape[0]):
                    W2D[irs + W.shape[1] * ips,
                        irp + W.shape[3] * ipp] = \
                        W[ips, irs, ipp, irp]

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
def collapse_array2D(s: float):
    """
    Auxiliary function to reshape the 2D array (signal) into 1D

    Pablo Oyola: poyola@us.es and Jose Rueda: jrrueda@us.es

    :param  s: 2D matrix

    :return s1d: Condensed 1D matrix
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
    """
    s2D = np.zeros((nx, ny))
    for irs in prange(ny):
        for ips in range(nx):
            s2D[ips, irs] = s1d[irs + ny * ips]
    return s2D