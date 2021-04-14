"""
Auxiliar functions for the INPASIM code

Jose Rueda: jrrueda@us.es
"""

import numpy as np
# import matplotlib.pyplot as plt


def line_fit_3D(x, y, z):
    """
    Fit a line to a 3D point cloud

    Jose Rueda: jrrueda@us.es

    extracted from:
    https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d

    Note, a regression is performed with OLS, so do not give a huge amount of
    points or your computer will be destroyed

    @param x: array of x coordinates
    @param y: array of y coordinates
    @param z: array of z coordinates

    @return out: dict containing:
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
