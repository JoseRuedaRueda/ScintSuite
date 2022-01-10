"""
Side auxiliar functions

Jose Rueda: jrrueda@us.es

Introduced in version 0.6.0
"""

import numpy as np


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

    @params a: Main dictionary
    @params b: Dictionary with the extra information to include in a
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
