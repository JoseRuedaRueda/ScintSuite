"""
SINPA geometry

Jose Rueda: jrrueda@us.es

Introduced in version 0.6.0
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_rotation_matrix(n, verbose=True):
    """
    Calculate a rotation matrix to leave n as ux

    Jose Rueda: jrrueda@us.es

    It gives that rotation matrix, R, such that ux' = R @ n, where ux' is the
    unitaty vector in the x direction

    @param n: unit vector

    @return M: Rotation matrix
    """
    # --- Check the unit vector
    modn = math.sqrt(np.sum(n * n))
    if abs(modn - 1) > 1e-2:
        print('The vector is not normalised, applying normalization')
        n /= modn
    # --- Calculate the normal vector to perform the rotation
    ux = np.array([1.0, 0.0, 0.0])
    u_turn = np.cross(n, ux)
    u_turn /= math.sqrt(np.sum(u_turn * u_turn))
    # --- Calculate the proper angle
    alpha = math.acos(-n[0])
    if verbose:
        print('The rotation angle is:', alpha)
        print('Please write in fortran this matrix but transposed!!')
    # --- Calculate the rotation
    r = R.from_rotvec(alpha * u_turn)
    return r.as_matrix()
