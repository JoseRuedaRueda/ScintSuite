"""
SINPA side function for the SINPA geometry manipulation

Jose Rueda: jrrueda@us.es

Introduced in version 0.6.0
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from ScintSuite._Machine import machine
from ScintSuite._Paths import Path
paths = Path(machine)


def calculate_rotation_matrix(n, u1=None, verbose=True):
    """
    Calculate a rotation matrix to leave n as ux

    Jose Rueda: jrrueda@us.es

    It gives that rotation matrix, R, such that ux' = R @ n, where ux' is the
    unitaty vector in the x direction. If u1 is provided, the rotation will
    also fulfil uy = R @ u1. In this way the scintillator will finish properly
    aligned after the rotation

    :param  n: unit vector
    :param  u1: Unit vector normal to n

    :return M: Rotation matrix
    """
    # --- Check the unit vector
    modn = math.sqrt(np.sum(n * n))
    if abs(modn - 1) > 1e-5:
        print('The vector is not normalised, applying normalization')
        n /= modn
    if u1 is not None:
        perp = np.sum(u1 * n)
        if perp > 1e-3:
            raise Exception('U1 is not normal to n, revise inputs')
    # --- Check if the normal vector is already not in the x direction
    if abs(abs(n[0]) - 1) < 1e-5:
        print('The scintillator is already in a plane of constant x')
        return np.eye(3)  # Return just the identity matrix
    # --- Calculate the normal vector to perform the rotation
    ux = np.array([1.0, 0.0, 0.0])
    u_turn = np.cross(n, ux)
    u_turn /= math.sqrt(np.sum(u_turn * u_turn))
    # --- Calculate the proper angle
    alpha = math.acos(n[0])
    if verbose:
        print('The rotation angle is:', alpha)
        # print('Please write in fortran this matrix but transposed!!')
    # --- Calculate the rotation
    r = R.from_rotvec(alpha * u_turn)
    rot1 = r.as_matrix()
    # --- Now perform the second rotation to orientate the scintillator
    if u1 is not None:
        uy = np.array([0.0, 1.0, 0.0])
        # Get u1 in the second system of coordinates
        u1_new = r.apply(u1)
        u_turn = np.cross(u1_new, uy)
        u_turn /= math.sqrt(np.sum(u_turn * u_turn))
        # --- Calculate the proper angle
        alpha = math.acos(u1_new[1])
        r = R.from_rotvec(alpha * u_turn)
        rot2 = r.as_matrix()
    else:
        rot2 = np.eye(3)
    return rot2 @ rot1, r
