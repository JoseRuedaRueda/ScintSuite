"""
Useful models and auxiliary functions

Here I placed routines useful for the analysis but which in principle are not
designed to be inserted in the main suite. For example a routine which given
the pitch a particle have in a given radial location, calculates the pitch
which will have at the FILD position
"""

import numpy as np


def pitch_at_other_place(R0, P0, R):
    """
    Given the pitch a a place R0, calculate the pitch at a location R

    Asumpion: Magnetic moment is conserved, B goes like 1/R

    Note: The conservation of magnetic moment has a v**2 so no info can be
    extracted on the pitch sign with this quick calculation

    Note 2: If P0 and R0 are arrays and R is a point, the result will be an
    array with the component i being the pitch at R of the ion which pitch at
    R0[i] was P0[i]. If P0 and R0 are single values and R an array, the
    component i of the output array is the pitch at the position R[i]. If all
    inputs are arrays... dragons can appear
    @param R0: the initial position
    @param P0: The initial pitch (as +- vpar/v)
    @return P: The pitch evaluated at that position
    """
    return np.sqrt(1 - R0 / R * (1 - P0 ** 2))
