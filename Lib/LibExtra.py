"""
Useful models and auxiliary functions

Here I placed routines useful for the analysis but which in principle are not
designed to be inserted in the main suite. For example a routine which given
the pitch a particle have in a given radial location, calculates the pitch
which will have at the FILD position
"""

import numpy as np
from LibMachine import machine
if machine == 'AUG':
    import LibDataAUG as ssdat


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


def TP_boundary(shot, z0, t, Rmin=1.5, Rmax=2.1, zmin=-0.9, zmax=0.9):
    """
    Approximate the TP TP_boundary

    jose rueda: jrrueda@us.es

    |pitch_TP|= sqrt(Rmin/R) where R min is the minimum R of the flux
    surface which pases by R

    @param shot: shot number
    @param z: height where to calculate the boundary
    """
    # --- Creatin of the grid
    r = np.linspace(Rmin, Rmax, int((Rmax - Rmin) * 100))
    z = np.linspace(zmin, zmax, int((zmax - zmin) * 100))

    R, Z = np.meshgrid(r, z)

    # --- Get rho
    rho = ssdat.get_rho(shot, R.flatten(), Z.flatten(), time=t)
    rho = np.reshape(rho, (z.size, r.size))
    # --- Do the gross approximation
    tp = np.zeros(r.size)
    iz = np.argmin(abs(z - z0))
    for i in range(r.size):
        # - Get the rho there
        rho1 = rho[iz, i]
        mask = abs(rho - rho1) < 0.01
        rr = R[mask]
        r0 = rr.min()
        print(r[i], r0)
        tp[i] = np.sqrt(1 - r0 / r[i])
    return r, tp
