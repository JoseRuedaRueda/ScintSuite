"""
Useful models and auxiliary functions

Here I placed routines useful for the analysis but which in principle are not
designed to be inserted in the main suite. For example a routine which given
the pitch a particle have in a given radial location, calculates the pitch
which will have at the FILD position
"""

import numpy as np
import math
import Lib.LibData as ssdat
try:
    from shapely.geometry import LineString
except ModuleNotFoundError:
    print('Shapely not found, you cannot calculate intersections')


# -----------------------------------------------------------------------------
# --- Pitch methods:
# -----------------------------------------------------------------------------
def pitch2pitch(P0, def_in: int = 0, def_out: int = 2):
    """
    Transform between the different pitch definitions

    Jose Rueda: jrrueda@us.es

    id list:
        -# 0: Pitch in degrees, co-current
        -# 1: Pitch in degrees, counter-current
        -# 2: Pitch as v_par / v. co-current
        -# 3: Pitch as v_par / v. cunter-current
    @param P0: The arrays containing the pitches we want to transform
    @param def_in: The id of the inputs pitches
    @param def_out: the id of the ouput pitches
    @return P1: Pitch in the other definition
    """
    if def_in == def_out:
        P1 = P0

    if def_in == 0:    # Input as co-current degrees
        if def_out == 1:
            P1 = P0 + 180.0
        elif def_out == 2:
            P1 = np.cos(P0)
        elif def_out == 3:
            P1 = -np.cos(P0)
    elif def_in == 1:  # Input as counter-current degrees
        if def_out == 0:
            P1 = P0 - 180.0
        elif def_out == 2:
            P1 = -np.cos(P0)
        elif def_out == 3:
            P1 = np.cos(P0)
    elif def_in == 2:  # Input as co-current
        if def_out == 0:
            P1 = np.arccos(P0) * 180. / np.pi
        elif def_out == 1:
            P1 = np.arccos(P0) * 180. / np.pi + 180.
        elif def_out == 3:
            P1 = -P0
    elif def_in == 3:  # Input as co-current
        if def_out == 0:
            P1 = np.arccos(P0) * 180. / np.pi - 180.
        elif def_out == 1:
            P1 = np.arccos(P0) * 180. / np.pi
        elif def_out == 2:
            P1 = -P0
    return P1


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


# -----------------------------------------------------------------------------
# --- Matrix filters
# -----------------------------------------------------------------------------
def neutron_filter(M, nsigma: int = 3):
    """
    Remove pixels affected by neutrons

    Jose Rueda: jrrueda@us.es

    @param M: Matrix with the counts to be filtered (np.array)
    @return Mo: Matrix filtered
    """
    Mo = M.copy()
    sx, sy = M.shape
    for ix in range(1, sx-1):
        for iy in range(1, sy-1):
            dummy = M[(ix-1):(ix+1), (iy-1):(iy+1)].copy()
            dummy[1, 1] = 0
            mean = np.mean(dummy)
            std = np.std(dummy)
            if Mo[ix, iy] > mean + nsigma * std:
                Mo[ix, iy] = mean

    return Mo


# -----------------------------------------------------------------------------
# --- Trapped passing boundary
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# --- Searching algorithms
# -----------------------------------------------------------------------------
def find_nearest_sorted(array, value):
    """
    Find the nearest element of an sorted array

    Taken from:
    https://stackoverflow.com/questions/2566412/
        find-nearest-value-in-numpy-array
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1])
                    < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


# -----------------------------------------------------------------------------
# --- Intersections
# -----------------------------------------------------------------------------
def find_2D_intersection(x1, y1, x2, y2, plot=False):
    """
    Get the intersection between curve (x1, y1) and (x2, y2)

    Jose Rueda: jrrueda@us.es

    Note: the precision will be the distance between the points of the array
    (x2, y2)

    taken from:
    https://stackoverflow.com/questions/28766692/
    intersection-of-two-graphs-in-python-find-the-x-value

    @param x1: x coordinate of the first curve, np.array
    @param y1: y coordinate of the first curve, np.array
    @param x2. x coordinate of the second curve, np.array
    @param y2: y coordinate of the second curve, np.array
    @param plot. If no intersection is found, plot the situation

    @return x: x coordinates of the intersection
    @return y: y coordinates of the intersection

    """
    first_line = LineString(np.column_stack((x1.flatten(), y1.flatten())))
    second_line = LineString(np.column_stack((x2.flatten(), y2.flatten())))
    intersection = first_line.intersection(second_line)
    try:  # just one point
        return intersection.xy
    except AssertionError:
        return LineString(intersection).xy
    except NotImplementedError:
        return LineString(intersection).xy
    except AttributeError:
        print('No intersection found')
        return None, None


# -----------------------------------------------------------------------------
# --- ELM filtering
# -----------------------------------------------------------------------------
def ELM_filter(s, s_timebase, tELM, offset=0.):
    """
    Remove time points affected by ELMs

    Jose Rueda: jrrueda@us.es

    @param s: np array with the signal
    @param s_timebase: np array with the timebase of the signal
    @param tELM: the dictionary created by the ss.dat.get_ELM_timebase
    @param offset: offset to be included in the onset of the elms
    """
    time = tELM['t_onset'] + offset
    dt = tELM['dt']
    flags = np.ones(len(s_timebase), dtype=np.bool)

    for i in range(tELM['n']):
        tmp_flags = (s_timebase >= time[i]) * (s_timebase <= (time[i] + dt[i]))
        flags *= ~tmp_flags
    return s[flags], s_timebase[flags], flags


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

    Please Pablo do not kill me for this extremelly uneficient way of doing
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
