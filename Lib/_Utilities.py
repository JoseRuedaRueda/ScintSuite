"""
Useful models and auxiliary functions

Here I placed routines useful for the analysis but which in principle are not
designed to be inserted in the main suite. For example a routine which given
the pitch a particle have in a given radial location, calculates the pitch
which will have at the FILD position
"""
import math
import numpy as np
import xarray as xr
import Lib.LibData as ssdat
import warnings
import logging
logger = logging.getLogger('ScintSuite.Utilities')
try:
    from shapely.geometry import LineString
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings('ignore',
                            category=ShapelyDeprecationWarning)
except ModuleNotFoundError:
    logger.warning('10: Shapely not found, you cannot calculate intersections')
except ImportError:
    logger.warning('12: Old version of shapely, but things should work')
try:
    from numba import njit, prange
except:
    logger.warning('10: You cannot use neutron filters')


# -----------------------------------------------------------------------------
# --- Pitch methods:
# -----------------------------------------------------------------------------
def pitch2pitch(P0: float, def_in: int = 0, def_out: int = 2)->np.ndarray:
    """
    Transform between the different pitch definitions

    Jose Rueda: jrrueda@us.es

    id list:
        -# 0: Pitch in degrees, co-current
        -# 1: Pitch in degrees, counter-current
        -# 2: Pitch as v_par / v. co-current
        -# 3: Pitch as v_par / v. cunter-current
    :param  P0: The arrays containing the pitches we want to transform
    :param  def_in: The id of the inputs pitches
    :param  def_out: the id of the ouput pitches
    :return P1: Pitch in the other definition
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


def pitch_at_other_place(R0, P0, R)->np.ndarray:
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
    :param  R0: the initial position
    :param  P0: The initial pitch (as +- vpar/v)
    :return P: The pitch evaluated at that position
    """
    return np.sqrt(1 - R0 / R * (1 - P0 ** 2))


# -----------------------------------------------------------------------------
# --- Matrix filters
# -----------------------------------------------------------------------------
def neutron_filter(M: np.ndarray, nsigma: int = 3)->np.ndarray:
    """
    Remove pixels affected by neutrons

    Jose Rueda: jrrueda@us.es

    :param  M: Matrix with the counts to be filtered (np.array)
    :return Mo: Matrix filtered
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

@njit(nogil=True, parallel=True)
def neutronAndDeadFilter(M: float, nsigma: float = 3.0, dead: bool = True):
    """
    Still too low, need something faster
    :param M:
    :param nsigma_neutrons:
    :param n_sigma_dead:
    :return:
    """
    nx, ny, nt = M.shape
    new_matrix = np.zeros((nx, ny, nt))
    for it in prange(nt):
        frame = M[:, :, it].copy()
        for ix in range(nsigma, nx-nsigma):
            for iy in range(nsigma, ny - nsigma):
                mean = frame[(ix-nsigma):(ix+nsigma),
                             (iy-nsigma):(iy+nsigma)].mean()
                std = frame[(ix-nsigma):(ix+nsigma),
                            (iy-nsigma):(iy+nsigma)].std()
                if frame[ix, iy] > mean + nsigma * std:
                    new_matrix[ix, iy, it] = mean
                elif frame[ix, iy] < mean - nsigma * std and dead:
                    new_matrix[ix, iy, it] = mean
                else:
                    new_matrix[ix, iy, it] = M[ix, iy, it]
        print(it)
    return new_matrix


# -----------------------------------------------------------------------------
# --- Trapped passing boundary
# -----------------------------------------------------------------------------
def TP_boundary(shot, z0, t, Rmin=1.5, Rmax=2.1, 
                zmin=-0.9, zmax=0.9)->xr.DataArray:
    """
    Approximate the TP TP_boundary

    jose rueda: jrrueda@us.es

    |pitch_TP|= sqrt(Rmin/R) where R min is the minimum R of the flux
    surface which pases by R

    :param  shot: shot number
    :param  z: height where to calculate the boundary
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
        tp[i] = np.sqrt(1 - r0 / r[i])
    output = xr.DataArray(tp, dims='R', coords={'R': r})
    output.attrs['long_name'] = '$\\lambda_{b}$'
    output.attrs['Description'] = '|pitch_TP|= sqrt(Rmin/R)'

    return output


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
def find_2D_intersection(x1, y1, x2, y2):
    """
    Get the intersection between curve (x1, y1) and (x2, y2)

    Jose Rueda: jrrueda@us.es

    Note: the precision will be the distance between the points of the array
    (x2, y2)

    taken from:
    https://stackoverflow.com/questions/28766692/
    intersection-of-two-graphs-in-python-find-the-x-value

    :param  x1: x coordinate of the first curve, np.array
    :param  y1: y coordinate of the first curve, np.array
    :param  x2. x coordinate of the second curve, np.array
    :param  y2: y coordinate of the second curve, np.array

    :return x: x coordinates of the intersection
    :return y: y coordinates of the intersection

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

    :param  s: np array with the signal
    :param  s_timebase: np array with the timebase of the signal
    :param  tELM: the dictionary created by the ss.dat.get_ELM_timebase
    :param  offset: offset to be included in the onset of the elms
    """
    time = tELM['t_onset'] + offset
    dt = tELM['dt']
    flags = np.ones(len(s_timebase), dtype=bool)

    for i in range(tELM['n']):
        tmp_flags = (s_timebase >= time[i]) * (s_timebase <= (time[i] + dt[i]))
        flags *= ~tmp_flags
    return s[flags], s_timebase[flags], flags


# -----------------------------------------------------------------------------
# --- Matrix manipulation
# -----------------------------------------------------------------------------
def distmat(a: np.ndarray, index: tuple)->np.ndarray:
    """
    Get index distance to a given point

    Rerturn a matrix of shape a.shape which values are the distances, in index
    length, to the point given by the tupple index

    taken from:
        https://stackoverflow.com/questions/61628380/calculate-distance-
        from-all-points-in-numpy-array-to-a-single-point-on-the-basis
    """
    i, j = np.indices(a.shape)
    return np.sqrt((i-index[0])**2 + (j-index[1])**2)
