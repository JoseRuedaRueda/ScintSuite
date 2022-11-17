"""Load the machine vessel"""
import os
import math
import get_gc            # Module to load vessel components
import aug_sfutils as sfutils
import numpy as np
import xarray as xr
import Lib.LibData.AUG.Equilibrium as equil
import Lib.LibData.AUG.DiagParam as params
import matplotlib.pyplot as plt
import Lib._Plotting as ssplt
import Lib._Utilities as ssextra
import Lib.errors as errors
from Lib._Paths import Path

import logging
logger = logging.logger('ScintSuite.Data')

try:
    import numba
except ModuleNotFoundError:
   REAL_NBI_GEOM = False
   logger.warning('Numba is not installed. Requirement for NBI geometry in AUG.')
else:
    import Lib.LibData.AUG._nbi_geom as nbigeom
    REAL_NBI_GEOM = True

pa = Path()


# -----------------------------------------------------------------------------
# --- Vessel
# -----------------------------------------------------------------------------
def poloidal_vessel(shot: int = 30585, simplified: bool = False):
    """
    Get coordinate of the poloidal projection of the vessel

    Jose Rueda: jrrueda@us.es

    :param  shot: shot number to be used
    :param  simplified: if true, a 'basic' shape of the poloidal vessel will be
    loaded, ideal for generate a 3D revolution surface from it
    """
    if simplified is not True:
        r = []
        z = []
        # Get vessel coordinates
        gc_r, gc_z = get_gc.get_gc(shot)
        for key in gc_r.keys():
            # print(key)
            r += list(gc_r[key][:])
            r.append(np.nan)
            z += list(gc_z[key][:])
            z.append(np.nan)
        return np.array((r, z)).transpose()
    else:
        file = os.path.join(pa.ScintSuite, 'Data', 'Vessel', 'AUG_pol.txt')
        return np.loadtxt(file, skiprows=4)


def toroidal_vessel(rot: float = -np.pi/8.0*3.0):
    """
    Return the coordinates of the AUG vessel

    Jose Rueda Rueda: ruejo@ipp.mpg.de

    Note: x = NaN indicate the separation between vessel block

    :param  rot: angle to rotate the coordinate system
    :return xy: np.array with the coordinates of the points [npoints, 2]
    """
    # --- Section 0: Read the data
    # The files are a series of 'blocks' representing each piece of the vessel,
    # each block is separated by an empty line. I will scan the file line by
    # line looking for the position of those empty lines:
    file = os.path.join(pa.ScintSuite, 'Data', 'Vessel', 'AUG_tor.txt')
    cc = 0
    nmax = 2000
    xy_vessel = np.zeros((nmax, 2))
    with open(file) as f:
        # read the comment block
        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
        # read the vessel components:
        for i in range(nmax):
            line = f.readline()
            if line == '\n':
                xy_vessel[cc, 0] = np.nan
            elif line == '':
                break
            else:
                dummy = line.split()
                xx = np.float(dummy[0])
                yy = np.float(dummy[1])
                xy_vessel[cc, 0] = xx * np.cos(rot) - yy * np.sin(rot)
                xy_vessel[cc, 1] = xx * np.sin(rot) + yy * np.cos(rot)
            cc += 1
    return xy_vessel[:cc-1, :]


# -----------------------------------------------------------------------------
# --- NBI coordinates
# -----------------------------------------------------------------------------
def NBI_diaggeom_coordinates(nnbi):
    """
    Just the coordinates manually extracted for shot 32312

    :param  nnbi: the NBI number
    :return coords: dictionary containing the coordinates of the initial and
    final points. '0' are near the source, '1' are near the central column
    """
    # --- Diaggeom parameters
    r0 = np.array([2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6])
    r1 = np.array([1.046, 1.046, 1.046, 1.046, 1.048, 2.04, 2.04, 1.048])

    z0 = np.array([0.022, 0.021, -0.021, -0.022,
                   -0.019, -0.149, 0.149, 0.019])
    z1 = np.array([-0.12, -0.145, 0.145, 0.12, -0.180, -0.6, 0.6, 0.180])

    phi0 = np.array([-32.725, -31.88, -31.88, -32.725,
                     145.58, 148.21, 148.21, 145.58]) * np.pi / 180.0
    phi1 = np.array([-13.81, 10.07, 10.07, -13.81,
                     -180.0, -99.43, -99.43, -180.0]) * np.pi / 180.0

    # --- Cartesian coordinates
    x0 = r0 * np.cos(phi0)
    x1 = r1 * np.cos(phi1)

    y0 = r0 * np.sin(phi0)
    y1 = r1 * np.sin(phi1)

    # --- Distance final-point source
    length = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)

    # --- Tangency radius:
    u = np.array((x1[nnbi-1] - x0[nnbi-1], y1[nnbi-1] - y0[nnbi-1],
                  z1[nnbi-1] - z0[nnbi-1]))
    u /= math.sqrt(np.sum(u ** 2))
    p0 = np.array((x0[nnbi-1], y0[nnbi-1], z0[nnbi-1]))
    t = - (p0[1] * u[1] + p0[0] * u[0]) / (u[0]**2 + u[1]**2)
    xt = p0[0] + u[0] * t
    yt = p0[1] + u[1] * t
    zt = p0[2] + u[2] * t
    rt = np.sqrt(xt**2 + yt**2)

    coords = {'phi0': phi0[nnbi-1], 'phi1': phi1[nnbi-1],
              'r0': r0[nnbi-1], 'r1': r1[nnbi-1],
              'x0': x0[nnbi-1], 'y0': y0[nnbi-1],
              'z0': z0[nnbi-1], 'x1': x1[nnbi-1],
              'y1': y1[nnbi-1], 'z1': z1[nnbi-1],
              'u': u,   # director vector
              'rt': rt,  # tangency radius
              'xt': xt, 'yt': yt, 'zt': zt,  # tangency point
              'length': length[nnbi-1]}
    return coords


def getNBIwindow(timeWindow: float, shotnumber: int,
                 nbion: int, nbioff: int = None,
                 simul: bool = True, pthreshold: float = 2.0):
    """
    Get the time window within the limits provide within the timeWindow that
    corresponds to the list nbiON that are turned on and the list nbioff.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  timeWindow: window of time to retrieve the NBI data (len=2).
    :param  shotnumber: Shot number from where to take the NBI timetraces.
    :param  nbion: list with the NBI number that should be ON.
    :param  nbioff: list with the NBIs that should be OFF.
    :param  simul: simultaneous flag. If True all the NBIs of nbion should be
    ON simultaenously.
    :param  pthreshold: power threshold to consider the beam is ON [MW].
    Default to 2.0 MW (to choose the 2.5MW standard beam.)
    """
    # --- Checking the time inputs.
    if len(timeWindow) == 1:
        timeWindow = np.array((timeWindow, np.inf))

    elif np.mod(len(timeWindow), 2) != 0:
        timeWindow[len(timeWindow)] = np.inf

    # Transforming the nbi inputs into ndarrays.
    if isinstance(nbion, np.ndarray):
        pass
    elif isinstance(nbion, (list, tuple)):
        nbion = np.array(nbion)
    else:
        nbion = np.array([nbion])  # it should be just a number
    if isinstance(nbioff, np.ndarray):
        pass
    elif isinstance(nbioff, (list, tuple)):
        nbioff = np.array(nbioff)
    else:
        nbioff = np.array([nbioff])  # it should be just a number

    # --- Opening the NBIs shotfile.
    try:
        sf = sfutils.SFREAD('NIS', shotnumber,
                            experiment='AUGD', edition=0)
    except:
        raise errors.DatabaseError(
            'Could not open NIS shotfile for #$05d' % shotnumber)

    # --- Transforming the indices of the NBIs into the AUG system (BOX, Beam)
    nbion_box = np.asarray(np.floor((nbion-1)/4), dtype=int)
    nbion_idx = np.asarray(nbion - (nbion_box)*4 - 1, dtype=int)
    if nbioff is not None:
        nbioff_box = np.asarray(np.floor(nbioff/4), dtype=int)
        nbioff_idx = np.asarray(nbioff - (nbioff_box+1)*4 - 1, dtype=int)

    # --- Reading the NBI data.
    pniq = sf('PNIQ')*1.0e-6
    timebase = sf.gettimebase('PNIQ')
    print(pniq.shape)
    t0_0 = np.abs(timebase-timeWindow[0]).argmin()
    t1_0 = np.abs(timebase-timeWindow[-1]).argmin()
    # Selecting the NBIs.
    pniq_on = pniq[t0_0:t1_0, nbion_idx, nbion_box] > pthreshold
    print(pniq_on.shape)
    if nbioff is not None:
        pniq_off = pniq[t0_0:t1_0, nbioff_idx, nbioff_box] > pthreshold
    timebase = timebase[t0_0:t1_0]

    # --- Reshaping the PNIQ into a 2D matrix for easier handling.

    if len(nbion) == 1:
        pniq_on = np.reshape(pniq_on, (len(pniq_on), 1))
    else:
        pniq_on = pniq_on.reshape((pniq_on.shape[0],
                                   pniq_on.shape[1]*pniq_on.shape[2]))

    if nbioff is not None:
        if len(nbioff) == 1:
            pniq_off = np.reshape(pniq_off, (len(pniq_off), 1))
        else:
            pniq_off = pniq_off.reshape((pniq_off.shape[0],
                                         pniq_off.shape[1]*pniq_off.shape[2]))

    if simul:  # if all the NBIs must be simultaneously ON.
        auxON = np.all(pniq_on, axis=1)
    else:  # if only one of the beams must be turned on.
        auxON = np.any(pniq_on, axis=1)

    # We take out the times at which the NBI_OFF are ON.
    if nbioff is not None:
        auxOFF = np.logical_not(np.any(pniq_off, axis=1))

        # Making the AND operation for all the times.
        aux = np.logical_and(auxON, auxOFF)
    else:
        aux = auxON

    # --- Loop over the time windows.
    nwindows = int(np.floor(len(timeWindow)/2))
    flags = np.zeros((pniq_on.shape[0],), dtype=bool)
    for ii in range(nwindows):
        t0 = np.abs(timebase-timeWindow[2*ii]).argmin()
        t1 = np.abs(timebase-timeWindow[2*ii + 1]).argmin()

        flags[t0:t1] = True

    # --- Filtering the outputs.
    aux = np.logical_and(flags, aux)
    data = pniq[t0_0:t1_0, nbion_idx, nbion_box]
    output = {
        'timewindow': timeWindow,
        'flags': aux,
        'time': timebase[aux],
        'data': data[aux, ...]
             }
    return output


def getNBI_timeTraces(shot: int, nbilist: int = None,
                      xArrayOutput: bool = False):
    """
    Get the NBI time trace

    Pablo Oyola ft. Jose Rueda

    :param  shot: shotnumber
    :param  nbilist: list of NBI to load, if xArrayOutput == True, will be
        ignored
    :param  xArrayOutput: if true, an xrarray will be returned

    :return: dict or xarray with the NBI power.
    """
    if nbilist is None:
        nbilist = (1, 2, 3, 4, 5, 6, 7, 8)

    nbilist = np.atleast_1d(nbilist)
    nbi_box = np.asarray(np.floor((nbilist-1)/4), dtype=int)
    nbi_idx = np.asarray(nbilist - (nbi_box)*4 - 1, dtype=int)
    try:
        sf = sfutils.SFREAD('NIS', shot,
                            edition=0, experiment='AUGD')

    except:
        raise errors.DatabaseError(
            'NBI shotfile cannot be opened for #%05d' % shot)

    pniq = sf('PNIQ')*1.0e-6
    timebase = sf.gettimebase('PNIQ')

    if xArrayOutput:
        nt, ni, nbox = pniq.shape
        dummy = np.zeros((nt, 8))
        dummy[:, 0:4] = pniq[:, :, 0].squeeze()
        dummy[:, 4:8] = pniq[:, :, 1].squeeze()

        output = xr.DataArray(dummy.T, dims=('number','t'),
                           coords={'number': np.arange(8) + 1,
                                   't': timebase})
        output.attrs['long_name'] = 'NBI power'
        output.attrs['units'] = 'MW'
    else:
        output = dict()
        for ii, nbinum in enumerate(nbilist):
            output[nbinum] = pniq[:, nbi_idx[ii], nbi_box[ii]]
        output['time'] = timebase
        output['total'] = np.sum(pniq, axis=(1, 2))
    return output


def getNBI_total(shot: int, tBeg: float = None, tEnd: float = None):
    """
    Gets the total NBI power from the NIS shotfile.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: shotnumber to get the NBI power.
    :param  tBeg: initial time to get the timetrace. If None, the initial time
    stored in the shotfile will be returned.
    :param  tEnd: final time to get the timetrace. If None, the final time
    stored in the shotfile will be returned.
    """
    sf = sfutils.SFREAD('NIS', shot)
    if not sf.status:
        raise errors.DatabaseError(
            'Cannot get the NIS shotfile for #%05d' % shot)

    pni = sf(name='PNI')
    time = sf.gettimebase('PNI')

    if tBeg is None:
        t0 = 0
    else:
        t0 = np.abs(time - tBeg).argmin()

    if tEnd is None:
        t1 = len(time)
    else:
        t1 = np.abs(time - tEnd).argmin()

    # cutting the data to the desired time range.
    pni = pni[t0:t1]
    time = time[t0:t1]

    output = {
        'power': pni,
        'time': time
    }

    return output


class NBI:
    """Class with the information and data from an NBI"""

    def __init__(self, nnbi: int, shot: int = 32312, diaggeom: bool=True):
        """
        Initialize the class

        @todo: Implement the actual algorithm to look at the shotfiles for the
        NBI geometry
        @todo: Create a new package to set this structure as machine
        independent??

        :param     nnbi: number of the NBI
        :param     shot: shot number
        :param     diaggeom: If true, values extracted manually from diaggeom
        """
        ## NBI number:
        self.number = nnbi
        ## Coordinates of the NBI
        self.coords = None
        ## Pitch information (injection pitch in each radial position)
        self.pitch_profile = None
        if diaggeom:
            self.coords = NBI_diaggeom_coordinates(nnbi)
        elif REAL_NBI_GEOM:
            self.coords = nbigeom.get_nbi_geom(nnbi)
        else:
            raise ValueError('NBI calculation of geometry is not available!')

    def calc_pitch_profile(self, shot: int, time: float, rmin: float = 1.3,
                           rmax: float = 2.2, delta: float = 0.04,
                           BtIp: float = params.IB_sign, deg: bool = False,
                           rho_string: str = 'rho_pol'):
        """
        Calculate the pitch profile of the NBI along the injection line

        If the 'pitch_profile' field of the NBI object is not created, it
        initialize it, else, it just append the new time point (it will insert
        the time point at the end, if first you call the function for t=1.0,
        then for 0.5 and then for 2.5 you will create a bit of a mesh, please
        use this function with a bit of logic)

        DISCLAIMER: We will not check if the radial position coincides with the
        previous data in the pitch profile structure, it is your responsibility
        to use consistent input when calling this function

        @todo implement using insert the insertion of the data on the right
        temporal position

        Note: To simplify, the calculation of the magnetic field is call in
        each time point (I am lazy to rewrite that part) so it's a bit
        not-efficient. If you are interesting in an optimum version, open an
        issue in gitlab

        :param  shot: Shot number
        :param  time: Time in seconds, can be an array
        :param  rmin: miminum radius to be considered during the calculation
        :param  rmax: maximum radius to be considered during the calculation
        :param  delta: the spacing of the points along the NBI [m]
        :param  BtIp: sign of the magnetic field respect to the current, the
        pitch will be defined as BtIp * v_par / v
        :param  deg: If true the pitch is acos(BtIp * v_par / v)
        """
        if self.coords is None:
            raise errors.NotValidInput('Sorry, NBI coordinates are needed!!!')
        # Get coordinate vector
        v = np.array([self.coords['x1'] - self.coords['x0'],
                      self.coords['y1'] - self.coords['y0'],
                      self.coords['z1'] - self.coords['z0']])
        normv = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        # make the vector with the desired length
        v *= delta / normv
        # estimate the number of steps
        nstep = np.int(normv / delta) + 10
        # 'walk' along the NBI
        point = np.array([self.coords['x0'], self.coords['y0'],
                          self.coords['z0']])
        R = np.zeros(nstep)
        Z = np.zeros(nstep)
        phi = np.zeros(nstep)
        flags = np.zeros(nstep, dtype=bool)
        for istep in range(nstep):
            Rdum = np.sqrt(point[0]**2 + point[1]**2)
            if (Rdum < rmax) and (Rdum > rmin):
                R[istep] = Rdum
                Z[istep] = point[2]
                phi[istep] = np.arctan2(point[1], point[0])
                flags[istep] = True
            if Rdum < rmin:
                break
            point += v
        # calculate the magnetic field
        R = R[flags]
        Z = Z[flags]
        phi = phi[flags]
        ngood = R.size
        t = np.array([time]).flatten()
        nt = t.size
        pitch = np.zeros((nt, ngood))
        rho = np.zeros((nt, ngood))
        for i in range(nt):
            br, bz, bt, bp = equil.get_mag_field(shot, R, Z, time=t[i])
            bx = -np.cos(0.5*np.pi - phi) * bt + np.cos(phi) * br
            by = np.sin(0.5*np.pi - phi) * bt + np.sin(phi) * br
            B = np.vstack((bx, by, bz))
            bnorm = np.sqrt(np.sum(B**2, axis=0))
            dummy = (bx * v[0] + by * v[1] + bz * v[2]) / delta / bnorm
            pitch[i, :] = BtIp * dummy.squeeze()
            # Get the rho:
            dummy = equil.get_rho(shot, R, Z, time=t[i], coord_out=rho_string)
            rho[i, :] = dummy.squeeze()
            if deg:
                pitch[i, :] = np.arccos(pitch) * 180.0 / np.pi
        # Now we have the pitch profiles, we just need to store the info at the
        # right place
        if self.pitch_profile is None:
            self.pitch_profile = {'t': t,
                                  'z': Z, 'R': R, 'pitch': pitch,
                                  'rho': rho,
                                  'rho_string': rho}

        else:
            if self.pitch_profile['R'].size != R.size:
                raise errors.NotValidInput(
                    'Have you changed delta from the last run?')
            # insert the date where it should be
            self.pitch_profile['t'] = \
                np.concatenate((self.pitch_profile['t'], t))
            # We assume the user has not change the grid
            self.pitch_profile['pitch'] = \
                np.vstack((self.pitch_profile['pitch'], pitch))
            self.pitch_profile['rho'] = \
                np.vstack((self.pitch_profile['rho'], rho))

    def calculate_intersection(self, R=None, Z=None, shot=None, t=None, rho=1,
                               precision=0.01, plot=False):
        """
        Calculate the intersection point(s) of the NBI line with a surface

        Note the calculation will be done in R,z projection. The surface to
        intersect can be defined directly by the set of points R,z (np.arrays)
        or shot number, time and desired rho (still to be implemented)

        :param  R: Arrays of R coordinates of the surface (option 1)
        :param  z: Arrays of R coordinates of the surface (option 1)
        :param  shot: shot number (option 2)
        :param  t: time to make the calculation (option 2)
        :param  rho: rho position to make the calculation (option 2)
        :param  precision: precision for the intersection calcualtion
        :param  plot: plot flag for the function find_2D_intersection()

        :return out: dict containing:
            -# 'x': x coordinates of the cut
            -# 'y': y coordinates of the cut
            -# 'z': z coordinates of the cut
            -# 'r': coordiantes of the cut
            -# 'phi': phi coordinates of the cut, in radians
            -# 'd': distance of the intersections to the NBI initial point
            -# 'n': number of intersections
        """
        npoints = int(self.coords['length'] / precision)
        xnbi = np.linspace(self.coords['x0'], self.coords['x1'], npoints)
        ynbi = np.linspace(self.coords['y0'], self.coords['y1'], npoints)
        znbi = np.linspace(self.coords['z0'], self.coords['z1'], npoints)
        rnbi = np.sqrt(xnbi**2 + ynbi**2)
        phinbi = np.arctan2(ynbi, xnbi)
        if (R is not None) and (Z is not None):
            rintersec, zintersec = \
                ssextra.find_2D_intersection(rnbi, znbi, R, Z, plot=plot)
            if rintersec is None:
                return None
            phic = np.zeros(len(rintersec))
            for i in range(phic.size):
                phic[i] = phinbi[np.argmin(abs(znbi - zintersec[i]))]
            xc = rintersec * np.cos(phic)
            yc = rintersec * np.sin(phic)
            d = np.zeros(len(rintersec))
            for i in range(d.size):
                d[i] = math.sqrt((xc[i] - xnbi[0])**2 + (yc[i] - ynbi[0])**2
                                 + (zintersec[i] - znbi[0])**2)
        out = {
            'x': xc,
            'y': yc,
            'z': zintersec,
            'r': rintersec,
            'phi': phic,
            'd': d,
            'n': d.size
        }
        return out

    # def generate_tarcker_markers(self, Nions, E: float = 93000., sE=2000.,
    #                              Rmin: float = 1.25, Rmax: float = 2.1, A=2.,
    #                              rc=None, lambda0: float = 2.5,
    #                              max_trials=2000):
    #     """
    #     Prepare markers along the NBI line
    #
    #     Jose Rueda: jrrueda@us.es
    #
    #     Gaussian distribution will be assumed for the energies
    #
    #     :param  Nions: Number of markers to generate
    #     :param  E: energy of the markers [eV]
    #     :param  sE: standard deviation of the energy of the markers [eV]
    #     :param  Rmin: minimum radius to launch the markers
    #     :param  Rmax: maximum radius to launch the markers
    #     :param  A: Mass number of the ions
    #     :param  rc: intersection coords of the NBI with the separatrix [x,y,z]
    #     :param  lambda0: decay length of the NBI weight in the plasma
    #     """
    #     unit = self.coords['u']
    #     p0 = np.array([self.coords['x0'], self.coords['y0'],
    #                    self.coords['z0']])
    #     # Initialise the random generator:
    #     rand = np.random.default_rng()
    #     gauss = rand.standard_normal
    #     # Initialise the arrays
    #     c = 0   # counter of good ions
    #     trials = 0  # Number of trials
    #     R = np.zeros(Nions, dtype=np.float64)
    #     z = np.zeros(Nions, dtype=np.float64)
    #     phi = np.zeros(Nions, dtype=np.float64)
    #     vR = np.zeros(Nions, dtype=np.float64)
    #     vz = np.zeros(Nions, dtype=np.float64)
    #     vt = np.zeros(Nions, dtype=np.float64)
    #     m = np.zeros(Nions, dtype=np.float64)
    #     q = np.zeros(Nions, dtype=np.float64)
    #     logw = np.zeros(Nions, dtype=np.float64)
    #     t = np.zeros(Nions, dtype=np.float64)
    #     if (rc is not None) and (lambda0 is not None):
    #         weight_markers = True
    #         d0 = math.sqrt((self.coords['x0'] - rc[0])**2
    #                        + (self.coords['y0'] - rc[1])**2
    #                        + (self.coords['z0'] - rc[2])**2)
    #     else:
    #         weight_markers = False
    #         d0 = 1.
    #
    #     while c < Nions:
    #         trials += 1
    #         a = np.random.random()
    #         p1 = p0 + a * unit * self.coords['length']
    #         R1 = np.sqrt(p1[0]**2 + p1[1]**2)
    #
    #         if weight_markers:  # to discard markers outside the sep
    #             d1 = math.sqrt((self.coords['x0'] - p1[0])**2
    #                            + (self.coords['y0'] - p1[1])**2
    #                            + (self.coords['z0'] - p1[2])**2)
    #             d_plasma = math.sqrt((rc[0] - p1[0])**2
    #                                  + (rc[1] - p1[1])**2
    #                                  + (rc[2] - p1[2])**2)
    #         else:
    #             d1 = 2.0 * d0  # to ensure we always enter the next if
    #
    #         if (R1 > Rmin) and (R1 < Rmax) and (d1 > d0):
    #             rand_E = E + sE * gauss()
    #             v = np.sqrt(2. * rand_E / A / sspar.mp) * sspar.c * unit
    #             R[c] = R1
    #             z[c] = p1[2]
    #             phi[c] = np.arctan2(p1[1], p1[0])
    #             vv = sstracker.cart2pol(p1, v)
    #             vR[c] = vv[0]
    #             vz[c] = vv[1]
    #             vt[c] = vv[2]
    #             m[c] = A
    #             q[c] = 1
    #             if weight_markers:
    #                 logw[c] = lambda0 * d_plasma
    #             else:
    #                 logw[c] = 0.
    #             t[c] = 0.0
    #             c += 1
    #         if trials > max_trials:
    #             print('All markers could not be generated')
    #             print('Maximum trials reached')
    #             c = Nions
    #     marker = {
    #         'R': R[:c],
    #         'z': z[:c],
    #         'phi': phi[:c],
    #         'vR': vR[:c],
    #         'vphi': vt[:c],
    #         'vz': vz[:c],
    #         'm': m[:c],
    #         'q': q[:c],
    #         'logw': logw[:c],
    #         't': t[:c]
    #     }
    #     return marker

    def plot_pitch_profile(self, line_params: dict = {},
                           ax_param={},
                           ax=None, x_axis: str = 'R'):
        """
        Plot the NBI pitch profile

        Jose Rueda: jrrueda@us.es

        :param  line_param: Dictionary with the line params
        :param  ax_param: Dictionary with the param fr ax_beauty
        :param  ax: axis where to plot, if none, open new figure
        :return : Nothing
        """
        ax_parameters = {
            'grid': 'both',
            'ylabel': '$\\lambda$',
            # 'fontsize': 14
        }
        if x_axis == 'R':
            ax_parameters['xlabel'] = 'R [m]'
        else:
            ax_parameters['xlabel'] = '$\\rho$'
        ax_parameters.update(ax_param)

        line_options = {
            'linewidth': 2,
            'label': 'NBI#' + str(self.number)
        }
        line_options.update(line_params)
        if self.pitch_profile is None:
            raise Exception('You must calculate first the pitch profile')
        ax_created = False
        if ax is None:
            fig, ax = plt.subplots()
            ax_created = True

        nt = self.pitch_profile['t'].size
        if nt == 1:
            if x_axis == 'R':
                x = self.pitch_profile['R']
            else:
                x = self.pitch_profile['rho'].flatten()

            ax.plot(x,
                    self.pitch_profile['pitch'].flatten(),
                    **line_options)
        else:
            for i in range(nt):
                if x_axis == 'R':
                    x = self.pitch_profile['R']
                else:
                    x = self.pitch_profile['rho'][i, :]
                ax.plot(x,
                        self.pitch_profile['pitch'][i, :],
                        **line_options,
                        label=line_options['label'] + ', t = '
                        + str(self.pitch_profile['t'][i]))
        if ax_created:
            ax = ssplt.axis_beauty(ax, ax_parameters)

        plt.legend()
        return ax

    def plot_central_line(self, projection: str = 'Poloidal', ax=None,
                          line_params: dict = {}, units: str = 'm'):
        """
        Plot the NBI line

        Jose Rueda: jrrueda@us.es

        :param  projection: 'Poloidal' (or 'pol') will plot the poloidal
        projection, 'Toroidal' (or 'tor') the toroidal one. Also, 3D supported
        :param  ax: ax where to plot the NBI line
        :param  line_params: line parameters for the function plt.plot()
        :param  units: Units to plot, m or cm supportted

        :return ax: The axis where the line was plotted
        """
        scales = {
            'cm': 100.,
            'm': 1.
        }
        # Initialize the plot parameters:
        line_options = {
            'linewidth': 2,
            'label': 'NBI#' + str(self.number)
        }
        line_options.update(line_params)
        # Open the axis
        created = False
        if ax is None:
            if projection.lower() != '3d':
                fig, ax = plt.subplots()
                created = True
            else:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                created = True

        # Plot the NBI:
        xx = np.linspace(self.coords['x0'], self.coords['x1'], 100)
        yy = np.linspace(self.coords['y0'], self.coords['y1'], 100)
        zz = np.linspace(self.coords['z0'], self.coords['z1'], 100)
        if (projection.lower() == 'poloidal') or (projection == 'pol'):
            rr = np.sqrt(xx**2 + yy**2)
            # If the plot was already there, do not change its axis limits to
            # plot this:
            if not created:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
            # Plot the nbi line:
            ax.plot(scales[units] * rr, scales[units] * zz, **line_options)
            # Set the axis back to its initial position
            if not created:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        elif (projection.lower() == 'toroidal') or (projection == 'tor'):
            # If the plot was already there, do not change its axis limits to
            # plot this:
            if not created:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
            # Plot the nbi line:
            ax.plot(scales[units] * xx, scales[units] * yy, **line_options)
            # Set the axis back to its initial position
            if not created:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        elif projection.lower() == '3d':
            ax.plot(scales[units] * xx, scales[units] * yy,
                    scales[units] * zz, **line_options)
        else:
            print('Projection: ', projection)
            raise errors.NotValidInput('Projection not understood!')
        return ax
