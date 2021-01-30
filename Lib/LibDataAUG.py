"""Routines to interact with the AUG database"""

# Module to load shotfiles
import dd
# Module to load vessel components
import get_gc
# Other libraries
import numpy as np
# import matplotlib.pyplot as plt
# Module to map the equilibrium
import map_equ as meq
# ---
import os
from LibPaths import Path
pa = Path()


# -----------------------------------------------------------------------------
# --- AUG parameters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In AUG shots numbers are written with 5 numbers 00001

## FILD diag names:
# fast-channels:
fild_diag = ['FHC', 'FHA', 'XXX', 'FHD', 'FHE']
fild_signals = ['FILD3_', 'FIPM_', 'XXX', 'Chan-', 'Chan-']
fild_number_of_channels = [20, 20, 99, 32, 64]


# -----------------------------------------------------------------------------
# --- Equilibrium and magnetic field
# -----------------------------------------------------------------------------
def get_mag_field(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
                  ed: int = 0, time: float = None, equ=None):
    """
    Wrapp to get AUG magnetic field

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param shot: Shot number
    @param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    @param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    @param equ: equilibrium object from the library map_equ
    @return br: Radial magnetic field (nt, nrz_in), [T]
    @return bz: z magnetic field (nt, nrz_in), [T]
    @return bt: toroidal magnetic field (nt, nrz_in), [T]
    @return bp: poloidal magnetic field (nt, nrz_in), [T]
    """
    # If the equilibrium object is not an input, let create it
    created = False
    if equ is None:
        equ = meq.equ_map(shot, diag=diag, exp=exp, ed=ed)
        created = True
    # Now calculate the field
    br, bz, bt = equ.rz2brzt(Rin, zin, t_in=time)
    bp = np.hypot(br, bz)
    # If we opened the equilibrium object, let's close it
    if created:
        equ.Close()
    return br, bz, bt, bp


def get_rho(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
            ed: int = 0, time: float = None, equ=None,
            coord_out: str = 'rho_pol'):
    """
    Wrapp to get AUG magnetic field

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param shot: Shot number
    @param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    @param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    @param equ: equilibrium object from the library map_equ
    @param coord_out: the desired rho coordinate, default rho_pol
    @return rho: The desired rho coordinate evaluated at the points
    """
    # If the equilibrium object is not an input, let create it
    created = False
    if equ is None:
        equ = meq.equ_map(shot, diag=diag, exp=exp, ed=ed)
        created = True
    # Now calculate the field
    rho = equ.rz2rho(Rin, zin, t_in=time, coord_out=coord_out,
                     extrapolate=True)
    # If we opened the equilibrium object, let's close it
    if created:
        equ.Close()
    return rho


# -----------------------------------------------------------------------------
# --- Vessel coordinates
# -----------------------------------------------------------------------------
def poloidal_vessel(shot: int = 30585, simplified: bool = False):
    """
    Get coordinate of the poloidal projection of the vessel

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param shot: shot number to be used
    @param simplified: if true, a 'basic' shape of the poloidal vessel will be
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
    Return the coordiates of the AUG vessel

    Jose Rueda Rueda: ruejo@ipp.mpg.de

    Note: x = NaN indicate the separation between vessel block

    @param rot: angle to rotate the coordinate system
    @return xy: np.array with the coordinates of the points [npoints, 2]
    """
    # --- Section 0: Read the data
    # The files are a series of 'blocks' representing each piece of the vessel,
    # each block is separated by an empty line. I will scan the file line by
    # line looking for the position of those empty lines:
    ## todo_ include rotation!! default -pi/8
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
def _NBI_diaggeom_coordinates(nnbi):
    """
    Just the coordinates manually extracted for shot 32312

    @param nnbi: the NBI number
    @return coords: dictionary containing the coordiates of the initial and
    final points. '0' are near the source, '1' are near the central column
    """
    r0 = np.array([2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6])
    r1 = np.array([1.046, 1.046, 1.046, 1.046, 1.048, 2.04, 2.04, 1.048])

    z0 = np.array([0.022, 0.021, -0.021, -0.022,
                   -0.019, -0.149, 0.149, 0.19])
    z1 = np.array([-0.12, -0.145, 0.145, 0.12, -0.180, -0.6, 0.6, 0.180])

    phi0 = np.array([-32.725, -31.88, -31.88, -32.725,
                     145.58, 148.21, 148.21, 145.58]) * np.pi / 180.0
    phi1 = np.array([-13.81, 10.07, 10.07, -13.81,
                     -180.0, -99.43, -99.43, -180.0]) * np.pi / 180.0

    x0 = r0 * np.cos(phi0)
    x1 = r1 * np.cos(phi1)

    y0 = r0 * np.sin(phi0)
    y1 = r1 * np.sin(phi1)

    coords = {'phi0': phi0[nnbi-1], 'phi1': phi1[nnbi-1],
              'x0': x0[nnbi-1], 'y0': y0[nnbi-1],
              'z0': z0[nnbi-1], 'x1': x1[nnbi-1],
              'y1': y1[nnbi-1], 'z1': z1[nnbi-1]}
    return coords


# -----------------------------------------------------------------------------
# --- Other shot files
# -----------------------------------------------------------------------------
def get_fast_channel(diag: str, diag_number: int, channels, shot: int):
    """
    Get the signal for the fast channels (PMT, APD)

    Jose Rueda Rueda: jrrueda@us.es

    @param diag: diagnostic: 'FILD' or 'INPA'
    @param diag_number: 1-5
    @param channels: channel number we want, or arry with channels
    @param shot: shot file to be opened
    """
    # Check inputs:
    if not ((diag == 'FILD') or (diag != 'INPA')):
        raise Exception('No understood diagnostic')

    # Load diagnostic names:
    if diag == 'FILD':
        if (diag_number > 5) or (diag_number < 1):
            print('You requested: ', diag_number)
            raise Exception('Wrong fild number')
        diag_name = fild_diag[diag_number - 1]
        signal_prefix = fild_signals[diag_number - 1]
        nch = fild_number_of_channels[diag_number - 1]

    # Look which channels we need to load:
    try:    # If we received a numpy array, all is fine
        nch_to_load = channels.size
        ch = channels
    except AttributeError:  # If not, we need to create it
        ch = np.array([channels])
        nch_to_load = ch.size

    # Open the shot file
    fast = dd.shotfile(diag_name, shot)
    data = []
    for ic in range(nch):
        real_channel = ic + 1
        if real_channel in ch:
            name_channel = signal_prefix + "{0:02}".format(real_channel)
            channel_dat = fast.getObjectData(name_channel.encode('UTF-8'))
            data.append(channel_dat)
        else:
            data.append(None)
    # get the time base (we will use last loaded channel)
    time = fast.getTimeBase(name_channel.encode('UTF-8'))
    print('Number of requested channels: ', nch_to_load)
    return {'time': time, 'signal': data}


# -----------------------------------------------------------------------------
# --- Classes
# -----------------------------------------------------------------------------
class NBI:
    """Class with the information and data from an NBI"""

    def __init__(self, nnbi: int, shot: int = 32312, diaggeom=True):
        """
        Initialize the class

        @todo: Implement the actual algorithm to look at the shotfiles for the
        NBI geometry
        @todo: Create a new package to set this structure as machine
        independent??

        @param    nnbi: number of the NBI
        @type:    int

        @param    shot: shot number
        @type:    int

        @param    diaggeom: If true, values extracted manually from diaggeom
        """
        ## Coordinates of the NBI
        self.coords = None
        ## Pitch information (injection pitch in each radial position)
        self.pitch_profile = None
        if diaggeom:
            self.coords = _NBI_diaggeom_coordinates(nnbi)
        else:
            raise Exception('Sorry, option not jet implemented')

    def calc_pitch_profile(self, shot: int, time: float, rmin: float = 1.4,
                           rmax: float = 2.2, delta: float = 0.04,
                           BtIp: int = -1.0, deg: bool = False):
        """
        Calculate the pitch profile of the NBI along the injection line

        If the 'pitch_profile' field of the NBI object is not created, it
        initialise it, else, it just append the new time point (it will insert
        the time point at the end, if first you call the function for t=1.0,
        then for 0.5 and then for 2.5 you will create a bit of a mesh, please
        use this function with a bit of logic)

        DISCLAIMER: We will not check if the radial position concides with the
        previous data in the pitch profile structure, it is your responsability
        to use consistent input when calling this function

        @todo implement using insert the insertion of the data on the right
        temporal position

        @param shot: Shot number
        @param time: Time in seconds
        @param rmin: miminum radius to be considered during the calculation
        @param rmax: maximum radius to be considered during the calculation
        @param delta: the spacing of the points along the NBI [m]
        @param BtIp: sign of the magnetic field respect to the current, the
        pitch will be defined as BtIp * v_par / v
        @param deg: If true the pitch is acos(BtIp * v_par / v)
        """
        if self.coords is None:
            raise Exception('Sorry, NBI coordinates are needed!!!')
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
        flags = np.zeros(nstep, dtype=np.bool)
        for istep in range(nstep):
            Rdum = np.sqrt(point[0]**2 + point[1]**2)
            if Rdum < rmin:
                break
            if Rdum < rmax:
                R[istep] = Rdum
                Z[istep] = point[2]
                phi[istep] = np.arctan2(point[1], point[0])
                flags[istep] = True
            point = point + v
        # calculate the magnetic field
        R = R[flags]
        Z = Z[flags]
        phi = phi[flags]
        ngood = R.size
        pitch = np.zeros(ngood)
        br, bz, bt, bp = get_mag_field(shot, R, Z, time=time)
        bx = -np.cos(0.5*np.pi - phi) * bt + np.cos(phi) * br
        by = np.sin(0.5*np.pi - phi) * bt + np.sin(phi) * br
        B = np.vstack((bx, by, bz))
        bnorm = np.sqrt(np.sum(B**2, axis=0))
        pitch = (bx * v[0] + by * v[1] + bz * v[2]) / delta / bnorm
        pitch = BtIp * pitch.squeeze()
        if deg:
            pitch = np.arccos(pitch) * 180.0 / np.pi
        # Now we have the pitch profiles, we just need to store the info at the
        # right place
        if self.pitch_profile is None:
            self.pitch_profile = {'t': np.array(time),
                                  'z': Z, 'R': R, 'pitch': pitch}

        else:
            # number of already present times:
            nt = len(self.pitch_profile['t'])
            # see if the number of points along the NBI matches
            npoints = self.pitch_profile['R'].size
            if npoints / nt != R.size:
                raise Exception('Have you changed delta from the last run?')
            # insert the date where it should be
            self.pitch_profile['t'] = \
                np.vstack((self.pitch_profile['t'], time))
            self.pitch_profile['z'] = \
                np.vstack((self.pitch_profile['z'], Z))
            self.pitch_profile['R'] = \
                np.vstack((self.pitch_profile['R'], R))
            self.pitch_profile['pitch'] = \
                np.vstack((self.pitch_profile['pitch'], pitch))
