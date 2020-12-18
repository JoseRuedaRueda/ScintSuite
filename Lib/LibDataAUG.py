"""Routines to interact with the AUG database"""

# Module to load the equilibrium
# from sf2equ_20200525 import EQU
# # Module to map the equilibrium
# import mapeq_20200507 as meq
# Module to load vessel components
import get_gc
# Other libraries
import numpy as np
# import matplotlib.pyplot as plt
# Module to map the equilibrium
import map_equ as meq

# -----------------------------------------------------------------------------
# AUG paramters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In AUG shots numbers are written with 5 numbers 00001


# def get_mag_field(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
#                   ed: int = 0, tiniEQU: float = None, tendEQU: float = None,
#                   time: float = None, equ=None):
#     """
#     Wrapper to get AUG magnetic field

#     Jose Rueda: jose.rueda@ipp.mpg.de

#     Adapted from: https://www.aug.ipp.mpg.de/aug/manuals/map_equ/

#     @param shot: Shot number
#     @param Rin: Array of R positions (in pairs with zin) [m]
#     @param zin: Array of z positions (in pairs with Rin) [m]
#     @param diag: Diag for AUG database, default EQH
#     @param exp: experiment, default AUGD
#     @param ed: edition, default 0 (last)
#     @param tiniEQU: Initial time to load the equilibrium object, only valid
#     if equ is not pass as input (in s)
#     @param tendEQU: End time to load the equilibrium object, only valid if
#     equ is not pass as input (in s)
#     @param time: Array of times where we want to calculate the field (the
#     field would be calculated in a time as close as possible to this
#     @param equ: equilibrium object of clas EQU (see
#     https://www.aug.ipp.mpg.de/aug/manuals/map_equ/equ/html/classsf2equ__20200525_1_1EQU.html)
#     @return br: Radial magnetic field (nt, nrz_in), [T]
#     @return bz: z magnetic field (nt, nrz_in), [T]
#     @return bt: toroidal magnetic field (nt, nrz_in), [T]
#     @return bp: poloidal magnetic field (nt, nrz_in), [T]
#     """
#     # If the equilibrium object is not an input, let create it
#     created = False
#     if equ is None:
#         equ = EQU(shot, diag=diag, exp=exp, ed=ed, tbeg=tiniEQU,
#                   tend=tendEQU)
#         created = True
#     # Check if the shot number is correct
#     else:
#         if equ.shot != shot:
#             print('Shot number of the received equilibrium does not match!')
#             br = 0
#             bz = 0
#             bt = 0
#             bp = 0
#             return br, bz, bt, bp
#     # Now calculate the field
#     br, bz, bt = meq.rz2brzt(equ, r_in=Rin, z_in=zin, t_in=time)
#     bp = np.hypot(br, bz)
#     # If we opened the equilibrium object, let's close it
#     if created:
#         equ.close()
#     return br, bz, bt, bp


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


def plot_vessel(ax, projection: str = 'poloidal', line_properties: dict = {},
                nshot: int = 30585):
    """
    Plot AUG vessel

    José Rueda Rueda

    Poloidal plot of the vessel is directly extracted from IPP tutorial:
    https://www.aug.ipp.mpg.de/aug/manuals/map_equ/

    @param ax: axes where to plot
    @param projection: 'poloidal' or 'toroidal'
    @param line_properties: dictionary with the argument for the function
    plot of matplat lib (example, color, linewidth...)
    @return: Vessel plotted in the selected axes
    """
    ## todo plot toroidal vessel
    # Make sure that the color property is in the line_properties option,
    # if not, python will plot every part of the vessel in a different color
    # and we will have a funny output...
    if 'color' not in line_properties:
        line_properties['color'] = 'k'

    if projection == 'poloidal':
        # Get vessel coordinates
        gc_r, gc_z = get_gc.get_gc(nshot)
        for key in gc_r.keys():
            # print(key)
            ax.plot(gc_r[key], gc_z[key], **line_properties)
    elif projection == 'toroidal':
        print('Sorry, this option is not jet implemented. Talk to Jose Rueda')
    else:
        print('Not recognised argument')

    return


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


class NBI:
    """Class with the information and data from an NBI"""

    def __init__(self, nnbi: int, shot: int = 32312, diaggeom=True, t=None):
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

        @param    t: time [s] if it is present, the pitch along the injection
        line will be calculated
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
        nstep = np.int(normv / delta)
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


# def get_NBI_geometry(nnbi=3, shot=30585):
#     """
#     Get gometry of the NBI line
#
#     Jose Rueda Rueda: jose.rueda@ipp.mpg.de
#
#     Extracted from the IDL routines of FIDASIM4
#
#     @params shot: shot number
#     """

    # # Number of NBIs:
    # nsrc = 8
    # # Parameters extracted from aug web page
    # # R0(P):: Distance horizontal beam crossing to - torus axis [m]
    # r0 = [284.2, 284.2, 284.2, 284.2, 329.6, 329.6, 329.6, 329.6]
    # # PHI: angle between R and box [rad]
    # phi = [15.0, 15.0, 15.0, 15.0, 18.90, 18.90, 18.90, 18.9] / 180.0 * np.pi
    # # THETA:   angle towards P (horizontal beam crossing) [rad]
# theta = [33.75, 33.75, 33.75, 33.75, 29., 29.0, 29.0, 29.0] / 180.0 * np.pi
    # theta[4:7] += np.pi  # for NBI box 2
    # theta -= np.pi / 8. * 3.   # rotate by 3 sectors (new coordinate system)
    # # ALPHA: horizontal angle between Box-axis and source [rad]
    # alpha = 4.1357 * [1., -1., -1., 1., 1., -1., -1., 1.] / 180.0 * np.pi
    # # distance between P0 and Px!
    # delta_x = [-50., -50., -50., -50., -50., 50., 50., -50.] / 100.0
    # # radius of tangency:
    # rtan = [0.53, 0.93, 0.93, 0.53, 0.84, 1.29, 1.29, 0.84]
    # # vertical angle between box-axis and source [rad]
    # beta = [-4.8991, -4.8991, 4.8991, 4.8991,
    #         -4.8991, -6.6555, 6.6555, 4.8991] * np.pi / 180.0
    # # @todo make the reading of beta from the database to work!!!
