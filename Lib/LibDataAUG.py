"""Routines to interact with the AUG database"""

import dd                # Module to load shotfiles
import get_gc            # Module to load vessel components
import numpy as np
import map_equ as meq    # Module to map the equilibrium
import os
import matplotlib.pyplot as plt
import LibPlotting as ssplt
from LibPaths import Path
from scipy.interpolate import interpn
pa = Path()


# -----------------------------------------------------------------------------
# --- AUG parameters
# -----------------------------------------------------------------------------
## Length of the shot numbers
shot_number_length = 5  # In AUG shots numbers are written with 5 numbers 00001

## Field and current direction
## @todo> This is hardcored here, at the end there are only 2 weeks of reverse
# field experiments in  the whole year, but if necesary, we could include here
# some kind of method to check the sign calling the AUG database
Bt_sign = 1   # +1 Indicates the positive phi direction (counterclockwise)
It_sign = -1  # -1 Indicates the negative phi direction (clockwise)
IB_sign = Bt_sign * It_sign

# -----------------------------------------------------------------------------
#                           FILD PARAMETERS
# -----------------------------------------------------------------------------
# All values except for beta, are extracted from the paper:
# J. Ayllon-Guerola et al. 2019 JINST14 C10032
# betas are taken to be -12.0 for AUG
# @todo include jet FILD as 'fild6', Jt-60 as 'fild7' as MAST-U as 'fild8'?
fild1 = {'alpha': 0.0,   # Alpha angle [deg], see paper
         'beta': -12.0,  # beta angle [deg], see FILDSIM doc
         'sector': 8,    # The sector where FILD is located
         'r': 2.180,     # Radial position [m]
         'z': 0.3,       # Z position [m]
         'phi_tor': 169.75,  # Toroidal position, [deg]
         'path': '/p/IPP/AUG/rawfiles/FIT/',  # Path for the video files
         'camera': 'PHANTOM',  # Type of used camera
         'extension': '_v710.cin',  # Extension of the video file, none for png
         'label': 'FILD1',  # Label for the diagnostic, for FILD6 (rFILD)
         'diag': 'FHC',  # name of the diagnostic for the fast channel
         'channel': 'FILD3_',  # prefix of the name of each channel (shotfile)
         'nch': 20}  # Number of fast channels

fild2 = {'alpha': 0.0, 'beta': -12.0, 'sector': 3, 'r': 2.180,
         'z': 0.3, 'phi_tor': 57.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD2/',
         'extension': '', 'label': 'FILD2', 'diag': 'FHA', 'channel': 'FIPM_',
         'nch': 20, 'camera': 'CCD'}

fild3 = {'alpha': 72.0, 'beta': -12.0, 'sector': 13, 'r': 1.975,
         'z': 0.765, 'phi_tor': 282.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD3/',
         'extension': '', 'label': 'FILD3', 'diag': 'xxx', 'channel': 'xxxxx',
         'nch': 99, 'camera': 'CCD'}

fild4 = {'alpha': 0.0, 'beta': -12.0, 'sector': 8, 'r': 2.035,
         'z': -0.462, 'phi_tor': 169.75,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD4/',
         'extension': '', 'label': 'FILD4', 'diag': 'FHD', 'channel': 'Chan-',
         'nch': 32, 'camera': 'CCD'}

fild5 = {'alpha': -48.3, 'beta': -12.0, 'sector': 7, 'r': 1.772,
         'z': -0.798, 'phi_tor': 147.25,
         'path': '/afs/ipp-garching.mpg.de/augd/augd/rawfiles/FIL/FILD5/',
         'extension': '', 'label': 'FILD5', 'diag': 'FHE', 'channel': 'Chan-',
         'nch': 64, 'camera': 'CCD'}

fild6 = {'alpha': 0.0, 'beta': 171.3, 'sector': 8, 'r': 2.180,
         'z': 0.3, 'phi_tor': 169.75,
         'path': '/p/IPP/AUG/rawfiles/FIT/',
         'extension': '_v710.cin', 'label': 'RFILD',
         'diag': 'FHC', 'channel': 'FILD3_', 'nch': 20, 'camera': 'CCD'}

FILD = (fild1, fild2, fild3, fild4, fild5, fild6)
## FILD diag names:
# fast-channels:
fild_diag = ['FHC', 'FHA', 'XXX', 'FHD', 'FHE', 'FHC']
fild_signals = ['FILD3_', 'FIPM_', 'XXX', 'Chan-', 'Chan-', 'FILD3_']
fild_number_of_channels = [20, 20, 99, 32, 64, 20]


# -----------------------------------------------------------------------------
# --- Equilibrium and magnetic field
# -----------------------------------------------------------------------------
def get_mag_field(shot: int, Rin, zin, diag: str = 'EQH', exp: str = 'AUGD',
                  ed: int = 0, time: float = None, equ=None):
    """
    Wrapp to get AUG magnetic field

    Jose Rueda: jrrueda@us.es

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
    Wrap to get AUG magnetic field

    Jose Rueda: jrrueda@us.es

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


def get_psipol(shot: int, Rin, zin, diag='EQH', exp: str = 'AUGD',
               ed: int = 0, time: float = None, equ=None):
    """
    Wrap to get AUG poloidal flux field

    Jose Rueda: jrrueda@us.es
    ft.
    Pablo Oyola: pablo.oyola@ipp.mpg.de

    @param shot: Shot number
    @param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    @param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    @param diag: Diag for AUG database, default EQH
    @param exp: experiment, default AUGD
    @param ed: edition, default 0 (last)
    @param time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this
    @param equ: equilibrium object from the library map_equ
    @return psipol: Poloidal flux evaluated in the input grid.
    """
    # If the equilibrium object is not an input, let create it
    created = False
    if equ is None:
        equ = meq.equ_map(shot, diag=diag, exp=exp, ed=ed)
        created = True

    equ.read_pfm()
    i = np.argmin(np.abs(equ.t_eq - time))
    PFM = equ.pfm[:, :, i].squeeze()
    psipol = interpn((equ.Rmesh, equ.Zmesh), PFM, (Rin, zin))

    # If we opened the equilibrium object, let's close it
    if created:
        equ.Close()

    return psipol


# -----------------------------------------------------------------------------
# --- Electron density and temperature profiles.
# -----------------------------------------------------------------------------
def get_ne(shotnumber: int, time: float, exp: str = 'AUGD', diag: str = 'IDA',
           edition: int = 0):
    """
    Wrap to get AUG electron density.

    Pablo Oyola: pablo.oyola@ipp.mpg.de

    @param shot: Shot number
    @param time: Time point to read the profile.
    @param exp: Experiment name.
    @param diag: diagnostic from which 'ne' will extracted.
    @param edition: edition of the shotfile to be read.

    @return output: a dictionary containing the electron density evaluated
    in the input times and the corresponding rhopol base.
    """
    # --- Opening the shotfile.
    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp, edition=edition)
    except:
        raise NameError('The shotnumber %d is not in the database'%shotnumber)

    # --- Reading from the database
    ne = sf(name='ne')

    # The area base is usually constant.
    rhop = ne.area.data[0, :]

    # Getting the time base since for the IDA shotfile, the whole data
    # is extracted at the time.
    timebase = sf(name='time')

    # Making the grid.
    TT, RR = np.meshgrid(time, rhop)

    # Interpolating in time to get the input times.
    ne_out = interpn((timebase, rhop), ne.data, (TT.flatten(), RR.flatten()))

    ne_out = ne_out.reshape(RR.shape)

    # Output dictionary:
    output = {'data': ne_out, 'rhop': rhop}

    # --- Closing the shotfile.
    sf.close()

    return output


def get_Te(shotnumber: int, time: float, exp: str = 'AUGD', diag: str = 'CEZ',
           edition: int = 0):
    """
    Wrap to get AUG ion temperature.

    Pablo Oyola: pablo.oyola@ipp.mpg.de

    @param shot: Shot number
    @param time: Time point to read the profile.
    @param exp: Experiment name.
    @param diag: diagnostic from which 'Te' will extracted.
    @param edition: edition of the shotfile to be read.

    @return output: a dictionary containing the electron temp. evaluated
    in the input times and the corresponding rhopol base.
    """
    # --- Opening the shotfile.
    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp, edition=edition)
    except:
        raise NameError('The shotnumber %d is not in the database'%shotnumber)

    # --- Reading from the database
    te = sf(name='Te')

    # The area base is usually constant.
    rhop = te.area.data[0, :]

    # Getting the time base since for the IDA shotfile, the whole data
    # is extracted at the time.
    timebase = sf(name='time')

    # Making the grid.
    TT, RR = np.meshgrid(time, rhop)

    # Interpolating in time to get the input times.
    te_out = interpn((timebase, rhop), te.data,
                     (TT.flatten(), RR.flatten()))

    te_out = te_out.reshape(RR.shape)
    # Output dictionary:
    output = {'data': te_out, 'rhop': rhop}

    sf.close()

    return output


# -----------------------------------------------------------------------------
# --- Vessel coordinates
# -----------------------------------------------------------------------------
def poloidal_vessel(shot: int = 30585, simplified: bool = False):
    """
    Get coordinate of the poloidal projection of the vessel

    Jose Rueda: jrrueda@us.es

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
    Return the coordinates of the AUG vessel

    Jose Rueda Rueda: ruejo@ipp.mpg.de

    Note: x = NaN indicate the separation between vessel block

    @param rot: angle to rotate the coordinate system
    @return xy: np.array with the coordinates of the points [npoints, 2]
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
def _NBI_diaggeom_coordinates(nnbi):
    """
    Just the coordinates manually extracted for shot 32312

    @param nnbi: the NBI number
    @return coords: dictionary containing the coordinates of the initial and
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
        @param    shot: shot number
        @param    diaggeom: If true, values extracted manually from diaggeom
        """
        ## NBI number:
        self.number = nnbi
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
        initialize it, else, it just append the new time point (it will insert
        the time point at the end, if first you call the function for t=1.0,
        then for 0.5 and then for 2.5 you will create a bit of a mesh, please
        use this function with a bit of logic)

        DISCLAIMER: We will not check if the radial position coincides with the
        previous data in the pitch profile structure, it is your responsibility
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

    def plot_pitch_profile(self, line_param: dict = {'linewidth': 2},
                           ax_param={'grid': 'both', 'xlabel': 'R [cm]',
                                     'ylabel': '$\\lambda$', 'fontsize': 14},
                           ax=None):
        """
        Plot the NBI pitch profile

        Jose Rueda: jrrueda@us.es

        @param line_param: Dictionary with the line params
        @param ax_param: Dictionary with the param fr ax_beauty
        @param ax: axis where to plot, if none, open new figure
        @return : Nothing
        """
        if self.pitch_profile is None:
            raise Exception('You must calculate first the pitch profile')
        if ax is None:
            fig, ax = plt.subplots()
            ax_created = True
        ax.plot(self.pitch_profile['R'], self.pitch_profile['pitch'],
                **line_param, label='NBI#'+str(self.number))

        if ax_created:
            ax = ssplt.axis_beauty(ax, ax_param)
        try:
            plt.legend(fontsize=ax_param['fontsize'])
        except KeyError:
            print('You did not set the fontsize in the input params...')
