"""Load the machine vessel"""
import get_gc            # Module to load vessel components
import dd                # Module to load shotfiles
import numpy as np
import os
from Lib.LibPaths import Path
import Lib.LibData.AUG.Equilibrium as equil
import Lib.LibData.AUG.DiagParam as params
import matplotlib.pyplot as plt
import Lib.LibPlotting as ssplt
pa = Path()


# -----------------------------------------------------------------------------
# --- Vessel
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


def getNBIwindow(timeWindow: float, shotnumber: int,
                 nbion: int, nbioff: int = None,
                 simul: bool = True, pthreshold: float = 2.0):
    """
    Get the time window within the limits provide within the timeWindow that
    corresponds to the list nbiON that are turned on and the list nbioff.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param timeWindow: window of time to retrieve the NBI data.
    @param shotnumber: Shot number from where to take the NBI timetraces.
    @param nbion: list with the NBI number that should be ON.
    @param nbioff: list with the NBIs that should be OFF.
    @param simul: simultaneous flag. If True all the NBIs of nbion should be
    ON simultaenously.
    @param pthreshold: power threshold to consider the beam is ON [MW].
    Default to 2.0 MW (to choose the 2.5MW standard beam.)
    """
    # --- Checking the time inputs.
    if len(timeWindow) == 1:
        timeWindow = np.array((timeWindow, np.inf))

    elif np.mod(len(timeWindow), 2) != 0:
        timeWindow[len(timeWindow)] = np.inf

    # --- Opening the NBIs shotfile.
    try:
        sf = dd.shotfile(diagnostic='NIS', pulseNumber=shotnumber,
                         experiment='AUGD', edition=0)
    except:
        raise Exception('Could not open NIS shotfile for #$05d' % shotnumber)

    # --- Transforming the indices of the NBIs into the AUG system (BOX, Beam)
    nbion_box = np.asarray(np.floor(nbion/4), dtype=int)
    nbion_idx = np.asarray(nbion - (nbion_box+1)*4 - 1, dtype=int)
    if nbioff is not None:
        nbioff_box = np.asarray(np.floor(nbioff/4), dtype=int)
        nbioff_idx = np.asarray(nbioff - (nbioff_box+1)*4 - 1, dtype=int)

    # --- Reading the NBI data.
    pniq = np.transpose(sf.getObjectData(b'PNIQ'), (2, 0, 1))*1.0e-6
    timebase = sf.getTimeBase(b'PNIQ')
    sf.close()

    t0_0 = np.abs(timebase-timeWindow[0]).argmin()
    t1_0 = np.abs(timebase-timeWindow[-1]).argmin()
    # Selecting the NBIs.
    pniq_on = pniq[t0_0:t1_0, nbion_box, nbion_idx] > pthreshold
    if nbioff is not None:
        pniq_off = pniq[t0_0:t1_0, nbioff_box, nbioff_idx] > pthreshold
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
    nwindows = np.floor(len(timeWindow)/2)
    flags = np.zeros((pniq_on.shape[0],), dtype=bool)
    for ii in np.arange(nwindows, dtype=int):
        t0 = np.abs(timebase-timeWindow[2*ii]).argmin()
        t1 = np.abs(timebase-timeWindow[2*ii + 1]).argmin()

        flags[t0:t1] = True

    # --- Filtering the outputs.
    aux = np.logical_and(flags, aux)
    data = pniq[t0_0:t1_0, nbion_box,  nbion_idx]
    output = {
        'timewindow': timeWindow,
        'flags': aux,
        'time': timebase[aux],
        'data': data[aux, ...]
             }
    return output


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
            raise Exception('Sorry, option not yet implemented')

    def calc_pitch_profile(self, shot: int, time: float, rmin: float = 1.1,
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

        @param shot: Shot number
        @param time: Time in seconds, can be an array
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
            if (Rdum < rmax) and (Rdum > rmin):
                R[istep] = Rdum
                Z[istep] = point[2]
                phi[istep] = np.arctan2(point[1], point[0])
                flags[istep] = True
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
                raise Exception('Have you changed delta from the last run?')
            # insert the date where it should be
            self.pitch_profile['t'] = \
                np.concatenate((self.pitch_profile['t'], t))
            # We assume the user hs not change the grid
            # self.pitch_profile['z'] = \
            #     np.vstack((self.pitch_profile['z'], Z))
            # self.pitch_profile['R'] = \
            #     np.vstack((self.pitch_profile['R'], R))
            self.pitch_profile['pitch'] = \
                np.vstack((self.pitch_profile['pitch'], pitch))
            self.pitch_profile['rho'] = \
                np.vstack((self.pitch_profile['rho'], rho))

    def plot_pitch_profile(self, line_params: dict = {},
                           ax_param={},
                           ax=None, x_axis: str = 'R'):
        """
        Plot the NBI pitch profile

        Jose Rueda: jrrueda@us.es

        @param line_param: Dictionary with the line params
        @param ax_param: Dictionary with the param fr ax_beauty
        @param ax: axis where to plot, if none, open new figure
        @return : Nothing
        """
        ax_parameters = {
            'grid': 'both',
            'ylabel': '$\\lambda$',
            'fontsize': 14
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
                        label=line_options['label'] + ', t = ' \
                        + str(self.pitch_profile['t'][i]))
        if ax_created:
            ax = ssplt.axis_beauty(ax, ax_parameters)

        plt.legend(fontsize=ax_parameters['fontsize'])

    def plot_central_line(self, projection: str = 'Poloidal', ax=None,
                          line_params: dict = {}, units: str = 'm'):
        """
        Plot the NBI line

        Jose Rueda: jrrueda@us.es

        @param projection: 'Poloidal' (or 'pol') will plot the poloidal
        projection, 'Toroidal' (or 'tor') the toroidal one
        @param ax: ax where to plot the NBI line
        @param line_params: line parameters for the function plt.plot()
        @param units: Units to plot, m or cm supportted

        @return ax: The axis where the line was plotted
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
            fig, ax = plt.subplots()
            created = True

        # Plot the NBI:
        if (projection == 'Poloidal') or (projection == 'pol'):
            xx = np.linspace(self.coords['x0'], self.coords['x1'], 100)
            yy = np.linspace(self.coords['y0'], self.coords['y1'], 100)
            zz = np.linspace(self.coords['z0'], self.coords['z1'], 100)
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
        elif (projection == 'Toroidal') or (projection == 'tor'):
            xx = np.array((self.coords['x0'], self.coords['x1']))
            yy = np.array((self.coords['y0'], self.coords['y1']))
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
        else:
            print('Projection: ', projection)
            raise Exception('Projection not understood!')
        return ax
