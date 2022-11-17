"""Contains the methods and classes to interact with iHIBPsim - Orbits"""
import os
import logging
import numpy as np
import Lib._Plotting as ssplt
import Lib._Parameters as sspar
import matplotlib.pyplot as plt

logger = logging.getLogger('ScintSuite.ihibpsim')

class orbit:
    def __init__(self, orbitData: dict, identifier: int = None, \
                 mass: np.float64 = None, charge: np.float64 = None):
        self.data = orbitData

        if id is not None:
            self.ID = identifier
        else:
            self.ID = np.arange(orbitData['R'].shape[0])

        if mass is None:
            self.mass = orbitData['m']
        else:
            self.mass = mass

        if charge is None:
            self.charge = orbitData['q']
        else:
            self.charge = charge

        self.getNaturalParameter()

    @property
    def size(self):
        return self.data['R'].shape[0]

    def plot(self, view: str = '2D', ax_params: dict = {}, ax=None,
             line_params: dict = {}, shaded3d_options: dict = {},
             imin: int = 0, imax: int = None, plot_vessel: bool = True):
        """
        Plot the orbit

        Jose Rueda: jrrueda@us.es
        ft.
        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  view: '2D' to plot, (R,z), (x,y). '3D' to plot the 3D orbit
        :param  ax_params: options for the function axis_beauty
        :param  line_params: options for the line plot (markers, colors and so on)
        :param  ax: axes where to plot, if none, new ones will be created. Note,
        if the '2D' mode is used, ax should be a list of axes, the first one for
        the Rz projection
        :param  shaded3d_options: dictionary with the options for the plotting of
        the 3d vessel
        :param  plot_vessel. Flag to plot the vessel or not
        """

        # --- Initialise the plotting parameters
        ax_options = {
            'ratio': 'equal',
            'fontsize': 16,
            'grid': 'both',
        }
        ax_options.update(ax_params)
        line_options = {
            'linewidth': 2
        }
        line_options.update(line_params)

        # --- Get cartesian coordinates:

        x = self.data['x']
        y = self.data['y']
        flag_ax_was_none = False
        if imax is None:
            imax = len(x)
        if imax > len(x):
            imax = len(x)
        if view == '2D':
            # Open the figure
            if ax is None:
                fig, ax = plt.subplots(1, 2)
                flag_ax_was_none = True
            # Plot the Rz, projection
            ax[0].plot(self.data['R'][imin:imax],
                        self.data['z'][imin:imax],
                        label='ID: ' + str(self.ID),
                        **line_options)
            # plot the initial and final points in a different color
            ax[0].plot(self.data['R'][imax-1], self.data['z'][imax-1],
                       'o', color='r')
            ax[0].plot(self.data['R'][imin], self.data['z'][imin],
                       'o', color='g')
            # Plot the xy projection
            ax[1].plot(x[imin:imax], y[imin:imax],
                        label='ID: ' + str(self.ID), **line_options)
            # plot the initial and final points in a different color
            ax[1].plot(x[imax-1], y[imax-1], 'o', color='r')
            ax[1].plot(x[imin], y[imin], 'o', color='g')
        else:
            # Open the figure
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            # Plot the orbit
            ax.plot(x[imin:imax], y[imin:imax],
                    self.data['z'][imin:imax],
                    **line_options)
            ax.scatter(x[imin], y[imin], self.data['z'][imin], color = 'g', label ='initial')
            ax.scatter(x[imax-1], y[imax-1], self.data['z'][imax-1], color = 'r', label ='Final')
            ax.legend()


        if flag_ax_was_none:
            if view == '2D':
                # Poloidal projection.
                ax_options['xlabel'] = 'R [m]'
                ax_options['ylabel'] = 'z [m]'
                if plot_vessel:
                    ssplt.plot_vessel(ax=ax[0])
                ax[0] = ssplt.axis_beauty(ax[0], ax_options)

                # XY projection.
                ax_options['xlabel'] = 'x [m]'
                ax_options['ylabel'] = 'y [m]'
                if plot_vessel:
                    ssplt.plot_vessel(projection='toroidal', ax=ax[1])
                ax[1] = ssplt.axis_beauty(ax[1], ax_options)
                plt.tight_layout()
            else:
                if plot_vessel:
                    ssplt.plot_vessel(ax=ax, projection='3D',
                                      params3d=shaded3d_options)
                ax_options['xlabel'] = 'x [m]'
                ax_options['ylabel'] = 'y [m]'
                ax_options['zlabel'] = 'z [m]'
                ax = ssplt.axis_beauty(ax, ax_options)
        plt.gcf().show()
        return ax

    def plotTimeTraces(self, ax=None, ax_params: dict = {},
                       line_params: dict = {},
                       legend_on: bool = True, plot_coords: bool = False,
                       ax_coords=None):
        """
        Plot the time traces of some orbit parameters

        This routine plots the time traces of some of the magnetic coordinates.
        As of May21, it plots only energy, pitch-angle, toroidal canonical
        momentum and magnetic moment, R, Z, and phi

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        Jose Rueda Rueda - jrrueda@us.es

        :param  ax: set of axis to plot the timetraces. As of Feb21, it must be
        an array of 4 axis to plot.
        :param  ax_params: options to make beautiful plots.
        :param  line_params: set of options to be sent when using plot
        routines.
        :param  legend_on: flag to determine when the legend is plot.
        :param  plot_coords: flag to plot the evolusion of r, z and phi
        :param  ax_coords: axis to plot the coordinate evolution
        """
        # --- Initialise the plotting parameter
        ax_opt = {
            'fontsize': 16,
            'grid': 'both',
        }
        line_opt = {
            'linewidth': 2,
        }
        ax_opt.update(ax_params)
        line_opt.update(line_params)

        # --- Preparing axis array
        if (ax is None) or (ax.shape[0] != 4):
            # Open the figure if not provided.
            fig, ax = plt.subplots(nrows=4, sharex=True)

            # --- Setting up the labels.
            ax_opt['ylabel'] = 'E [keV]'
            ax[0] = ssplt.axis_beauty(ax[0], ax_opt)

            ax_opt['ylabel'] = '$\\lambda$ [-]'
            ax[1] = ssplt.axis_beauty(ax[1], ax_opt)

            ax_opt['ylabel'] = '$\\mu$ [J/T]'
            ax[2] = ssplt.axis_beauty(ax[2], ax_opt)

            ax_opt['ylabel'] = '$P_\\phi$ [kg$m^2$/s]'
            ax_opt['xlabel'] = 'Time [s]'
            ax[3] = ssplt.axis_beauty(ax[3], ax_opt)

        # --- Make the plot
        ax[0].plot(self.data['time'], self.data['K']/sspar.ec*1e-3,
                   label='ID: ' + str(self.ID), **line_opt)
        ax[1].plot(self.data['time'], self.data['pitch'],
                   label='ID: ' + str(self.ID), **line_opt)
        ax[2].plot(self.data['time'], self.data['mu'],
                   label='ID: ' + str(self.ID), **line_opt)
        ax[3].plot(self.data['time'], self.data['Pphi'],
                   label='ID: ' + str(self.ID), **line_opt)
        plt.gcf().show()    # show the figure
        # --- Plot the coordinate evolution, if needed
        if plot_coords:
            if (ax_coords is None) or (ax_coords.shape[0] != 3):
                # Open the figure if not provided.
                fig2, ax_coords = plt.subplots(nrows=3, sharex=True)

                # --- Setting up the labels.
                del ax_opt['xlabel']
                ax_opt['ylabel'] = 'R [m]'
                ax_coords[0] = ssplt.axis_beauty(ax_coords[0], ax_opt)

                ax_opt['ylabel'] = 'z [m]'
                ax_coords[1] = ssplt.axis_beauty(ax_coords[1], ax_opt)

                ax_opt['ylabel'] = '$\\phi$ [rad]'
                ax_opt['xlabel'] = 'Time [s]'
                ax_coords[2] = ssplt.axis_beauty(ax_coords[2], ax_opt)
            # --- Making the plot
            ax_coords[0].plot(self.data['time'], self.data['R'],
                              label='ID: ' + str(self.ID), **line_opt)
            ax_coords[1].plot(self.data['time'], self.data['z'],
                              label='ID: ' + str(self.ID), **line_opt)
            ax_coords[2].plot(self.data['time'], self.data['phi'],
                              label='ID: ' + str(self.ID), **line_opt)
            plt.gcf().show()    # show the figure
        if legend_on:
            plt.legend()

        return ax

    """
    The following routines handle extra parameters that can be computed
    from an input magnetic fields.
    Calculation of pitch-angle, toroidal canonical momentum, magnetic
    dipole momentum, ...

    A magnetic field input is here required.
    """

    def setMagnetics(self, magn, calcMomenta=True, magMomentumOrder=0,
                     magMomentumGyrocenter=False, magToroidalUseVpar=False,
                     IpBt: float = 1.0):
        """
        Sets the magnetic field to compute magnetic field related variables,
        i.e., pitch-angle, toroidal canonical momentum, magnetic momentum,...

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  magn: Magnetic fields containing the magnetic field and the
        poloidal flux to be able to map the markers info into the variables.
        :param  calcMomenta: Calls the function to make the calculation of the
        toroidal canonical momentum, pitch-angle and magnetic momentum.
        :param  magMomOrder: Order of the calculation of the magnetic momentum.
        @see{orbit::calculateMagMomentum}
        :param  IpBt: sign convention for the pitch-angle definition.
        """

        self.magObject = magn

        if calcMomenta:
            self.calculatePitchAngle(IpBt)
            if magMomentumGyrocenter:
                self.calculateGyrocenter()
            self.calculateToroidalCanonicalMomentum(usevpar=magToroidalUseVpar,
                                                    IpBt=IpBt)
            self.calculateMagMoment(order=magMomentumOrder,
                                    gyrocenter=magMomentumGyrocenter)

    def calculateGyrocenter(self):
        """
        Calculate the coordinates (approx) of the gyrocenter

        Jose Rueda: jrrueda@us.es

        Note: it will take a step of size rl from the actual orbit position
        if rl is larger respect to the scale of the magnetic field, this is
        not really correct

        Side effect, r_l, larmor radius and omega_c, cyclotron frecuency
        will be added to the orbit dictionary

        :return : fields xc, yc, rc, zc, phic at the orbit data dict
        """
        br, bz, bphi = self.magObject.getBfield(self.data['R'],
                                                self.data['z'],
                                                self.data['phi'],
                                                self.data['time'])
        # --- Field in cartesian cooridnates
        b = np.sqrt(br**2 + bz**2 + bphi**2)
        bx = br * np.cos(self.data['phi'])\
            - bphi * np.sin(self.data['phi'])
        by = br * np.sin(self.data['phi'])\
            + bphi * np.cos(self.data['phi'])
        # --- velocity in cartesian coordinates
        vx = self.data['vR'] * np.cos(self.data['phi']) \
            - self.data['vt'] * np.sin(self.data['phi'])
        vy = self.data['vR'] * np.sin(self.data['phi']) \
            + self.data['vt'] * np.cos(self.data['phi'])
        v = np.sqrt(vx**2 + vy**2 + self.data['vz']**2)
        # --- coordinates
        x = self.data['R'] * np.cos(self.data['phi'])
        y = self.data['R'] * np.sin(self.data['phi'])
        # --- approximate gyro-frequency and rl
        self.data['omega_c'] = self.charge * b / self.mass
        self.data['r_l'] = v * np.sqrt((1.0 - self.data['pitch']**2))\
            / self.data['omega_c']
        # --- vector pointing towards gyrocenter in cartesian coordinates
        ux = (vy * bz - self.data['vz'] * by) * self.charge
        uy = (self.data['vz'] * bx - vx * bz) * self.charge
        uz = (vx * by - vy * bx) * self.charge
        u = np.sqrt(ux**2 + uy**2 + uz**2)
        ux *= self.data['r_l'] / u
        uy *= self.data['r_l'] / u
        uz *= self.data['r_l'] / u
        # --- coordinates of the gyrocenter
        self.data['xc'] = x + ux
        self.data['yc'] = y + uy
        self.data['zc'] = self.data['z'] + uz
        self.data['rc'] = np.sqrt(self.data['xc']**2 + self.data['yc']**2)
        self.data['phic'] = np.arctan2(self.data['yc'], self.data['xc'])
        return

    def calculatePitchAngle(self, ipbt):
        """
        For the orbits stored in the class, the routine computes the
        pitch-angle.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  ipbt: sign convention to define the direction of the pitch
        angle.
        """

        # Interpolate the magnetic field in the point.
        br, bz, bphi = self.magObject.getBfield(self.data['R'],
                                                self.data['z'],
                                                self.data['phi'],
                                                self.data['time'])
        # Get the absolute magnetic field and velocities.
        babs = np.sqrt(br**2 + bz**2 + bphi**2)
        vabs = np.sqrt(self.data['vR']**2 +
                       self.data['vt']**2 +
                       self.data['vz']**2)

        self.data['pitch'] = ipbt * ((self.data['vR'] * br +
                                      self.data['vz'] * bz +
                                      self.data['vt'] * bphi) /
                                     (babs*vabs))

    def calculateToroidalCanonicalMomentum(self, usevpar=False, IpBt=-1.0):
        """
        For the orbits stored in the class, the routine computes the
        toroidal canonical momentum.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  ipbt: sign convention to define the direction of the pitch
        angle.
        """

        self.data['psipol'] = self.magObject.getPsipol(self.data['R'],
                                                       self.data['z'],
                                                       self.data['phi'],
                                                       self.data['time'])
        if usevpar:
            v = np.sqrt(self.data['vt']**2 + self.data['vR']**2
                        + self.data['vt']**2)
            self.data['Pphi'] = self.mass * v * self.data['pitch'] * IpBt\
                * self.data['R'] - self.charge * self.data['psipol']
        else:
            self.data['Pphi'] = self.mass * self.data['vt'] *\
                self.data['R'] - self.charge * self.data['psipol']

    def calculateMagMoment(self, order: int = 0, gyrocenter=False):
        """
        For the orbits stored in the class, the routine computes the
        magnetic dipole momentum. The order parameter establish up to which
        level of approximation the magnetic dipole moment.

        Levels of approximation:
            1) 0th order: typical formula -> mu = mass*v_perp/(2*B)
            2) 1st order: Second order correction from the LittleJonh's
            paper:
            R.G. Littlejohn, "Variational principles of guiding centre
            motion", J. Plasma Physics (1983) - Equation (31)

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  order: approximation order of the magnetic moment.
        :param  gyrocenter: if true the field at the gyrocenter will be used
        """

        if gyrocenter:
            # --- field at gyrocenter
            brc, bzc, bphic = self.magObject.getBfield(self.data['rc'],
                                                       self.data['zc'],
                                                       self.data['phic'],
                                                       self.data['time'])
            # Get the absolute magnetic field and velocities.
            babs = np.sqrt(brc**2 + bzc**2 + bphic**2)
            self.data['mu'] = self.data['K']/babs *\
                (1.0 - self.data['pitch']**2)
        else:
            # Interpolate the magnetic field in the point.
            br, bz, bphi = self.magObject.getBfield(self.data['R'],
                                                    self.data['z'],
                                                    self.data['phi'],
                                                    self.data['time'])
            # Get the absolute magnetic field and velocities.
            babs = np.sqrt(br**2 + bz**2 + bphi**2)

            if order == 0:
                self.data['mu'] = self.data['K']/babs *\
                                 (1.0 - self.data['pitch']**2)
        return

    def getNaturalParameter(self):
        """
        Computes the natural parameter of the orbit, i.e., the path that the
        orbit has covered along the orbit.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.data['s'] = np.zeros((self.size))
        self.data['x'] = self.data['R']*np.cos(self.data['phi'])
        self.data['y'] = self.data['R']*np.sin(self.data['phi'])
        for ii in range(1, self.size-1):
            dx = self.data['x'][ii] - self.data['x'][ii - 1]
            dy = self.data['y'][ii] - self.data['y'][ii - 1]
            dz = self.data['z'][ii] - self.data['z'][ii - 1]
            ds = np.sqrt(dx**2 + dy**2 + dz**2)

            self.data['s'][ii] = self.data['s'][ii - 1] + ds

        return

    """Class methods overload."""

    def __getitem__(self, idx):
        """
        Overload of the method to be able to access the data in the orbit data.
        It returns the whole data of a given orbit.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  idx: orbit number.
        :return self.data[idx]: Orbit dictionary.
        """
        return self.data[idx]


def plotOrbits(orbitList, view: str = '2D', ax_params: dict = {}, ax=None,
               line_params: dict = {}, shaded3d_options: dict = {},
               imin: int = 0, imax: int = None):
    """
    Given an input list of orbits, this routine will plot all the orbits into
    the same axis and return an unique axis object after it.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    :param  orbitList: list of orbits to plot.
    :param  view: '2D' to plot, (R,z), (x,y). '3D' to plot the 3D orbit
    :param  ax_params: options for the function axis_beauty
    :param  line_params: options for the line plot (markers, colors and so on)
    :param  ax: axes where to plot, if none, new ones will be created. Note,
    if the '2D' mode is used, ax should be a list of axes, the first one for
    the Rz projection
    :param  shaded3d_options: dictionary with the options for the plotting of
    the 3d vessel
    """
    # --- Initialise the plotting parameters
    ax_options = {
        'ratio': 'equal',
        'fontsize': 14,
    }
    ax_options.update(ax_params)
    line_options = {
        'linewidth': 1
    }
    line_options.update(line_params)
    if orbitList is None:
        raise Exception('The input object is empty!')

    ax = orbitList[1].plot(view=view, ax_options=ax_options,
                           line_options=line_options,
                           shaded3d_options=shaded3d_options,
                           imin=imin, imax=imax)

    for ii in range(2, len(orbitList)):
        ax = orbitList[ii].plot(view=view, ax_options=ax_options, ax=ax,
                                line_options=line_options,
                                shaded3d_options=shaded3d_options,
                                imin=imin, imax=imax)

    return ax


def plotTimeTraces(orbitList, magn, ax=None, ax_params: dict = {},
                   line_params: dict = {}, grid: bool = True,
                   legend_on: bool = True):
    """
    Given a list of orbits, this routine will plot all the orbits time-traces
    into a single figure and return the axis array.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    :param  ax: set of axis to plot the timetraces. As of Feb21, it must be
    an array of 4 axis to plot.
    :param  ax_params: options to make beautiful plots.
    :param  line_params: set of options to be sent when using plot
    routines.
    @grid: establish if plotting the grids in the axis.
    @legend_on: flag to determine when the legend is plot.
    """
    # --- Initialise the plotting parameters
    ax_options = {
        'fontsize': 16,
        'grid': 'both',
    }
    ax_options.update(ax_params)
    line_options = {
        'linewidth': 2
    }
    line_options.update(line_params)
    if orbitList is None:
        raise Exception('The input object is empty!')

    orbitList[1].setMagnetics(magn)
    ax = orbitList[1].plotTimeTraces(ax_options=ax_options,
                                     line_options=line_options, grid=grid,
                                     legend_on=legend_on)

    for ii in range(2, len(orbitList)):
        orbitList[ii].setMagnetics(magn)
        ax = orbitList[ii].plotTimeTraces(ax=ax, ax_options=ax_options,
                                          line_options=line_options, grid=grid,
                                          legend_on=legend_on)

    return ax


class orbitFile:
    """Class to read and work orbits generated by the i-HIBPsim libraries."""
    def __init__(self, filename: str, load_all: bool = True):
        """
        Initialization of the orbit class. Loads the header.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  filename: Filename containing the orbit data. If provided,
                         this routine will read and store the header info.

        The header is data is stored at the end of the file:
            [...ORBIT DATA...]
            int32: List of steps recorded for each particle. (x nOrbits)
            int32: ID list stored in the file.               (x nOrbits)
            int32: Number of orbits stored                    = nOrbits
            int32: Number of elements stored per orbit step   = nCh
            int32: Orbit version number.

        :param  load_all: This flags sets the behaviour of orbits[1] to return
        either the whole data (if set to True) or only the spatial part (if set
        to False).
        """
        self.initialized = False

        if not os.path.isfile(filename):
            raise FileNotFoundError('File %s not found for orbits!'%filename)

        fid = open(filename, 'rb')
        fid.seek(-4*4, sspar.SEEK_END)
        self.nOrbits = np.fromfile(fid, 'int32', 1)[0]
        self.version = np.fromfile(fid, 'int32', 3)
        self.nCh = 10  # Number of particle characteristics.

        hdr_offset = - (4 * 4 + self.nOrbits * 4 * 2)
        fid.seek(hdr_offset, sspar.SEEK_END)
        self.stepsPerOrbit = np.fromfile(fid, 'int32', self.nOrbits)
        self.idList = np.fromfile(fid, 'int32', self.nOrbits)

        if np.all(self.idList == 1):
            logger.warning('Generating automatically the ID list')
            self.idList = np.arange(len(self.idList))+1

        # Checking the file.
        if self.nOrbits == 0:
            raise Exception('No orbits stored in the file!')

        # Data in the file seems valid. Let's prepare the offsets
        # so the data access is quicker and easier.
        self.offsets = np.cumsum(self.stepsPerOrbit, dtype=np.uint32)*8*self.nCh
        self.offsets = np.concatenate(([0], self.offsets))

        # Rewind the file to the beginnig.
        fid.seek(0, sspar.SEEK_BOF)
        self.fid = fid
        self.initialized = True

        # Shortcut to access via __getitem__ overload.
        self.loadAll = True

        # Starting the rest of the class data.
        self.nxtOrbit = 0

        self.chainLoadFlag = False

    def __del__(self):
        self.fid.close()

    def loadOrbit(self, id: int = None, full_info: bool = True):
        """
        Loads from the file the orbits that are specified from the input.
        If some or all the 'id' provided do not exist, the corresponding
        is not loaded.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        WARNING: consider that the reading Fortran -> Python will transp-
        ose the matrix order.

        :param  id: Vector of identifiers of the particle orbits to
        load from the file.

        :param  full_info: If false, only the trajectory will be loaded, if true
        also the velocity, weight, charge and mass will be loaded.

        :return orbOut: class with the orbit information. @see{orbit}.
        """
        if id is None:
            return self.loadAllOrbits(full_info=full_info)

        output = {}
        if self.initialized and not self.chainLoadFlag:
            # Getting the ID index location:
            id_location = np.argmin(np.abs(self.idList-id))
            offset = self.offsets[id_location]

            # Set the cursor at the appropriate point from the beginning.
            self.fid.seek(offset, sspar.SEEK_BOF)

            # Reading from the file.
            orbitData = np.fromfile(self.fid, 'float64',
                                    self.stepsPerOrbit[id_location] *
                                    self.nCh)

            orbitData = orbitData.reshape((self.nCh,
                                           self.stepsPerOrbit[id_location]),
                                          order='F').T
            # Adding to the dictionary:
            output['R'] = orbitData[:, 0]
            output['z'] = orbitData[:, 1]
            output['phi'] = orbitData[:, 2]
            output['m'] = orbitData[0, 6]*sspar.amu2kg
            output['q'] = orbitData[0, 7]*sspar.ec
            output['time'] = orbitData[:, 9]

            if full_info:
                output['vR'] = orbitData[:, 3]
                output['vt'] = orbitData[:, 4]
                output['vz'] = orbitData[:, 5]
                output['logW'] = orbitData[:, 8]

                # Computing the kinetic energy:
                output['K'] = orbitData[:, 3] ** 2 + \
                              orbitData[:, 4] ** 2 + \
                              orbitData[:, 5] ** 2

                output['K'] *= output['m']*0.5

        orbOutput = orbit(output, identifier=id)
        return orbOutput

    def loadAllOrbits(self, num_orbits: int = None, full_info: bool = True):
        """
        This routine just loads all the particle orbits into a list of orbit
        class.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  num_orbits: it changes the behaviour. If not none, then it
        will only load the first 'num_orbits' of orbits from file.
        :param  full_info: Determine if velocity components are also needed to
        be stored.

        """

        if num_orbits is None:
            num_orbits = self.size
        else:
            num_orbits = np.min((num_orbits, self.size))

        output = []
        if self.initialized:
            self.initChainLoad()

            for i in range(num_orbits):
                try:
                    output.append(self.getNextOrbit())
                except:
                    break
            self.endChainLoad()

        return output

    def __getitem__(self, ii: int):
        """
        Loads from the file the orbits that are specified from the input.
        If some or all the 'id' provided do not exist, the corresponding
        is not loaded.
        This routine is just an overloaded operator that acts a wrapper
        for the 'loadOrbit'.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  id: Identifiers of the particle orbits to load from the
        file.

        :return (orbits, id): dictionary and ID list read from the file.
        @see{orbits::loadOrbit} for more info.
        """
        return self.loadOrbit(ii, full_info=self.loadAll)

    def switchLoadAll(self, flag_loadAll: bool = True):
        """
        This routine allows to access the internal variable for loading
        orbit data. This will change the behaviour of the __index__
        overloaded procedure to read only the spatial data or the full
        information, containing velocities and weighting evolution.

        :param  flag_loadAll: if set to True, all the data will be loaded
        from the file, containing the spatial part (R, z, phi) as well as
        velocities and weighting. Otherwise, only the spatial part will be
        loaded.
        """
        self.loadAll = flag_loadAll

    def initChainLoad(self):
        """
        The data in the orbit file can be accessed using a serial reading
        approach: one orbit after the other. This starts the appropriate
        flag in the class and allows the fast serial reading.
        """
        if self.initialized:
            self.chainLoadFlag = True
            self.nxtOrbit = 0

            # Rewind the file to the origin.
            self.fid.seek(0, sspar.SEEK_BOF)

        return

    def getNextOrbit(self):
        """
        The data in the orbit file can be accessed using a serial reading
        approach: one orbit after the other. This starts the appropriate
        flag in the class and allows the fast serial reading.

        :return output: dictionary with the orbit information:
            -# 'R', 'z', 'phi': for the position
            -# 'vR', 'vz', 'vt': for the velocity
            -# 'q', 'm': charge and mass
            -# 'logw': logarithmic weight
            -# 'time': time of each point
        """
        if not self.initialized:
            raise Exception('Orbit object not initialized')

        if not self.chainLoadFlag:
            raise Warning('The orbit file is not opened for serial access')

        # Reached the end of the file.
        if self.nxtOrbit >= self.idList.size:
            raise Exception('End of the file orbit reached!')

        orbitData = np.fromfile(self.fid, np.float64,
                                (self.stepsPerOrbit[self.nxtOrbit]) *
                                self.nCh)

        orbitData = orbitData.reshape((self.nCh,
                                       self.stepsPerOrbit[self.nxtOrbit]),
                                      order='F')

        output = {}
        # Adding to the dictionary:
        output['R'] = orbitData[0, :]
        output['z'] = orbitData[1, :]
        output['phi'] = orbitData[2, :]
        output['m'] = orbitData[6, 0]*sspar.amu2kg
        output['q'] = orbitData[7, 0]*sspar.ec
        output['time'] = orbitData[9, :]

        if self.loadAll:
            output['vR'] = orbitData[3, :]
            output['vt'] = orbitData[4, :]
            output['vz'] = orbitData[5, :]
            output['logW'] = orbitData[8, :]

            # Computing the kinetic energy:
            output['K'] = orbitData[3, :]**2 + orbitData[4, :]**2 + \
                orbitData[5, :]**2
            output['K'] *= output['m']*0.50

        idPctl = self.idList[self.nxtOrbit]
        self.nxtOrbit += 1  # Next orbit.

        return orbit(output, identifier=np.array((idPctl)))

    def endChainLoad(self):
        """This allows to arbitrarily search for orbits along the file."""
        if self.initialized:
            self.chainLoadFlag = False
            self.nxtOrbit = 0

        return

    def setChainPoint(self, pos: int = 0):
        """
        In the serial reading mode, this routine will change the position
        that will be read in the following iteration.

        :param  pos: Position index in the list of orbits to be read next.
        By default, it comes to the beginning.

        :return pos_real: return the actual position that has been been set.
        """
        if self.initialized and self.chainLoadFlag:
            if pos < 0:
                actual_pos = 0
            elif pos > self.size:
                actual_pos = self.size
            else:
                actual_pos = pos

            self.nxtOrbit = actual_pos

            return actual_pos

        return -1

    @property
    def size(self):
        """
        Return the number of orbits stored in the file.

        :return size: size of the ID list, i.e., number of particle orbits
        stored in the file.
        """
        return self.idList.size

    def plot(self, id=1, view: str = '2D', ax_options: dict = {}, ax=None,
             line_options: dict = {}, shaded3d_options: dict = {},
             imin: int = 0, imax: int = None):
        """
        Plot the orbit

        Jose Rueda: jrrueda@us.es
        ft.
        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  id: ID of the orbit. If we are in the serial reading, it will be
        ignored, and the next orbit will be loaded instead.
        :param  view: '2D' to plot, (R,z), (x,y). '3D' to plot the 3D orbit
        :param  ax_options: options for the function axis_beauty
        :param  line_options: options for the line plot (markers, colors and so on)
        :param  ax: axes where to plot, if none, new ones will be created. Note,
        if the '2D' mode is used, ax should be a list of axes, the first one for
        the Rz projection
        :param  shaded3d_options: dictionary with the options for the plotting of
        the 3d vessel
        """
        if not self.initialized:
            raise Exception('The orbit object has not been initialized.')
        # --- Reading the orbits.
        if self.chainLoadFlag:
            orbit = self.getNextOrbit()
        else:
            orbit = self.loadOrbit(id)

        ax1 = orbit.plot(view=view, ax_params=ax_options, ax=ax,
                         line_params=line_options,
                         shaded3d_options=shaded3d_options,
                         imin=imin, imax=imax)

        return ax1
