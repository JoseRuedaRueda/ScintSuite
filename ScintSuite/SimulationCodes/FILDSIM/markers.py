"""
Routines to read and plot the markers and orbits files

Introduced in version 0.6.0
"""
import os
import numpy as np
import ScintSuite._Plotting as ssplt
import matplotlib.pyplot as plt
from ScintSuite._Machine import machine
from ScintSuite.decorators import deprecated
import ScintSuite._Paths as p
paths = p.Path(machine)
del p


@deprecated('Plase use the Strike Object from the common library')
class Strikes:
    """
    Class to interact with the FILDSIM strike point.

    Jose Rueda Rueda: jrrueda@us.es
    """

    def __init__(self, runID: str = None,
                 file: str = None, verbose: bool = True):
        """
        Load the strike points and initialise the object.

        :param  runID: runID of the SINPA simulation
        :param  file: if a filename is provided, data will be loaded from this
        file, ignoring the SINPA folder structure (and runID)
        :param  verbose. flag to print some info in the command line
        """
        # --- Load the strike points
        if file is None:
            name = '_strike_points.dat'
            file = os.path.join(paths.FILDSIM, 'results',
                                runID + name)
        self.file = file
        self.load_strike_points(verbose=verbose)

    def load_strike_points(self, verbose: bool = True):
        """
        Load the strike points used to calculate the map.

        Jose Rueda: ruejo@ipp.mpg.de

        :param  runID: runID of the FILDSIM simulation
        :param  plate: plate to collide with (Collimator or Scintillator)
        :param  file: if a filename is provided, data will be loaded from this
        file, ignoring the SINPA folder structure (and runID)
        :param  verbose. flag to print some info in the command line
        """
        if verbose:
            print('Reading strike points: ', self.file)
        dummy = np.loadtxt(self.file, skiprows=3)
        self.header = {
            'pitch': np.unique(dummy[:, 1]),
            'gyroradius': np.unique(dummy[:, 0])
        }
        self.header['npitch'] = self.header['pitch'].size
        self.header['ngyr'] = self.header['gyroradius'].size
        # --- Order the strike points in gyroradius and pitch angle
        self.data = np.empty((self.header['npitch'],
                              self.header['ngyr']),
                             dtype=np.ndarray)
        self.header['counters'] = np.zeros((self.header['npitch'],
                                            self.header['ngyr']),
                                           dtype=int)
        self.header['scint_limits'] = {  # for later histogram making
            'xmin': 300.,
            'xmax': -300.,
            'ymin': 300.,
            'ymax': -300.
        }
        ncolum = 0
        for ir in range(self.header['ngyr']):
            for ip in range(self.header['npitch']):
                self.data[ip, ir] = dummy[
                    (dummy[:, 0] == self.header['gyroradius'][ir])
                    * (dummy[:, 1] == self.header['pitch'][ip]), 2:]
                self.header['counters'][ip, ir], ncolums = \
                    self.data[ip, ir].shape
                ncolum = max(ncolum, ncolums)
                # Update the scintillator limit for the histogram
                if self.header['counters'][ip, ir] > 0:
                    self.header['scint_limits']['xmin'] = \
                        min(self.header['scint_limits']['xmin'],
                            self.data[ip, ir][:, 2].min())
                    self.header['scint_limits']['xmax'] = \
                        max(self.header['scint_limits']['xmax'],
                            self.data[ip, ir][:, 2].max())
                    self.header['scint_limits']['ymin'] = \
                        min(self.header['scint_limits']['ymin'],
                            self.data[ip, ir][:, 3].min())
                    self.header['scint_limits']['ymax'] = \
                        max(self.header['scint_limits']['ymax'],
                            self.data[ip, ir][:, 3].max())
        # Check with version of FILDSIM was used
        if ncolum == 7:
            print('Old FILDSIM format, initial position NOT included')
            old = True
        elif ncolum == 10:
            print('New FILDSIM format, initial position included')
            old = False
        else:
            print('Detected number of columns: ', ncolum)
            raise Exception('Error loading file, not recognised columns')
        # Write some help
        self.header['info'] = {
            'gyrophase': {
                'i': 0,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Initial gyrophase',
                'shortName': '$\\alpha$',
            },
            'x': {
                'i': 1,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'X Strike position',
                'shortName': 'x',
            },
            'y': {
                'i': 2,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Y Strike position',
                'shortName': 'y',
            },
            'z': {
                'i': 3,  # Column index in the file
                'units': ' [cm]',   # Units
                'longName': 'Z Strike position',
                'shortName': 'z'
            },
            'remap_rl': {
                'i': 4,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Remapped Larmor radius',
                'shortName': '$r_l$',
            },
            'remap_pitch': {
                'i': 5,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Remapped pitch angle',
                'shortName': '$\\lambda$',
            },
            'phi': {
                'i': 6,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Incident angle',
                'shortName': '$\\phi$',
            },

        }
        if not old:
            self.header['info']['xi'] = {
                'i': 7,   # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'X initial position',
                'shortName': '$x_{i}$',
            }
            self.header['info']['yi'] = {
                'i': 8,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Y initial position',
                'shortName': '$y_{i}$',
            }
            self.header['info']['zi'] = {
                'i': 9,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Z initial position',
                'shortName': '$z_{i}$',
            }
        if verbose:
            print('Total number of strike points: ',
                  np.sum(self.header['counters']))

    def calculate_scintillator_histogram(self, delta: float = 0.1):
        """
        Calculate the spatial histograms (x,y)Scintillator.

        Jose Rueda: jrrueda@us.es

        :param  delta: bin width for the histogram
        """
        colum_pos = self.header['info']['x']['i']  # column of the x position
        # --- define the grid for the histogram
        xbins = np.arange(self.header['scint_limits']['xmin'],
                          self.header['scint_limits']['xmax'],
                          delta)
        ybins = np.arange(self.header['scint_limits']['ymin'],
                          self.header['scint_limits']['ymax'],
                          delta)
        data = np.zeros((xbins.size-1, ybins.size-1))
        for ig in range(self.header['ngyr']):
            for ia in range(self.header['npitch']):
                if self.header['counters'][ia, ig] > 0:
                    H, xedges, yedges = \
                        np.histogram2d(self.data[ia, ig][:, colum_pos + 1],
                                       self.data[ia, ig][:, colum_pos + 2],
                                       bins=(xbins, ybins))
                    data += H
        xcen = 0.5 * (xedges[1:] + xedges[:-1])
        ycen = 0.5 * (yedges[1:] + yedges[:-1])
        data /= delta**2
        self.ScintHistogram = {
            'xcen': xcen,
            'ycen': ycen,
            'xedges': xedges,
            'yedges': yedges,
            'H': data
        }

    def plot2D(self, per=0.1, ax=None, mar_params: dict = {},
               gyr_index=None, pitch_index=None,
               ax_params: dict = {}):
        """
        Scatter plot of the strike points.

        :param  per: ratio of markers to be plotted (1=all of them)
        :param  ax: axes where to plot
        :param  mar_params: Dictionary with the parameters for the markers
        :param  gyr_index: index (or indeces if given as an np.array) of
        gyroradii to plot
        :param  pitch_index: index (or indeces if given as an np.array) of
        alphas to plot
        :param  ax_params: parameters for the axis beauty routine. Only applied
        if the axis was created inside the routine
        """
        # --- Initialise plotting options
        mar_options = {
            'marker': '.',
            'color': 'k'
        }
        mar_options.update(mar_params)
        ax_options = {
            'grid': 'both',
            'xlabel': '$x_{s}$ [cm]',
            'ylabel': '$y_{s}$ [cm]',
        }
        ax_options.update(ax_params)
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Chose the variable we want to plot
        column_to_plot = self.header['info']['x']['i']

        # --- Plot the markers:
        npitch, ngyr = self.header['counters'].shape

        # See which gyroradius / pitch we need
        if gyr_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyr_index, (list, np.ndarray)):
                index_gyr = gyr_index
            else:  # it should be just a number
                index_gyr = np.array([gyr_index])
        if pitch_index is None:  # if None, use all gyroradii
            index_pitch = range(npitch)
        else:
            # Test if it is a list or array
            if isinstance(pitch_index, (list, np.ndarray)):
                index_pitch = pitch_index
            else:  # it should be just a number
                index_pitch = np.array([pitch_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_pitch:
                if self.header['counters'][ia, ig] > 0:
                    flags = np.random.rand(
                        self.header['counters'][ia, ig]) < per
                    x = self.data[ia, ig][flags, column_to_plot + 1]
                    y = self.data[ia, ig][flags, column_to_plot + 2]
                    ax.scatter(x, y, **mar_options)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)

    def plot1D(self, var='y', gyr_index=None, pitch_index=None, ax=None,
               ax_params: dict = {}, nbins: int = 20):
        """
        Plot (and calculate) the histogram of the selected variable.

        Jose Rueda: jrrueda@us.es

        :param  var: variable to plot
        """
        # --- Get the index:
        self.header['info'][var]['i']
        # --- Default plotting options
        ax_options = {
            'grid': 'both',
            'xlabel': self.header['info'][var]['shortName']
                    + self.header['info'][var]['units'],
            'ylabel': '',
        }
        ax_options.update(ax_params)
        # --- Create the axes
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Chose the variable we want to plot
        column_to_plot = self.header['info'][var]['i']

        # --- Plot the markers:
        nalpha, ngyr = self.header['counters'].shape

        # See which gyroradius / pitch we need
        if gyr_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyr_index, (list, np.ndarray)):
                index_gyr = gyr_index
            else:  # it should be just a number
                index_gyr = np.array([gyr_index])
        if pitch_index is None:  # if None, use all gyroradii
            index_pitch = range(nalpha)
        else:
            # Test if it is a list or array
            if isinstance(pitch_index, (list, np.ndarray)):
                index_pitch = pitch_index
            else:  # it should be just a number
                index_pitch = np.array([pitch_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_pitch:
                if self.header['counters'][ia, ig] > 0:
                    dat = self.data[ia, ig][:, column_to_plot]
                    H, xe = np.histogram(dat, bins=nbins)
                    # Normalise H
                    delta = xe[1] - xe[0]
                    H = H.astype(np.float64)
                    H /= delta
                    xc = 0.5 * (xe[:-1] + xe[1:])
                    name = str(round(self.header['gyroradius'][ig], 2))\
                        + '[cm]' \
                        + str(round(self.header['pitch'][ia], 2)) + '[º]'
                    ax.plot(xc, H, label=name)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)
        ax.legend()

    def plot_scintillator_histogram(self, ax=None, ax_params: dict = {}):
        """
        Plot the histogram of the scintillator strikes

        Jose Rueda: jrrueda@us.es
        """
        # --- Check inputs
        if self.ScintHistogram is None:
            raise Exception('You need to calculate first the histogram')
        # --- Initialise potting options
        ax_options = {
            'xlabel': '$x_{s}$ [cm]',
            'ylabel': '$y_{s}$ [cm]',
        }
        ax_options.update(ax_params)
        # --- Open the figure (if needed)
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Plot the matrix
        ax.imshow(self.ScintHistogram['H'].T,
                  extent=[self.ScintHistogram['xcen'][0],
                          self.ScintHistogram['xcen'][-1],
                          self.ScintHistogram['ycen'][0],
                          self.ScintHistogram['ycen'][-1]],
                  origin='lower')
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)

    def scatter(self, varx='y', vary='z', gyr_index=None,
                alpha_index=None, ax=None, ax_params: dict = {},
                includeW: bool = False):
        """
        Scatter plot of two variables of the strike points

        Jose Rueda: jrrueda@us.es

        :param  var: variable to plot
        """
        # --- Get the index:
        xcolumn_to_plot = self.header['info'][varx]['i']
        ycolumn_to_plot = self.header['info'][vary]['i']
        # --- Default plotting options
        ax_options = {
            'grid': 'both',
            'xlabel': self.header['info'][varx]['shortName']
            + self.header['info'][varx]['units'],
            'ylabel': self.header['info'][vary]['shortName']
            + self.header['info'][vary]['units']
        }
        ax_options.update(ax_params)
        # --- Create the axes
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Plot the markers:
        nalpha, ngyr = self.header['counters'].shape

        # See which gyroradius / pitch we need
        if gyr_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyr_index, (list, np.ndarray)):
                index_gyr = gyr_index
            else:  # it should be just a number
                index_gyr = np.array([gyr_index])
        if alpha_index is None:  # if None, use all gyroradii
            index_alpha = range(nalpha)
        else:
            # Test if it is a list or array
            if isinstance(alpha_index, (list, np.ndarray)):
                index_alpha = alpha_index
            else:  # it should be just a number
                index_alpha = np.array([alpha_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_alpha:
                if self.header['counters'][ia, ig] > 0:
                    x = self.data[ia, ig][:, xcolumn_to_plot]
                    y = self.data[ia, ig][:, ycolumn_to_plot]
                    w = np.ones(self.header['counters'][ia, ig])
                    ax.scatter(x, y, w)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)


class Orbits:
    """
    Class to interact with the FILDSIM orbits.

    Jose Rueda: jrrueda@us.es
    """

    def __init__(self, orbits_file, orbits_index_file: str = None):
        """
        Read FILDSIM orbits.

        ajvv: avanvuuren@us.es and Jose Rueda: jrrueda@us.es

        :param  orbits_file: full path to the orbits file
            eg. /path/to/runid_example_orbits.dat
        :param  orbits_index_file: full path to the orbits_index file, if None,
        the name will be deducedd from the orbits file
            eg. /path/to/runid_example_orbits_index.dat

        create self.data: list with orbit trajectories where each tracetory
                       is given as an array with shape:
                       (number of trajectory points, 3).
                       The second index refers to the x, y and z coordinates
                       of the trajectory points.
        """
        orbits = []
        if orbits_index_file is None:
            orbits_index_file = orbits_file[:-4] + '_index.dat'
        orbits_index_data = np.loadtxt(orbits_index_file)
        orbits_data = np.loadtxt(orbits_file)

        # Check if there are no orbits
        if orbits_data.size == 0:
            raise Exception('No Orbits found in the file')
        if len(orbits_index_data) == 1:
            orbits_index_data = [orbits_index_data]

        ii = 0
        for i in orbits_index_data:
            orbits.append(orbits_data[ii:(ii + int(i)), :])
            ii += int(i)

        self.data = orbits

    def plot(self, per: float = 0.5, ax3D=None, axarr=None, dpi: int = 100,
             marker_params: dict = {}, line_params: dict = {},
             plot2D: bool = True, plot3D: bool = True):
        '''
        Plot the output FILDSIM orbits.

        ajvv: avanvuuren@us.es and Jose Rueda: jrrueda@us.es

        ----------
        :param  per: percentaje of orbits to plot
        :param  ax3D: 3D axis to plot the orbits, if None, new will be made
        :param  axarr: array of axis to plot projections, if None, will be made
        :param  dpi: dpi to render the figures, only used if the axis are
            createdby this function
        :param  line_params: Parameters for plot orbit lines,
                            Default: linestyle = 'solid' or color = 'red'
        :param  marker_params: Parameters for plot orbit end points,
                            Default: maker = 'circle' or color = 'red'
        '''
        # Default plot parameters:
        marker_options = {
            'markersize': 3,
            # 'fillstyle': 'none',
            'color': 'r',
            'marker': 'o',
            'linestyle': 'none'
        }
        marker_options.update(marker_params)
        line_options = {
            'color': 'red',
            'marker': ''
        }
        line_options.update(line_params)

        # --- Open the figure
        created_3D = False
        if ax3D is None and plot3D:
            fig = plt.figure(figsize=(6, 10), facecolor='w', edgecolor='k',
                             dpi=dpi)
            ax3D = fig.add_subplot(111, projection='3d')
            ax3D.set_xlabel('X [cm]')
            ax3D.set_ylabel('Y [cm]')
            ax3D.set_zlabel('Z [cm]')
            created_3D = True
        if axarr is None and plot2D:
            fig2, axarr = plt.subplots(nrows=1, ncols=3, figsize=(18, 10),
                                       facecolor='w', edgecolor='k', dpi=dpi)
            ax2D_xy = axarr[0]  # topdown view, i.e should see pinhole surface
            ax2D_xy.set_xlabel('X [cm]')
            ax2D_xy.set_ylabel('Y [cm]')
            ax2D_xy.set_title('Top down view (X-Y plane)')
            ax2D_yz = axarr[1]  # front view, i.e. should see scintilator plate
            ax2D_yz.set_xlabel('Y [cm]')
            ax2D_yz.set_ylabel('Z [cm]')
            ax2D_yz.set_title('Front view (Y-Z plane)')
            ax2D_xz = axarr[2]  # side view, i.e. should see slit plate surface
            ax2D_xz.set_xlabel('X [cm]')
            ax2D_xz.set_ylabel('Z [cm]')
            ax2D_xz.set_title('Side view (Y-Z plane)')
            created_2D = True
        else:
            ax2D_xy = axarr[0]  # topdown view, i.e should see pinhole surface
            ax2D_yz = axarr[1]  # front view, i.e. should see scintilator plate
            ax2D_xz = axarr[2]  # side view, i.e. should see slit plate surface
            created_2D = False

        for orbit in self.data:
            r = np.random.rand()
            if r < per:
                xline = orbit[:, 0]
                yline = orbit[:, 1]
                zline = orbit[:, 2]
                if plot3D:
                    ax3D.plot3D(xline, yline, zline, **line_options)
                if plot2D:
                    ax2D_xy.plot(xline, yline, **line_options)
                    ax2D_xy.plot(xline[0], yline[0], **marker_options)
                    ax2D_xy.plot(xline[-1], yline[-1], **marker_options)

                    ax2D_yz.plot(yline, zline, **line_options)
                    ax2D_yz.plot(yline[0], zline[0], **marker_options)
                    ax2D_yz.plot(yline[-1], zline[-1], **marker_options)

                    ax2D_xz.plot(xline, zline, **line_options)
                    ax2D_xz.plot(xline[0], zline[0], **marker_options)
                    ax2D_xz.plot(xline[-1], zline[-1], **marker_options)

        if created_2D and plot2D:
            fig2.tight_layout()
            fig2.show()
        if created_3D and plot3D:
            fig.tight_layout()
            fig.show()
