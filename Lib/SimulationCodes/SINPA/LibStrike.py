"""
Contains the class to read and interact with the strikes

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 0.6.0
"""
import os
import numpy as np
# import Lib.LibParameters as sspar
from Lib.LibMachine import machine
from Lib.LibPaths import Path
from Lib.SimulationCodes.SINPA.StrikesHeader import order_INPA as infoINPA
from Lib.SimulationCodes.SINPA.StrikesHeader import order_FILD as infoFILD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Lib.LibPlotting as ssplt
paths = Path(machine)


# -----------------------------------------------------------------------------
# --- Colum order for the strikes files
# -----------------------------------------------------------------------------

class Strikes:
    """
    StrikePoint class.

    Jose Rueda: jrrueda@us.es
    """

    def __init__(self, runID: str = None, plate: str = 'Scintillator',
                 file: str = None, verbose: bool = True):
        """
        Initialise the object reading data from a SINPA file.

        Jose Rueda: jrrueda@us.es

        @param runID: runID of the SINPA simulation
        @param plate: plate to collide with (Collimator or Scintillator)
        @param file: if a filename is provided, data will be loaded from this
        file, ignoring the SINPA folder structure (and runID)
        @param verbose. flag to print some info in the command line
        """
        # --- Load the strike points
        if file is None:
            if plate.lower() == 'scintillator':
                name = 'StrikePoints.bin'
            elif plate.lower() == 'collimator':
                name = 'CollimatorStrikePoints.bin'
            elif plate.lower() == 'signalscintillator':
                name = 'SignalStrikePoints.bin'
            elif plate.lower() == 'signalcollimator':
                name = 'SignalCollimatorStrikePoints.bin'
            else:
                raise Exception('Plate not understood, revise inputs')

            file = os.path.join(paths.SINPA, 'runs', runID, 'results', name)
        self.file = file
        if verbose:
            print('Reading file: ', file)
        self.read_file(verbose=verbose, plate=plate)
        # --- Initialise the rest of the object
        ## Histogram of Scintillator strikes
        self.ScintHistogram = None

    def read_file(self, verbose=False, plate: str = 'Scintillator'):
        """
        Read the strike points from a SINPA simulation

        Jose Rueda: jrrueda@us.es

        @param filename: filename of the file
        @param verbose: flag to print information on the file
        """
        fid = open(self.file, 'rb')
        header = {
            'versionID1': np.fromfile(fid, 'int32', 1)[0],
            'versionID2': np.fromfile(fid, 'int32', 1)[0],
        }
        header['info'] = {}

        if header['versionID1'] < 1:
            # Keys of what we have in the file:
            header['runID'] = np.fromfile(fid, 'S50', 1)[:]
            header['ngyr'] = np.fromfile(fid, 'int32', 1)[0]
            header['Gyroradius'] = np.fromfile(fid, 'float64', header['ngyr'])
            header['nalpha'] = np.fromfile(fid, 'int32', 1)[0]
            header['Alphas'] = np.fromfile(fid, 'float64', header['nalpha'])
            header['FILDSIMmode'] = \
                np.fromfile(fid, 'int32', 1)[0].astype(np.bool)
            header['ncolumns'] = np.fromfile(fid, 'int32', 1)[0]
            header['counters'] = \
                np.zeros((header['nalpha'], header['ngyr']), np.int)
            data = np.empty((header['nalpha'], header['ngyr']),
                            dtype=np.ndarray)
            header['scint_limits'] = {
                'xmin': 300.,
                'xmax': -300.,
                'ymin': 300.,
                'ymax': -300.
            }
            # get the information
            scints = ['scintillator',  'signalscintillator']
            if plate.lower() == 'signalcollimator':
                plate = 'collimator'
            if header['FILDSIMmode']:
                header['info'].update(infoFILD[header['versionID1']]
                                      [plate.lower()])
            else:
                header['info'].update(infoINPA[header['versionID1']]
                                      [plate.lower()])
            if plate.lower() in scints:
                ycolum = header['info']['ys']['i']
                zcolum = header['info']['zs']['i']
            for ig in range(header['ngyr']):
                for ia in range(header['nalpha']):
                    header['counters'][ia, ig] = \
                        np.fromfile(fid, 'int32', 1)[0]
                    if header['counters'][ia, ig] > 0:
                        data[ia, ig] = \
                            np.reshape(np.fromfile(fid, 'float64',
                                                   header['ncolumns'] *
                                                   header['counters'][ia, ig]),
                                       (header['counters'][ia, ig],
                                        header['ncolumns']),
                                       order='F')
                        if plate.lower() in scints:
                            header['scint_limits']['xmin'] = \
                                min(header['scint_limits']['xmin'],
                                    data[ia, ig][:, ycolum].min())
                            header['scint_limits']['xmax'] = \
                                max(header['scint_limits']['xmax'],
                                    data[ia, ig][:, ycolum].max())
                            header['scint_limits']['ymin'] = \
                                min(header['scint_limits']['ymin'],
                                    data[ia, ig][:, zcolum].min())
                            header['scint_limits']['ymax'] = \
                                max(header['scint_limits']['ymax'],
                                    data[ia, ig][:, zcolum].max())
        self.header = header
        self.data = data
        if verbose:
            print('Total number of strike points: ',
                  np.sum(header['counters']))
            print('SINPA version: ', header['versionID1'], '.',
                  header['versionID2'])
        fid.close()

    def calculate_scintillator_histogram(self, delta: float = 0.1,
                                         includeW: bool = True,
                                         kind_separation: bool = False):
        """
        Calculate the spatial histograms (x,y)Scintillator

        Jose Rueda: jrrueda@us.es

        @param delta: bin width for the histogram
        @param includeW: flag to include weight
        @param kind_separation: To separate between different kinds (if we have
        the markers coming from FIDASIM). Notice that in the results, k=0 will
        mean the total signal, or the mapping signal
        """
        colum_pos = self.header['info']['xs']['i']  # column of the x position
        if kind_separation:
            column_kind = self.header['info']['kind']['i']
        if includeW:
            weight_column = self.header['info']['w']['i']
        # --- define the grid for the histogram
        xbins = np.arange(self.header['scint_limits']['xmin'],
                          self.header['scint_limits']['xmax'],
                          delta)
        ybins = np.arange(self.header['scint_limits']['ymin'],
                          self.header['scint_limits']['ymax'],
                          delta)
        data = np.zeros((xbins.size-1, ybins.size-1))
        if kind_separation:
            if includeW:
                w = self.data[0, 0][:, weight_column]
            else:
                w = np.ones(self.header['counters'][0, 0])
            self.ScintHistogram = {5: {}, 6: {}, 7: {}, 8: {}}
            for k in [5, 6, 7, 8]:
                f = self.data[0, 0][:, column_kind].astype(np.int) == k
                H, xedges, yedges = \
                    np.histogram2d(self.data[0, 0][f, colum_pos + 1],
                                   self.data[0, 0][f, colum_pos + 2],
                                   bins=(xbins, ybins), weights=w[f])
                xcen = 0.5 * (xedges[1:] + xedges[:-1])
                ycen = 0.5 * (yedges[1:] + yedges[:-1])
                H /= delta**2
                data += H
                # Save the value of the counts for each kind
                self.ScintHistogram[k] = {
                    'xcen': xcen,
                    'ycen': ycen,
                    'xedges': xedges,
                    'yedges': yedges,
                    'H': H
                }
            # Save the total values
            self.ScintHistogram[0] = {
                'xcen': xcen,
                'ycen': ycen,
                'xedges': xedges,
                'yedges': yedges,
                'H': data
            }
        else:   # mapping tipe
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nalpha']):
                    if self.header['counters'][ia, ig] > 0:
                        if includeW:
                            w = self.data[ia, ig][:, weight_column]
                        else:
                            w = np.ones(self.header['counters'][ia, ig])
                        H, xedges, yedges = \
                            np.histogram2d(self.data[ia, ig][:, colum_pos + 1],
                                           self.data[ia, ig][:, colum_pos + 2],
                                           bins=(xbins, ybins), weights=w)
                        data += H
            xcen = 0.5 * (xedges[1:] + xedges[:-1])
            ycen = 0.5 * (yedges[1:] + yedges[:-1])
            data /= delta**2
            self.ScintHistogram = {
                0: {
                    'xcen': xcen,
                    'ycen': ycen,
                    'xedges': xedges,
                    'yedges': yedges,
                    'H': data
                }
            }

    def plot3D(self, per=0.1, ax=None, mar_params={},
               gyr_index=None, alpha_index=None,
               where: str = 'Head'):
        """
        Plot the strike points in a 3D axis as scatter points

        Jose Rueda: jrrueda@us.es

        @param per: ratio of markers to be plotted (1=all of them)
        @param ax: axes where to plot
        @param mar_params: Dictionary with the parameters for the markers
        @param gyr_index: index (or indeces if given as an np.array) of
        gyroradii to plot
        @param alpha_index: index (or indeces if given as an np.array) of
        alphas to plot
        @param where: string indicating where to plot: 'head', 'NBI',
        'ScintillatorLocalSystem'. First two are in absolute
        coordinates, last one in the scintillator coordinates (see SINPA
        documentation) [Head will plot the strikes in the collimator or
        scintillator]
        """
        # --- Default markers
        mar_options = {
            'marker': '.',
            'color': 'k'
        }
        mar_options.update(mar_params)
        # --- Create the axes
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            created = True
        else:
            created = False
        # --- Chose the variable we want to plot
        if where == 'Head' or where == 'head':
            column_to_plot = self.header['info']['x']['i']
        elif where == 'NBI':
            column_to_plot = self.header['info']['xnbi']['i']
        elif where == 'ScintillatorLocalSystem':
            column_to_plot = self.header['info']['xs']['i']
        else:
            raise Exception('Not understood what do you want to plot')

        # --- Plot the markers:
        nalpha, ngyr = self.header['counters'].shape
        minx = +100.  # Dummy variables to set a decent axis limit
        miny = +100.0
        minz = +100.0
        maxx = -300.0
        maxy = -300.0
        maxz = -300.0
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
                    flags = np.random.rand(
                        self.header['counters'][ia, ig]) < per
                    x = self.data[ia, ig][flags, column_to_plot]
                    minx = min(minx, x.min())
                    maxx = max(maxx, x.max())
                    y = self.data[ia, ig][flags, column_to_plot + 1]
                    miny = min(miny, y.min())
                    maxy = max(maxy, y.max())
                    z = self.data[ia, ig][flags, column_to_plot + 2]
                    minz = min(minz, z.min())
                    maxz = max(maxz, z.max())
                    ax.scatter(x, y, z, **mar_options)
        # Set axis limits and beuty paramters
        if created:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_zlim(minz, maxz)
            # Get rid of colored axes planes
            # (https://stackoverflow.com/questions/11448972/
            #  changing-the-background-color-of-the-axes-planes-of-a-
            #  matplotlib-3d-plot)
            # First remove fill
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Now set color to white (or whatever is "invisible")
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')

            # Bonus: To get rid of the grid as well:
            ax.grid(False)

    def plot2D(self, per=0.1, ax=None, mar_params: dict = {},
               gyr_index=None, alpha_index=None,
               ax_params: dict = {}):
        """
        Plot the strike points in a 2D axis as scatter points

        Jose Rueda: jrrueda@us.es

        Note: This is only defined for the scintillator strike points, which
        are in a plane, not for the 3D collimator strike points. It will
        rise an exception if you try to use it for the collimator points

        @param per: ratio of markers to be plotted (1=all of them)
        @param ax: axes where to plot
        @param mar_params: Dictionary with the parameters for the markers
        @param gyr_index: index (or indeces if given as an np.array) of
        gyroradii to plot
        @param alpha_index: index (or indeces if given as an np.array) of
        alphas to plot
        @param ax_params: parameters for the axis beauty routine. Only applied
        if the axis was created inside the routine
        """
        # --- Default plotting options
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
        # --- Create the axes
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Chose the variable we want to plot
        column_to_plot = self.header['info']['xs']['i']

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
                    flags = np.random.rand(
                        self.header['counters'][ia, ig]) < per
                    x = self.data[ia, ig][flags, column_to_plot + 1]
                    y = self.data[ia, ig][flags, column_to_plot + 2]
                    ax.scatter(x, y, **mar_options)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)
            fig.show()

    def plot1D(self, var='beta', gyr_index=None, alpha_index=None, ax=None,
               ax_params: dict = {}, nbins: int = 20, includeW: bool = False):
        """
        Plot (and calculate) the histogram of the selected variable

        Jose Rueda: jrrueda@us.es

        @param var: variable to plot
        """
        # --- Get the index:
        column_to_plot = self.header['info'][var]['i']
        column_of_W = self.header['info']['w']['i']
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
                    dat = self.data[ia, ig][:, column_to_plot]
                    if includeW:
                        w = self.data[ia, ig][:, column_of_W]
                    else:
                        w = np.ones(self.header['counters'][ia, ig])
                    H, xe = np.histogram(dat, weights=w, bins=nbins)
                    # Normalise H
                    H /= xe[1] - xe[0]
                    xc = 0.5 * (xe[:-1] + xe[1:])
                    ax.plot(xc, H)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)

    def plot_scintillator_histogram(self, ax=None, ax_params: dict = {},
                                    cmap=None, kind=0):
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
        if cmap is None:
            cmap = ssplt.Gamma_II()
        # --- Open the figure (if needed)
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Plot the matrix
        ax.imshow(self.ScintHistogram[kind]['H'].T,
                  extent=[self.ScintHistogram[kind]['xcen'][0],
                          self.ScintHistogram[kind]['xcen'][-1],
                          self.ScintHistogram[kind]['ycen'][0],
                          self.ScintHistogram[kind]['ycen'][-1]],
                  origin='lower', cmap=cmap)
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)

    def scatter(self, varx='beta', vary='dmin', gyr_index=None,
                alpha_index=None, ax=None, ax_params: dict = {},
                includeW: bool = False):
        """
        Scatter plot of two variables of the strike points

        Jose Rueda: jrrueda@us.es

        @param var: variable to plot
        """
        # --- Get the index:
        xcolumn_to_plot = self.header['info'][varx]['i']
        ycolumn_to_plot = self.header['info'][vary]['i']
        column_of_W = self.header['info']['w']['i']
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
                    if includeW:
                        w = self.data[ia, ig][:, column_of_W]
                    else:
                        w = np.ones(self.header['counters'][ia, ig])
                    ax.scatter(x, y, w)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)
