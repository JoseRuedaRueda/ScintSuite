"""
Strike object for SINPA and FILDSIM codes.

Maintaned by Jose Rueda: jrrueda@us.es

Contains the Strike object, which stores the information of the strike points
calculated by the code and plot the different information on it
"""
import os
import warnings
import numpy as np
# import Lib.LibParameters as sspar
from Lib.LibMachine import machine
from Lib.LibPaths import Path
from Lib.SimulationCodes.Common.strikeHeader import orderStrikes as order
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Lib.LibPlotting import axis_beauty, axisEqual3D, clean3Daxis
import Lib.LibPlotting as ssplt
from copy import deepcopy
paths = Path(machine)


def readSINPAstrikes(filename: str, verbose: bool = False):
    """
    Read the strike points from a SINPA simulation

    Jose Rueda: jrrueda@us.es

    @param filename: filename of the file
    @param verbose: flag to print information on the file

    Note: in order to load the proper header (with information on which
    variables are stored in the file), the code will guess which kind of file
    it is by the name of the file. Therefore, official name of the SINPA files
    should not be changed, if not, this routine does no longer work
    """
    # --- Identify which can of file we are dealing with:
    if filename.endswith('spmap'):
        plate = 'scintillator'
    elif filename.endswith('spcmap'):
        plate = 'collimator'
    elif filename.endswith('spsignal'):
        plate = 'signalscintillator'
    elif filename.endswith('spcsignal'):
        plate = 'collimator'
    else:
        raise Exception('File not understood. Has you chenged the ext???')

    # --- Open the file and read
    with open(filename, 'rb') as fid:
        header = {
            'versionID1': np.fromfile(fid, 'int32', 1)[0],
            'versionID2': np.fromfile(fid, 'int32', 1)[0],
        }
        if header['versionID1'] < 1:
            # Keys of what we have in the file:
            header['runID'] = np.fromfile(fid, 'S50', 1)[:]
            header['ngyr'] = np.fromfile(fid, 'int32', 1)[0]
            header['gyroradius'] = np.fromfile(fid, 'float64', header['ngyr'])
            header['nXI'] = np.fromfile(fid, 'int32', 1)[0]
            header['XI'] = np.fromfile(fid, 'float64', header['nXI'])
            header['FILDSIMmode'] = \
                np.fromfile(fid, 'int32', 1)[0].astype(np.bool)
            header['ncolumns'] = np.fromfile(fid, 'int32', 1)[0]
            header['counters'] = \
                np.zeros((header['nXI'], header['ngyr']), np.int)
            data = np.empty((header['nXI'], header['ngyr']),
                            dtype=np.ndarray)
            header['scint_limits'] = {
                'xmin': 300.,
                'xmax': -300.,
                'ymin': 300.,
                'ymax': -300.
            }
            # get the information
            scints = ['scintillator',  'signalscintillator']
            if header['FILDSIMmode']:
                header['info'] = \
                    deepcopy(order['sinpa_FILD'][header['versionID1']
                                                 ][plate.lower()])
            else:
                header['info'] = \
                    deepcopy(order['sinpa_INPA'][header['versionID1']
                                                 ][plate.lower()])
            if plate.lower() in scints:
                ycolum = header['info']['ys']['i']
                zcolum = header['info']['zs']['i']
            for ig in range(header['ngyr']):
                for ia in range(header['nXI']):
                    header['counters'][ia, ig] = \
                        np.fromfile(fid, 'int32', 1)[0]
                    if header['counters'][ia, ig] > 0:
                        data[ia, ig] = np.reshape(
                            np.fromfile(fid, 'float64',
                                        header['ncolumns']
                                        * header['counters'][ia, ig]),
                            (header['counters'][ia, ig],
                             header['ncolumns']), order='F')
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
        if verbose:
            print('Total number of strike points: ',
                  np.sum(header['counters']))
            print('SINPA version: ', header['versionID1'], '.',
                  header['versionID2'])
            print('Average number of strike points per centroid: ',
                  int(header['counters'].mean()))
    return header, data


def readFILDSIMstrikes(filename: str, verbose: bool = False):
    """
    Load the strike points from a FILDSIM simulation.

    Jose Rueda: ruejo@ipp.mpg.de

    @param runID: runID of the FILDSIM simulation
    @param plate: plate to collide with (Collimator or Scintillator)
    @param file: if a filename is provided, data will be loaded from this
    file, ignoring the SINPA folder structure (and runID)
    @param verbose. flag to print some info in the command line
    """
    if verbose:
        print('Reading strike points: ', filename)
    dummy = np.loadtxt(filename, skiprows=3)
    header = {
        'XI': np.unique(dummy[:, 1]),
        'gyroradius': np.unique(dummy[:, 0])
    }
    header['nXI'] = header['XI'].size
    header['ngyr'] = header['gyroradius'].size
    # --- Order the strike points in gyroradius and pitch angle
    data = np.empty((header['nXI'], header['ngyr']), dtype=np.ndarray)
    header['counters'] = np.zeros((header['nXI'], header['ngyr']),
                                  dtype=np.int)
    header['scint_limits'] = {  # for later histogram making
        'xmin': 300.,
        'xmax': -300.,
        'ymin': 300.,
        'ymax': -300.
    }
    nmarkers, ncolum = dummy.shape
    for ir in range(header['ngyr']):
        for ip in range(header['nXI']):
            data[ip, ir] = dummy[
                (dummy[:, 0] == header['gyroradius'][ir])
                * (dummy[:, 1] == header['XI'][ip]), 2:]
            header['counters'][ip, ir], ncolums = data[ip, ir].shape
            # Update the scintillator limit for the histogram
            if header['counters'][ip, ir] > 0:
                header['scint_limits']['xmin'] = \
                    min(header['scint_limits']['xmin'],
                        data[ip, ir][:, 2].min())
                header['scint_limits']['xmax'] = \
                    max(header['scint_limits']['xmax'],
                        data[ip, ir][:, 2].max())
                header['scint_limits']['ymin'] = \
                    min(header['scint_limits']['ymin'],
                        data[ip, ir][:, 3].min())
                header['scint_limits']['ymax'] = \
                    max(header['scint_limits']['ymax'],
                        data[ip, ir][:, 3].max())
    # Check with version of FILDSIM was used
    if ncolum == 9:
        print('Old FILDSIM format, initial position NOT included')
        versionID = 0
    elif ncolum == 12:
        print('New FILDSIM format, initial position included')
        versionID = 1
    else:
        print('Detected number of columns: ', ncolum)
        raise Exception('Error loading file, not recognised columns')
    # Write some help
    header['info'] = order['fildsim_FILD'][versionID]
    # Check number of markers
    total_counter = np.sum(header['counters'])
    if nmarkers != total_counter:
        warnings.warn('Total number of markers not matching!!!')
        print('Total number of strike points: ', nmarkers)
        print('Total number of counters: ', total_counter)
    if verbose:
        print('Total number of strike points: ', total_counter)
        print('Average number of strike points per centroid: ',
              int(header['counters'].mean()))
    # Small retrocompatibility part
    # Just for old FILDSIM user which may have their routines based on the
    # Strike map points object, make a copy of the XI values as in the old
    # notation (they are just 10 numbers, so it will not be the end of the
    # world)
    header['npitch'] = header['nXI']
    header['pitch'] = header['XI']
    return header, data


# -----------------------------------------------------------------------------
# --- Main Object
# -----------------------------------------------------------------------------
class Strikes:
    """
    StrikePoint class.

    Jose Rueda: jrrueda@us.es

    Stores the information of the strike points calculated by the code and plot
    the different information on it
    """

    def __init__(self, runID: str = None, type: str = 'MapScintillator',
                 file: str = None, verbose: bool = True, code: str = 'SINPA'):
        """
        Initialise the object reading data from a SINPA file.

        Jose Rueda: jrrueda@us.es

        @param runID: runID of the simulation
        @param type: file to load (mapcollimator, mapscintillator,
            signalcollimator or signalscintillator).Not used if code=='FILDSIM'
        @param file: if a filename is provided, data will be loaded from this
            file, ignoring the code folder structure (and runID)
        @param verbose. flag to print some info in the command line
        @param code: name of the code where the data is coming from
        """
        # --- Get the name of the file
        if file is None:
            if code.lower() == 'sinpa':
                # Guess the name of the file
                if type.lower() == 'mapscintillator':
                    name = runID + '.spmap'
                elif type.lower() == 'mapcollimator':
                    name = runID + '.spcmap'
                elif type.lower() == 'signalscintillator':
                    name = runID + '.spsignal'
                elif type.lower() == 'signalcollimator':
                    name = runID + '.spcsignal'
                else:
                    raise Exception('Type not understood, revise inputs')
                file = os.path.join(paths.SINPA, 'runs', runID, 'results',
                                    name)
            elif code.lower() == 'fildsim':
                name = '_strike_points.dat'
                file = os.path.join(paths.FILDSIM, 'results',
                                    runID + name)
        self.file = file
        # --- read the file
        if verbose:
            print('Reading file: ', file)
        if code.lower() == 'sinpa':
            self.header, self.data = readSINPAstrikes(file, verbose)
        elif code.lower() == 'fildsim':
            self.header, self.data = readFILDSIMstrikes(file, verbose)
        else:
            raise Exception('Code not understood')
        # --- Initialise the rest of the object
        ## Histogram of Scintillator strikes
        self.ScintHistogram = None
        ## Code used
        self.code = code

    def calculate_scintillator_histogram(self, delta: float = 0.1,
                                         includeW: bool = False,
                                         kind_separation: bool = False,
                                         xmin: float = None,
                                         xmax: float = None,
                                         ymin: float = None,
                                         ymax: float = None):
        """
        Calculate the spatial histograms (x,y)Scintillator

        Jose Rueda: jrrueda@us.es

        @param delta: bin width for the histogram.
        @param includeW: flag to include weight
        @param kind_separation: To separate between different kinds (if we have
            the markers coming from FIDASIM). Notice that in the results, k=0
            will mean the total signal, or the mapping signal
        @param xmin: minimum value for the xaxis of the histogram.
        @param ymin: minimum value for the yaxis of the histogram.
        @param xmax: maximum value for the xaxis of the histogram.
        @param ymax: maximum value for the yaxis of the histogram.

        Note: in principle, xmin, xmax, ymin, ymax whould not be given, as
        they will be taken from the strike point data. This variables are here
        just in case you want to manually define some grid to compare different
        sims.

        Note2: in the codes y,z are the variables which define the scintillator
        plane. Here y is renamed as x and y as z to be coherent with the FILD
        calibration database.
        """
        if self.code.lower() == 'sinpa':
            colum_pos = self.header['info']['xs']['i']  # column of the x
        else:
            colum_pos = self.header['info']['x']['i']  # column of the x
        if kind_separation:
            column_kind = self.header['info']['kind']['i']
        if includeW:
            weight_column = self.header['info']['w']['i']
        # --- define the grid for the histogram
        if xmin is None:
            xmin = self.header['scint_limits']['xmin']
        if xmax is None:
            xmax = self.header['scint_limits']['xmax']
        if ymin is None:
            ymin = self.header['scint_limits']['ymin']
        if ymax is None:
            ymax = self.header['scint_limits']['ymax']

        xbins = np.arange(xmin, xmax, delta)
        ybins = np.arange(ymin, ymax, delta)
        data = np.zeros((xbins.size-1, ybins.size-1))
        if kind_separation:  # Only for FIDASIM strike points
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
        else:   # mapping type, FILDSIM compatible
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
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
               gyr_index=None, XI_index=None,
               where: str = 'Head'):
        """
        Plot the strike points in a 3D axis as scatter points

        Jose Rueda: jrrueda@us.es

        @param per: ratio of markers to be plotted (1=all of them)
        @param ax: axes where to plot
        @param mar_params: Dictionary with the parameters for the markers
        @param gyr_index: index (or indeces if given as an np.array) of
            gyroradii to plot
        @param XI_index: index (or indeces if given as an np.array) of
            XIs (pitch or R) to plot
        @param where: string indicating where to plot: 'head', 'NBI',
        'ScintillatorLocalSystem'. First two are in absolute
        coordinates, last one in the scintillator coordinates (see SINPA
        documentation) [Head will plot the strikes in the collimator or
        scintillator]. For oldFILDSIM, use just head
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
        if where.lower() == 'head':
            column_to_plot = self.header['info']['x']['i']
        elif where.lower() == 'nbi':
            column_to_plot = self.header['info']['xnbi']['i']
        elif where.lower() == 'scintillatorlocalsystem':
            column_to_plot = self.header['info']['xs']['i']
        else:
            raise Exception('Not understood what do you want to plot')

        # --- Plot the markers:
        nXI, ngyr = self.header['counters'].shape
        minx = +100.0  # Dummy variables to set a decent axis limit
        miny = +100.0
        minz = +100.0
        maxx = -300.0
        maxy = -300.0
        maxz = -300.0
        # See which gyroradius / pitch (R) we need
        if gyr_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyr_index, (list, np.ndarray)):
                index_gyr = gyr_index
            else:  # it should be just a number
                index_gyr = np.array([gyr_index])
        if XI_index is None:  # if None, use all gyroradii
            index_XI = range(nXI)
        else:
            # Test if it is a list or array
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:  # it should be just a number
                index_XI = np.array([XI_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_XI:
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
            # Set the aspect ratio to equal
            axisEqual3D(ax)
            # Get rid of the colored panes
            clean3Daxis(ax)

    def plot1D(self, var='beta', gyr_index=None, XI_index=None, ax=None,
               ax_params: dict = {}, nbins: int = 20, includeW: bool = False):
        """
        Plot (and calculate) the histogram of the selected variable

        Jose Rueda: jrrueda@us.es

        @param var: variable to plot
        @param gyr_index: index (or indeces if given as an np.array) of
            gyroradii to plot
        @param XI_index: index (or indeces if given as an np.array) of
            XIs (pitch or R) to plot
        @param ax: axes where to plot
        @param ax_params: parameters for the axis beauty
        @param nbins: number of bins for the 1D histogram
        @param includeW: include weight for the histogram
        """
        # --- Get the index:
        column_to_plot = self.header['info'][var]['i']
        if includeW:
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
        nXI, ngyr = self.header['counters'].shape

        # See which gyroradius / pitch we need
        if gyr_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyr_index, (list, np.ndarray)):
                index_gyr = gyr_index
            else:  # it should be just a number
                index_gyr = np.array([gyr_index])
        if XI_index is None:  # if None, use all gyroradii
            index_XI = range(nXI)
        else:
            # Test if it is a list or array
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:  # it should be just a number
                index_XI = np.array([XI_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_XI:
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
        @param ax: axes where to plot
        @param ax_params: parameters for the axis beauty
        @param cmap: color map to be used, if none -> Gamma_II()
        @param nbins: number of bins for the 1D histogram
        @param kind: kind of markers to consider (for FILDSIM just 0, default)
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

    def scatter(self, varx='y', vary='z', gyr_index=None,
                XI_index=None, ax=None, ax_params: dict = {},
                mar_params: dict = {}, per: float = 0.5,
                includeW: bool = False, xscale=1.0, yscale=1.0):
        """
        Scatter plot of two variables of the strike points

        Jose Rueda: jrrueda@us.es

        @param varx: variable to plot in the x axis
        @param vary: variable to plot in the y axis
        @param per: ratio of markers to be plotted (1=all of them)
        @param ax: axes where to plot
        @param mar_params: Dictionary with the parameters for the markers
        @param gyr_index: index (or indeces if given as an array) of
            gyroradii to plot
        @param XI_index: index (or indeces if given as an array) of
            XIs (so pitch or R) to plot
        @param ax_params: parameters for the axis beauty routine. Only applied
            if the axis was created inside the routine
        @param xscale: Scale to multiply the variable plotted in the xaxis
        @param yscale: Scale to multiply the variable plotted in the yaxis

        Note: The units will not be updates adter the scaling, so you will need
        to change manually the labels via the ax_params()
        """
        # --- Get the index:
        xcolumn_to_plot = self.header['info'][varx]['i']
        ycolumn_to_plot = self.header['info'][vary]['i']
        if includeW:
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
        mar_options = {
            'marker': '.',
            'color': 'k'
        }
        mar_options.update(mar_params)
        # --- Create the axes
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Plot the markers:
        nXI, ngyr = self.header['counters'].shape

        # See which gyroradius / pitch we need
        if gyr_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyr_index, (list, np.ndarray)):
                index_gyr = gyr_index
            else:  # it should be just a number
                index_gyr = np.array([gyr_index])
        if XI_index is None:  # if None, use all gyroradii
            index_XI = range(nXI)
        else:
            # Test if it is a list or array
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:  # it should be just a number
                index_XI = np.array([XI_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_XI:
                if self.header['counters'][ia, ig] > 0:
                    flags = np.random.rand(
                        self.header['counters'][ia, ig]) < per
                    x = self.data[ia, ig][flags, xcolumn_to_plot]
                    y = self.data[ia, ig][flags, ycolumn_to_plot]
                    if includeW:
                        w = self.data[ia, ig][flags, column_of_W]
                    else:
                        w = np.ones(self.header['counters'][ia, ig])
                    ax.scatter(x * xscale, y * yscale, w, **mar_options)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)
