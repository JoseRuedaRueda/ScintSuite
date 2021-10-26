"""
Contains the Orbit object

Introduced in version 0.6.0
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import Lib.LibParameters as sspar
from mpl_toolkits.mplot3d import Axes3D
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)


class OrbitClass:
    """Class to interact with the orbit file"""

    def __init__(self, runID: str = None, verbose: bool = True,
                 file: str = None):
        """
        Initialize the orbit object

        @param runID: runID of the simulation
        @param verbose: flag to plot information of the reading
        @param file: full file to read (if provided, runID will be ignored)
        """
        if file is None:
            file = \
                os.path.join(paths.SINPA, 'runs', runID, 'results/Orbits.bin')
        self.file = file
        self.read_file(file, verbose=verbose)

    def read_file(self, filename, verbose: bool = True):
        """
        Read the orbits from an orbit file

        Jose Rueda: jrrueda@us.es

        @param filename: full name of the file to be read
        @param verbose: flag to print information
        """
        fid = open(filename, 'rb')
        fid.seek(-4, sspar.SEEK_END)
        nOrbits = np.fromfile(fid, 'int32', 1)[0]
        if verbose:
            print('Number of orbits to read: ', nOrbits)
        self.nOrbits = nOrbits
        fid.seek(0, sspar.SEEK_BOF)
        self.versionID1 = np.fromfile(fid, 'int32', 1)[0]
        self.versionID1 = np.fromfile(fid, 'int32', 1)[0]
        self.runID = np.fromfile(fid, 'S50', 1)[:]
        self.counters = np.zeros(nOrbits, np.int)
        self.data = np.empty(nOrbits, dtype=np.ndarray)
        self.kindOfCollision = np.zeros(nOrbits, np.int)

        for i in range(nOrbits):
            self.counters[i] = np.fromfile(fid, 'int32', 1)[0]
            self.kindOfCollision[i] = np.fromfile(fid, 'int32', 1)[0]
            self.data[i] = \
                np.reshape(np.fromfile(fid, 'float64', self.counters[i] * 3),
                           (self.counters[i], 3), order='F')
        fid.close()
        if verbose:
            print('Hitting collimator: ', np.sum(self.kindOfCollision == 0))
            print('Hitting scintillator: ', np.sum(self.kindOfCollision == 2))
            print('Wrong markers: ', np.sum(self.kindOfCollision == 9))

    def size(self):
        """Get the number of loaded orbits."""
        return self.nOrbits

    def __getitem__(self, idx):
        """
        Overload of the method to be able to access the data in the orbit data.

        It returns the whole data of a geometry elements

        Copied from PabloOrbit object (see iHIBSIM library)

        @param idx: element number

        @return self.data[idx]: Element dictionary
        """
        return self.data[idx]

    def plot3D(self, per=0.1, ax=None, line_params={}, imax=1000, kind=2):
        """
        Plot the strike points in a 3D axis as scatter points

        Jose Rueda: jrrueda@us.es

        @param per: ratio of markers to be plotted (1=all of them)
        @param ax: axes where to plot (if none, they are created)
        @param line_params: Parameters for plt.plot()
        @param imax: maximum number of steps to plot
        @param kindOfCollision: type of orbit to plot:
            -0: Collimator colliding
            -2: Scintillator colliding
            -9: Not colliding
        """
        # --- Default markers
        line_options = {
            'color': 'k'
        }
        line_options.update(line_params)
        # --- Create the axes
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            created = True
        else:
            created = False

        # --- Plot the markers:

        for i in range(self.nOrbits):
            if self.kindOfCollision[i] == kind:
                random_number = np.random.rand()
                if random_number < per:
                    imax_plot = min(imax, self.counters[i])
                    ax.plot(self.data[i][:imax_plot, 0],
                            self.data[i][:imax_plot, 1],
                            self.data[i][:imax_plot, 2], **line_options)
        # --- Set properly the axis
        if created:
            # Get rid of colored axes planes
            # (https://stackoverflow.com/questions/11448972/
            # changing-the-background-color-of-the-axes-planes-of
            # -a-matplotlib-3d-plot)
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
