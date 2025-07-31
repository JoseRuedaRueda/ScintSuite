"""
Contains the Orbit object

Introduced in version 0.6.0
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import ScintSuite._Parameters as sspar
from mpl_toolkits.mplot3d import Axes3D
from ScintSuite._Machine import machine
from ScintSuite._Paths import Path
import ScintSuite._Plotting as ssplt
paths = Path(machine)


class OrbitClass:
    """Class to interact with the orbit file"""

    def __init__(self, runID: str = None, verbose: bool = True,
                 file: str = None):
        """
        Initialize the orbit object

        :param  runID: runID of the simulation
        :param  verbose: flag to plot information of the reading
        :param  file: full file to read (if provided, runID will be ignored)
        """
        if file is None:
            file = os.path.join(paths.SINPA, 'runs', runID, 'results',
                                runID + '.orb')
        self.file = file
        self.read_file(file, verbose=verbose)

    def read_file(self, filename, verbose: bool = True):
        """
        Read the orbits from an orbit file

        Jose Rueda: jrrueda@us.es

        :param  filename: full name of the file to be read
        :param  verbose: flag to print information
        """
        fid = open(filename, 'rb')
        fid.seek(-12, sspar.SEEK_END)
        self.m = np.fromfile(fid, 'float64', 1)[0] * sspar.amu2kg
        nOrbits = np.fromfile(fid, 'int32', 1)[0]
        if verbose:
            print('Number of orbits to read: ', nOrbits)
        self.nOrbits = nOrbits
        fid.seek(0, sspar.SEEK_BOF)
        self.versionID1 = np.fromfile(fid, 'int32', 1)[0]
        self.versionID1 = np.fromfile(fid, 'int32', 1)[0]
        self.runID = np.fromfile(fid, 'S50', 1)[:]
        self.kindOfFile = np.fromfile(fid, 'int32', 1)[0]
        self.rl = np.zeros(nOrbits)
        self.xi = np.zeros(nOrbits)
        self.counters = np.zeros(nOrbits, int)
        self.data = np.empty(nOrbits, dtype=np.ndarray)
        self.kindOfCollision = np.zeros(nOrbits, int)

        for i in range(nOrbits):
            self.counters[i] = np.fromfile(fid, 'int32', 1)[0]
            self.rl[i] = np.fromfile(fid, 'float64', 1)[0]
            self.xi[i] = np.fromfile(fid, 'float64', 1)[0]
            self.kindOfCollision[i] = np.fromfile(fid, 'int32', 1)[0]
            self.data[i] = {
                'position': np.reshape(np.fromfile(fid, 'float64',
                                                   self.counters[i] * 3),
                                       (self.counters[i], 3), order='F')
            }
            if self.kindOfFile == 69:
                self.data[i]['velocity'] =\
                    np.reshape(np.fromfile(fid, 'float64',
                                           self.counters[i] * 3),
                               (self.counters[i], 3), order='F')
                self.data[i]['energy'] = \
                    0.5 * self.m * np.sum(self.data[i]['velocity']**2, axis=1)\
                    / sspar.ec / 1000.0

        self.header = {
            'units': {
                'position': 'cm',
                'velocity': 'cm/s',
                'mass': 'kg',
                'energy': 'keV'
             }
        }
        fid.close()
        if verbose:
            if self.kindOfFile == 69:
                print('Large Orbit file')
            else:
                print('Short orbit file')
            print('Hitting collimator: ', np.sum(self.kindOfCollision == 0))
            print('Hitting scintillator: ', np.sum(self.kindOfCollision == 2))
            print('Wrong markers: ', np.sum(self.kindOfCollision == 9)
                  + np.sum(self.kindOfCollision == 1))

    def size(self):
        """Get the number of loaded orbits."""
        return self.nOrbits

    def __getitem__(self, idx):
        """
        Overload of the method to be able to access the data in the orbit data.

        It returns the whole data of a geometry elements

        Copied from PabloOrbit object (see iHIBSIM library)

        :param  idx: element number

        :return self.data[idx]: Element dictionary
        """
        return self.data[idx]

    def plot3D(self, per=0.1, ax=None, line_params={}, imax=1000, kind=(2,), factor=1.0):
        """
        Plot the strike points in a 3D axis as scatter points

        Jose Rueda: jrrueda@us.es

        :param  per: ratio of markers to be plotted (1=all of them)
        :param  ax: axes where to plot (if none, they are created)
        :param  line_params: Parameters for plt.plot()
        :param  imax: maximum number of steps to plot
        :param  kindOfCollision: type of orbit to plot: (tuple or list)
            -0: Collimator colliding
            -1: Foil colliding but not scintillator collision
            -2: Scintillator colliding
            -9: Not colliding
        :param  factor: factor to multiply the coordinates by
        @ToDo: Include the rl, xi selector
        """
        # --- Default plotting options
        line_options = {
            'color': 'k'
        }
        line_options.update(line_params)
        # --- ensure we have a list:
        if not isinstance(kind, (tuple, list, np.ndarray)):
            kind = (kind,)
        # --- Create the axes
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            created = True
        else:
            created = False

        # --- Plot the markers:
        for i in range(self.nOrbits):
            if self.kindOfCollision[i] in kind:
                random_number = np.random.rand()
                if random_number < per:
                    imax_plot = min(imax, self.counters[i])
                    ax.plot(self.data[i]['position'][:imax_plot, 0]*factor,
                            self.data[i]['position'][:imax_plot, 1]*factor,
                            self.data[i]['position'][:imax_plot, 2]*factor,
                            **line_options)
        # --- Set properly the axis
        if created:
            ssplt.clean3Daxis(ax)
            ssplt.axisEqual3D(ax)
        return ax

    def plotEnergy(self, iorbit, ax=None, line_params: dict = {},
                   ax_params: dict = {}):
        """
        Plot the energy of the orbit given by iorbit

        :param  iorbit: number (or list) of orbits to plots
        :param  ax: axis where to plot, if none, new will be created
        :param  line_params: dicttionary with the line parameters
        """
        # --- Initialise plotting options
        ax_options = {
            'xlabel': 'Step',
            'ylabel': 'Energy [keV]',
            'grid': 'both'
        }

        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False

        if isinstance(iorbit, (list, np.ndarray)):
            orbits_to_plot = iorbit
        else:  # it should be just a number
            orbits_to_plot = np.array([iorbit])

        for i in orbits_to_plot:
            ax.plot(self.data[i]['energy'], **line_params, label=str(i))

        if created:
            ax = ssplt.axis_beauty(ax, ax_options)
        return ax

    def save_orbits_to_txt(self, kind=(2,), units: str = 'mm', seperated: bool = False ):
        """
        Save each individual orbit in a text file structred with collums for
        x y z positions of the orbits, to be easily uplaoded in CAD software

        Anton van Vuuren: avanvuuren@us.es

        :param  kindOfCollision: type of orbit to plot: (tuple or list)
            -0: Collimator colliding
            -1: Foil colliding but not scintillator collision
            -2: Scintillator colliding
            -9: Not colliding
        :param  units: Units in which to savethe orbit positions.
        """
        # --- Check the scale
        if units not in ['m', 'cm', 'mm']:
            raise Exception('Not understood units?')
        possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
        factor = possible_factors[units]

        # --- save the orbit steps:
        if seperated:
            for i in range(self.nOrbits):
                if self.kindOfCollision[i] in kind:
                    with open('orb_run_%s_rl_%.2f_xi_%.2f.txt'
                            % (self.runID[0].strip().decode("utf-8"),
                                self.rl[i],
                                np.rad2deg(np.arccos(self.xi[i]))), 'w') as f:
                        for p in range(len(self.data[i]['position'][:, 0])):
                            f.write('%f %f %f \n'
                                    % (self.data[i]['position'][p, 0] * factor,
                                    self.data[i]['position'][p, 1] * factor,
                                    self.data[i]['position'][p, 2] * factor)
                                    )
        else:
            with open('orb_run_%s.txt'
                    % (self.runID[0].strip().decode("utf-8")), 'w') as f:
                for i in range(self.nOrbits):
                    if self.kindOfCollision[i] in kind:
                        for p in range(len(self.data[i]['position'][:, 0])):
                            f.write('%f %f %f \n'
                                    % (self.data[i]['position'][p, 0] * factor,
                                    self.data[i]['position'][p, 1] * factor,
                                    self.data[i]['position'][p, 2] * factor)
                                    )