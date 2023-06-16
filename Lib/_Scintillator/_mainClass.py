"""Scintillator class."""
import numpy as np
import Lib.errors as errors
import matplotlib.pyplot as plt
from Lib._Mapping._Common import XYtoPixel
from Lib._TimeTrace._roipoly import roipoly
from Lib._Scintillator._efficiency import ScintillatorEfficiency
__all__ = ['Scintillator']

import logging
logger = logging.getLogger('ScintSuite.Scintillator')


# ------------------------------------------------------------------------------
# --- Scintillator object
# ------------------------------------------------------------------------------
class Scintillator(XYtoPixel):
    """
    Class with the scintillator information.

    :author: Jose Rueda-Rueda jrrueda@us.es

    As for the other XYtoPixel coordinated, x1,x2 define the scintillator plate
    x3 is the normal to the plate, and the object proyection in the camera
    sensor is given by x,y

    :param     file: Path to the file with the scintillator geometry
    :param     format: Code to which the file belongs, FILDSIM or SINPA
    :param     material: Defaults to 'TG-green'
    :param     particle: Defaults to 'D' (deuterium)
    :param     thickness: thickness of the plate in mu-m

    :Example:
        >>> # Read the AUG FILD scintillator plate
        >>> import os
        >>> import Lib as ss
        >>> fileScint = os.path.join(ss.paths.ScintSuite, 'Data', 'Plates',
        >>>                          'FILD', 'AUG', 'AUG01.pl')
        >>> scint = ss.scint.Scintillator(file=fileScint)
    """

    def __init__(self, file: str = None, format: str = 'FILDSIM',
                 material: str = 'TgGreenA',
                 particle: str = 'D', thickness: int = 9):
        """Initialize the class."""
        # Init the parent class
        XYtoPixel.__init__(self)

        # ---- Allocate space for latter
        ## Material used in the scintillator plate
        self.material = None
        ## Code defining the geometry
        self.code = None
        ## Name of the scintillator plate given in the simulation
        self.name = None
        ## Name of the geometry file
        self.geomFile = None
        ## incident particle, for efficiency
        self.particle = particle
        ## Material thickness
        self.thickness = thickness
        ## Scintillator efficiency
        self.efficiency = None

        # ---- if possible read the geometry
        if file is not None:
            self._read_geometry(file, format)
        # ---- if possible read the efficiency
        try:
            self._read_efficiency(material=material, particle=particle,
                                  thickness=thickness)
        except (FileNotFoundError, OSError):
            pass

    # --------------------------------------------------------------------------
    # ---- Geometry
    # --------------------------------------------------------------------------
    def _read_geometry(self, file: str ,  format: str):
        """
        Read the geometry file.

        :author: Jose Rueda Rueda - jrrueda@us.es

        :param  file: file to load
        :param  format: 'FILDSIM' or 'SINPA', the format of the file read
        """
        self.geomFile = file
        ## Code defining the geometry
        self.code = format.lower()
        # Read the file
        if format.lower() == 'fildsim':
            with open(file) as f:
                # Dummy line with description
                f.readline()
                # Line with the scintillator name
                dummy = f.readline()
                ## Name of the scintillator plate given in the simulation
                self.name = dummy[5:-1]
                # Line with the number of vertices
                dummy = f.readline()
                ## Number of vertices
                self.n_vertices = int(dummy[11:-1])
                # Skip the data with the vertices and the normal vector
                for i in range(self.n_vertices + 3):
                    f.readline()
                ## Units in which the scintillator data is loaded:
                dummy = f.readline()
                self.units = dummy[:-1]

            ## Coordinates of the vertex of the scintillator (X,Y,Z).
            self._coord_real['x3'], self._coord_real['x1'], \
                self._coord_real['x2'] = \
                np.loadtxt(file, skiprows=3, delimiter=',',
                           max_rows=self.n_vertices, unpack=True)

            ## Normal vector
            self.normal_vector = np.loadtxt(file, skiprows=4 + self.n_vertices,
                                            delimiter=',', max_rows=1)

            # Transforming to meters.
            factor = {'m': 1.0,
                      'cm': 0.01,
                      'mm': 0.001,
                      'inch': 0.01/2.54}
            
            if self.units not in factor:
                factor = 0.01
                logger.warning('XX: Not found units, assuming cm.')
            else:
                factor = factor[self.units]

            self._coord_real['x1'] *= factor
            self._coord_real['x2'] *= factor
            self._coord_real['x3'] *= factor
            self.units = 'm'


        elif format.lower() == 'sinpa':
            with open(file, 'r') as f:
                self.name = f.readline().strip(),
                self.description = [f.readline().strip(), f.readline().strip()]
                self.n_vertices = int(np.loadtxt(file, max_rows=1, skiprows=4,
                                      comments='!')) * 3
                self._coord_real['x3'], self._coord_real['x1'], \
                    self._coord_real['x2'] = \
                    np.loadtxt(file, skiprows=5, comments='!', unpack=True)
        else:
            raise errors.NotValidInput('Not recognised code')

    # --------------------------------------------------------------------------
    # --- Reading the efficiency
    # --------------------------------------------------------------------------
    def _read_efficiency(self, material: str = 'TgGreenA',
                         particle: str = 'D',
                         thickness: int = 9):
        """
        Read the efficiency file.

        :author: Jose Rueda-Rueda - jrrueda@us.es

        :param material: Name of the material to load
        :param particle: Name of the specie to load
        :param thickness: Thickness of the scintillator material

        :Notes:
        - The particle, material and thickness will be used to deduce the name
          of the fille to load: `<material>/<particle>_<thickness>.dat`
        """
        self.material = material
        self.particle = particle
        self.thickness = thickness
        self.efficiency = \
            ScintillatorEfficiency(material=material, particle=particle,
                                   thickness=thickness, verbose=False)

    # --------------------------------------------------------------------------
    # ---- Mask for video timetraces
    # --------------------------------------------------------------------------
    def get_path_pix(self):
        """
        Return the path covered by the scintillator in pixel coordinates.

        This path allows latter to easily define a mask to integrate the video
        along it

        :author: Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        xdum = self._coord_pix['x']
        ydum = self._coord_pix['y']
        if self.code == 'fildsim':
            # FILDSIM geometry does not close the last line, so we have to add
            # it manually
            x = np.concatenate((xdum, np.array([self._coord_pix['x'][0]])))
            y = np.concatenate((ydum, np.array([self._coord_pix['y'][0]])))
        else:
            x = np.array([xdum[0], xdum[1], xdum[2], xdum[1],
                          xdum[0], xdum[2]])
            y = np.array([ydum[0], ydum[1], ydum[2], ydum[1],
                          ydum[0], ydum[2]])
            for i in range(1, int(self.n_vertices/3)):
                x = np.concatenate((x,
                                   np.array([xdum[3*i], xdum[3*i + 1],
                                             xdum[3*i+2], xdum[3*i + 1],
                                             xdum[3*i], xdum[3*i + 2]])))
                y = np.concatenate((y,
                                   np.array([ydum[3*i], ydum[3*i + 1],
                                             ydum[3*i+2], ydum[3*i + 1],
                                             ydum[3*i], ydum[3*i + 2]])))

        return np.array((x, y)).T[1:, ...]

    def get_roi(self):
        """
        Return a roipoly object with the countour of the scintillator.

        :author: Pablo Oyola

        :return: roipoly class
        """
        return roipoly(path=self.get_path_pix())

    # --------------------------------------------------------------------------
    # ---- Plotting
    # --------------------------------------------------------------------------
    def plot_pix(self, ax=None, line_params: dict = {}):
        """
        Plot the scintillator, in pixels, in the axes ax.

        :param  ax: axes where to plot
        :param  line_params: dictionary with the parameters to plot
        :return ax: axes used to plot
        """
        plt_options = {
            'color': 'w',
            'marker': '',
        }
        plt_options.update(line_params)
        if ax is None:
            fig, ax = plt.subplots()
        xdum = self._coord_pix['x']
        ydum = self._coord_pix['y']
        if self.code == 'fildsim':
            # FILDSIM geometry does not close the last line, so we have to add
            # it manually
            x = np.concatenate((xdum, np.array([self._coord_pix['x'][0]])))
            y = np.concatenate((ydum, np.array([self._coord_pix['y'][0]])))
        else:
            x = np.array([xdum[0], xdum[1], xdum[2], xdum[1],
                          xdum[0], xdum[2]])
            y = np.array([ydum[0], ydum[1], ydum[2], ydum[1],
                          ydum[0], ydum[2]])
            for i in range(1, int(self.n_vertices/3)):
                x = np.concatenate((x,
                                   np.array([xdum[3*i], xdum[3*i + 1],
                                             xdum[3*i+2], xdum[3*i + 1],
                                             xdum[3*i], xdum[3*i + 2]])))
                y = np.concatenate((y,
                                   np.array([ydum[3*i], ydum[3*i + 1],
                                             ydum[3*i+2], ydum[3*i + 1],
                                             ydum[3*i], ydum[3*i + 2]])))
        ax.plot(x, y, **plt_options)
        plt.draw()
        return ax

    def plot_real(self, ax=None, line_params: dict = {}, units: str=None):
        """
        Plot the scintillator, in real coordinates in the axes ax.

        :param  ax: axes where to plot
        :param  line_params: dictionary with the parameters to plot
        :return ax: the axes where the scintilator was drawn
        """
        plt_options = {
            'color': 'w',
            'marker': '',
        }
        plt_options.update(line_params)

        # Selectin the units for plotting.
        factor = { 'm': 1.0,
                   'cm': 100.0,
                   'mm': 1000.0,
                   'inch': 100.0/2.54
                 }.get(units)

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self._coord_real['x1']*factor,
                self._coord_real['x2']*factor, **plt_options)
        plt.draw()
        return ax
