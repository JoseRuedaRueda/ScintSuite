"""Scintillator class."""
import numpy as np
import Lib.errors as errors
import matplotlib.pyplot as plt
from Lib._Mapping._Common import XYtoPixel
from Lib._TimeTrace._roipoly import roipoly
from Lib._Scintillator._efficiency import ScintillatorEfficiency
__all__ = ['Scintillator']


# ------------------------------------------------------------------------------
# --- Scintillator object
# ------------------------------------------------------------------------------
class Scintillator(XYtoPixel):
    """
    Class with the scintillator information.

    As for the other XYtoPixel coordinated, x1,x2 define the scintillator plate
    x3 is the normal to the plate, and the object proyection in the camera
    sensor is given by x,y
    """

    def __init__(self, file: str = None, format: str = 'FILDSIM',
                 material: str = 'TgGreenA', particle='D', thickness=9):
        """
        Initialize the class.

        - Geometry files
        @param    file: Path to the file with the scintillator geometry
        @param    format: Code to which the file belongs, FILDSIM or SINPA
        - Efficiency related attributes
        @param    material: Defaults to 'TG-green'
        @param    particle: Defaults to 'D' (deuterium)
        @param    thickness: thickness of the plate in mu-m
        """
        XYtoPixel.__init__(self)

        # --- Allocate space for latter
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

        # --- if possible read the geometry
        if file is not None:
            self._read_geometry(file, format)
        # --- if possible read the efficiency
        try:
            self._read_efficiency(material=material, particle=particle,
                                  thickness=thickness)
        except (FileNotFoundError, OSError):
            pass

    # --------------------------------------------------------------------------
    # --- Reading geometry
    # --------------------------------------------------------------------------
    def _read_geometry(self, file,  format):
        """
        Read the geometry file
        @param file:
        @param format:
        @return:
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
    def _read_efficiency(self, material: str = 'TgGreenA', particle='D',
                         thickness=9):
        """

        :param material:
        :param particle:
        :param thickness:
        :return:
        """
        self.material = material
        self.particle = particle
        self.thickness = thickness
        self.efficiency = \
            ScintillatorEfficiency(material=material, particle=particle,
                                   thickness=thickness, verbose=False)
    # --------------------------------------------------------------------------
    # --- Mask for video timetraces
    # --------------------------------------------------------------------------
    def get_path_pix(self):
        """
        Returns the path covered by the scintillator in pixel coordinates.

        This path allows latter to easily define a mask to integrate the video
        along it

        Pablo Oyola - pablo.oyola@ipp.mpg.de
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

        return np.array((x,y)).T[1:, ...]

    def get_roi(self):
        """
        Return a roipoly object with the countour of the scintillator
        :return: roipoly class
        """
        return roipoly(path=self.get_path_pix())

    # --------------------------------------------------------------------------
    # --- Plotting
    # --------------------------------------------------------------------------
    def plot_pix(self, ax=None, line_params: dict = {}):
        """
        Plot the scintillator, in pixels, in the axes ax.

        @param ax: axes where to plot
        @param line_params: dictionary with the parameters to plot
        @return: ax axes used to plot
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

    def plot_real(self, ax=None, line_params: dict = {}):
        """
        Plot the scintillator, in real coordinates in the axes ax.

        @param ax: axes where to plot
        @param line_params: dictionary with the parameters to plot
        @return: Nothing, just update the plot
        """
        plt_options = {
            'color': 'w',
            'marker': '',
        }
        plt_options.update(line_params)

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self._coord_real['x1'], self._coord_real['x2'], **plt_options)
        plt.draw()
        return ax
