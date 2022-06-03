"""Scintillator class."""
import numpy as np
import matplotlib.pyplot as plt
from Lib._Mapping._Common import XYtoPixel
import Lib.errors as errors
__all__ = ['Scintillator']


class Scintillator(XYtoPixel):
    """
    Class with the scintillator information.

    Note, the notation is given by FILDSIM, and it is a bit misleading,
    in FILDSIM x,y,z axis are defined, the scintillator lies in a plane of
    constant x, so the only variables to play with are y,z. However, x,
    y are always used to refer to x horizontal and vertical direction in the
    camera sensor. We have to live with this. Just ignore the x coordinates
    of the scintillator data and work with y,z as they were x,y
    """

    def __init__(self, file: str, format: str = 'FILDSIM',
                 material: str = 'TG-green'):
        """
        Initialize the class.

        @param    file: Path to the file with the scintillator geometry
        @param    format: Code to which the file belongs, FILDSIM or SINPA
        @param    material: Defaults to 'TG-green'
        """
        XYtoPixel.__init__(self)
        ## Material used in the scintillator plate
        self.material = material
        ## Code (format) of the plate
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
            self.x, self.y, self.z =\
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
                self.x, self.y, self.z =\
                    np.loadtxt(file, skiprows=5, comments='!', unpack=True)
        else:
            raise errors.NotValidInput('Not recognised code')
        ## Coordinates of the vertex of the scintillator in pixels
        self._coord_pix = None
        self._coord_real = {
            'x1': self.y,
            'x2': self.z,
            'x3': self.x
        }

    def get_path_pix(self):
        """
        Returns the path covered by the scintillator in pixel coordinates.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        xdum = self._coord_pix['x1']
        ydum = self._coord_pix['x2']
        if self.code == 'fildsim':
            # FILDSIM geometry does not close the last line, so we have to add
            # it manually
            x = np.concatenate((xdum, np.array([self.xpixel[0]])))
            y = np.concatenate((ydum, np.array([self.ypixel[0]])))
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

        return x, y

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
            x = np.concatenate((xdum, np.array([self.xpixel[0]])))
            y = np.concatenate((ydum, np.array([self.ypixel[0]])))
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
        return ax

    def plot_real(self, ax=None, line_params: dict = {}):
        """
        Plot the scintillator, in cm, in the axes ax.

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
        ax.plot(self.y, self.z, **plt_options)
        return ax
