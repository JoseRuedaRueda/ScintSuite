"""Scintillator class."""
import numpy as np
import matplotlib.pyplot as plt
import Lib.LibMap.Common as common


class Scintillator:
    """
    Class with the scintillator information.

    Note, the notation is given by FILDSIM, and it is a bit misleading,
    in FILDSIM x,y,z axis are defined, the scintillator lies in a plane of
    constant x, so the only variables to play with are y,z. However, x,
    y are always used to refer to x horizontal and vertical direction in the
    camera sensor. We have to live with this. Just ignore the x coordinates
    of the scintillator data and work with y,z as they were x,y
    """

    def __init__(self, file: str, material: str = 'TG-green'):
        """
        Initialize the class.

        @param    file: Path to the file with the scintillator geometry
        @param    material: Defaults to 'TG-green'
        """
        ## Material used in the scintillator plate
        self.material = material
        # Read the file
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
            self.orig_units = dummy[:-1]

        ## Coordinates of the vertex of the scintillator (X,Y,Z). In cm
        self.coord_real = np.loadtxt(file, skiprows=3, delimiter=',',
                                     max_rows=self.n_vertices)
        ## Normal vector
        self.normal_vector = np.loadtxt(file, skiprows=4 + self.n_vertices,
                                        delimiter=',', max_rows=1)
        ## Coordinates of the vertex of the scintillator in pixels
        self.xpixel = None
        self.ypixel = None
        # We want the coordinates in cm, if 'cm' is not the unit, apply the
        # corresponding transformation. (Void it is interpreter as cm)
        factors = {'cm': 1., 'm': 100., 'mm': 0.1, 'inch': 2.54}
        if self.orig_units in factors:
            self.coord_real = self.coord_real * factors[self.orig_units]
        else:
            print('Not recognised unit, possible wrong format file!!!')
            print('Maybe you are using and old FILDSIM file, so do not panic')
            return

    def plot_pix(self, ax=None, plt_par: dict = {}):
        """
        Plot the scintillator, in pixels, in the axes ax.

        @param ax: axes where to plot
        @param plt_par: dictionary with the parameters to plot
        @return: Nothing, just update the plot
        """
        if 'color' not in plt_par:
            plt_par['color'] = 'r'
        if 'markerstyle' not in plt_par:
            plt_par['marker'] = ''
        if 'linestyle' not in plt_par:
            plt_par['linestyle'] = '--'
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.xpixel, self.ypixel, **plt_par)

    def plot_real(self, ax=None, plt_par: dict = {}):
        """
        Plot the scintillator, in cm, in the axes ax.

        @param ax: axes where to plot
        @param plt_par: dictionary with the parameters to plot
        @return: Nothing, just update the plot
        """
        if 'color' not in plt_par:
            plt_par['color'] = 'r'
        if 'markerstyle' not in plt_par:
            plt_par['marker'] = ''
        if 'linestyle' not in plt_par:
            plt_par['linestyle'] = '--'
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.coord_real[:, 1], self.coord_real[:, 2], **plt_par)

    def calculate_pixel_coordinates(self, calib):
        """
        Transform the real coordinates of the map into pixels.

        Jose Rueda Rueda: jrrueda@us.es

        @param calib: a CalParams() object with the calibration info
        @return: Nothing, just update the plot
        """
        dummyx = self.coord_real[:, 1]
        dummyy = self.coord_real[:, 2]

        self.xpixel, self.ypixel = \
            common.transform_to_pixel(dummyx, dummyy, calib)
        return
