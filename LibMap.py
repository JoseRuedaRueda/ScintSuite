"""@package LibMap
Module to remap the scintillator

It contains the routines to load and aling the strike maps, as well as
perform the remapping
"""
# import time
import numpy as np


# def remap_grid(timing=True):
#     if timing:
#         tic = time.time()
#
#     if timing:
#         toc = time.time()
#         print('Elapsed time: ', toc - tic)


def transform_to_pixel(x, y, grid_param):
    """
    Transform from X,Y coordinates (points along the scintillator) to pixels
    in the camera

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

    @param x: Array of positions to be transformed, x coordinate
    @param y: Array of positions to be transformed, y coordinate
    @param grid_param: Object containing all the information for the
    transformation, see class GridParams()
    @return xpixel: x positions in pixels
    @return ypixel: y position in pixels
    @todo Include a model to take into account the distortion
    """
    alpha = grid_param.deg_pix * np.pi / 180
    xpixel = (np.cos(alpha) * x - np.sin(alpha) * y) * grid_param.xscale_pix + \
            grid_param.xshift_pix
    ypixel = (np.sin(alpha) * x + np.cos(alpha) * y) * grid_param.yscale_pix + \
            grid_param.yshift_pix

    return xpixel, ypixel


class StrikeMap:
    """
    Class with the information of the strike map
    """

    def __init__(self, flag, file):
        """
        Initialise the class

        @param flag: 0  means fild, 1 means INPA, 2 means iHIBP
        @param file: Full path to file with the strike map
        @todo Eliminate flag and extract info from file name??
        """
        ## X-position, in pixles, of the strike map
        self.xpixel = None
        ## Y-Position, in pixels, of the strike map
        self.ypixel = None

        if flag == 0:
            # Read the file
            dummy = np.loadtxt(file, skiprows=3)
            # See which rows has collimator factor larger than zero (ie see for
            # which combination of energy and pitch some markers has arrived)
            ind = dummy[:, 7] > 0
            # Initialise the class
            ## Gyroradius of map points
            self.gyroradius = dummy[ind, 0]
            ## Pitch of map points
            self.pitch = dummy[ind, 1]
            ## x coordinates of map points
            self.x = dummy[ind, 2]
            ## y coordinates of map points
            self.y = dummy[ind, 3]
            ## z coordinates of map points
            self.z = dummy[ind, 4]
            ## Average initial gyrophase of map markers
            self.avg_ini_gyrophase = dummy[ind, 5]
            ## Number of markers striking in this area
            self.n_strike_points = dummy[ind, 6]
            ## Collimator factor as defined in FILDSIM
            self.collimator_factor = dummy[ind, 7]
            ## Average incident angle of the FILDSIM markers
            self.avg_incident_angle = dummy[ind, 8]


class GridParams:
    """
    Class with the information to relate points in the camera sensor with
    points in the scintillator
    """

    def __init__(self):
        """
        Initializer of the class
        """
        # Image parameters: To transform from pixel to cm on the scintillator
        ## cm/pixel in the x direction
        self.xscale_im = 0
        ## cm/pixel in the y direction
        self.yscale_im = 0
        ## Offset to aling 0,0 of the sensor with the scintillator (x direction)
        self.xshift_im = 0
        ## Offset to aling 0,0 of the sensor with the scintillator (y direction)
        self.yshift_im = 0
        ## Rotation angle to transform from the sensor to the scintillator
        self.deg_im = 0
        # To transform the from real coordinates to pixel (see
        # transform_to_pixel())
        ## pixel/cm in the x direction
        self.xscale_pix = 0
        ## pixel/cm in the y direction
        self.yscale_pix = 0
        ## Offset to aling 0,0 of the sensor with the scintillator (x direction)
        self.xshift_pix = 0
        ## Offset to aling 0,0 of the sensor with the scintillator (y direction)
        self.yshift_pix = 0
        ## Rotation angle to transform from the sensor to the scintillator
        self.deg_pix = 0


class Scintillator:
    """
    Class with the scintillator information.

    Note, the notation is given by FILDSIM, and it is a bit misleading,
    in FILDSIM x,y,z axis are defined, the scintillator lies in a plane of
    constant x, so the only variables to play with are y,z. However, x,
    y are always used to refer to x horizontal and vertical direction in the
    camera sensor. We have to live with this. Just ignore the x coordintaes
    of the scintillator data and work with y,z as they were x,y
    """

    def __init__(self, file, material='TG-green'):
        ## Material used in the scintillator plate
        self.material = material
        # Read the file
        with open(file) as f:
            # Dummy line with description
            f.readline()
            # Line with the scinllator name
            dummy = f.readline()
            ## Name of the scintillator plate given in the simulation
            self.name = dummy[5:-1]
            # Line with the number of vertices
            dummy = f.readline()
            ## Number of vertices
            self.n_vertices = int(dummy[11:-1])
            # Skip the data with the vertices and the normal vector
            for i in range(self.n_vertices+3):
                f.readline()

            ## Units in which the scintillator data is loaded:
            dummy = f.readline()
            self.orig_units = dummy[:-1]

        ## Coordinates of the vertex of the scintillator (X,Y,Z). In cm
        self.coord_real = np.loadtxt(file, skiprows=3, delimiter=',',
                                     max_rows=self.n_vertices)
        ## Normal vector
        self.normal_vector = np.loadtxt(file, skiprows=4+self.n_vertices,
                                        delimiter=',', max_rows=1)
        ## Coordinates of the vertex of the scintillator in pixels
        self.coord_pix = None
        # We want the coordinates in cm, if 'cm' is not the unit, apply the
        # corresponding transformation. (Void it is interpreter as cm)
        factors = {'cm': 1, 'm': 0.01, 'inch': 2.54}
        if self.orig_units in factors:
            self.coord_real = self.coord_real * factors[self.orig_units]
        else:
            print('Not recognised unit, possible wrong format file!!!')
            print('Maybe you are using and old FILDSIM file, so do not panic')
            return

    def plot_px(self, ax):
        """
        Plot the scintillator, in pixels, in the axes ax
        @param ax: axes where to plot
        @return: Nothing, just update the plot
        """
        ax.plt(self.coord_pix[:, 1], self.coord_pix[:, 2], '--r')

    def plot_real(self, ax):
        """
        Plot the scintillator, in cm, in the axes ax
        @param ax: axes where to plot
        @return: Nothing, just update the plot
        """
        ax.plt(self.coord_real[:, 1], self.coord_real[:, 2], '--r')
