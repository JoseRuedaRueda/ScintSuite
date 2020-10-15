# Library to perform the mapping  operation
import time
import numpy as np


def remap_grid(timing=True):
    if timing:
        tic = time.time()

    if timing:
        toc = time.time()
        print('Elapsed time: ', toc - tic)


def transform_to_pixel(x, y, grid_param):
    """
    Transform from X,Y coordinates (points along the scintillator) to pixels
    in the camera
    @param x: Array of positions to be transformed, x coordinate
    @param y: Array of positions to be transformed, y coordinate
    @param grid_param: Object containing all the information for the
    transformation, see class GridParams()
    @return: x,y in pixels
    @todo: Include a model to take into account the distortion
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
    @param flag: 0  means fild, 1 means INPA, 2 means iHIBP
    @param file: Full path to file with the strike map
    @todo: Eliminate flag and extract info from file name??
    """

    def __init__(self, flag, file):
        self.xpixel = None
        self.ypixel = None
        if flag == 0:
            # Read the file
            dummy = np.loadtxt(file, skiprows=3)
            # See which rows has collimator factor larger than zero (ie see for
            # which combination of energy and pitch some markers has arrived)
            ind = dummy[:, 7] > 0
            # Initialise the class
            self.gyroradius = dummy[ind, 0]
            self.pitch = dummy[ind, 1]
            self.x = dummy[ind, 2]
            self.y = dummy[ind, 3]
            self.z = dummy[ind, 4]
            self.avg_ini_gyrophase = dummy[ind, 5]
            self.n_strike_points = dummy[ind, 6]
            self.collimator_factor = dummy[ind, 7]
            self.avg_incident_angle = dummy[ind, 8]


class GridParams:
    """
    Class with the information to relate points in the camera sensor with
    points in the scintillator
    """

    def __init__(self):
        # Image parameters
        self.xscale_im = 0
        self.yscale_im = 0
        self.xshift_im = 0
        self.yshift_im = 0
        self.deg_im = 0
        # To transform the from real coordinates to pixel (see
        # transform_to_pixel())
        self.xscale_pix = 0
        self.yscale_pix = 0
        self.xshift_pix = 0
        self.yshift_pix = 0
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
        self.material = material
        # Read the file
        self.coord_real = np.loadtxt(file, skiprows=3, delimiter=',')
        self.coord_pix = None

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
