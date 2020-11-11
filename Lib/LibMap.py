"""@package LibMap
Module to remap the scintillator

It contains the routines to load and aling the strike maps, as well as
perform the remapping
"""
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interp
import LibPlotting as ssplt 


def transform_to_pixel(x, y, grid_param):
    """
    Transform from X,Y coordinates (points along the scintillator) to pixels
    in the camera

    Jose Rueda Rueda: jose.rueda@ipp.mpg.de

    @param x: Array of positions to be transformed, x coordinate
    @param y: Array of positions to be transformed, y coordinate
    @param grid_param: Object containing all the information for the
    transformation, see class CalParams()
    @return xpixel: x positions in pixels
    @return ypixel: y position in pixels
    @todo Include a model to take into account the distortion
    """
    alpha = grid_param.deg * np.pi / 180
    xpixel = (np.cos(alpha) * x - np.sin(alpha) * y) * grid_param.xscale + \
                grid_param.xshift
    ypixel = (np.sin(alpha) * x + np.cos(alpha) * y) * grid_param.yscale + \
                grid_param.yshift

    return xpixel, ypixel


def get_points(fig, scintillator, plt_flag: bool = True, npoints: int = 3):
    """
    Get the points of the scintillator via ginput method

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param fig: Axis where the scintillator is drawn
    @param scintillator: Scintillator object
    @param plt_flag: flag to plot
    @param npoints: number of points in the selection
    @return index: array with the index of each point, to be located in the
    scintillator.coord_real
    """
    print('Select the points in the scintillator')
    points = fig.ginput(npoints)
    index = np.zeros(npoints, dtype=int)
    for i in range(npoints):
        relative_pos = scintillator.coord_real[:, 1:3] - points[i]
        # print(relative_pos)
        index[i] = int(
            np.argmin(relative_pos[:, 1] ** 2 + relative_pos[:, 0] ** 2))
        # print(index[i])
        if plt_flag:
            plt.plot(scintillator.coord_real[index[i], 1],
                     scintillator.coord_real[index[i], 2], 'o')
            plt.text(scintillator.coord_real[index[i], 1],
                     scintillator.coord_real[index[i], 2], str(i + 1))
    return index


# def transformation_factors(index, scintillator, fig): DEPRECATED
#     """
#     Calculate the factor to translate from points in the scintillator to
#     pixels in the camera

#     Jose Rueda: jose.rueda@ipp.mpg.de

#     @param index: index array generated by get_points()
#     @param scintillator: Scintillator object
#     @param fig: the figure where the calibration image is plotted
#     @return:
#     """
#     # Number of points to be used
#     npoints = index.size
#     # Point on the calibration frame
#     print('Select the points on the calibration frame, in the same order as '
#           'before')
#     points_frame = fig.ginput(npoints)
#     # Initialise the variables
#     alpha = 0  # Rotation to be applied
#     mag = 0  # Magnification
#     offset = np.zeros(2)
#     for i in range(npoints):
#         # We will use the pair formed by each point and the following,
#         # for the case of the last point in the list, just take the next one
#         i2 = i + 1
#         if i2 == npoints:
#             i2 = 0
#         # Distance in the real scintillator
#         d_real = np.sqrt((scintillator.coord_real[index[i], 2] -
#                           scintillator.coord_real[index[i2], 2]) ** 2 +
#                          (scintillator.coord_real[index[i], 1] -
#                           scintillator.coord_real[index[i2], 1]) ** 2)
#         # Distance in the sensor
#         dummy = np.array(points_frame[i]) - np.array(points_frame[i2])
#         d_pix = np.sqrt(dummy[1] ** 2 + dummy[0] ** 2)
#         # Cumulate the magnification factor (we will normalise at the end)
#         mag = mag + d_pix / d_real
#         # Calculate the angles
#         alpha_real = math.atan2(scintillator.coord_real[index[i], 2] -
#                                 scintillator.coord_real[index[i2], 2],
#                                 scintillator.coord_real[index[i], 1] -
#                                 scintillator.coord_real[index[i2], 1])
#         # If alpha == 180, it can be also -180, atan2 fails here, check which
#         # one is the case
#         if int(alpha_real * 180 / np.pi) == 180:
#             print('Correcting angle')
#             if scintillator.coord_real[index[i2],
#                                        1] > scintillator.coord_real[
#                 index[i], 1]:
#                 alpha_real = - alpha_real

#         alpha_px = math.atan2(dummy[1], dummy[0])
#         alpha = alpha + (alpha_px - alpha_real)
#         # Transform the coordinate to estimate the offset
#         x_new = (scintillator.coord_real[index[i], 1] *
#                  math.cos(alpha_px - alpha_real) -
#                  scintillator.coord_real[index[i], 2] *
#                  math.sin(alpha_px - alpha_real)) * d_pix / d_real
#         y_new = (scintillator.coord_real[index[i], 1] *
#                  math.sin(alpha_px - alpha_real) +
#                  scintillator.coord_real[index[i], 2] *
#                  math.cos(alpha_px - alpha_real)) * d_pix / d_real
#         offset = offset + np.array(points_frame[i]) - np.array((x_new, y_new))
#         # print(alpha_px*180/np.pi, alpha_real*180/np.pi)
#         # print((alpha_px-alpha_real)*180/np.pi)
#     # Normalise magnification and angle
#     mag = mag / npoints
#     alpha = alpha / npoints
#     offset = offset / npoints
#     return mag, alpha, offset


def calculate_transformation_factors(scintillator, fig, plt_flag: bool = True):
    """
    Calculate the transformation factor to align the strike map with the
    camera sensor

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param scintillator: Scintillator object
    @param fig: figure where the calibration image is drawn
    @param plt_flag: flag to plot the selected points to align the scintillator
    @return xmag: magnification factor in the x direction to align strike map
    @return mag: magnification factor in the y direction to align the strike map
    @return alpha: rotation angle to orientate the strike map
    @return offset: offset to place the strike map
    @todo check with a non inverted png if the minus I included to align the
    scintillator is needed or not
    """

    # Plot the scintillator
    fig_scint, ax_scint = plt.subplots()
    scintillator.plot_real(ax_scint)

    # Select the points in the scintillator
    index = get_points(fig_scint, scintillator, plt_flag)
    npoints = index.size
    print(npoints)
    # Select the corresponding points in the reference frame
    print('Select the points on the calibration frame, in the same order as '
          'before')
    points_frame = fig.ginput(npoints)
    print(points_frame)
    # Define the vectors which will give us the reference
    v21_real = np.array((scintillator.coord_real[index[0], 1] -
                         scintillator.coord_real[index[1], 1],
                         scintillator.coord_real[index[0], 2] -
                         scintillator.coord_real[index[1], 2], 0))
    v23_real = np.array((scintillator.coord_real[index[2], 1] -
                         scintillator.coord_real[index[1], 1],
                         scintillator.coord_real[index[2], 2] -
                         scintillator.coord_real[index[1], 2], 0))
    print(v21_real)
    print(points_frame[0][0])
    a = np.array((points_frame[0][0], points_frame[0][1], 0))
    print(a)
    print(a.shape)

    v21_pix = np.array([points_frame[0][0], points_frame[0][1], 0]) - \
                  np.array([points_frame[1][0], points_frame[1][1], 0])

    v23_pix = np.array([points_frame[2][0], points_frame[2][1], 0]) - \
                  np.array([points_frame[1][0], points_frame[1][1], 0])

    # See if an inversion of one of the axis is needed or not.
    normal_real = np.cross(v21_real, v23_real)
    print(normal_real)
    normal_pix = np.cross(v21_pix, v23_pix)
    print(normal_pix)
    # If the normals has opposite signs, an inversion must be done
    if normal_pix[2] * normal_real[2] < 0:
        sign = -1.0
    else:
        sign = 1.0
    # With this sign in mind, now we can proceed with the calculation of the
    # ratio and the gyration angle
    # Initialise the variables
    alpha = 0  # Rotation to be applied
    mag = 0  # Magnification
    offset = np.zeros(2)
    for i in range(npoints):
        # We will use the pair formed by each point and the following,
        # for the case of the last point in the list, just take the next one
        i2 = i + 1
        if i2 == npoints:
            i2 = 0
        # Distance in the real scintillator
        d_real = np.sqrt((scintillator.coord_real[index[i], 2] -
                          scintillator.coord_real[index[i2], 2]) ** 2 +
                         (scintillator.coord_real[index[i], 1] -
                          scintillator.coord_real[index[i2], 1]) ** 2)
        # Distance in the sensor
        dummy = np.array(points_frame[i]) - np.array(points_frame[i2])
        d_pix = np.sqrt(dummy[1] ** 2 + dummy[0] ** 2)
        # Cumulate the magnification factor (we will normalise at the end)
        mag = mag + d_pix / d_real
        # Calculate the angles
        alpha_r = -  math.atan2(scintillator.coord_real[index[i], 2] -
                                scintillator.coord_real[index[i2], 2],
                                sign * scintillator.coord_real[index[i], 1] -
                                sign * scintillator.coord_real[index[i2], 1])
        # If alpha == 180, it can be also -180, atan2 fails here, check which
        # one is the case
        if int(alpha_r * 180 / np.pi) == 180:
            print('Correcting angle')
            if scintillator.coord_real[index[i2], 1] > scintillator.coord_real[
                                                        index[i], 1]:
                alpha_r = - alpha_r

        alpha_px = - math.atan2(dummy[1], dummy[0])
        alpha = alpha + (alpha_px - alpha_r)
        # Transform the coordinate to estimate the offset
        x_new = (scintillator.coord_real[index[i], 1] *
                 math.cos(alpha_px - alpha_r) -
                 scintillator.coord_real[index[i], 2] *
                 math.sin(alpha_px - alpha_r)) * d_pix / d_real * sign
        y_new = (scintillator.coord_real[index[i], 1] *
                 math.sin(alpha_px - alpha_r) +
                 scintillator.coord_real[index[i], 2] *
                 math.cos(alpha_px - alpha_r)) * d_pix / d_real
        offset = offset + np.array(points_frame[i]) - np.array((x_new, y_new))
        # print(alpha_px*180/np.pi, alpha_real*180/np.pi)
        # print((alpha_px-alpha_real)*180/np.pi)
    # Normalise magnification and angle
    mag = mag / npoints
    xmag = sign * mag
    alpha = alpha / npoints * 180 / np.pi
    offset = offset / npoints
    return xmag, mag, alpha, offset[0], offset[1]


def remap(smap, timing=True):
    """
    Remap a frame

    Jose Rueda: jose.rueda@ipp.mpg.de

    @param smap: StrikeMap() object with the strike map
    @param timing: Flag to time the routine or not
    """
    # --- 0: Check inputs
    if smap.xpixel is None:
        print('Please, transform to pixel the strike map before!!!')
        error = 1
        return error, 0
    
    
    if timing:
        tic = time.time()

    

    if timing:
        toc = time.time()
        print('Elapsed time: ', toc - tic)


class CalibrationDatabase:
    """
    Class with with the database of parameter to align the scintillator
    """

    def __init__(self, filename: str, n_header: int = 5):
        """
        Read the calibration database, to align the strike maps

        See @ref database for a full documentation of each field

        @param filename: Complete path to the file with the calibrations
        @param n_header: Number of header lines
        @return database: Dictionary containing the database information
        """
        ## Name of file with the data
        self.file = filename
        ## Header of the file
        self.header = []
        ## Dictionary with the data from the calibration. See @ref database
        ## for a full description of the meaning of each field
        self.data = {'ID': [], 'camera': [], 'shot1': [], 'shot2': [],
                     'xshift': [], 'yshift': [], 'xscale': [], 'yscale': [],
                     'deg': [], 'cal_type': [], 'diag_ID': []}
        # Open the file
        with open(filename) as f:
            for i in range(n_header):
                # Lines with description
                self.header = self.header + [f.readline()]
            # Database itself
            for line in f:
                dummy = line.split()
                self.data['ID'] = self.data['ID'] + [int(dummy[0])]
                self.data['camera'] = self.data['camera'] + [dummy[1]]
                self.data['shot1'] = self.data['shot1'] + [int(dummy[2])]
                self.data['shot2'] = self.data['shot2'] + [int(dummy[3])]
                self.data['xshift'] = self.data['xshift'] + [float(dummy[4])]
                self.data['yshift'] = self.data['yshift'] + [float(dummy[5])]
                self.data['xscale'] = self.data['xscale'] + [float(dummy[6])]
                self.data['yscale'] = self.data['yscale'] + [float(dummy[7])]
                self.data['deg'] = self.data['deg'] + [float(dummy[8])]
                self.data['cal_type'] = self.data['cal_type'] + [dummy[9]]
                self.data['diag_ID'] = self.data['diag_ID'] + [int(dummy[10])]

    def append_to_database(self, camera: str, shot1: int, shot2: int,
                           xshift: float, yshift: float, xscale: float,
                           yscale: float, deg: float, cal_type: str,
                           diag_ID: str):
        """
        Add a new entry to the database

        @param camera:
        @param shot1:
        @param shot2:
        @param xshift:
        @param yshift:
        @param xscale:
        @param yscale:
        @param deg:
        @param cal_type:
        @param diag_ID:
        """
        self.data['ID'] = self.data['ID'] + [self.data['ID'][-1] + 1]
        self.data['camera'] = self.data['camera'] + [camera]
        self.data['shot1'] = self.data['shot1'] + [shot1]
        self.data['shot2'] = self.data['shot2'] + [shot2]
        self.data['xshift'] = self.data['xshift'] + [xshift]
        self.data['yshift'] = self.data['yshift'] + [yshift]
        self.data['xscale'] = self.data['xscale'] + [xscale]
        self.data['yscale'] = self.data['yscale'] + [yscale]
        self.data['deg'] = self.data['deg'] + [deg]
        self.data['cal_type'] = self.data['cal_type'] + [cal_type]
        self.data['diag_ID'] = self.data['diag_ID'] + [diag_ID]

    def write_database_to_txt(self, file: str = ''):
        """
        Write database into a txt.

        If no name is given, the name of the loaded file will be used but a
        'new' will be added. Example: if the file from where the info has
        been loaded is 'calibration.txt' the new file would be
        'calibration_new.txt'. This is just to be save and avoid overwriting
        the original database.

        @param file: name of the file where to write the results
        @return : a file created inth your information
        """
        if file == '':
            file = self.file[:-4] + '_new.txt'
        with open(file, 'w') as f:
            # Write the header
            for i in range(len(self.header)):
                f.write(self.header[i])
            # Write the database information
            for i in range(len(self.data['ID'])):
                line = str(self.data['ID'][i]) + ' ' + \
                       self.data['camera'][i] + ' ' + \
                       str(self.data['shot1'][i]) + ' ' + \
                       str(self.data['shot2'][i]) + ' ' + \
                       str(self.data['xshift'][i]) + ' ' + \
                       str(self.data['yshift'][i]) + ' ' + \
                       str(self.data['xscale'][i]) + ' ' + \
                       str(self.data['yscale'][i]) + ' ' + \
                       str(self.data['deg'][i]) + ' ' + \
                       self.data['cal_type'][i] + ' ' + \
                       str(self.data['diag_ID'][i]) + ' ' + '\n'
                f.write(line)
            print('File ' + file + ' writen')

    def get_calibration(self, shot, camera, cal_type, diag_ID):
        """
        Give the calibration parameter of a precise database entry

        @param shot: Shot number for which we want the calibration
        @param camera: Camera used
        @param cal_type: Type of calibration we want
        @param diag_ID: ID of the diagnostic we want
        @return cal: CalParams() object
        """

        flags = np.zeros(len(self.data['ID']))
        for i in range(len(self.data['ID'])):
            if (self.data['shot1'][i] <= shot) * \
                    (self.data['shot2'][i] >= shot) * \
                    (self.data['camera'][i] == camera) * \
                    (self.data['cal_type'][i] == cal_type) * \
                    (self.data['diag_ID'][i] == diag_ID):
                flags[i] = True

        n_true = sum(flags)

        if n_true == 0:
            print('No entry find in the database, revise database')
            return
        elif n_true > 1:
            print('Several entries fulfull the condition')
            print('Possible entries:')
            print(self.data['ID'][flags])
        else:
            dummy = np.argmax(np.array(flags))
            cal = CalParams()
            cal.xscale = self.data['xscale'][dummy]
            cal.yscale = self.data['yscale'][dummy]
            cal.xshift = self.data['xshift'][dummy]
            cal.yshift = self.data['yshift'][dummy]
            cal.deg = self.data['deg'][dummy]

            return cal


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
        ## Associated diagnostic
        if flag == 0:
            self.diag = 'FILD'
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
            ## Matrix to translate from pixels in the camera to gyroradius
            self.gyr_interp = None
            ## Matrix to translate from pixels in the camera to pitch
            self.pit_interp = None
            ## Matrix to translate from pixels in the camera to collimator factor
            self.col_interp = None

    def plot_real(self, ax, plt_param: dict = None, line_param: dict = None):
        """
        Plot the strike map (x,y = dimensions in the scintillator)

        Jose Rueda: jose.rueda@ipp.mpg.de

        @param ax: Axes where to plot
        @param plt_param: Parameters for plot beauty, example: markersize=6
        @return: Strike maps over-plotted in the axis

        """
        # Define some standard parameters:
        if plt_param is None:
            plt_param = {}
        if 'markersize' not in plt_param:
            plt_param['markersize'] = 6
        if 'fillstyle' not in plt_param:
            plt_param['fillstyle'] = 'none'
        if 'color' not in plt_param:
            plt_param['color'] = 'w'
        if 'marker' not in plt_param:
            plt_param['marker'] = 'o'
        if 'linestyle' not in plt_param:
            plt_param['linestyle'] = 'none'

        if 'color' not in line_param:
            line_param['color'] = 'w'
        if 'markerstyle' not in line_param:
            line_param['marker'] = ''

        # Draw the lines of constant gyroradius, energy, or rho (depending on
        # the particular diagnostic) [These are the 'horizontal' lines]
        if hasattr(self, 'gyroradius'):
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_param)
        else:
            return
            ## @todo: talk with Pablo about his strike maps and his coordinates

        # Draw the lines of constant pitch (depending of the diagnostic,
        # in INPA would be constant radius [these are the vertical lines]
        if hasattr(self, 'pitch'):
            uniq = np.unique(self.pitch)
            n = len(uniq)
            for i in range(n):
                flags = self.pitch == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_param)
        else:
            return
            ## todo: change == by a < tol??

        # Plot some markers in the grid position
        ## todo include labels energy/pitch in the plot
        ax.plot(self.y, self.z, **plt_param)

    def plot_pix(self, ax, plt_param: dict = None, line_param: dict = {}):
        """
        Plot the strike map (x,y = pixels on the camera)

        Jose Rueda: jose.rueda@ipp.mpg.de

        @param ax: Axes where to plot
        @param plt_param: Parameters for plot beauty, example: markersize=6
        @return: Strike maps over-plotted in the axis
        """
        # Define some standard parameters:
        if plt_param is None:
            plt_param = {}
        if 'markersize' not in plt_param:
            plt_param['markersize'] = 6
        if 'fillstyle' not in plt_param:
            plt_param['fillstyle'] = 'none'
        if 'color' not in plt_param:
            plt_param['color'] = 'w'
        if 'marker' not in plt_param:
            plt_param['marker'] = 'o'
        if 'linestyle' not in plt_param:
            plt_param['linestyle'] = 'none'

        if 'color' not in line_param:
            line_param['color'] = 'w'
        if 'markerstyle' not in line_param:
            line_param['marker'] = ''

        # Draw the lines of constant gyroradius, energy, or rho (depending on
        # the particular diagnostic) [These are the 'horizontal' lines]
        if hasattr(self, 'gyroradius'):
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_param)
        else:
            return
            ## @todo: talk with Pablo about his strike maps and his coordinates

        # Draw the lines of constant pitch (depending of the diagnostic,
        # in INPA would be constant radius [these are the vertical lines]
        if hasattr(self, 'pitch'):
            uniq = np.unique(self.pitch)
            n = len(uniq)
            for i in range(n):
                flags = self.pitch == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_param)
        else:
            return
            ## todo: change == by a < tol??

        # Plot some markers in the grid position
        ## todo include labels energy/pitch in the plot
        ax.plot(self.xpixel, self.ypixel, **plt_param)

    def calculate_pixel_coordinates(self, calib):
        """
        Transform the real coordinates of the map into pixels

        Jose Rueda Rueda: jose.rueda@ipp.mpg.de

        @param calib: a CalParams() object with the calibration info
        """
        self.xpixel, self.ypixel = transform_to_pixel(self.y, self.z, calib)
    
    def interp_grid(self, frame_shape, method=2, plot=False):
        """
        Interpolate grid values on the frames
    
        Error codes:
            - 0: Scintillator map pixel position not calculated before
            - 1: Interpolating method not recognised
    
        @param smap: StrikeMap() object
        @param frame_shape: Size of the frame used for the calibration (in px)
        @param method: method to calculate the interpolation:
            - 0: griddata nearest (not recomended)
            - 1: griddata linear
            - 2: griddata cubic
        """
        # --- 0: Check inputs
        if self.xpixel is None:
            print('Please, transform to pixel the strike map before!!!')
            error = 1
            return error
        # --- 1: Create grid for the interpolation
        grid_x, grid_y = np.mgrid[0:frame_shape[0], 0:frame_shape[1]]
        # --- 2: Interpolate the gyroradius
        dummy = np.column_stack((self.xpixel, self.ypixel))
        if method == 0:
            if self.diag == 'FILD':
                self.gyr_interp = scipy_interp.griddata(dummy,
                                                        self.gyroradius,
                                                        (grid_x, grid_y),
                                                        method='nearest',
                                                        fill_value=1000)
                self.gyr_interp = self.gyr_interp.transpose()
                self.pit_interp = scipy_interp.griddata(dummy,
                                                        self.pitch,
                                                        (grid_x, grid_y),
                                                        method='nearest',
                                                        fill_value=1000)
                self.pit_interp = self.pit_interp.transpose()
                self.col_interp = scipy_interp.griddata(dummy,
                                                        self.collimator_factor,
                                                        (grid_x, grid_y),
                                                        method='nearest',
                                                        fill_value=1000)
                self.col_interp = self.col_interp.transpose()
        elif method == 1:
            if self.diag == 'FILD':
                self.gyr_interp = scipy_interp.griddata(dummy,
                                                        self.gyroradius,
                                                        (grid_x, grid_y),
                                                        method='linear',
                                                        fill_value=1000)
                self.gyr_interp = self.gyr_interp.transpose()
                self.pit_interp = scipy_interp.griddata(dummy,
                                                        self.pitch,
                                                        (grid_x, grid_y),
                                                        method='linear',
                                                        fill_value=1000)
                self.pit_interp = self.pit_interp.transpose()
                self.col_interp = scipy_interp.griddata(dummy,
                                                        self.collimator_factor,
                                                        (grid_x, grid_y),
                                                        method='linear',
                                                        fill_value=1000)
                self.col_interp = self.col_interp.transpose()
        elif method == 2:
            if self.diag == 'FILD':
                self.gyr_interp = scipy_interp.griddata(dummy,
                                                        self.gyroradius,
                                                        (grid_x, grid_y),
                                                        method='cubic',
                                                        fill_value=1000)
                self.gyr_interp = self.gyr_interp.transpose()
                self.pit_interp = scipy_interp.griddata(dummy,
                                                        self.pitch,
                                                        (grid_x, grid_y),
                                                        method='cubic',
                                                        fill_value=1000)
                self.pit_interp = self.pit_interp.transpose()
                self.col_interp = scipy_interp.griddata(dummy,
                                                        self.collimator_factor,
                                                        (grid_x, grid_y),
                                                        method='cubic',
                                                        fill_value=1000)
                self.col_interp = self.col_interp.transpose()
        else:
            error = 2
            print('Method not understood')
            return error

        # --- Plot
        if plot:
            if self.diag == 'FILD':
                fig, axes = plt.subplots(2, 2)
                # Plot the scintillator grid
                self.plot_pix(axes[0, 0], line_param={'color': 'k'})
                axes[0, 0].set_xlim(0, frame_shape[0])
                axes[0, 0].set_ylim(0, frame_shape[1])
                # Plot the interpolated gyroradius
                c1 = axes[0, 1].contourf(grid_x, grid_y,
                                         self.gyr_interp.transpose(),
                                         cmap=ssplt.Gamma_II())
                axes[0, 1].set_xlim(0, frame_shape[0])
                axes[0, 1].set_ylim(0, frame_shape[1])
                fig.colorbar(c1, ax=axes[0, 1], shrink=0.9)
                # Plot the interpolated gyroradius
                c2 = axes[1, 0].contourf(grid_x, grid_y,
                                         self.pit_interp.transpose(),
                                         cmap=ssplt.Gamma_II())
                axes[1, 0].set_xlim(0, frame_shape[0])
                axes[1, 0].set_ylim(0, frame_shape[1])
                fig.colorbar(c2, ax=axes[1, 0], shrink=0.9)
                # Plot the interpolated gyroradius
                c3 = axes[1, 1].contourf(grid_x, grid_y,
                                         self.col_interp.transpose(),
                                         cmap=ssplt.Gamma_II())
                axes[1, 1].set_xlim(0, frame_shape[0])
                axes[1, 1].set_ylim(0, frame_shape[1])
                fig.colorbar(c3, ax=axes[1, 1], shrink=0.9)


class CalParams:
    """
    Class with the information to relate points in the camera sensor with
    points in the scintillator
    """

    def __init__(self):
        """
        Initializer of the class
        """
        # To transform the from real coordinates to pixel (see
        # transform_to_pixel())
        ## pixel/cm in the x direction
        self.xscale = 0
        ## pixel/cm in the y direction
        self.yscale = 0
        ## Offset to align 0,0 of the sensor with the scintillator (x direction)
        self.xshift = 0
        ## Offset to align 0,0 of the sensor with the scintillator (y direction)
        self.yshift = 0
        ## Rotation angle to transform from the sensor to the scintillator
        self.deg = 0


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

    def __init__(self, file, material='TG-green'):
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
        factors = {'cm': 1, 'm': 0.01, 'inch': 2.54}
        if self.orig_units in factors:
            self.coord_real = self.coord_real * factors[self.orig_units]
        else:
            print('Not recognised unit, possible wrong format file!!!')
            print('Maybe you are using and old FILDSIM file, so do not panic')
            return

    def plot_pix(self, ax, plt_par: dict):
        """
        Plot the scintillator, in pixels, in the axes ax
        
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
        ax.plot(self.xpixel, self.ypixel, **plt_par)

    def plot_real(self, ax, plt_par: dict):
        """
        Plot the scintillator, in cm, in the axes ax
        
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
        ax.plot(self.coord_real[:, 1], self.coord_real[:, 2], **plt_par)

    def calculate_pixel_coordinates(self, calib):
        """
        Transform the real coordinates of the map into pixels

        Jose Rueda Rueda: jose.rueda@ipp.mpg.de

        @param calib: a CalParams() object with the calibration info
        @return: Nothing, just update the plot
        """
        dummyx = self.cord_real[:, 1]
        dummyy = self.cord_real[:, 2]

        self.xpixel, self.ypixel = transform_to_pixel(dummyx, dummyy, calib)
