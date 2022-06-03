"""
Parent class for the strike map

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 0.10.0
"""

from Lib._StrikeMap._readMaps import readSmap
from Lib._Mapping._Common import XYtoPixel
from Lib._SideFunctions import createGrid
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as scipy_interp
from Lib.decorators import deprecated
import Lib.errors as errors
from tqdm import tqdm
from Lib._basicVariable import BasicVariable
import logging
logger = logging.getLogger('ScintSuite.StrikeMap')


class GeneralStrikeMap(XYtoPixel):
    """
    General class for StrikeMap handling

    Jose Rueda Rueda: jrrueda@us.es

    Public Methods:
        - setRemapVariables: Set the

    """

    def __init__(self, file, variables_to_remap: tuple = None,
                 code: str = None):
        """

        """
        # --- Init the parent class
        XYtoPixel.__init__(self)
        # --- Read the data
        self._header, self._data = readSmap(file, code=code)
        # --- Extract a copy of the x,y,z position, for the XYtoPixel to work
        self._coord_real = {
            'x1': self._data['x1'].data,
            'x2': self._data['x2'].data,
            'x3': self._data['x3'].data,
            'units': self._data['x1'].units
        }
        # --- Set the variables to later remap and interpolation
        if variables_to_remap is None:
            if self._header['diagnostic'] == 'INPA':
                variables_to_remap = ('R0', 'gyroradius')
            elif self._header['diagnostic'] == 'FILD':
                variables_to_remap = ('pitch', 'gyroradius')
            elif self._header['diagnostic'] == 'iHIBP':
                variables_to_remap = ('x1', 'x2')
                # ToDo. Set here the to proper iHIBP variables
        self.setRemapVariables(variables_to_remap, verbose=False)
        self._grid_interp = None  # allocate the atribute for latter
        self._map_interpolators = None
        # set tje shape of the map
        self._shape = self._header['shape']
        try:
            self._MC_variables = self._header['MC_variables']
        except KeyError:
            self._MC_variables = None

    def setRemapVariables(self, variables_to_remap: tuple,
                          verbose: bool = True):
        """
        Set the selected variables to perform the integration.

        Notice: calling this functions will update the interpolators

        @param variables_to_remap: tuple contianing the name of the 2 selected
            variables
        """
        self._to_remap = [self._data[name] for name in variables_to_remap]
        if verbose:
            print('Please call interp_grid to update the interpolators')

    def interp_grid(self, frame_shape, method: int = 2,
                    verbose: bool = False,
                    grid_params: dict = {}, MC_number: int = 100,
                    variables_to_interpolate: tuple = None,
                    limitation: float = 10.0):
        """
        Interpolate grid values on the frames.

        Jose Rueda Rueda: jrrueda@us.es

        @param frame_shape: Size of the frame used for the calibration (in px)
        @param method: method to calculate the interpolation:
            - 1: griddata linear (you can also write 'linear')
            - 2: griddata cubic  (you can also write 'cubic')
        @param verbose: flag to print some info along the way
        @param grid_params: grid options for the transformationn matrix grid
        @param MC_number: Number of MC markers for the transformation matrix,
        if this number < 0, the transformation matrix will not be calculated
        @param variables_to_interpolate: Variables we want to interpolate in
            the camera sensor. If none, just the self._to_remap variables will
            be used

        Info on the dictionary self.grid_interp:
            'nameX': Values of x variable in each pixel
            'collimator_factor': Collimator values of each pixel
            'interpolators': {
                'nameX': interpolator pixel -> x
            },
            'transformation_matrix': {
                'var1_var2': 4D tensor for the remap to var1_var2
            }
        """
        # --- Check inputs
        if self._coord_pix['x'] is None:
            raise Exception('Transform to pixel the strike map before')
        if variables_to_interpolate is None:
            variables_to_interpolate = \
                (self._to_remap[0].name, self._to_remap[1].name)
        # Prepare the options and interpolators for later
        if method == 1 or str(method).lower() == 'linear':
            met = 'linear'
            interpolator = scipy_interp.LinearNDInterpolator
        elif method == 2 or str(method).lower() == 'cubic':
            met = 'cubic'
            interpolator = scipy_interp.CloughTocher2DInterpolator
        else:
            raise errors.NotValidInput('Not recognized interpolation method')
        if verbose:
            print('Using %s interpolation of the grid' % met)
        if self._grid_interp is not None:
            if self._grid_interp['method'] != met:
                text = 'Interpolation method must equal the previous one'
                raise errors.NotValidInput(text)
        if len(variables_to_interpolate) > 2 and MC_number > 0:
            MC_number = 0
            text = '6: transformation matrix can not be computing for more'\
                + 'than 2 variables. Skiping the calculation'
            logger.warning(text)
        # --- Set default settigns
        grid_options = {
            'ymin': 1.2,
            'ymax': 5.5,
            'dy': 0.1,
            'xmin': 20.0,
            'xmax': 90.0,
            'dx': 2.0
        }
        grid_options.update(grid_params)
        # --- Create the grid for the interpolation
        grid_x, grid_y = np.mgrid[0:frame_shape[1], 0:frame_shape[0]]
        # --- 2: Interpolate the grid
        # Prepare the grid for te griddata method:
        dummy = np.column_stack((self._coord_pix['x'],
                                 self._coord_pix['y']))
        # Initialise the structure
        # this function can be called latter to interpolate other variable, so
        # we need to be sure to do not overwrite everything
        if self._grid_interp is None:
            self._grid_interp = dict.fromkeys(variables_to_interpolate)
            self._grid_interp['interpolators'] = \
                dict.fromkeys(variables_to_interpolate)
            self._grid_interp['method'] = met
        else:  # the dict was there so just include the new fields
            new_dict = dict.fromkeys(variables_to_interpolate)
            self._grid_interp.update(new_dict)
            self._grid_interp['interpolators'].update(new_dict)

        for coso in variables_to_interpolate:  # For you lina ;)
            try:
                dummy2 = \
                    scipy_interp.griddata(
                        dummy, self._data[coso].data,
                        (grid_x, grid_y), method=met, fill_value=1000.0
                    )
                self._grid_interp[coso] = dummy2.copy().T
                # Now define the interpolators for latter use
                grid = list(zip(self._coord_pix['x'],
                                self._coord_pix['y']))
                self._grid_interp['interpolators'][coso] = \
                    interpolator(grid, self._data[coso].data,
                                 fill_value=1000.0)
            except KeyError:  # the ihibp does not have coll factor
                logger.warning('6: %s not found!!! skiping' % coso)
        # --- Calculate the transformation matrix
        if MC_number > 0:
            self._calculate_transformation_matrix(
                MC_number, variables_to_interpolate, grid_options,
                frame_shape, limitation)

    def export_spatial_coordinates(self, Geom=None, units: str = 'mm',
                                   file_name_save: str = None,
                                   filename: str = 'Map.txt'):
        """
        Store the map coordinate in a simple txt.

        The srike map points are stored in three columns representing the X,
        Y and Z coordinates, which can then be easily loaded in CAD software.

        Anton van Vuuren: avanvuuren@us.es
        revised by jrrueda@us.es for version 0.10.0 and the new Smap object

        Note that this is just giving you the coordenates in a simple way to
        be loaded by CAD software (for example) No extra information is stored
        in the file

        @param: Geometry object with which to apply anti rotation
            and translation to recover the map in absoulte coordinates.
            If no Geometry object is given the strike map coordinates will be
            saved in the scintillator reference system (ie, pure coordinates of
            the smap file)
        @param units: Units in which to save the strikemap positions.
        @param file_name_save: name of the txt file to save the data. This is
            just kept for retrocompatibility issues, please use the input
            filename. If present, filename input will be ignored
        @param filename: name of the text file to store the strike map in
        """
        # --- Initialisate the settings
        if file_name_save is not None:
            filename = file_name_save
        # - Check the scale
        if units not in ['m', 'cm', 'mm', 'inch']:
            raise Exception('Not understood units?')
        possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0, 'inch': 0.254}
        # Factor to pass from the smap coordinate units to the desired one
        factor = possible_factors[units] \
            / possible_factors[self._coord_real['units']]
        # - Get the necessary roto-translation
        if Geom is not None:
            rot = Geom.ExtraGeometryParams['rotation']
            tras = Geom.ExtraGeometryParams['ps']
        else:
            rot = np.identity(3)
            tras = 0.
        # --- Save the map
        with open(file_name_save, 'w') as f:
            for xm, ym, zm in zip(self.x, self.y, self.z):
                point_rotated = rot.T @ (np.array([xm, ym, zm])) + tras
                f.write('%f %f %f \n' % (point_rotated[0] * factor,
                                         point_rotated[1] * factor,
                                         point_rotated[2] * factor))

    def plot_var(self, varname: str, varname2: str = None):
        """
        Perform a basic and quick plot of a variable data

        @param varname: name of the variable
        """
        # Get the variable
        var = self._data[varname]
        params = {
            'xlabel': 'Point ID',
            'ylabel': var.plot_label
        }
        if varname2 is not None:
            var2 = self._data[varname2]
            params = {
                'xlabel': var.plot_label,
                'ylabel': var2.plot_label
            }

        fig, ax = plt.subplots()
        if varname2 is None:
            ax.plot(var.data, 'x')
        else:
            ax.scatter(var.data, var2.data)
        return ax

    def _plot_real(self, ax=None,
                   marker_params: dict = {}, line_params: dict = {},
                   factor: float = 1.0):
        """
        Plot the strike map (x,y = dimensions in the scintillator).

        Jose Rueda: jrrueda@us.es

        Note, this just plot the grid of points, latter, the children will
        call this function and add the labels and so on

        @param ax: Axes where to plot
        @param markers_params: parameters for plt.plot() to plot the markers
        @param line_params: parameters for plt.plot() to plot the markers
        @param factor: scaling factor to plot the data. Dimensions will be
            multiplied by this factor. Notice that this is just to compare
            strike maps from different codes for the situations in which a code
            operate in cm and the oder in m.
        """
        # Default plot parameters:
        marker_options = {
            'markersize': 3,
            'fillstyle': 'none',
            'color': 'k',
            'marker': 'o',
            'linestyle': 'none'
        }
        marker_options.update(marker_params)
        line_options = {
            'color': 'k',
            'marker': ''
        }
        line_options.update(line_params)

        if ax is None:
            fig, ax = plt.subplots()
        # Draw the line of constant yToRemap variable, if you use default
        # settings this would be the lines of constant gyroradius
        # for FILD/INPA
        for i in range(self._MC_variables[0].data.size):
            flags = (self(self._MC_variables[0].name)
                     == self._MC_variables[0].data[i])
            ax.plot(self('x1')[flags] * factor,
                    self('x2')[flags] * factor,
                    **line_options)
        # Draw the lines of contant xToRemap variable, if you use default
        # settings those would be lines of contant pitch/R0 (for FILD/INPA)
        for i in range(self._MC_variables[1].data.size):
            flags = (self(self._MC_variables[1].name)
                     == self._MC_variables[1].data[i])
            ax.plot(self('x1')[flags] * factor,
                    self('x2')[flags] * factor,
                    **line_options)

        # Plot some markers in the grid position
        ax.plot(self('x1') * factor, self('x2') * factor,
                **marker_options)
        plt.draw()
        return ax

    def _plot_pix(self, ax=None,
                  marker_params: dict = {}, line_params: dict = {}):
        """
        Plot the strike map (x,y = dimensions in the scintillator).

        Jose Rueda: jrrueda@us.es

        Note, this just plot the grid of points, latter, the children will
        call this function and add the labels and so on

        @param ax: Axes where to plot
        @param markers_params: parameters for plt.plot() to plot the markers
        @param line_params: parameters for plt.plot() to plot the markers
        """
        # Default plot parameters:
        marker_options = {
            'markersize': 3,
            'fillstyle': 'none',
            'color': 'k',
            'marker': 'o',
            'linestyle': 'none'
        }
        marker_options.update(marker_params)
        line_options = {
            'color': 'k',
            'marker': ''
        }
        line_options.update(line_params)

        if ax is None:
            fig, ax = plt.subplots()
        # Draw the line of constant MC var 0 variable, if you use default
        # settings this would be the lines of constant pitch/R0
        # for FILD/INPA
        for i in range(self._MC_variables[0].data.size):
            flags = (self(self._MC_variables[0].name)
                     == self._MC_variables[0].data[i])
            ax.plot(self._coord_pix['x'][flags],
                    self._coord_pix['y'][flags],
                    **line_options)
        # Draw the lines of contant MC var 1 variable, if you use default
        # settings those would be lines of contant gyroradius (for FILD/INPA)
        for i in range(self._MC_variables[1].data.size):
            flags = (self(self._MC_variables[1].name)
                     == self._MC_variables[1].data[i])
            ax.plot(self._coord_pix['x'][flags],
                    self._coord_pix['y'][flags],
                    **line_options)

        # Plot some markers in the grid position
        ax.plot(self._coord_pix['x'].data, self._coord_pix['y'].data,
                **marker_options)
        plt.draw()
        return ax

    def _calculate_transformation_matrix(self, MC_number: int,
                                         variables: tuple,
                                         grid_options: dict,
                                         frame_shape: tuple,
                                         limitation: float = 10.0):
        """
        Calculate the transformation matrix from camera pixel to phase space

        @param MC_number: number of MC markers to launch for the calculation
        @param variables: set of 2 variables to be selected. Notice that they
            should be present in the _grid_interp (should be just 2 strings)
        @param grid_options: dictionary containing grid parameters, 'x' will
            apply for the first variable in the variable tuple and 'y' for the
            second
        @param frame_shape: shape of the camera sensor
        @param limitation: Maximum memory size for the transformation matrix
        """
        # Create the transformation matrix key, if not present
        if 'transformation_matrix' not in self._grid_interp.keys():
            self._grid_interp['transformation_matrix'] = {}

        # Initialise the random number generator
        rand = np.random.default_rng()
        generator = rand.uniform
        # Greate the grid
        nx, ny, xedges, yedges = createGrid(**grid_options)
        # Initialise the transformation matrix
        memory_size = nx * ny * frame_shape[0] * frame_shape[1] \
            * 8 / 1024 / 1024 / 1024
        if memory_size > limitation:
            print(memory_size)
            text = 'The requiring matrix will consume %2.1f Gb, this is above'\
                % memory_size\
                + 'the threshold. Increase it if you really want to proceed'
            raise errors.NotValidInput(text)
        transform = np.zeros((nx, ny,
                              frame_shape[0], frame_shape[1]), dtype='float64')
        print('Calculating transformation matrix')
        # This can be slightly confusing, but x,y are coordinates in the
        # camera sensor, and X,Y in the phase space.
        for i in tqdm(range(frame_shape[0])):
            for j in range(frame_shape[1]):
                # Generate markers coordinates in the chip, note the
                # first dimmension of the frame is y-pixel
                # (IDL heritage)
                x_markers = j + generator(size=MC_number)
                y_markers = i + generator(size=MC_number)
                # Calculate the r-pitch coordinates
                X_markers = \
                    self._grid_interp['interpolators'][variables[0]](x_markers,
                                                                     y_markers)
                Y_markers = \
                    self._grid_interp['interpolators'][variables[1]](x_markers,
                                                                     y_markers)
                # make the histogram in the X-Y space
                H, xedges, yedges = \
                    np.histogram2d(X_markers, Y_markers,
                                   bins=[xedges, yedges])
                transform[:, :, i, j] = H.copy()
        # Normalise the transformation matrix
        transform /= MC_number
        transform /= (grid_options['dx'] * grid_options['dy'])
        self._grid_interp['transformation_matrix'][
            variables[0] + '_' + variables[1]] = transform

    @deprecated('Please use directly export_spatial_coordinates. Tbr in 0.11')
    def map_to_txt(self, Geom=None,
                   units: str = 'mm',
                   file_name_save: str = 'Map.txt'):
        """
        Wrapper to export_spatial_coordinates, just for retrocompatibility,
        to be removed in version 0.11.0
        """
        self.export_spatial_coordinates(Geom=Geom, units=units,
                                        file_name_save=file_name_save)

    @property
    def shape(self):
        return self._shape

    @property
    def code(self):
        return self._header['code']

    @property
    def diagnostic(self):
        return self._header['diagnostic']

    @property
    def file(self):
        return self._header['file']

    @property
    def MC_variables(self):
        return self._MC_variables

    @property
    def variables(self):
        return self._header['variables']['name']

    def __call__(self, name):
        return self._data[name].data
