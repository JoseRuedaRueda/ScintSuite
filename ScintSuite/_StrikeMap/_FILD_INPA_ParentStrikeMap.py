"""
Parent Strike Map for INPA and FILD diagnostic

Jose Rueda: jrrueda@us.es

Introduced in version 0.10.0
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interp
import ScintSuite.errors as errors
import ScintSuite._Plotting as ssplt
from tqdm import tqdm
from ScintSuite._Paths import Path
from ScintSuite._SideFunctions import createGrid
from ScintSuite.decorators import deprecated
from ScintSuite._basicVariable import BasicVariable
from ScintSuite._Mapping._Common import _fit_to_model_
from ScintSuite.SimulationCodes.Common.strikes import Strikes
from ScintSuite._StrikeMap._ParentStrikeMap import GeneralStrikeMap
from ScintSuite.SimulationCodes.FILDSIM.execution import get_energy
from ScintSuite.SimulationCodes.SINPA._execution import guess_strike_map_name
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --- Initialise auxiliary elements
logger = logging.getLogger('ScintSuite.StrikeMap')


class FILDINPA_Smap(GeneralStrikeMap):
    """
    Parent class for INPA and FILD strike maps.

    Jose Rueda Rueda: jrrueda@us.es

    Public Methods (* means inherited from the father):
        - *calculate_pixel_coordinates: calculate the map coordinates in the camera
        - *setRemapVariables: Set the variables to be used when remapping
        - *interp_grid: Interpolate the smap variables in a given camera frame
        - *export_spatial_coordinates: save grid point into a .txt
        - *plot_var: perform a quick plot of a variable (or pair) of the map
        - *plot_pix: plot the strike map in the camera space
        - *plot_real: plot the scintillator in the real space
        - calculate_energy: calculate the energy associated with each gyroradius
        - load_strike_points: Load the points used to create the map
        - calculate_phase_space_resolution: calculate the resolution associated
          with the phase-space variables of the map
        - plot_phase_space_resolution: plot the resolution in the phase space
        - remap_strike_points: remap the loaded strike points
        - remap_external_strike_points: remap any strike points
        - plot_phase_space_resolution_fits: plot the resolution in the phase space
        - plot_collimator_factors: plot the resolution in the phase space
        - plot_instrument_function: Plot the instrument function

    Private method (* means inherited from the father):
        - *_calculate_transformation_matrix: Calculate the transformation matrix
        - _calculate_instrument_function_interpolators: calculate the
            interpolators for the instrument function
        - _calculate_mapping_interpolators: Calculate the interpolators to map
            the strike points

    Properties (* means inherited from the father):
        - *shape: grid size used in the calculation. Eg, for standard FILD,
            shape=[npitch, ngyroradius]
        - *code: Code used to calculate the map
        - *diagnostic: Detector associated with this map
        - *file: full path to the file where the map is storaged
        - *MC_variables: monte carlo variables used to create the map

    Calls (* means inherited from the father):
        - *variables: smap('my var') will return the values of the variable
    """

    def __init__(self, file: str = None, variables_to_remap: tuple = None,
                 code: str = None, theta: float = None, phi: float = None,
                 GeomID: str = None, diagnostic: str = 'INPA',
                 verbose: bool = True, decimals: int = 1):
        """
        Initailise the Strike map object.

        Just call the init of the parent object but with the possibility of
        telling the theta and phi, so the code look in the remap database and
        find the closet map

        :param file: strike map file to load (option 1 to load the map)
        :param variables_to_remap: pair of variables selected for the remap.
            By default:
                    - FILD: ('pitch', 'gyroradius')
                    - INPA: ('R0', 'gyroradius')
                    - iHIBP: ('x1', 'x2')
        :param code: code used to calculate the map. If None, would be
            guessed automatically
        :param theta: theta angle of the database (option 2 to load the map)
        :param phi: phi angle of the database (option 2 to load the map)
        :param  GeomID: geometry ID of the database (option 2 to load the map)
        :param  decimals: Number of decimasl to look in the database (opt 2)
        :param  diagnostic: diagnostic to look in the database (opt 2)
        :param  verbose: print some information in the terminal
        """
        if (theta is not None) and (phi is not None):
            if verbose:
                logger.warning('Theta and phi present, ignoring filename')
            name = guess_strike_map_name(phi, theta, geomID=GeomID,
                                         decimals=decimals)
            file = os.path.join(Path().ScintSuite, 'Data', 'RemapStrikeMaps',
                                diagnostic, GeomID, name)
            if not os.path.isfile(file):
                logger.error('Looking for %s' % file)
                raise errors.NotFoundStrikeMap('you need to calculate the map')
        GeneralStrikeMap.__init__(self,  file,
                                  variables_to_remap=variables_to_remap,
                                  code=code)
        # --- Allocate for latter
        self.strike_points = None
        self.fileStrikePoints = None
        self._resolutions = None
        self._interpolators_instrument_function = None
        self.instrument_function = None

    def calculate_energy(self, B: float, A: float = 2.01410178,
                         Z: float = 1.0):
        """
        Calculate the energy associated to each centroid (in keV)

        Jose Rueda: jrrueda@us.es

        :param  B: magnetif field modulus
        :param  A: mass of the ion, in amu
        :param  Z: charge, in e units
        """
        dummy = get_energy(self('gyroradius'), B=B, A=A, Z=Z) / 1000.0
        self._data['e0'] = BasicVariable(name='e0', units='keV', data=dummy)
        self._optionsForEnergy = {
            'B': B,
            'A': A,
            'z': Z,
        }

    def load_strike_points(self, file=None, verbose: bool = True,
                           calculate_pixel_coordinates: bool = False,
                           remap_in_pixel_space: bool = False,
                           remap: bool = True):
        """
        Load the strike points used to calculate the map.

        Jose Rueda: ruejo@ipp.mpg.de

        :param  file: File to be loaded. If none, name will be deduced from the
            self.file variable, so the strike points are supposed to be in
            the same folder than the strike map
        :param  verbose: Flag to plot some information about the strike points
        :param  calculate_pixel_coordinates: if true the pixel coordinates will
            be calculated just after loading the points (using the camera
            calibration storaged in the strike map)
        :param  remap_in_pixel_space: in SINPA, the remap is not done in fortran
            but must be done in python. By default, it is done just after
            loading the points. If this flag is true, the remap will be done
            using the pixel coordinates instead of the strike in the
            scintillator.
        :param  remap: Flag to remap the strike points just after loading them
            introduced in version 1.3.7, before it was always done
        """
        # See if the strike points where already there
        if self.strike_points is not None:
            logger.warning('11: Strike points present, overwritting.')
        # Get the object we need to fill and the file to be load
        if file is None:
            if self._header['code'].lower() == 'sinpa':
                (filename, extension) = self.file.rsplit('.', 1)
                file = filename + '.spmap'
            elif self._header['code'].lower() == 'fildsim':
                file = self.file[:-14] + 'strike_points.dat'
            elif self._header['code'].lower() == 'ihibpsim':
                raise errors.NotImplementedError('Sorry, not done')

        # Load the strike points
        self.fileStrikePoints = file
        self.strike_points =\
            Strikes(file=file, verbose=verbose, code=self.code)
        # If desired, calculate pixel coordinates
        if calculate_pixel_coordinates:
            self.strike_points.calculate_pixel_coordinates(
                self.CameraCalibration)
        # If the code was SINPA, perform the remap, as it is not done in
        # fortran:
        if self._header['code'].lower() == 'sinpa' and remap:
            logger.info('Remapping the strike points')
            self.remap_strike_points(remap_in_pixel_space=remap_in_pixel_space)

    def calculate_phase_space_resolution(self, diag_params: dict = {},
                                         min_statistics: int = 100,
                                         adaptative: bool = True,
                                         calculate_uncertainties: bool = False,
                                         confidence_level: float = 0.9544997,
                                         bin_per_sigma: int = 4,
                                         variables: tuple = None,
                                         verbose: bool = False):
        """
        Calculate the resolution associated with each point of the map.

        Notice that it is though for 2D diagnostics, so there are 2 directions
        in the scintillator associated with 2 variables of the phase space
        For more complciated situations, you need to define another function

        Jose Rueda Rueda: jrrueda@us.es

        :param  diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch. It contains:
                -dx: x space used by default in the fit. (pitch in fild)
                -dy: y space used by default in the fit. (gyroradius in fild)
                -x_method: Function to use in the x fit, default Gauss
                -y_method: Function to use in the y fit, default Gauss
            Accepted methods are:
                - Gauss: Gaussian fit
                - sGauss: squewed Gaussian fit
        :param  min_statistics: Minimum number of points for a given r,p to make
            the fit (if we have less markers, this point will be ignored)
        :param  adaptative: If true, the bin width will be adapted such that the
            number of bins in a sigma of the distribution is bin_per_sigma. If
            this is the case, dx, dy of the diag_options will no longer have an
            impact
        :param  calculate_uncertainties: flag to calcualte the uncertainties of
            the fit
        :param  confidence_level: confidence level for the uncertainty
            determination
        :param  variables: Variables where to calculate the resolutions. By
            default, the ones selected for the remapping will be used
        :param  verbose: Flag to print some information
        """
        if self.strike_points is None:
            logger.info('Trying to load the strike points')
            self.load_strike_points()
        # --- Prepare options:
        diag_options = {
            'dx': 1.0,
            'dy': 0.1,
            'x_method': 'Gauss',
            'y_method': 'sGauss'
        }
        diag_options.update(diag_params)
        # Select the variables
        if variables is None:
            variables = [v.name for v in self._to_remap]
        # --- Get the columns we need
        # Get the number of pairs of strike points launched
        nx, ny = self.shape
        # get the names
        namex = 'remap_' + variables[0]
        namey = 'remap_' + variables[1]
        # Get the physical units
        unitsx = self.strike_points.header['info'][namex]['units']
        unitsy = self.strike_points.header['info'][namey]['units']
        # first check if the remapped strike points are there
        if (namex not in self.strike_points.header['info'].keys()
                or namey not in self.strike_points.header['info'].keys()):
            raise Exception('Non remap data in the strike points object!')
        else:
            iix = self.strike_points.header['info'][namex]['i']
            iiy = self.strike_points.header['info'][namey]['i']
        # Get the names of the variables to fit
        names = {
            'Gauss': ['amplitude', 'center', 'sigma'],
            'sGauss': ['amplitude', 'center', 'sigma', 'gamma']
        }
        xnames = names[diag_options['x_method']]
        ynames = names[diag_options['y_method']]

        # allocate the variables
        self._resolutions = {
            'variables': (
                BasicVariable(name=variables[0], units=unitsx),
                BasicVariable(name=variables[1], units=unitsy)),
            'npoints': np.zeros((nx, ny)),
        }
        for key, names in zip(variables, (xnames, ynames)):
            # For the resolution
            self._resolutions[key] = \
                {n: np.full((nx, ny), np.nan) for n in names}
            # For the uncertainty
            self._resolutions['unc_' + key] = \
                {n: np.full((nx, ny), np.nan) for n in names}
            # For the normalization
            self._resolutions['norm_' + key] = np.full((nx, ny), np.nan)
            # For the general fits
            self._resolutions['fits_' + key] = \
                np.full((nx, ny), None, dtype=np.ndarray)
        self._resolutions['model_' + variables[0]] = diag_options['x_method']
        self._resolutions['model_' + variables[1]] = diag_options['y_method']
        # --- Core: Calculation of the resolution
        if verbose:
            logger.info('Calculating resolutions ...')
        for ix in tqdm(range(nx)):
            for iy in range(ny):
                # -- Select the data:
                data = self.strike_points.data[ix, iy]
                # if there is no enough data, skip this point
                if self.strike_points.header['counters'][ix, iy] < min_statistics:
                    continue
                # -- Prepare the basic bin edges
                # Prepare the bin edges according to the desired width
                if adaptative:
                    sigmax = np.std(data[:, iix])
                    dx = sigmax / float(bin_per_sigma)
                    sigmay = np.std(data[:, iiy])
                    dy = sigmay / float(bin_per_sigma)
                else:
                    dx = diag_options['dx']
                    dy = diag_options['dy']
                xedges = \
                    np.arange(start=data[:, iix].min() - dx,
                              stop=data[:, iix].max() + dx,
                              step=dx)
                yedges = \
                    np.arange(start=data[:, iiy].min() - dy,
                              stop=data[:, iiy].max() + dy,
                              step=dy)
                # -- fit the x variable
                params, self._resolutions['fits_' + variables[0]][ix, iy], \
                    self._resolutions['norm_' + variables[0]][ix, iy], unc = \
                    _fit_to_model_(
                        data[:, iix], bins=xedges,
                        model=diag_options['x_method'],
                        confidence_level=confidence_level,
                        uncertainties=calculate_uncertainties)
                # save the data in place
                for key in xnames:
                    self._resolutions[variables[0]][key][ix, iy] = params[key]
                    # For the uncertainty
                    self._resolutions['unc_' + variables[0]][key][ix, iy] = \
                        unc[key]

                # -- fit the y variable
                params, self._resolutions['fits_' + variables[1]][ix, iy], \
                    self._resolutions['norm_' + variables[1]][ix, iy], unc = \
                    _fit_to_model_(
                        data[:, iiy], bins=yedges,
                        model=diag_options['y_method'],
                        confidence_level=confidence_level,
                        uncertainties=calculate_uncertainties)
                # save the data in place
                for key in ynames:
                    self._resolutions[variables[1]][key][ix, iy] = params[key]
                    # For the uncertainty
                    self._resolutions['unc_' + variables[1]][key][ix, iy] =\
                        unc[key]
        # --- Prepare the Interpolators
        self._calculate_instrument_function_interpolators()
        # self._calculate_position_interpolators()

    def _calculate_instrument_function_interpolators(self):
        """
        Calculate the interpolators from phase to resolution parameters

        These interpolaros would be latter used in the calculation of the
        weight function. An example of them is the one which give the sigma
        of the weight function for each pair of value of gyroradius and pitch,
        in FILD

        Jose Rueda: jrrueda@us.es

        Notice, the interpolators will need the input variables in the same
        order in which they were called in the calculate resolution, ie, if
        tou call the calculate resolution with (pitch, gyroradius) [standard]
        The obtained interpolators would be such that if you want the value in
        the pitch 30, gyroradius 2.3, you need to call them as
        <interpolator> (30, 2.3)

        The used interpolator will be linear!
        """
        # --- Preparation phase:
        # Prepare grid
        xx, yy = \
            np.meshgrid(self.MC_variables[0].data,
                        self.MC_variables[1].data)
        # Flatten them into 1D arrays
        xxx = xx.flatten()
        yyy = yy.flatten()
        # Allocate the space for the interpolators

        dummy = {
            'variables': self._resolutions['variables'],
        }
        # --- Calculate the interpolators
        for var in self._resolutions['variables']:     # Loop in variables
            name = var.name
            dummy[name] = {}
            for key in self._resolutions[name].keys():  # Loop in parameters
                data = self._resolutions[name][key].T.flatten()  # parameter
                # The transpose is to match dimension of the mesh-grid
                # Now delete the NaN cell
                flags = np.isnan(data)
                if np.sum(~flags) > 4:  # If at least we have 4 points
                    dummy[name][key] = \
                        scipy_interp.LinearNDInterpolator(
                            np.vstack((xxx[~flags], yyy[~flags])).T,
                            data[~flags]
                        )
            # Trey with uncertainties
            unc_name = 'unc_' + name
            if unc_name in self._resolutions.keys():  # If the uncertainty is
                dummy[unc_name] = {}                 # calculated
                for key in self._resolutions[unc_name].keys():  # Loop in parameters
                    data = self._resolutions[unc_name][key].T.flatten()  # parameter
                    # The transpose is to match dimension of the mesh-grid
                    # Now delete the NaN cell
                    flags = np.isnan(data)
                    if np.sum(~flags) > 4:  # If at least we have 4 points
                        dummy[unc_name][key] = \
                            scipy_interp.LinearNDInterpolator(
                                np.vstack((xxx[~flags], yyy[~flags])).T,
                                data[~flags]
                            )
        # The collimator factor is not inside the resolution dictionary, so
        # we need an independent call
        # First thing is to ensure that the matrix is created in the same
        # variables
        data = self._data['collimator_factor_matrix'].data.T.flatten()

        flags = np.isnan(data)
        if np.sum(~flags) > 4:
            dummy['collimator_factor'] = \
                scipy_interp.LinearNDInterpolator(
                    np.vstack((xxx[~flags], yyy[~flags])).T,
                    data[~flags]
                )
        self._interpolators_instrument_function = dummy

    # def _calculate_position_interpolators(self):
    #     """
    #     Calculate the interpolators from phase to scintillator position
    #
    #     Jose Rueda: jrrueda@us.es
    #
    #     Notice, the interpolators will need the input variables in the same
    #     order in which they were called in the calculate resolution, ie, if
    #     tou call the calculate resolution with (pitch, gyroradius) [standard]
    #     The obtained interpolators would be such that if you want the value in
    #     the pitch 30, gyroradius 2.3, you need to call them as
    #     <interpolator> (30, 2.3)
    #
    #     The used interpolator will be linear!
    #     """
    #     # --- Preparation phase:
    #     # Prepare grid
    #     xx, yy = \
    #         np.meshgrid(self.strike_points.header[self._resolutions['variables'][0]],
    #                     self.strike_points.header[self._resolutions['variables'][1]])
    #     # Flatten them into 1D arrays
    #     xxx = xx.flatten()
    #     yyy = yy.flatten()
    #     # Allocate the space for the interpolators
    #     dummy = {
    #         'variables': self._resolutions['variables'],
    #     }
    #     # Allocate the matrices
    #     namex = self._resolutions['variables'][0]
    #     namey = self._resolutions['variables'][1]
    #     nx = self._header['n' + namex]
    #     ny = self._header['n' + namey]
    #     unique_x = self._header['unique_' + namex]
    #     unique_y = self._header['unique_' + namey]
    #     X1 = np.zeros(nx, ny)
    #     X2 = np.zeros(nx, ny)
    #
    #     # Fill the matrices
    #     for ix in range(nx):
    #         for iy in range(ny):
    #             flags = (self._data[namex] == unique_x[ix]) \
    #                 * (self._data[namey] == unique_y[ix])
    #             if np.sum(flags) == 1:
    #                 # By definition, flags can only have one True
    #                 # yes, x is smap.y... FILDSIM notation
    #                 X1[ix, iy] = self._data['x1'][flags]
    #                 X2[ix, iy] = self._data['x2'][flags]
    #             elif np.sum(flags) != 0:
    #                 print('Weird case')
    #                 print('Number of found points %i:' % np.sum(flags))
    #                 print('x = %f' % unique_x[ix])
    #                 print('y = %f' % unique_y[iy])
    #                 raise Exception('More than one point found')
    #     datax1 = X1.T.flatten()
    #     datax2 = X2.T.flatten()
    #     # Remove NaN
    #     flags = np.isnan(data)
    #     dummy['x1'] = scipy_interp.LinearNDInterpolator(
    #         np.vstack((xxx[~flags], yyy[~flags])).T,
    #         datax[~flags]
    #     )
    #     dummy['x2'] = scipy_interp.LinearNDInterpolator(
    #         np.vstack((xxx[~flags], yyy[~flags])).T,
    #         datax[~flags]
    #     )
    #     self._interpolators_scintillator_position = dummy

    def _calculate_mapping_interpolators(self,
                                         kernel: str = 'thin_plate_spline',
                                         degree=2,
                                         variables: tuple = None):
        """
        Calculate interpolators scintillator position -> phase space.

        If there is pixel data, it also calculate the interpolators of the
        pixel space

        Jose Rueda: jrrueda@us.es

        :param  kernel: kernel for the interpolator
        :param  degree: degree for the added polynomial
        :param  variables: variables to prepare the interpolators

        See RBFInterpolator of Scipy for full documentation
        """
        # --- Select the colums to be used
        # temporal solution to save the coordinates in the array
        coords = np.zeros((self._data['x1'].size, 2))
        coords[:, 0] = self._coord_real['x1'].copy()
        coords[:, 1] = self._coord_real['x2'].copy()

        # Allocate the space
        if variables is None:
            variables = (self._to_remap[0].name,
                         self._to_remap[1].name)

        if self._map_interpolators is None:
            self._map_interpolators = dict.fromkeys(variables)
        else:
            newDict = dict.fromkeys(variables)
            self._map_interpolators.update(newDict)

        for key in variables:
            # Sometimes necessary to avoid errors on Cobra, reason is unknown
            try:
                self._map_interpolators[key] = \
                    scipy_interp.RBFInterpolator(coords,
                                                 self._data[key].data,
                                                 kernel=kernel, degree=degree)
            except RuntimeError:
                self._map_interpolators[key] = \
                    scipy_interp.RBFInterpolator(coords,
                                                 self._data[key].data,
                                                 kernel=kernel, degree=degree)
        # --- If there is pixel informations, do the same for the pixel space
        try:
            # @ToDo: See why this modify R0 and e0 interpolators
            # coords[:, 0] = self._coord_pix['x'].copy()
            # coords[:, 1] = self._coord_pix['y'].copy()
            # var2 = [a + '_pix' for a in variables]
            # newDict2 = dict.fromkeys(var2)
            # print(newDict2)
            # self._map_interpolators.update(newDict2)
            # for key in variables:
            #     self._map_interpolators[key+'_pix'] = \
            #         scipy_interp.RBFInterpolator(coords,
            #                                      self._data[key].data,
            #                                      kernel=kernel, degree=degree)
            coords2 = np.zeros((self._data['x1'].size, 2))
            coords2[:, 0] = self._coord_pix['x'].copy()
            coords2[:, 1] = self._coord_pix['y'].copy()
            var2 = [a + '_pix' for a in variables]
            newDict2 = dict.fromkeys(var2)
            self._map_interpolators.update(newDict2)
            for key in variables:
                self._map_interpolators[key+'_pix'] = \
                    scipy_interp.RBFInterpolator(coords2,
                                                 self._data[key].data,
                                                 kernel=kernel, degree=degree)
        except (KeyError, AttributeError):
            pass

    def remap_strike_points(self, overwrite: bool = True,
                            remap_in_pixel_space: bool = False):
        """
        Remap the StrikePoints

        Jose Rueda: jrrueda@us.es

        :param  overwrite: if true, the variable data will be overwritten, even
            if that remap was already done
        :param  remap_in_pixel_space: flagg to decide if the remap will be done
            in pixel or real space
        """
        # --- See if the interpolators are defined
        if self._map_interpolators is None:
            logger.debug('Interpolators not calcualted. Calculating them')
            self._calculate_mapping_interpolators()
        # --- Proceed to remap
        # Get the shape of the map
        nx, ny = self.shape
        # Get the index of the colums containing the scintillation position
        if not remap_in_pixel_space:
            ix1 = self.strike_points.header['info']['x1']['i']
            ix2 = self.strike_points.header['info']['x2']['i']
        else:
            ix1 = self.strike_points.header['info']['xcam']['i']
            ix2 = self.strike_points.header['info']['ycam']['i']
        # Loop over the deseired variables
        var_list = [k for k in self._map_interpolators.keys() if k.endswith('pix')==False]
        for k in var_list:
            # See if we need to overwrite
            name = 'remap_' + k
            # Get the name of the interpolator to use
            if remap_in_pixel_space:
                interpolator = k + '_pix'
            else:
                interpolator = k
            was_there = False
            if name in self.strike_points.header['info'].keys():
                was_there = True
                if overwrite:
                    logger.warning('%s found in the object, overwritting' % k)
                    ivar = self.strike_points.header['info'][name]['i']
                else:
                    logger.info('%s found in the object, skipping' % k)
                    continue
            # Loop over the strike points pairs
            for ix in range(nx):
                for iy in range(ny):
                    if self.strike_points.header['counters'][ix, iy] > 0:
                        n_strikes = \
                            self.strike_points.header['counters'][ix, iy]
                        remap_data = np.zeros((n_strikes, 1))
                        remap_data[:, 0] = \
                            self._map_interpolators[interpolator](
                                self.strike_points.data[ix, iy][:, [ix1, ix2]])
                        # self.strike_points.data[ip, ir][:, iiy])
                        # append the remapped data to the object
                        if was_there:
                            self.strike_points.data[ix, iy][:, ivar] = \
                                remap_data
                        else:
                            self.strike_points.data[ix, iy] = \
                                np.append(self.strike_points.data[ix, iy],
                                          remap_data, axis=1)
            # Update the headers, if needed
            if not was_there:
                Old_number_colums = len(self.strike_points.header['info'])
                # Take the original variable as base for the dictionary
                extra_column = dict.fromkeys([name, ])
                extra_column[name] = {
                    'i': Old_number_colums,
                    'units': '@Todo',
                    'lonName': name,
                    'shortName': name
                }
                extra_column[name]['i'] = Old_number_colums
                # Update the header
                self.strike_points.header['info'].update(extra_column)

    def remap_external_strike_points(self, strikes, overwrite: bool = True):
        """
        Remap the signal (or any external) StrikePoints

        Jose Rueda: jrrueda@us.es

        Notice, this is in practical no used, use better the remap method of the
        strike points object if you want to remap externally calculated (signal)
        strike points
        """
        # --- See if the interpolators are defined
        if self._map_interpolators is None:
            logger.warning('27: Interpolators not calcualted. Calculating them')
            self._calculate_mapping_interpolators()
        # --- Proceed to remap
        # Get the shape of the map
        nx, ny = strikes.shape
        # Get the index of the colums containing the scintillation position
        ix1 = strikes.header['info']['x1']['i']
        ix2 = strikes.header['info']['x2']['i']
        # Try to get the pixel position
        try:
            ix1pix = strikes.header['info']['xcam']['i']
            ix2pix = strikes.header['info']['ycam']['i']
            camera = True
        except KeyError:
            camera = False
        # Loop over the deseired variables
        for k in self._map_interpolators.keys():
            # See if we need to overwrite
            name = 'remap_' + k
            was_there = False
            if name in strikes.header['info'].keys():
                was_there = True
                if overwrite:
                    logger.warning('%s found in the object, overwritting' % k)
                    ivar = strikes.header['info'][name]['i']
                else:
                    logger.info('%s found in the object, skipping' % k)
                    continue
            # Loop over the strike points pairs
            for ix in range(nx):
                for iy in range(ny):
                    if strikes.header['counters'][ix, iy] > 0:
                        n_strikes = \
                            strikes.header['counters'][ix, iy]
                        remap_data = np.zeros((n_strikes, 1))
                        if not k.endswith('pix'):
                            remap_data[:, 0] = \
                                self._map_interpolators[k](
                                    strikes.data[ix, iy][:, [ix1, ix2]])

                        elif k.endswith('pix') and camera:
                            remap_data[:, 0] = \
                                self._map_interpolators[k](
                                    strikes.data[ix, iy][:, [ix1pix, ix2pix]])
                        # self.strike_points.data[ip, ir][:, iiy])
                        # append the remapped data to the object
                        if was_there:
                            strikes.data[ix, iy][:, ivar] = remap_data.squeeze()
                        else:
                            strikes.data[ix, iy] = \
                                np.append(strikes.data[ix, iy],
                                          remap_data, axis=1)
            # Update the headers, if needed
            if not was_there:
                Old_number_colums = len(strikes.header['info'])
                # Take the original variable as base for the dictionary
                extra_column = dict.fromkeys([name, ])
                extra_column[name] = {
                    'i': Old_number_colums,
                    'units': '@Todo',
                    'longName': name,
                    'shortName': name
                }
                extra_column[name]['i'] = Old_number_colums
                # Update the header
                strikes.header['info'].update(extra_column)

    def  calculate_phaseSpace_to_pixelMatrix(self, gridPhaseSpace, gridPixel,
                                             limitation: float = 10.0,
                                             MC_number=300):
        """
        Calculate the transformation like matrix to go from the remap to the camera frame
        
        
        """
        # First calcualte the interpolators from phase space to pixel
        # Select the variable to interpolate
        xvar = self._data[gridPhaseSpace['xname']].data
        print(xvar.mean())
        yvar = self._data[gridPhaseSpace['yname']].data
        print(yvar.mean())
        gridPhaseSpace.pop('xname')
        gridPhaseSpace.pop('yname')
        
        # Get the pixel position
        xpix = self._coord_pix['x']
        ypix = self._coord_pix['y']
        
        # Construct the phase space grid
        nx, ny, xedges, yedges = createGrid(**gridPhaseSpace)
        nxpix, nypix, xedgespix, yedgespix = createGrid(**gridPixel)
        
        # Construct the interpolators
        interpolatorX = scipy_interp.LinearNDInterpolator(
            np.column_stack((xvar.flatten(), yvar.flatten())),
            xpix.flatten()
        )
        interpolatorY = scipy_interp.LinearNDInterpolator(
            np.column_stack((xvar.flatten(), yvar.flatten())),
            ypix.flatten()
        )      # 
        memory_size = nx * ny * nxpix * nypix \
            * 8 / 1024 / 1024 / 1024
        if memory_size > limitation:
            text = 'The requiring matrix will consume %2.1f Gb, this is above'\
                % memory_size\
                + 'the threshold. Increase it if you really want to proceed'
            raise errors.NotValidInput(text)
        transform = np.zeros((nxpix, nypix, nx, ny), dtype='float64')
        # Allocate the random number generator
        rand = np.random.default_rng()
        generator = rand.uniform
        # Loop over the grid
        for i in tqdm(range(nx)):
            for j in range(ny):
                # Create a set of MC markers
                x_markers = generator(xedges[i], xedges[i+1], MC_number)
                y_markers = generator(yedges[j], yedges[j+1], MC_number)
                # Calculate the pixel position
                xpix_markers= interpolatorX(x_markers, y_markers)
                ypix_markers= interpolatorY(x_markers, y_markers)
                # Histogram the pixel space
                H, _, _ = np.histogram2d(xpix_markers, ypix_markers,
                                         bins=[xedgespix, yedgespix])
                # Save it in place
                transform[:, :, i, j] = H.copy()
        # Normalise the matrix
        transform /= MC_number
        return transform
    # --------------------------------------------------------------------------
    # --- Plotting Block
    # --------------------------------------------------------------------------
    def plot_phase_space_resolution(self, ax_params: dict = {},
                                    cmap=None,
                                    nlev: int = 50,
                                    index_x: list = None,
                                    index_y: list = None,
                                    ax_lim: dict = {},
                                    cmap_lim: dict = {}):
        """
        Plot the phase space resolutions.

        Jose Rueda: jrrueda@us.es

        :param  ax_param: parameters for the axis beauty function. Note, labels
        of the color axis are hard-cored, if you want custom axis labels you
        would need to draw the plot on your own
        :param  cMap: is None, Gamma_II will be used
        :param  nlev: number of levels for the contour
        :param  index_gyr: if present, reslution would be plotted along
        gyroradius given by gyroradius[index_gyr]
        :param  ax_lim: Manually set the x and y axes, currently only works for making it bigger, not smaller
                       Should be given as ax_lim = {'xlim' : [x1,x2], 'ylim' : [y1,y2]}
        :param  cmap_lim: Manually set the upper limit for the color map
                         Should be given as cmap_lim = {'gyroradius' : ___, 'pitch' : ___}
        """
        # Initialise the plotting settings
        ax_options = {
            'xlabel': self.MC_variables[0].plot_label,
            'ylabel': self.MC_variables[1].plot_label,
        }
        ax_options.update(ax_params)
        if cmap is None:
            cmap = ssplt.Gamma_II()
        # --- Plot the resolution
        if (index_x is None) and (index_y is None):
            fig, ax = plt.subplots(1, 2, sharex=True)

            for var, subplot in zip(self._resolutions['variables'], ax):
                key = var.name
                xAxisPlot = self.MC_variables[0].data
                yAxisPlot = self.MC_variables[1].data
                res_matrix = self._resolutions[key]['sigma'].T
                if ax_lim:
                    if ax_lim["xlim"][0] < np.min(self.MC_variables[0].data):
                        n,m = res_matrix.shape
                        res_matrix_new = np.full((n,m+1))
                        res_matrix_new[:,1:] = res_matrix
                        res_matrix = res_matrix_new
                        xAxisPlot = np.insert(xAxisPlot,0,ax_lim["xlim"][0])
                    if ax_lim["xlim"][1] > np.max(self.MC_variables[0].data):
                        n,m = res_matrix.shape
                        res_matrix_new = np.full((n,m+1),np.nan)
                        res_matrix_new[:,:-1] = res_matrix
                        res_matrix = res_matrix_new
                        xAxisPlot = np.append(xAxisPlot,ax_lim["xlim"][1])
                    if ax_lim["ylim"][0] < np.min(self.MC_variables[1].data):
                        n,m = res_matrix.shape
                        res_matrix_new = np.full((n+1,m),np.nan)
                        res_matrix_new[1:,:] = res_matrix
                        res_matrix = res_matrix_new
                        yAxisPlot = np.insert(yAxisPlot,0,ax_lim["ylim"][0])
                    if ax_lim["ylim"][1] > np.max(self.MC_variables[1].data):
                        n,m = res_matrix.shape
                        res_matrix_new = np.full((n+1,m),np.nan)
                        res_matrix_new[:-1,:] = res_matrix
                        res_matrix = res_matrix_new
                        yAxisPlot = np.append(yAxisPlot,ax_lim["ylim"][1])

                try:
                    step = cmap_lim[key] / nlev
                    nlev_new = np.arange(0,cmap_lim[key] + step, step)
                except KeyError:
                    nlev_new = nlev

                cont = subplot.contourf(
                    xAxisPlot, yAxisPlot,
                    res_matrix,
                    levels=nlev_new, cmap=cmap
                )
                subplot = ssplt.axis_beauty(subplot, ax_options)
                # Now place the color var in the proper position
                divider = make_axes_locatable(subplot)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cont,
                             label='$\\sigma_{%s} [%s]$' % (key, var.units),
                             cax=cax)
            plt.tight_layout()
        else:   # Plot along the x - y variable
            if index_x is not None:
                fig, ax = plt.subplots(1, 2, sharex=True)
                # Test if it is a list or array
                if isinstance(index_x, (list, np.ndarray, tuple)):
                    pass
                else:  # it should be just a number
                    index_x = np.array([index_x])
                yAxis = self.MC_variables[1].data
                for var, subplot in zip(self._resolutions['variables'], ax):
                    key = var.name
                    cmap = ssplt.Gamma_II()
                    for i in index_x:
                        ssplt.p1D_shaded_error(
                            subplot, yAxis,
                            self._resolutions[key]['sigma'][i, :],
                            self._resolutions['unc_' + key]['sigma'][i, :],
                            color=cmap(i/self.MC_variables[0].data.size),
                            alpha=0.1,
                            line_param={
                                'label': str(self.MC_variables[0].data[i])
                            })
                    ax_options = {
                        'xlabel': self.MC_variables[1].plot_label,
                        'ylabel': '$\\sigma_{%s} [%s]$' % (key, var.units),
                    }
                    ax_options.update(ax_params)
                    subplot = ssplt.axis_beauty(subplot, ax_options)
            if index_y is not None:
                fig, ax = plt.subplots(1, 2, sharex=True)
                # Test if it is a list or array
                if isinstance(index_y, (list, np.ndarray, tuple)):
                    pass
                else:  # it should be just a number
                    index_y = np.array([index_y])
                xAxis = self.MC_variables[0].data
                for var, subplot in zip(self._resolutions['variables'], ax):
                    key = var.name
                    cmap = ssplt.Gamma_II()
                    for i in index_y:
                        ssplt.p1D_shaded_error(
                            subplot, xAxis,
                            self._resolutions[key]['sigma'][:, i],
                            self._resolutions['unc_' + key]['sigma'][:, i],
                            color=cmap(i/self.MC_variables[0].data.size),
                            alpha=0.1,
                            line_param={
                                'label': str(self.MC_variables[0].data[i])
                            })
                    ax_options = {
                        'xlabel': self.MC_variables[0].plot_label,
                        'ylabel': '$\\sigma_{%s} [%s]$' % (key, var.units),
                    }
                    ax_options.update(ax_params)
                    subplot = ssplt.axis_beauty(subplot, ax_options)

    @deprecated('Some input will change name in the final version')
    def plot_phase_space_resolution_fits(self, var: str = 'Gyroradius',
                                         ax_params: dict = {},
                                         ax=None, gyr_index=None, pitch_index=None,
                                         gyroradius=None, pitch=None,
                                         kind_of_plot: str = 'normal',
                                         include_legend: bool = False,
                                         XI_index=None,
                                         grid: bool = False):
        """
        Plot the fits done to calculate the resolution

        :param  var: variable to plot, Gyroradius or Pitch for FILD. Capital
        letters will be ignored
        :param  ax_param: dictoniary with the axis parameters axis_beauty()
        :param  ax: axis where to plot
        :param  gyr_index: index, or arrays of indeces, of gyroradius to plot
        :param  pitch_index: index, or arrays of indeces, of pitches to plot,
            this is outdated code, please use XI_index instead
        :param  gyroradius: gyroradius value or array of them to plot. If
        present, gyr_index will be ignored
        :param  pitch: idem to gyroradius bu for the pitch
        :param  kind_of_plot: kind of plot to make:
            - normal: scatter plot of the data and fit like a line
            - bar: bar plot of the data and file like a line
            - uncertainty: scatter plot of the data and shading area for the
                fit (3 sigmas)
            - just_fit: Just a line plot as the fit
        :param  include_legend: flag to include a legend
        :param  XI_index: equivalent to pitch_index, but with the new criteria
        """
        # --- Initialise plotting options and axis:
        default_labels = {
            'gyroradius': {
                'xlabel': 'Gyroradius [cm]',
                'ylabel': 'Counts [a.u.]'
            },
            'pitch': {
                'xlabel': 'Pitch [$\\degree$]',
                'ylabel': 'Counts [a.u.]'
            }
        }
        if grid:
            ax_options = {
                'grid': 'both',
            }
        else:
            ax_options = {}
        ax_options.update(default_labels[var.lower()])
        ax_options.update(ax_params)
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        if (pitch_index is None) and (XI_index is not None):
            pitch_index = XI_index
        # --- Localise the values to plot
        if gyroradius is not None:
            # test if it is a number or an array of them
            if isinstance(gyroradius, (np.ndarray)):
                gyroradius = gyroradius
            else:
                gyroradius = np.array([gyroradius])
            index_gyr = np.zeros(gyroradius.size, dtype=int)
            for i in range(index_gyr.size):
                index_gyr[i] = \
                    np.argmin(np.abs(self.MC_variables[1].data - gyroradius[i]))
            logger.debug('Found gyroradius: %.2f' % self.MC_variables[1].data[index_gyr])
        else:
            # test if it is a number or an array of them
            if gyr_index is not None:
                if isinstance(gyr_index, (list, np.ndarray)):
                    index_gyr = gyr_index
                else:
                    index_gyr = np.array([gyr_index])
            else:
                index_gyr = np.arange(self.MC_variables[1].data.size, dtype=int)

        if pitch is not None:
            # test if it is a number or an array of them
            if isinstance(pitch, (list, np.ndarray)):
                pitch = pitch
            else:
                pitch = np.array([pitch])
            index_pitch = np.zeros(pitch.size, dtype=int)
            for i in range(index_pitch.size):
                index_pitch[i] = \
                    np.argmin(np.abs(self.MC_variables[0].data - pitch[i]))
            logger.info('Found pitches: %.2f' %
                        self.MC_variables[0].data[index_pitch])
        else:
            # test if it is a number or an array of them
            if pitch_index is not None:
                if isinstance(pitch_index, (list, np.ndarray)):
                    index_pitch = pitch_index
                else:
                    index_pitch = np.array([pitch_index])
            else:
                index_pitch = np.arange(self.MC_variables[0].data.size, dtype=int)
        # --- Get the maximum value for the normalization

        # --- Plot the desired data
        # This is just to allow the user to ask the variable with capitals
        # letters or not

        for ir in index_gyr:
            for ip in index_pitch:
                # The lmfit model has included a plot function, but is slightly
                # not optimal so we will plot it 'manually'
                if self._resolutions['fits_' + var.lower()][ip, ir] is not None:
                    x = self._resolutions['fits_' + var.lower()][ip, ir].userkws['x']
                    deltax = x.max() - x.min()
                    x_fine = np.linspace(x.min() - 0.1 * deltax,
                                         x.max() + 0.1 * deltax)
                    name = 'rL: ' + str(round(self.MC_variables[1].data[ir], 1))\
                        + ' $\\lambda$: ' + \
                        str(round(self.MC_variables[0].data[ip], 1))
                    normalization = \
                        self._resolutions['norm_' + var.lower()][ip, ir]
                    y = self._resolutions['fits_' + var.lower()][ip, ir].eval(
                        x=x_fine) * normalization
                    if kind_of_plot.lower() == 'normal':
                        # plot the data as scatter plot
                        scatter = ax.scatter(
                            x,
                            normalization * self._resolutions['fits_' + var.lower()][ip, ir].data,
                            label='__noname__')
                        # plot the fit as a line
                        ax.plot(x_fine, y,
                                color=scatter.get_facecolor()[0, :3],
                                label=name)
                    elif kind_of_plot.lower() == 'bar':
                        bar = ax.bar(
                            x,
                            normalization * self._resolutions['fits_' + var.lower()][ip, ir].data,
                            label='__noname__', width=x[1]-x[0],
                            alpha=0.25)
                        ax.plot(x_fine, y,
                                color=bar.patches[0].get_facecolor()[:3],
                                label=name)
                    elif kind_of_plot.lower() == 'just_fit':
                        ax.plot(x_fine, y, label=name)
                    elif kind_of_plot.lower() == 'uncertainty':
                        scatter = ax.scatter(
                            x,
                            normalization * self._resolutions['fits_' + var.lower()][ip, ir].data,
                            label='__noname__')
                        dely = normalization \
                            * self._resolutions['fits_' + var.lower()][ip, ir].eval_uncertainty(sigma=3, x=x_fine)
                        ax.fill_between(x_fine, y-dely, y+dely, alpha=0.25,
                                        label='3-$\\sigma$ uncertainty band',
                                        color=scatter.get_facecolor()[0, :3])
                    else:
                        raise errors.NotValidInput(
                            'Not kind of plot not understood')
                else:
                    pass
                    # print('Not fits for rl: '
                    #       + str(round(self.unique_gyroradius[ir], 1))
                    #       + 'pitch: '
                    #       + str(round(self.unique_pitch[ip], 1)))
        if include_legend:
            ax.legend()
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)

    @deprecated('Some input will change name in the final version')
    def plot_collimator_factor(self, ax_param: dict = {}, cMap=None,
                               nlev: int = 20, ax_lim: dict = {},
                               cmap_lim: float = 0):
        """
        Plot the collimator factor.

        Jose Rueda: jrrueda@us.es

        @todo: Implement label size in colorbar

        :param  ax_param: parameters for the axis beauty function. Note, labels
        of the color axis are hard-cored, if you want custom axis labels you
        would need to draw the plot on your own
        :param  cMap: is None, Gamma_II will be used
        :param  nlev: number of levels for the contour
        :param  ax_lim: Manually set the x and y axes, currently only works for making it bigger, not smaller
                       Should be given as ax_lim = {'xlim' : [x1,x2], 'ylim' : [y1,y2]}
        :param  cmap_lim: Manually set the upper limit for the color map
        """
        # --- Initialise the settings:
        if cMap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = cMap
        ax_options = {
            'xlabel': '$\\lambda [\\degree]$',
            'ylabel': '$r_l [cm]$'
        }
        ax_options.update(ax_param)

        # --- Open the figure and prepare the map:
        fig, ax = plt.subplots(1, 1, figsize=(6, 10),
                               facecolor='w', edgecolor='k')

        coll_matrix = np.transpose(self('collimator_factor_matrix'))
        # In case you want to manually set the axis limits to something bigger
        xAxisPlot = self.MC_variables[0].data
        yAxisPlot = self.MC_variables[1].data
        if ax_lim:
            if ax_lim["xlim"][0] < np.min(self.MC_variables[0].data):
                n,m = coll_matrix.shape
                coll_matrix_new = np.zeros((n,m+1))
                coll_matrix_new[:,1:] = coll_matrix
                coll_matrix = coll_matrix_new
                xAxisPlot = np.insert(xAxisPlot,0,ax_lim["xlim"][0])
            if ax_lim["xlim"][1] > np.max(self.MC_variables[0].data):
                n,m = coll_matrix.shape
                coll_matrix_new = np.zeros((n,m+1))
                coll_matrix_new[:,:-1] = coll_matrix
                coll_matrix = coll_matrix_new
                xAxisPlot = np.append(xAxisPlot,ax_lim["xlim"][1])
            if ax_lim["ylim"][0] < np.min(self.MC_variables[1].data):
                n,m = coll_matrix.shape
                coll_matrix_new = np.zeros((n+1,m))
                coll_matrix_new[1:,:] = coll_matrix
                coll_matrix = coll_matrix_new
                yAxisPlot = np.insert(yAxisPlot,0,ax_lim["ylim"][0])
            if ax_lim["ylim"][1] > np.max(self.MC_variables[1].data):
                n,m = coll_matrix.shape
                coll_matrix_new = np.zeros((n+1,m))
                coll_matrix_new[:-1,:] = coll_matrix
                coll_matrix = coll_matrix_new
                yAxisPlot = np.append(yAxisPlot,ax_lim["ylim"][1])

        if cmap_lim:
            step = cmap_lim / nlev
            nlev = np.arange(0,cmap_lim + step, step)

        # Plot the gyroradius resolution
        a1 = ax.contourf(xAxisPlot,
                         yAxisPlot,
                         coll_matrix,
                         levels=nlev, cmap=cmap)
        fig.colorbar(a1, ax=ax, label='Collimating factor')
        ax = ssplt.axis_beauty(ax, ax_options)

        plt.tight_layout()
        return

    def plot_instrument_function(self, xs: float = None, ys: float = None,
                                 x: float = None, y: float = None,
                                 ax=None, interpolation: str = 'bicubic',
                                 ax_params: dict = {},
                                 cmap=None):
        """

        :param ax_param:
        :param cmap:
        :return:
        """
        par = {
            'xs': xs,
            'ys': ys,
            'x': x,
            'y': y,
            'method': 'nearest'
        }
        par2 = {}
        for k in par.keys():
            if par[k] is not None:
                par2[k] = par[k]
        # - Open the figure, if needed
        if ax is None:
            fig, ax = plt.subplots()
        # - Get the color map
        if cmap is None:
            cmap = ssplt.Gamma_II()
        # - Plot the stuff

        self.instrument_function.sel(**par2).plot.imshow(ax=ax, cmap=cmap,
                                                         interpolation=interpolation)
        ax = ssplt.axis_beauty(ax, ax_params,)
        return ax
