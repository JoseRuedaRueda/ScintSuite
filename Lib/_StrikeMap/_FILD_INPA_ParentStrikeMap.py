"""
Parent Strike Map for INPA and FILD diagnostic

Jose Rueda: jrrueda@us.es

Introduced in version 0.10.0
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interp
import Lib.errors as errors
import Lib._Plotting as ssplt
from tqdm import tqdm
from Lib._StrikeMap._ParentStrikeMap import GeneralStrikeMap
from Lib.SimulationCodes.Common.strikes import Strikes
from Lib.SimulationCodes.SINPA.execution import guess_strike_map_name
from Lib.SimulationCodes.FILDSIM.execution import get_energy
from Lib._basicVariable import BasicVariable
from Lib._Mapping._Common import _fit_to_model_
from Lib._Paths import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable


class FILDINPA_Smap(GeneralStrikeMap):
    """
    Parent class for INPA and FILD strike maps.

    Jose Rueda Rueda: jrrueda@us.es

    New public methods respect to the parent class
        - load_strike_points: Load the points used to create the map
        - calculate_phase_space_resolution: calculate the resolution associated
          with the phase-space variables of the map
        - plot_phase_space_resolution: plot the resolution in the phase space
        - remap_strike_points: remap the loaded strike points
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
        """
        if (theta is not None) and (phi is not None):
            if verbose:
                print('Theta and phi present, ignoring filename')
            name = guess_strike_map_name(phi, theta, geomID=GeomID,
                                         decimals=decimals)
            file = os.path.join(Path().ScintSuite, 'Data', 'RemapStrikeMaps',
                                diagnostic, GeomID, name)
            if not os.path.isfile(file):
                print(file)
                raise errors.NotFoundStrikeMap('you need to calculate the map')
        GeneralStrikeMap.__init__(self,  file,
                                  variables_to_remap=variables_to_remap,
                                  code=code)

    def calculate_energy(self, B: float, A: float = 2.01410178,
                         Z: float = 1.0):
        """
        Calculate the energy associated to each centroid (in keV)

        Jose Rueda: jrrueda@us.es

        @param B: magnetif field modulus
        @param A: mass of the ion, in umas
        """
        dummy = get_energy(self('gyroradius'), B=B, A=A, Z=Z) / 1000.0
        self._data['e0'] = BasicVariable(name='e0', units='keV', data=dummy)

    def load_strike_points(self, file=None, verbose: bool = True):
        """
        Load the strike points used to calculate the map.

        Jose Rueda: ruejo@ipp.mpg.de

        @param file: File to be loaded. If none, name will be deduced from the
            self.file variable, so the strike points are supposed to be in
            the same folder than the strike map
        @param verbose: Flag to plot some information about the strike points
        @param newFILDSIM: Flag to decide if we are using the new FILDSIM or
            the old one
        """
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
        # If the code was SINPA, perform the remap, as it is not done in
        # fortran:
        if self._header['code'].lower() == 'sinpa':
            self.remap_strike_points()

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

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch. It contains:
                -dx: x space used by default in the fit. (pitch in fild)
                -dy: y space used by default in the fit. (gyroradius in fild)
                -x_method: Function to use in the x fit, default Gauss
                -y_method: Function to use in the y fit, default Gauss
            Acepted methods are:
                - Gauss: Gaussian fit
                - sGauss: squewed Gaussian fit
        @param min_statistics: Minimum number of points for a given r,p to make
            the fit (if we have less markers, this point will be ignored)
        @param adaptative: If true, the bin width will be adapted such that the
            number of bins in a sigma of the distribution is 4. If this is the
            case, dpitch, dgyr, will no longer have an impact
        @param confidence_level: confidence level for the uncertainty
            determination
        @param calculate_uncertainties: flag to calcualte the uncertainties of
            the fit
        """
        if self.strike_points is None:
            print('Trying to load the strike points')
            self.load_strike_points()
        # --- Prepare options:
        diag_options = {
            'dx': 1.0,
            'dy': 0.1,
            'x_method': 'Gauss',
            'y_method': 'Gauss'
        }
        diag_options.update(diag_params)
        # Select the variables
        if variables is None:
            if self.diagnostic == 'FILD':
                variables = ('pitch', 'gyroradius')
            elif self.diagnostic == 'INPA':
                variables = ('R0', 'gyroradius')
            else:
                raise errors.NotImplementedError('To be done')
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
                dict.fromkeys(names, np.full((nx, ny), np.nan))
            # For the uncertainty
            self._resolutions['unc_' + key] = \
                dict.fromkeys(names, np.full((nx, ny), np.nan))
            # For the normalization
            self._resolutions['norm_' + key] = np.full((nx, ny), np.nan)
            # For the general fits
            self._resolutions['fits_' + key] = \
                np.full((nx, ny), None, dtype=np.ndarray)
        self._resolutions['model_' + variables[0]] = diag_options['x_method']
        self._resolutions['model_' + variables[1]] = diag_options['y_method']
        # --- Core: Calculation of the resolution
        if verbose:
            print('Calculating resolutions ...')
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
                params, self._resolutions['fits_' + variables[0]], \
                    self._resolutions['norm_' + variables[0]], unc = \
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
                params, self._resolutions['fits_' + variables[1]], \
                    self._resolutions['norm_' + variables[1]], unc = \
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

    def plot_phase_space_resolution(self, ax_params: dict = {},
                                    cmap=None,
                                    nlev: int = 50,
                                    index_x: list = None,
                                    index_y: list = None):
        """
        Plot the phase space resolutions.

        Jose Rueda: jrrueda@us.es

        @param ax_param: parameters for the axis beauty function. Note, labels
        of the color axis are hard-cored, if you want custom axis labels you
        would need to draw the plot on your own
        @param cMap: is None, Gamma_II will be used
        @param nlev: number of levels for the contour
        @param index_gyr: if present, reslution would be plotted along
        gyroradius given by gyroradius[index_gyr]
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
            xAxisPlot = self.MC_variables[0].data
            yAxisPlot = self.MC_variables[1].data
            for var, subplot in zip(self._resolutions['variables'], ax):
                key = var.name
                cont = subplot.contourf(
                    xAxisPlot, yAxisPlot,
                    self._resolutions[key]['sigma'].T,
                    levels=nlev, cmap=cmap
                )
                subplot = ssplt.axis_beauty(subplot, ax_options)
                # Now place the color var in the proper position
                divider = make_axes_locatable(subplot)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cont,
                             label='$\\sigma_{%s} [%s]$' % (key, var.units),
                             cax=cax)
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
                # The transpose is to match dimenssion of the meshgrid
                # Now delete the NaN cell
                flags = np.isnan(data)
                if np.sum(~flags) > 4:  # If at least we have 4 points
                    dummy[name][key] = \
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

        Jose Rueda: jrrueda@us.es

        @param kernel: kernel for the interpolator
        @param degree: degree for the added polynomial

        See RBFInterpolator of Scipy for full documentation
        """
        # --- Select the colums to be used
        # temporal solution to save the coordinates in the array
        coords = np.zeros((self._data['x1'].size, 2))
        coords[:, 0] = self._coord_real['x1']
        coords[:, 1] = self._coord_real['x2']

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
            self._map_interpolators[key] = \
                scipy_interp.RBFInterpolator(coords,
                                             self._data[key].data,
                                             kernel=kernel, degree=degree)

    def remap_strike_points(self, overwrite: bool = True):
        """
        Remap the StrikePoints

        Jose Rueda: jrrueda@us.es
        """
        # --- See if the interpolators are defined
        if self._map_interpolators is None:
            print('Interpolators not calcualted. Calculating them')
            self._calculate_mapping_interpolators()
        # --- Proceed to remap
        # Get the shape of the map
        nx, ny = self.shape
        # Get the index of the colums containing the scintillation position
        ix1 = self.strike_points.header['info']['x1']['i']
        ix2 = self.strike_points.header['info']['x2']['i']
        # Loop over the deseired variables
        for k in self._map_interpolators.keys():
            # See if we need to overwrite
            name = 'remap_' + k
            was_there = False
            if name in self.strike_points.header['info'].keys():
                was_there = True
                if overwrite:
                    print('%s found in the object, overwritting' % k)
                    ivar = self.strike_points.header['info'][name]['i']
                else:
                    print('%s found in the object, skipping' % k)
                    continue
            # Loop over the strike points pairs
            for ix in range(nx):
                for iy in range(ny):
                    if self.strike_points.header['counters'][ix, iy] > 0:
                        n_strikes = \
                            self.strike_points.header['counters'][ix, iy]
                        remap_data = np.zeros((n_strikes, 1))
                        remap_data[:, 0] = \
                            self._map_interpolators[k](
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
                    'ShortName': name
                }
                extra_column[name]['i'] = Old_number_colums
                # Update the header
                self.strike_points.header['info'].update(extra_column)

    def remap_external_strike_points(self, strikes, overwrite: bool = True):
        """
        Remap the signal (or any external) StrikePoints

        Jose Rueda: jrrueda@us.es
        """
        # --- See if the interpolators are defined
        if self._map_interpolators is None:
            print('Interpolators not calcualted. Calculating them')
            self._calculate_mapping_interpolators()
        # --- Proceed to remap
        # Get the shape of the map
        nx, ny = strikes.shape
        # Get the index of the colums containing the scintillation position
        ix1 = strikes.header['info']['x1']['i']
        ix2 = strikes.header['info']['x2']['i']
        # Loop over the deseired variables
        for k in self._map_interpolators.keys():
            # See if we need to overwrite
            name = 'remap_' + k
            was_there = False
            if name in strikes.header['info'].keys():
                was_there = True
                if overwrite:
                    print('%s found in the object, overwritting' % k)
                    ivar = strikes.header['info'][name]['i']
                else:
                    print('%s found in the object, skipping' % k)
                    continue
            # Loop over the strike points pairs
            for ix in range(nx):
                for iy in range(ny):
                    if strikes.header['counters'][ix, iy] > 0:
                        n_strikes = \
                            strikes.header['counters'][ix, iy]
                        remap_data = np.zeros((n_strikes, 1))
                        remap_data[:, 0] = \
                            self._map_interpolators[k](
                                strikes.data[ix, iy][:, [ix1, ix2]])
                        # self.strike_points.data[ip, ir][:, iiy])
                        # append the remapped data to the object
                        if was_there:
                            strikes.data[ix, iy][:, ivar] = remap_data
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
                    'lonName': name,
                    'ShortName': name
                }
                extra_column[name]['i'] = Old_number_colums
                # Update the header
                strikes.header['info'].update(extra_column)

    def plot_pix(self, ax):
        return self._plot_pix(ax=ax)
