"""
FILD strike map

Contains the Strike Map object fully adapted to FILD
"""
import os
import logging
import numpy as np
import xarray as xr
from scipy.signal import convolve
from ScintSuite._SideFunctions import createGrid, gkern
from ScintSuite._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap
from ScintSuite.SimulationCodes.FILDSIM.execution import get_energy
import ScintSuite.errors as errors
logger = logging.getLogger('ScintSuite.FILDsmap')
try:
    import lmfit
except ModuleNotFoundError:
    pass


class Fsmap(FILDINPA_Smap):
    """
    FILD Strike map.

    Jose Rueda-Rueda: jrrueda@us.es

    In the current stage, it does not contain any particular method, all of them
    are inherited from the FILDINPA_Smap object. In the future some routines
    related with fine plotting will be added.

    Public Methods (* means inherited from the father):
       - *calculate_pixel_coordinates: calculate the map coordinates in the camera
       - *setRemapVariables: Set the variables to be used when remapping
       - *interp_grid: Interpolate the smap variables in a given camera frame
       - *export_spatial_coordinates: save grid point into a .txt
       - *plot_var: perform a quick plot of a variable (or pair) of the map
       - *plot_pix: plot the strike map in the camera space
       - *plot_real: plot the scintillator in the real space
       - *calculate_energy: calculate the energy associated with each gyroradius
       - *load_strike_points: Load the points used to create the map
       - *calculate_phase_space_resolution: calculate the resolution associated
          with the phase-space variables of the map
       - *plot_phase_space_resolution: plot the resolution in the phase space
       - *remap_strike_points: remap the loaded strike points
       - *remap_external_strike_points: remap any strike points
       - *plot_phase_space_resolution_fits: plot the resolution in the phase space
       - *plot_collimator_factors: plot the resolution in the phase space
       - *plot_instrument_function: Plot the instrument function
       - build_weight_matrix: Build the instrument function


    Private method (* means inherited from the father):
       - *_calculate_transformation_matrix: Calculate the transformation matrix
       - *_calculate_instrument_function_interpolators: calculate the
            interpolators for the instrument function
       - *_calculate_mapping_interpolators: Calculate the interpolators to map
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

    # --------------------------------------------------------------------------
    # --- Weight function calculation / plot
    # --------------------------------------------------------------------------
    def build_weight_matrix(self, grid_options_scint,
                            grid_options_pinhole,
                            efficiency=None,
                            B=1.8, A=2.0, Z=1, 
                            cutoff=None):
        """
        Build FILD weight function

        Jose Rueda Rueda: jrrueda@us.es

        Introduced in version 1.1.0
        Based on the old build_weight_matrix from the FILDSIM library

        :param  xscint: x
        :param  yscint: y
        :param  xpin: x
        :param  ypin: y
        :param  efficiency: ScintillatorEfficiency() object. If None, efficiency
        will not be included
        :param  spoints: path pointing to the strike point file. Not needed if smap
        is a strike map object with the resolutions already calculated
        :param  diag_params: Parametes for the resolution calculation, useless if
        the input strike map has the resolutions already calcualted See
        StrikeMap.calculate_resolutions() for the whole list of options
        :param  B: Magnetic field, used to translate between radius and energy, for
        the efficiency evaluation
        :param  A: Mass in amu, used to translate between radius and energy, for the
        efficiency evaluation
        :param  Z: charge in elecrton charges, used to translate between radius and
        energy, for the efficiency evaluation
        :param  cutoff: cutoff value for the weight matrix. If None, no cutoff is
        applied. If a value is given, all pixels in the scintillator grid with a
        weight lower than cutoff*max(weight (for each pixel in the pinhole)) will
        be set to zero. This kills fake correlations detected in tomographic 
        reconstructions

        DISCLAIMER: Only fully tested when the gyroradius or the energy variable
        is the y variable of the strike map. Please consider to use this
        criteria as default
        """
        # --- Check the StrikeMap
        if self.strike_points is None:
            logger.info('Strikes not loaded, loading')
            self.load_strike_points()
        if self._resolutions is None:
            self.calculate_phase_space_resolution()
        # --- Prepare the grid
        nxs, nys, xedgess, yedgess = createGrid(**grid_options_scint)
        xcenterss = (xedgess[:-1] + xedgess[1:]) * 0.5
        ycenterss = (yedgess[:-1] + yedgess[1:]) * 0.5
        nxp, nyp, xedgesp, yedgesp = createGrid(**grid_options_pinhole)
        xcentersp = (xedgesp[:-1] + xedgesp[1:]) * 0.5
        ycentersp = (yedgesp[:-1] + yedgesp[1:]) * 0.5
        # --- Get the names / models of the variables
        names = [self._resolutions['variables'][k].name for k in range(2)]
        models = [self._resolutions['model_' + dum] for dum in names]
        logger.info('Calculating FILD weight matrix')
        logger.debug('Applied models: ' + models[0] + ' ' +  models[1])
        # --- Prepare model and efficiency
        # Prepare the model based on the resolution calculation:
        # Just for consitency, use the same one we use for the resolution
        # calculation
        models_func = {
            'Gauss': lmfit.models.GaussianModel().func,
            'sGauss': lmfit.models.SkewedGaussianModel().func
        }

        # prepare the grid
        XX, YY = np.meshgrid(xcenterss, ycenterss)
        # Prepare the parameters we will interpolate:
        parameters_to_consider = {
            'Gauss': ['sigma', 'center'],
            'sGauss': ['sigma', 'gamma', 'center']
        }
        # Make the comparison if efficiency is None or not, to avoid doing it
        # inside the loop:
        if efficiency is not None:
            for k in range(2):
                if k == 0:
                    dummy = xcentersp
                else:
                    dummy = ycentersp
                if names[k] == 'energy' or names[k] == 'e0':
                    eff = efficiency(dummy).values
                elif names[k] == 'gyroradius':
                    energy = get_energy(dummy, B, A, Z) / 1000.0
                    eff = efficiency(energy).values
            logger.info('Considering scintillator efficiency in W')
        else:
            eff = np.ones(ycentersp.size)
        # Build the weight matrix. We will use brute force, I am sure that there
        # is a tensor product implemented in python which does the job in a more
        # efficient way
        res_matrix = np.zeros((nxs, nys, nxp, nyp))

        for kk in range(nyp):
            for ll in range(nxp):
                # Interpolate sigmas, gammas and collimator_factor
                x_parameters = {}
                for k in parameters_to_consider[models[0]]:
                    x_parameters[k] = \
                        self._interpolators_instrument_function[names[0]][k](
                                xcentersp[ll], ycentersp[kk])
                y_parameters = {}
                for k in parameters_to_consider[models[1]]:
                    y_parameters[k] = \
                        self._interpolators_instrument_function[names[1]][k](
                                xcentersp[ll], ycentersp[kk])

                col_factor = \
                    self._interpolators_instrument_function['collimator_factor'](
                        xcentersp[ll], ycentersp[kk]) / 100.0
                if col_factor > 0.0:
                    # Calculate the contribution:
                    dummy = col_factor * \
                            models_func[models[0]](XX.flatten(),
                                                   **x_parameters) * \
                            models_func[models[1]](YY.flatten(),
                                                   **y_parameters) \
                            * eff[kk]
                    # The model extend over the whole plate, which can produce
                    # some numerical errors. We will set to zero the pixels with 
                    # less weight than the cutoff (relative to maximal weight)
                    if cutoff is not None:
                        mask = dummy / np.max(dummy) < cutoff
                        dummy[mask] = 0.0
                    res_matrix[:, :, ll, kk] = np.reshape(dummy, XX.shape).T
                else:
                    res_matrix[:, :, ll, kk] = 0.0
        res_matrix[np.isnan(res_matrix)] = 0.0
        # --- Now fill the weight function
        self.instrument_function = xr.DataArray(
                res_matrix, dims=('xs', 'ys', 'x', 'y'),
                coords={'xs': xcenterss, 'ys': ycenterss, 'x': xcentersp,
                        'y': ycentersp}
        )
        self.instrument_function['xs'].attrs['long_name'] = names[0].capitalize()
        self.instrument_function['x'].attrs['long_name'] = names[0].capitalize()
        self.instrument_function['y'].attrs['long_name'] = names[1].capitalize()
        self.instrument_function['ys'].attrs['long_name'] = names[1].capitalize()


    def build_numerical_weight_matrix(self, strikes=None,
                                      variablesScint: tuple = ('pitch', 'gyroradius'),
                                      sigmaOptics: float = 0.0,
                                      verbose: bool = True,
                                      normFactor: float = 1.0,
                                      energyFit=None,
                                      efficiency=None,
                                      B: float = 1.8,
                                      Z: float = 1.0,
                                      A: float = 2.01410,) -> None:
        """
        Build the FILD numerical weight function.

        For a complete documentation of how each submatrix is defined from the
        physics point of view, please see full and detailed INPA notes

        :param  strikes: FILDSIM strikes.Notice that it can also be just a 
            string pointing towards the strike file
        :param  variablesScint: tuple of variable to spawn the scintillator space
        :param  variablesFI: tuple of variables to spawn the FI space
        :param  weigt: name of the weight to be selected
        :param  gridFI: grid for the variables in the FI phase space
        :param  sigmaOptics: fine resolution sigma of the optical system
        :param  verbose: flag to incldue information in the console
        :param  normFactor: Overal factor to scale the weight matrix

        Notes:
        - Scintillator grid cannot be included as input because is taken from
            the transformation matrix
        - The normFactor is though to be used as the constant value of the
            FBM set for the FIDASIM simulation, to eliminate this constant dummy
            factor

        TODO: Include fine resolution depending of the optical axis
        """
        # Block 0: Loading and settings ----------------------------------------
        # --- Check inputs
        if self.CameraCalibration is None:
            raise errors.NotFoundCameraCalibration('I need the camera calib')
        if self._grid_interp.keys() is None:
            raise errors.NotValidInput('Need to calculate interpolators first')
        if 'transformation_matrix' not in self._grid_interp.keys():
            raise errors.NotValidInput('Need to calculate T matrix first')
        # --- Check the strike grid
        diffgyroradius = np.diff(self.strike_points.header['gyroradius'])
        if np.std(diffgyroradius)/diffgyroradius[0] > 0.01:
            raise Exception('The strikes are not equally spaced in gyroradius')
        diffpitch = np.diff(self.strike_points.header['XI'])
        if np.std(diffpitch)/diffpitch[0] > 0.01:
            raise Exception('The strikes are not equally spaced in pitch')
        # --- Check the efficiency
        if efficiency is not None and energyFit is not None:
            raise Exception('You cannot use both efficiency and energyFit')
        # --- Load/put in place the strikes
        if self.strike_points is None:
            if isinstance(strikes, (str,)):
                if os.path.isfile(strikes):
                    self.strike_points = \
                        Strikes(file=strikes, verbose=verbose, code='SINPA')
                else:
                    self.strike_points = \
                        Strikes(runID=strikes, verbose=verbose, code='SINPA')
            elif isinstance(strikes, Strikes):
                self.strike_points = strikes
        else:
            logger.info('Using Smap strikes')
        # --- Calculate camera position, if not done outside:
        if 'xcam' not in self.strike_points.header['info'].keys():
            self.strike_points.calculate_pixel_coordinates(
                self.CameraCalibration)
        else:
            text = 'Strikes already contain pixel position!' +\
                'Assuming you did it with the same calibration than the map'
            logger.warning(text)
        # --- Prepare the names
        nameT = variablesScint[0] + '_' + variablesScint[1]
        # --- Get the Tmatrix
        Tmatrix = self._grid_interp['transformation_matrix'][nameT]
        gridT = self._grid_interp['transformation_matrix'][nameT+'_grid']
        camera_frame = Tmatrix.shape[2:4]
        # --- Get the index of the different colums
        jpx = self.strike_points.header['info']['ycam']['i']
        jpy = self.strike_points.header['info']['xcam']['i']
        jX = self.strike_points.header['info']['remap_pitch']['i']
        jY = self.strike_points.header['info']['remap_gyroradius']['i']
        # Block 1: Preparation phase -------------------------------------------
        # --- Prepare the grids
        # - Edges
        pxEdges = np.arange(camera_frame[0]+1) - 0.5
        pyEdges = np.arange(camera_frame[1]+1) - 0.5
        # - Centers
        pxCen = 0.5 * (pxEdges[:-1] + pxEdges[1:])
        pyCen = 0.5 * (pxEdges[:-1] + pxEdges[1:])
        # - Velocity space grid
        xCen = self.strike_points.header['XI']  # Pinhole
        yCen = self.strike_points.header['gyroradius']
        # - Volume
        xvol = xCen[1] - xCen[0]
        yvol = yCen[1] - yCen[0]
        # By definition, the space in pixel space is 1, so there is no need of
        # calculate that volume.
        P2F = np.zeros((Tmatrix.shape[2], Tmatrix.shape[3],
                        xCen.size, yCen.size))
        # --- Prepare the data
        for jxpinhole in range(xCen.size):
            for jypinhole in range(yCen.size):
                try:
                    nStrikes = self.strike_points.data[jxpinhole, jypinhole].shape[0]
                    if nStrikes == 0:
                        continue
                except AttributeError:
                    continue
                # Get the collimator factor
                col_factor = \
                    self._interpolators_instrument_function['collimator_factor'](xCen[jxpinhole], yCen[jypinhole]) / 100.0
                if col_factor <= 0.0:
                    continue
                
                if efficiency is not None:
                    energy = get_energy(yCen[jypinhole], B, A, Z) / 1000.0
                    eff = efficiency(energy).values
                else:
                    eff = 1.0
                w = eff * col_factor * np.ones(nStrikes)/nStrikes

                # --- Get the ideal camera frame
                H, xpixel, ypixel = np.histogram2d(
                    self.strike_points.data[jxpinhole, jypinhole][:, jpx],
                    self.strike_points.data[jxpinhole, jypinhole][:, jpy],
                    bins=[pxEdges, pyEdges], weights=w)
                # --- Defocus camera frame if needed
                if sigmaOptics > 0.01:
                    kernel = gkern(int(6.0*sigmaOptics)+1, sig=sigmaOptics)
                    H = convolve(H, kernel, mode='same')
                    # set to zero the negative celss due to numerical errors
                    H[H < 0.0] = 0.0
                # --- Place in position the camera frame
                P2F[:, :, jxpinhole, jypinhole] = H.copy()
        # --- Prepare the weight matrix
        #vol = xvol * yvol. Was a bug to introduce this factor. The FI
        # distribution is already normalised to this volume
        vol = 1.0
        WF = np.tensordot(Tmatrix, P2F, axes=2) / normFactor / vol
        # save it
        self.instrument_function =\
            xr.DataArray(WF,
                         dims=('xs', 'ys', 'x', 'y'),
                         coords={'xs': gridT['x'],
                                 'ys': gridT['y'], 'x': xCen, 'y': yCen})
        self.instrument_function['xs'].attrs['long_name'] = \
            variablesScint[0].capitalize()
        self.instrument_function['x'].attrs['long_name'] = 'Pitch'
        self.instrument_function['y'].attrs['long_name'] = 'Gyroradius'
        self.instrument_function['ys'].attrs['long_name'] = \
            variablesScint[1].capitalize()
        # Now perform the energy scaling:
        if energyFit is not None:
            logger.info('Adding energy scaling')
            if isinstance(energyFit, str):
                # The user give us the energy fit file
                fit = lmfit.model.load_modelresult(energyFit)
            else:
                # Assume we have a fit object
                fit = energyFit
            # So, now, we need to see which scintillator variable is energy
            # or gyroradius
            if ('energy' in variablesScint) or ('e0' in variablesScint):
                dumNames = np.array(variablesScint)
                i = np.where((dumNames == 'e0') | (dumNames == 'energy'))[0]
                if i == 0:
                    keyToEval = 'xs'
                else:
                    keyToEval = 'ys'
                xToEval = self.instrument_function[keyToEval]
                scaleFactor = fit.eval(x=xToEval.values)

            elif 'gyroradius' in variablesScint:
                dumNames = np.array(variablesScint)
                i = np.where((dumNames == 'gyroradius'))[0]
                if i == 0:
                    keyToEval = 'xs'
                else:
                    keyToEval = 'ys'
                # Now move to energy
                xToEval = self.instrument_function[keyToEval]
                energyToEval = get_energy(xToEval.values, B, A, Z)
                scaleFactor = fit.eval(x=energyToEval)

            scaleFactor = xr.DataArray(scaleFactor, dims=keyToEval,
                                       coords={keyToEval: xToEval})
            self.instrument_function = self.instrument_function * scaleFactor

        # --- Add some units
        # TODO: This is a bit hardcored, so would be better in another way
        units = {
            'r0': 'm',
            'gyroradius': 'cm',
            'gyroradius0': 'cm',
            'energy': 'keV',
            'pitch': ''
            }
        for k in self.instrument_function.coords.keys():
            try:
                self.instrument_function[k].attrs['units'] =\
                    units[self.instrument_function[k].attrs['long_name'].lower()]
            except KeyError:
                pass
        # --- Add other metadata:
        self.instrument_function.attrs['sigmaOptics'] = sigmaOptics
        self.instrument_function.attrs['normFactor'] = normFactor
        self.instrument_function.attrs['B'] = B
        self.instrument_function.attrs['Z'] = Z
        self.instrument_function.attrs['A'] = A

        return
    

    def build_parameters_xarray(self,calculate_uncertainties=False):
        """
        Put all the fitting parameters into an xarray and WF

        Alex Reyner Viñolas: alereyvinn@alum.us.es
        """
        # --- Check the StrikeMap
        if self.strike_points is None:
            logger.info('Strikes not loaded, loading')
            self.load_strike_points()
        if self._resolutions is None:
            self.calculate_phase_space_resolution(calculate_uncertainties=calculate_uncertainties)

        # --- Get the names / models of the variables
        names = [self._resolutions['variables'][k].name for k in range(2)]
        models = [self._resolutions['model_' + dum] for dum in names]
        logger.info("Calculating FILD fit parameters matrix -> \
                    self._resolutions['fit_xarrays']")
        logger.debug('Applied models: ' + models[0] + ' ' +  models[1])

        # Prepare the parameters that will be included in the xarray:
        parameters_to_consider = {
            'Gauss': ['sigma', 'center'],
            'sGauss': ['sigma', 'gamma', 'center']
        }

        #  --- Define the x and y coordinates
        xcoords = self.MC_variables[0].data
        ycoords = self.MC_variables[1].data

        # --- Build the dataset
        self._resolutions['fit_xarrays'] = xr.Dataset(coords={'x':xcoords,'y':ycoords})
        self._resolutions['fit_xarrays'].coords['y'].attrs['long_name'] = 'Gyroradius'
        self._resolutions['fit_xarrays'].coords['y'].attrs['units'] = 'cm'   
        self._resolutions['fit_xarrays'].coords['x'].attrs['long_name'] = 'Pitch'
        self._resolutions['fit_xarrays'].coords['x'].attrs['units'] = 'º'    
        for i in range(2):
            name = names[i]
            model = models[i]
            for j in parameters_to_consider[model]:
                dummy = self._resolutions[name][j]
                self._resolutions['fit_xarrays'][name+'_'+j] = (['x','y'],dummy)
        self._resolutions['fit_xarrays']['coll_factor'] = (['x','y'],
                                          self('collimator_factor_matrix'))
        

        # --- Add uncs if they exist
        try:
            self._resolutions['unc_xarrays'] = xr.Dataset(coords={'x':xcoords,'y':ycoords})
            for i in range(2):
                name = names[i]
                model = models[i]
                for j in parameters_to_consider[model]:
                    dummy = self._resolutions['unc_'+name][j]
                    self._resolutions['unc_xarrays'][name+'_'+j] = (['x','y'],dummy)
        except:
            pass
        
