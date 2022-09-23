"""
FILD strike map

Contains the Strike Map object fully adapted to FILD
"""
import logging
import numpy as np
import xarray as xr
from Lib._SideFunctions import createGrid
from Lib._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap
from Lib.SimulationCodes.FILDSIM.execution import get_energy
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
                            B=1.8, A=2.0, Z=1):
        """
        Build FILD weight function

        Jose Rueda Rueda: jrrueda@us.es

        Introduced in version 1.1.0
        Based on the old build_weight_matrix from the FILDSIM library

        @param xscint: x
        @param yscint: y
        @param xpin: x
        @param ypin: y
        @param efficiency: ScintillatorEfficiency() object. If None, efficiency
        will not be included
        @param spoints: path pointing to the strike point file. Not needed if smap
        is a strike map object with the resolutions already calculated
        @param diag_params: Parametes for the resolution calculation, useless if
        the input strike map has the resolutions already calcualted See
        StrikeMap.calculate_resolutions() for the whole list of options
        @param B: Magnetic field, used to translate between radius and energy, for
        the efficiency evaluation
        @param A: Mass in amu, used to translate between radius and energy, for the
        efficiency evaluation
        @param Z: charge in elecrton charges, used to translate between radius and
        energy, for the efficiency evaluation
        @param only_gyroradius: flag to decide if the output will be the matrix
        just relating giroradius in the pinhole and the scintillator, ie, pitch
        integrated

        DISCLAIMER: Only fully tested when the gyroradius or the energy variable
        is the y variable of the strike map. Please consider to use this
        criteria as default
        """
        # --- Check the StrikeMap
        if self.strike_points is None:
            logger.info('Strikes not loaded, loading')
            self.load_strike_points()
        if self._resolutions is None:
            self.calculate_phase_space_resolution(diag_params=diag_parameters)
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
                    eff = efficiency.interpolator(dummy)
                elif names[k] == 'gyroradius':
                    energy = get_energy(dummy, B, A, Z)
                    eff = efficiency.interpolator(energy)
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
                for k in parameters_to_consider[models[0]]:
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
                    res_matrix[:, :, ll, kk] = np.reshape(dummy, XX.shape).T
                else:
                    res_matrix[:, :, ll, kk] = 0.0
        res_matrix[np.isnan(res_matrix)] = 0.0
        # --- Now fill the weight function
        self.instrument_function = xr.DataArray(
                res_matrix, dims = ('xs', 'ys', 'xp', 'yp'),
                coords = {'xs':xcenterss, 'ys': ycenterss, 'xp': xcentersp,
                        'yp': ycentersp}
        )
        self.instrument_function['xs'].attrs['long_name'] = names[0].capitalize()
        self.instrument_function['xp'].attrs['long_name'] = names[0].capitalize()
        self.instrument_function['yp'].attrs['long_name'] = names[1].capitalize()
        self.instrument_function['ys'].attrs['long_name'] = names[1].capitalize()



