"""
Strike map for the INPA diagnostic

Jose Rueda: jrrueda@us.es
"""
import os
import lmfit
import logging
import numpy as np
import xarray as xr
import Lib.LibData as ssdat
import Lib.errors as errors
from tqdm import tqdm
from Lib._Machine import machine
from scipy.signal import convolve
from Lib._basicVariable import BasicVariable
from Lib._SideFunctions import createGrid, gkern
from Lib.SimulationCodes.FILDSIM import get_energy
from Lib.SimulationCodes.Common.strikes import Strikes
from Lib._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap

logger = logging.getLogger('ScintSuite.INPAstrikeMap')


class Ismap(FILDINPA_Smap):
    """
    INPA Strike map.

    Jose Rueda-Rueda: jrrueda@us.es

    Strike map object adapted to the INPA diagnostic

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
       - * plot_instrument_function: Plot the instrument function
       - build_weight_matrix: Build the instrument function
       - getRho: get the rho coordinates associated to each point

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

    def __init__(self, file: str = None, variables_to_remap: tuple = None,
                 code: str = None, theta: float = None, phi: float = None,
                 GeomID: str = 'iAUG01', verbose: bool = True,
                 decimals: int = 1, rho_pol: BasicVariable = None,
                 rho_tor: BasicVariable = None):
        """
        Initialise the INPA strike map.

        This is essentially the same initialisation than the parent strike map
        but it allows to externally set the rho values. This is useful for the
        case of the remap, where we would need to recalculate the associated
        rho in each time point, which can be time consuming, to avoid this,
        we just calculate all time points before the remap (so just one
        equilibrium must be loaded) and then include the rho to each loaded map

        :param  file: strike map file to load (option 1 to load the map)
        :param  variables_to_remap: pair of variables selected for the remap.
            By default:
                    - FILD: ('pitch', 'gyroradius')
                    - INPA: ('R0', 'gyroradius')
                    - iHIBP: ('x1', 'x2')
        :param  code: code used to calculate the map. If None, would be
            guessed automatically
        :param  theta: theta angle of the database (option 2 to load the map)
        :param  phi: phi angle of the database (option 2 to load the map)
        :param  GeomID: geometry ID of the database (option 2 to load the map)
        :param  decimals: Number of decimasl to look in the database (opt 2)
        :param  diagnostic: diagnostic to look in the database (opt 2)
        :param  verbose: print some information in the terminal
        :param  rho_pol: rho coordinates of the map points
        :param  rho_tor: rho coordinates of the map points
        """
        FILDINPA_Smap.__init__(self, file=file,
                               variables_to_remap=variables_to_remap,
                               code=code, theta=theta, phi=phi, GeomID=GeomID,
                               verbose=verbose, decimals=decimals)
        # If needed, place rho in place
        if rho_pol is not None:
            self._data['rho_pol'] = rho_pol
        if rho_tor is not None:
            self._data['rho_tor'] = rho_pol
        # Allocate space for latter
        self.secondaryStrikes = None

    # --------------------------------------------------------------------------
    # --- Database interaction
    #---------------------------------------------------------------------------
    def getRho(self, shot, time, coord: str = 'rho_pol',
               extra_options: dict = {},):
        """
        Get the rho coordinates associated to each strike point

        Jose Rueda: jrrueda@us.es

        :param  shot: shot number to load the equilibrium
        :param  time: time point to load the equilibrium
        :param  coord: coordinate: rho_pol or rho_tor
        :param  extra_options: dicctionary with extra options to initialise the
            equilibrium
        """
        # Initialise the equilibrium options
        if machine.lower() == 'aug':
            options = {
                'diag': 'EQH',
                'exp': 'AUGD',
                'ed': 0,
            }
        else:
            options = {}
        options.update(extra_options)

        rho = ssdat.get_rho(shot, self('R0'),
                            self('z0'), time=time,
                            coord_out=coord,
                            **options).squeeze()
        rmag, zmag, time = ssdat.get_mag_axis(shot=shot, time=time)

        flags = self('R0') < rmag
        rho[flags] *= -1.0

        self._data[coord] = BasicVariable(
            name=coord,
            units='',
            data=rho,
        )

    # --------------------------------------------------------------------------
    # --- Weight function calculation
    # --------------------------------------------------------------------------
    def build_weight_matrix(self, strikes,
                            variablesScint: tuple = ('R0', 'e0'),
                            variablesFI: tuple = ('R0', 'e0'),
                            weight: str = 'weight0',
                            gridFI: dict = None,
                            sigmaOptics: float = 4.5,
                            verbose: bool = True,
                            normFactor: float = 1.0,
                            energyFit = None,
                            B: float = 1.8,
                            Z: float = 1.0,
                            A: float = 2.01410,
                            ):
        """
        Build the INPA weight function

        For a complete documentation of how each submatrix is defined from the
        physics point of view, please see full and detailed INPA notes

        :param  strikes: SINPA strikes from the FIDASIM simulation with constant
            FBM. Notice that it can also be just a string pointing towards the
            strike file
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
        if strikes is None:
            raise errors.NotValidInput('I need some strikes!')
        if self.CameraCalibration is None:
            raise errors.NotFoundCameraCalibration('I need the camera calib')
        if self._grid_interp.keys() is None:
            raise errors.NotValidInput('Need to calculate interpolators first')
        if 'transformation_matrix' not in self._grid_interp.keys():
            raise errors.NotValidInput('Need to calculate T matrix first')
        # --- Initialise default grid
        if gridFI is None:
            gridFI = {
                'xmin': 1.55,
                'xmax': 2.20,
                'dx': 0.015,
                'ymin': 15.0,
                'ymax': 100.0,
                'dy': 1.0
            }
        # --- Load/put in place the strikes
        if isinstance(strikes, (str,)):
            if os.path.isfile(strikes):
                self.secondaryStrikes = \
                    Strikes(file=strikes, verbose=verbose, code='SINPA')
            else:
                self.secondaryStrikes = \
                    Strikes(runID=strikes, verbose=verbose, code='SINPA')
        elif isinstance(strikes, Strikes):
            self.secondaryStrikes = strikes
        else:
            raise errors.NotValidInput('Error in the input strikes')
        # --- Calculate camera position, if not done outside:
        if 'xcam' not in self.secondaryStrikes.header['info'].keys():
            self.secondaryStrikes.calculate_pixel_coordinates(
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
        jpx = self.secondaryStrikes.header['info']['ycam']['i']
        jpy = self.secondaryStrikes.header['info']['xcam']['i']
        jX = self.secondaryStrikes.header['info'][variablesFI[0]]['i']
        jY = self.secondaryStrikes.header['info'][variablesFI[1]]['i']
        jkind = self.secondaryStrikes.header['info']['kind']['i']
        jw = self.secondaryStrikes.header['info'][weight]['i']
        # Block 1: Preparation phase -------------------------------------------
        # --- Prepare the grids
        # - Edges
        pxEdges = np.arange(camera_frame[0]+1) - 0.5
        pyEdges = np.arange(camera_frame[1]+1) - 0.5
        nx, ny, xedges, yedges = createGrid(**gridFI)
        kindEdges = [4.5, 5.5, 6.5, 7.5, 8.5]
        # - Centers
        pxCen = 0.5 * (pxEdges[:-1] + pxEdges[1:])
        pyCen = 0.5 * (pxEdges[:-1] + pxEdges[1:])
        xCen = 0.5 * (xedges[:-1] + xedges[1:])
        yCen = 0.5 * (yedges[:-1] + yedges[1:])
        # - Volume
        xvol = xCen[1] - xCen[0]
        yvol = yCen[1] - yCen[0]
        # By definition, the space in pixel space is 1, so there is no need of
        # calculate that volume.
        # --- Prepare the data
        nStrikes = self.secondaryStrikes.data[0, 0].shape[0]
        dummy = np.zeros((nStrikes, 5))

        dummy[:, 0] = self.secondaryStrikes.data[0, 0][:, jpx]
        dummy[:, 1] = self.secondaryStrikes.data[0, 0][:, jpy]
        dummy[:, 2] = self.secondaryStrikes.data[0, 0][:, jX]
        dummy[:, 3] = self.secondaryStrikes.data[0, 0][:, jY]
        dummy[:, 4] = self.secondaryStrikes.data[0, 0][:, jkind]
        if weight is not None:
            w = self.secondaryStrikes.data[0, 0][:, jw]
        else:
            w = np.ones(nStrikes)

        # --- Calculate the pixel position
        # Block 2: Calculation phase -------------------------------------------
        edges = [pxEdges, pyEdges, xedges, yedges, kindEdges]
        H, edges_hist = np.histogramdd(
                dummy,
                bins=edges,
                weights=w,
        )
        # Add the finite focus of the optics
        if sigmaOptics > 0.01:
            logger.info('Adding finite focus')
            kernel = gkern(int(6.0*sigmaOptics)+1, sig=sigmaOptics)
            for kx in tqdm(range(H.shape[2])):
                for ky in range(H.shape[3]):
                    for kj in range(H.shape[4]):
                        H[..., kx, ky, kj] = \
                            convolve(H[..., kx, ky, kj].squeeze(), kernel,
                                     mode='same')
        else:
            logger.info('Not considering finite focusing')

        # Now perform the tensor product
        vol = xvol * yvol
        W = np.tensordot(Tmatrix, H, axes=2) / vol / normFactor
        # save it
        self.instrument_function =\
            xr.DataArray(W,
                         dims=('xs', 'ys', 'x', 'y', 'kind'),
                         coords={'kind': [5, 6, 7, 8], 'xs': gridT['x'],
                                 'ys': gridT['y'], 'x': xCen, 'y': yCen})
        self.instrument_function['kind'].attrs['long_name'] = 'Kind'
        self.instrument_function['xs'].attrs['long_name'] = \
            variablesScint[0].capitalize()
        self.instrument_function['x'].attrs['long_name'] = \
            variablesFI[0].capitalize()
        self.instrument_function['y'].attrs['long_name'] = \
            variablesFI[1].capitalize()
        self.instrument_function['ys'].attrs['long_name'] = \
            variablesScint[1].capitalize()
        # Now perform the energy scaling:
        if energyFit is not None:
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
            'rho': ''
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
