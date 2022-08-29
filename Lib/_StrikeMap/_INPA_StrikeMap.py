"""
Strike map for the INPA diagnostic

Jose Rueda: jrrueda@us.es
"""
import Lib.LibData as ssdat
from Lib._Machine import machine
from Lib._basicVariable import BasicVariable
from Lib._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap


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

        @param file: strike map file to load (option 1 to load the map)
        @param variables_to_remap: pair of variables selected for the remap.
            By default:
                    - FILD: ('pitch', 'gyroradius')
                    - INPA: ('R0', 'gyroradius')
                    - iHIBP: ('x1', 'x2')
        @param code: code used to calculate the map. If None, would be
            guessed automatically
        @param theta: theta angle of the database (option 2 to load the map)
        @param phi: phi angle of the database (option 2 to load the map)
        @param GeomID: geometry ID of the database (option 2 to load the map)
        @param decimals: Number of decimasl to look in the database (opt 2)
        @param diagnostic: diagnostic to look in the database (opt 2)
        @param verbose: print some information in the terminal
        @param rho_pol: rho coordinates of the map points
        @param rho_tor: rho coordinates of the map points
        """
        FILDINPA_Smap.__init__(self, file=file,
                               variables_to_remap=variables_to_remap,
                               code=code, theta=theta, phi=phi, GeomID=GeomID,
                               verbose=verbose, decimals=decimals)

        if rho_pol is not None:
            self._data['rho_pol'] = rho_pol
        if rho_tor is not None:
            self._data['rho_tor'] = rho_pol

    def getRho(self, shot, time, coord: str = 'rho_pol',
               extra_options: dict = {},):
        """
        Get the rho coordinates associated to each strike point

        Jose Rueda: jrrueda@us.es

        @param shot: shot number to load the equilibrium
        @param time: time point to load the equilibrium
        @param coord: coordinate: rho_pol or rho_tor
        @param extra_options: dicctionary with extra options to initialise the
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
