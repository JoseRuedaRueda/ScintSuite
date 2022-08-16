"""
FILD strike map

Contains the Strike Map object fully adapted to FILD
"""

from Lib._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap


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

    pass
