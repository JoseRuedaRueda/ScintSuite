"""
Run FILDSIM and calculate resolutions

Created as an example to use the library without the graphical user interface.
In this case we will execute FILDSIM and calculate the resolutions

DISCLAIMER: This was created for version 0.2.4, it is possible that some
function has been changed and the script does not work at all now.
(or we have a more fancy and quick way to do this stuff). If this happens,
contact Jose Rueda (jrrueda@us.es) by email and he will update this 'tutorial'

Note: I modify it a bit for 0.4.2, but I did not test it. Plese, if you do,
send me an email with the results (jrrueda@us.es)
You must execute first the function paths.py!
"""

import ScintSuite.as ss
import os


# -----------------------------------------------------------------------------
namelist_path = ss.paths.FILDSIM   # Paths to save namelist
# -----------------------------------------------------------------------------
# Section 0: FILDSIM Settings
# -----------------------------------------------------------------------------
# Note, in the function LibMap.remap_all_loaded_frames_FILD, you have the
# example of how to calculate the phi and theta angles which correspond to a
# given shot and time point (it is already automatized)
# Here we will just manually select the theta and phi.
# Also the suite include default paths to save FILDSIM namelist and results,
# but here we will manually choose them

FILDSIM_namelist = {
    'config': {
        'runid': 'kiwi',
        'result_dir': ss.paths.FILDSIM + '/results/',
        'backtrace': False,
        'n_gyroradius': 11,
        'n_pitch': 10,
        'save_orbits': 0,
        'verbose': True,
    },
    'input_parameters': {
        'n_ions': 6000,
        'step': 0.01,
        'helix_length': 10.0,
        'gyroradius': [1.5, 1.75, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
        'pitch_angle': [85., 80., 70., 60., 50., 40., 30., 20., 10, 0.],
        'gyrophase_range': [0., 3.14],
        'start_x': [-0.025, 0.025],
        'start_y': [-0.1, 0.1],
        'start_z': [0.0, 0.0],
        'theta': 0.0,
        'phi': 0.0
    },
    'plate_setup_cfg': {
        'geometry_dir': ss.paths.FILDSIM + './geometry/',
        'n_scintillator': 1,
        'n_slits': 6
    },
    'plate_files': {
        'scintillator_files': ['aug_fild1_scint.pl'],
        'slit_files': ['aug_fild1_pinhole_1_v2.pl',
                       'aug_fild1_pinhole_2_v2.pl',
                       'aug_fild1_slit_1_v2.pl',
                       'aug_fild1_slit_back_v2.pl',
                       'aug_fild1_slit_lateral_1_v2.pl',
                       'aug_fild1_slit_lateral_2_v2.pl']
    }
}

# Write namelist
ss.fildsim.write_namelist(FILDSIM_namelist, namelist_path)
namelist_name = os.path.join(namelist_path,
                             FILDSIM_namelist['config']['runid'] + '.cfg')
# Run FILDSIM
ss.fildsim.run_FILDSIM(namelist_name)

# Load the result of the simulation
base_name = FILDSIM_namelist['config']['result_dir']\
    + FILDSIM_namelist['config']['runid']
strike_map_file = base_name + '_strike_map.dat'
strike_points_file = base_name + '_strike_points.dat'
# Load the strike map
Smap = ss.mapping.StrikeMap('FILD', strike_map_file)
# Load the strike points used to calculate the map
Smap.load_strike_points(strike_points_file)
# Calculate the resolution: Default is 1 degree of bin width for the pitch
# histograms, 0.1 for gyro-radius, Gaussian for pitch and skew Gaussian for
# gyro-radius.
Smap.calculate_resolutions()   # Default call,
# Example changing the binning and settings skewGaussian for boths
# Smap.calculate_resolutions(diag_params = {'dpitch':2.0, 'dgyr':0.25,
#                                           'p_method': 'sGauss'},
#                            g_method='sGauss')
# Example changing the minimum of markers needed to consider making the fit,
# default is 100
# Smap.calculate_resolutions(diag_params = {'dpitch':2.0, 'dgyr':0.25,
#                                           'p_method': 'sGauss'},
#                            g_method='sGauss', min_statistics=500)
Smap.plot_resolutions()   # Default call for plotting the resolutions
