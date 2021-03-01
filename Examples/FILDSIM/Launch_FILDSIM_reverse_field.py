"""
Run FILDSIM and calculate resolutions

Created as an example to use the library without the graphical user interface.
In this case we will execute FILDSIM and calculate the resolutions, for the
case of reverse field campaing

DISCLAIMER: This was revised on the 24/02/2021. Since them several
improvement may have been done, it is possible that some function has been
changed and the script does not work at all now. (or we have a more fancy and
quick way to do this stuff). If this happens, contact
jose rueda (jrrueda@us.es) by email and he will update this 'tutorial'

You must execute first the function paths.py!
"""

import Lib as ss
import numpy as np
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

# All default values could be omitted, I just write them to show all one can
# set
FILDSIM_namelist = {
    'runID': 'test_runID',
    'result_dir': ss.paths.FILDSIM + '/results/',
    'backtrace': '.false.',                       # Default, you can omit this
    'N_gyroradius': 11,                           # Default
    'N_pitch': 10,                                # Default
    'save_orbits': 0,                             # Default
    'verbose': '.true.',                         # Default
    'N_ions': 4000,                              # Default
    'step': 0.01,                                 # Default
    'helix_length': 10.0,                         # Default
    'gyroradius': [1.5, 1.75, 2., 3., 4., 5., 6., 7., 8., 9., 10.],   # Default
    'pitch': [85., 80., 70., 60., 50., 40., 30., 20., 10, 0.],
    'min_gyrophase': 1.,
    'max_gyrophase': 1.8,
    'start_x': [-0.025, 0.025],                                       # Default
    'start_y': [-0.1, 0.1],                                           # Default
    'start_z': [0.0, 0.0],                                            # Default
    'theta': 0.0,                                                     # Default
    'phi': 0.0,                                                       # Default
    'geometry_dir': ss.paths.FILDSIM + './geometry/',
    'N_scintillator': 1,                                              # Default
    'N_slits': 6,                                                     # Default
    'scintillator_files': ['aug_fild1_scint.pl'],                     # Default
    'slit_files': ['aug_rfildb_pinhole1.pl',                       # Default
                   'aug_rfildb_pinhole_2.pl',                       # Default
                   'aug_rfildb_slit_1.pl',                          # Default
                   'aug_rfildb_slit_back.pl',                       # Default
                   'aug_rfildb_slit_lateral_1.pl',                  # Default
                   'aug_rfildb_slit_lateral_2.pl']}                 # Default

# Write namelist
ss.fildsim.write_namelist(namelist_path, **FILDSIM_namelist)
namelist_name = os.path.join(namelist_path,
                             FILDSIM_namelist['runID'] + '.cfg')
# Run FILDSIM
ss.fildsim.run_FILDSIM(ss.paths.FILDSIM, namelist_name)

# Load the result of the simulation
base_name = FILDSIM_namelist['result_dir'] + FILDSIM_namelist['runID']
strike_map_file = base_name + '_strike_map.dat'
strike_points_file = base_name + '_strike_points.dat'
# Load the strike map
Smap = ss.mapping.StrikeMap('FILD', strike_map_file)
# Load the strike points used to calculate the map
Smap.load_strike_points(strike_points_file)
# Calculate the resolution: Default is 1 degree of bin width for the pitch
# histograms, 0.1 for gyro-radius, Gaussians for pitch and skew Gaussian for
# gyro-radius.
Smap.calculate_resolutions()  # Default call,
# Example changing the binning and settings skewGaussian for boths
# Smap.calculate_resolutions(dpitch=2.0, dgyr=0.25, p_method='sGauss',
#                            g_method='sGauss')
# Example changing the minimum of markers needed to consider making the fit,
# default is 20
# Smap.calculate_resolutions(dpitch=2.0, dgyr=0.25, p_method'Gauss',
#                            g_method='sGauss', min_statistics=50)
