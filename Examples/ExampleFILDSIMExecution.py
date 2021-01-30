"""
Run FILDSIM and calculate resolutions

Created as an example to use the library without the graphical user interface.
In this case we will execute FILDSIM and calculate the resolutions

DISCLAIMER: This was created on the 15/01/2020. Since them several
improvement may have been done, it is possible that some function has been
changed and the script does not work at all now. (or we have a more fancy and
quick way to do this stuff). If this happens, contact
jose rueda (jose.rueda@ipp.mpg.de) by email and he will update this 'tutorial'

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
# given shot and time point (it is already automatised)
# Here we will just manually select the theta and phi.
# Also the suite include default paths to save FILDSIM namelist and results,
# but here we will manually choose them

# All default values could be ommited, I just write them to show all one can
# set
FILDSIM_namelist = {
    'runID': 'test_runID',
    'result_dir': ss.paths.FILDSIM + '/results/',
    'backtrace': '.false.',                       # Default, you can omit this
    'N_gyroradius': 11,                           # Default
    'N_pitch': 10,                                # Default
    'save_orbits': 0,                             # Default
    'verbose': '.true.',                         # Default
    'N_ions': 30000,                              # Default
    'step': 0.01,                                 # Default
    'helix_length': 10.0,                         # Default
    'gyroradius': [1.5, 1.75, 2., 3., 4., 5., 6., 7., 8., 9., 10.],   # Default
    'pitch': [85., 80., 70., 60., 50., 40., 30., 20., 10, 0.],        # Default
    'min_gyrophase': 1.0,                                             # Default
    'max_gyrophase': 1.8,                                             # Default
    'start_x': [-0.025, 0.025],                                       # Default
    'start_y': [-0.1, 0.1],                                           # Default
    'start_z': [0.0, 0.0],                                            # Default
    'theta': 0.0,                                                     # Default
    'phi': 0.0,                                                       # Default
    'geometry_dir': ss.paths.FILDSIM + './geometry/',
    'N_scintillator': 1,                                              # Default
    'N_slits': 6,                                                     # Default
    'scintillator_files': ['aug_fild1_scint.pl'],                     # Default
    'slit_files': ['aug_fild1_pinhole_1_v2.pl',                       # Default
                   'aug_fild1_pinhole_2_v2.pl',                       # Default
                   'aug_fild1_slit_1_v2.pl',                          # Default
                   'aug_fild1_slit_back_v2.pl',                       # Default
                   'aug_fild1_slit_lateral_1_v2.pl',                  # Default
                   'aug_fild1_slit_lateral_2_v2.pl']}                 # Default

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
# hisograms, 0.1 for gyroadius, gaussians for pitch and skewGaussian for
# gyroradius.
Smap.calculate_resolutions()  # Default call,
# Example changing the binning and settings skewGaussian for boths
# Smap.calculate_resolutions(dpitch=2.0, dgyr=0.25, p_method='sGauss',
#                            g_method='sGauss')
# Example changing the minimum of markers needed to consider making the fit,
# default is 20
# Smap.calculate_resolutions(dpitch=2.0, dgyr=0.25, p_method'Gauss',
#                            g_method='sGauss', min_statistics=50)

# The calculation of the resolution is done internally, the objetive of the
# suite is calculated them as hey are needed for tomography, but not to be a
# full interface to FILDSIM (at this stage, the corresponging request can be
# opened in gitlab). Therefore a nice automatic plotting of the fitted curves
# is not implemented. But, as this is an example, I will write here the
# fitting in a pedestrian way, in order to show what the routine does, and in
# order to do some plotting:
plot_gaussian = True
if plot_gaussian:
    desired_gyr = 3.0       # Desired strike points to fit
    desired_pitch = 40.0
    dpitch = 1.0            # bin width for the histograms
    dgyr = 0.1
    # Select the data
    data = Smap.strike_points['Data'][
        (Smap.strike_points['Data'][:, 0] == desired_gyr) *
        (Smap.strike_points['Data'][:, 1] == desired_pitch), :]
    npoints = len(data[:, 0])
    print('We have: ', npoints, ' strike points')
    # Prepare the bin edged according to the desired width
    edges_pitch = \
        np.arange(start=data[:, 7].min() - dpitch,
                  stop=data[:, 7].max() + dpitch,
                  step=dpitch)
    edges_gyr = \
        np.arange(start=data[:, 6].min() - dgyr,
                  stop=data[:, 6].max() + dgyr,
                  step=dgyr)
    par_p, result = ss.mapping._fit_to_model_(data[:, 7], bins=edges_pitch,
                                              model='Gauss')
    par_g, result = ss.mapping._fit_to_model_(data[:, 6], bins=edges_gyr,
                                              model='sGauss')
