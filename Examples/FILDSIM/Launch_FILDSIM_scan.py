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
nmarkers = [1000, 5000, 11000, 50000, 100000]
FILDSIM_namelist = {
    'result_dir': ss.paths.FILDSIM + '/results/',
    'backtrace': '.false.',                       # Default, you can omit this
    'N_gyroradius': 11,                           # Default
    'N_pitch': 10,                                # Default
    'save_orbits': 0,                             # Default
    'verbose': '.true.',                         # Default
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
    'phi': 0.0,
    'geometry_dir': ss.paths.FILDSIM + './geometry/'}

# -----------------------------------------------------------------------------
# --- Section 1: Run FILDSIM
# -----------------------------------------------------------------------------
for i in range(len(nmarkers)):
    FILDSIM_namelist['N_ions'] = nmarkers[i]
    FILDSIM_namelist['runID'] = str(nmarkers[i])
    # Write namelist
    ss.fildsim.write_namelist(namelist_path, **FILDSIM_namelist)
    namelist_name = os.path.join(namelist_path,
                                 FILDSIM_namelist['runID'] + '.cfg')
    # Run FILDSIM
    ss.fildsim.run_FILDSIM(ss.paths.FILDSIM, namelist_name)

# -----------------------------------------------------------------------------
# --- Section 2: Analyse the results
# -----------------------------------------------------------------------------
SMAPS = []

for i in range(len(nmarkers)):
    # Load the result of the simulation
    FILDSIM_namelist['runID'] = str(nmarkers[i])
    base_name = FILDSIM_namelist['result_dir'] + FILDSIM_namelist['runID']
    strike_map_file = base_name + '_strike_map.dat'
    strike_points_file = base_name + '_strike_points.dat'
    # Load the strike map
    Smap = ss.mapping.StrikeMap('FILD', strike_map_file)
    # Load the strike points used to calculate the map
    Smap.load_strike_points(strike_points_file)
    # Calculate the resolutions
    Smap.calculate_resolutions()  # Default call,
    # Save the result in the SMAP list
    SMAPS.append(Smap)
    del Smap
