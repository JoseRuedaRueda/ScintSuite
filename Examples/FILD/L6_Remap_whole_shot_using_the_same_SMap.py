"""
Remap video from FILD cameras using just one strike map

Lesson 6 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise and timetraces remap of the whole video, but
this time using a given strike map. As just one Smap will be used, the MC
remapping will be employed

Please write the proper path to your file in the Smap_file

jose Rueda: jrrueda@us.es
Lina Velarde: lvelarde@us.es

Note: Written for version 0.5.3. Revised for version 1.0.0
"""
import ScintSuite.as ss
import numpy as np
import matplotlib.pyplot as plt
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 44732
diag_ID = 1  # 6 for rFILD
t1 = 0.1     # Initial time to be loaded, [s]
t2 = 0.6     # Final time to be loaded [s]
t0 = 0.28    # Time instant to plot the remapped frame
limitation = False  # If true, the suite will not allow to load more than
limit = 2048        # 'limit' Mb of data. To avoid overloading the resources
flag_MC = False   # If MC remap is desired

# - Noise subtraction settings:
subtract_noise = False   # Flag to apply noise subtraction
tn1 = 0.1     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.15     # Final time to average the frames for noise subtraction [s]

# - Remapping options:
# Smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/39612_1p99s_strike_map.dat'  #!#
Smap_file = '/home/jqw5960/SINPA/runs/ajustefino/results/ajustefino.map'

save_remap = False
par = {
    'variables_to_remap': ('pitch','energy'),
    'ymin': 30,     # Minimum energy [in keV]
    'ymax': 70,     # Maximum energy [in keV]
    'dy': 0.5,
    'xmin': 20.0,     # Minimum pitch angle [in degrees]
    'xmax': 90.0,     # Maximum pitch angle [in degrees]
    'dx': 2.0,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 1}  # Precision for the strike map (1 is more than enough)

# - Plotting options:
plot_profiles_in_time = True   # Plot the time evolution of pitch and r
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
vid = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID)
# - read the frames:
tdummy = time()
print('Reading camera frames: ', shot, '...')
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)
print('Elapsed time [s]: ', time() - tdummy)
# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)
# -----------------------------------------------------------------------------
# --- Section 3: Load and prepare the strike map
# -----------------------------------------------------------------------------
# Load the strike map
smap = ss.mapping.StrikeMap('FILD', Smap_file)
# As we are using a specific strike map, we need to manually specify that we
# want the remap in energy and for that we first need to calculate it
vid._getB()
meanB = np.mean(vid.BField.B.data)
smap.calculate_energy(meanB)
# When the strike map is specified, the energy variable should be called as e0
smap.setRemapVariables(('pitch','e0'))
# Calculate pixel coordinates of the map
smap.calculate_pixel_coordinates(vid.CameraCalibration)

if flag_MC:
    MC_number = 150  # number of MC markers per pixel
    # Calculate the relation pixel - gyr and pitch
    grid = {'ymin': par['ymin'], 'ymax': par['ymax'], 'dy': par['dy'],
            'xmin': par['xmin'], 'xmax': par['xmax'], 'dx': par['dx']}
    smap.interp_grid(vid.exp_dat['frames'].shape[0:2], method=par['method'],
                    MC_number=MC_number, grid_params=grid, limitation=20)
# Include this map in the remapping parameters:
par['map'] = smap

# -----------------------------------------------------------------------------
# --- Section 6: Proceed with the remap
# -----------------------------------------------------------------------------
# - Remap frames:
vid.remap_loaded_frames(par)
# - Plot:
vid.plot_frame_remap(t=t0)
if plot_profiles_in_time:
    b = vid.integrate_remap(xmax=par['xmax'],ymax=par['ymax']) # rL max is 18
    # Integral in XI
    fig, ax = plt.subplots()
    b['integral_over_x'].plot()
    # Integral in rL
    fig2, ax2 = plt.subplots()
    b['integral_over_y'].plot()

# -----------------------------------------------------------------------------
# --- Section 7: Export data
# -----------------------------------------------------------------------------
# - Export remapped data
if save_remap:
    vid.export_remap()
