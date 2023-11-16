"""
Remap video from FILD cameras

Lesson 4 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise

jose Rueda: jrrueda@us.es

Note; Written for version 0.3.0. Checked for version 1.0.0
"""
import ScintSuite as ss
import ScintSuite._Mapping as ssmap
from ScintSuite._Mapping._Calibration import CalParams
import matplotlib.pyplot as plt
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 78452
diag_ID = 1  # 6 for rFILD
t1 = 0     # Initial time to be loaded, [s]
t2 = 0.002     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = False   # Flag to apply noise subtraction
tn1 = 0.20     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.23     # Final time to average the frames for noise subtraction [s]
flag_copy = False  # If true, a copy of the frames will be done while
#                  # substracting the noise

# - Filter options:
apply_filter = False  # Flag to apply filter to the frames
kind_of_filter = 'median'
options_filter = {
    'size': 1        # Size of the window to apply the filter
}
# If you want a gaussian one
# kind_of_filter = 'gaussian'
# options_filter = {
#     'sigma': 1        # sigma of the gaussian for the convolution (in pixels)
# }

smap = ssmap.StrikeMap(0, '/home/poley/NoTivoli/ScintSuite/SINPA/runs/scint_2023_test1_ur/results/scint_2023_test1_ur.map')
#smap = ssmap.StrikeMap(0, '/home/jansen/SINPA/runs/77971@0.4_ur/results/77971@0.4_ur.map')
cal = CalParams()
cal.xscale = 18317
cal.yscale = 18317
cal.xshift = 100
cal.yshift = 775.6
cal.deg = 0
cal.camera = 'CCD'
smap.calculate_pixel_coordinates(cal)
smap.calculate_energy(1.1462052172)
smap.setRemapVariables(('pitch', 'e0'), verbose=False)
# - Remapping options:
save_remap = False  # If true, the remap will be saved in a netCDF file
par = {
    'ymin': 1.0,      # Minimum gyroradius [in cm]
    'ymax': 45.0,     # Maximum gyroradius [in cm]
    'dy': 4.0,        # Interval of the gyroradius [in cm]
    'xmin': 20,     # Minimum pitch angle [in degrees]
    'xmax': 60,     # Maximum pitch angle [in degrees]
    'dx': 0.5,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 1,
    'smap_folder': '/home/jansen/SINPA/runs/77971@0.4_ur/results',
    'map': smap
    }  # Precision for the strike map (1 is more than enough)
# Note, if the smap_folder variable is not present, the program will look for
# the strike maps in the path given by ss.paths.StrikeMaps
# - Plotting options:
plot_profiles_in_time = False   # Plot the time evolution of pitch and r
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
filename='/home/poley/NoTivoli/ScintSuite/11111.mat'
#filename='/videodata/pcfild002/data/fild002/75620.mat'
vid = ss.vid.FILDVideo(file=filename)

# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise and filter frames
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2, flag_copy=flag_copy)

if apply_filter:
    vid.filter_frames(kind_of_filter, options_filter)
# -----------------------------------------------------------------------------
# --- Section 4: Remap
# -----------------------------------------------------------------------------
# - Remap frames:
vid.remap_loaded_frames(par)

# - Plot:
if plot_profiles_in_time:
    b = vid.integrate_remap(xmax=par['xmax'],ymax=par['ymax']) # rL max is 18
    # Integral in XI
    fig, ax = plt.subplots()
    b['integral_over_x'].plot()
    # Integral in rL
    fig2, ax2 = plt.subplots()
    b['integral_over_y'].plot()
# - Export remapped data
if save_remap:
    vid.export_remap()
