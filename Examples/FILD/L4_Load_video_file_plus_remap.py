"""
Remap video from FILD cameras

Lesson 4 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise

jose Rueda: jrrueda@us.es

Note; Written for version 0.3.0. Checked for version 0.9.0
"""
import Lib as ss
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 39612
diag_ID = 1  # 6 for rFILD
t1 = 0.2     # Initial time to be loaded, [s]
t2 = 1.0     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.20     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.23     # Final time to average the frames for noise subtraction [s]
flag_copy = False  # If true, a copy of the frames will be done while
#                  # substracting the noise

# - Filter options:
apply_filter = True  # Flag to apply filter to the frames
kind_of_filter = 'median'
options_filter = {
    'size': 1        # Size of the window to apply the filter
}
# If you want a gaussian one
# kind_of_filter = 'gaussian'
# options_filter = {
#     'sigma': 1        # sigma of the gaussian for the convolution (in pixels)
# }

# - Remapping options:
save_remap = False  # If true, the remap will be saved in a netCDF file
par = {
    'rmin': 1.2,      # Minimum gyroradius [in cm]
    'rmax': 10.5,     # Maximum gyroradius [in cm]
    'dr': 0.1,        # Interval of the gyroradius [in cm]
    'pmin': 20.0,     # Minimum pitch angle [in degrees]
    'pmax': 90.0,     # Maximum pitch angle [in degrees]
    'dp': 1.0,    # Pitch angle interval
    # Parameters for the pitch-gryroradius profiles
    'rprofmin': 1.5,     # Minimum gyroradius for the pitch profile calculation
    'rprofmax': 4.0,    # Maximum gyroradius for the pitch profile calculation
    'pprofmin': 20.0,    # Minimum pitch for the gyroradius profile calculation
    'pprofmax': 90.0,    # Maximum pitch for the gyroradius profile calculation
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 1}  # Precision for the strike map (1 is more than enough)
# Note, if the smap_folder variable is not present, the program will look for
# the strike maps in the path given by ss.paths.StrikeMaps
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
    vid.plot_profiles_in_time()
# - Export remapped data
if save_remap:
    vid.export_remap()
