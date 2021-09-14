"""
Remap and compare two frames

Lesson 9 from the FILD experimental analysis. Video files will be loaded,
noise will be subtracted, a median filter will be applied, two experimental
frames will be plotted and their remaps compared

Notes: just for simplicity, although we just want two frames, a large interval
of the video video will be loaded (to have the dark frames to subtract the
noise)

jose Rueda: jrrueda@us.es

Note: Written for version 0.2.6.
"""
import Lib as ss
import matplotlib.pyplot as plt
import numpy as np
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 38866
diag_ID = 4  # 6 for rFILD (DLIF)
t1 = 5.75     # Initial time to be loaded, [s]
t2 = 7.5     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 5.75     # Initial time to average the frames for noise subtraction [s]
tn2 = 5.9     # Final time to average the frames for noise subtraction [s]

# - Filter options:
apply_filter = True  # Flag to apply filter to the frames
kind_of_filter = 'median'
options_filter = {
    'size': 2        # Size of the window to apply the filter
}
# If you want a Gaussian one
# kind_of_filter = 'gaussian'
# options_filter = {
#     'sigma': 1        # sigma of the Gaussian for the convolution (in pixels)
# }

# - Frames to compare
tf1 = 6.8  # time of the first frame to be used [s]
tf2 = 7.1  # time of the second frame to be used [s]

# - Remapping options:
calibration_database = './Data/Calibrations/FILD/calibration_database.txt'
camera = 'CCD'      # CCD for other FILDs
par = {
    'rmin': 1.2,      # Minimum gyroradius [in cm]
    'rmax': 10.5,     # Maximum gyroradius [in cm]
    'dr': 0.05,        # Interval of the gyroradius [in cm]
    'pmin': 20.0,     # Minimum pitch angle [in degrees]
    'pmax': 90.0,     # Maximum pitch angle [in degrees]
    'dp': 1.0,    # Pitch angle interval
    # Parameters for the pitch-gryroradius profiles
    'rprofmin': 1.5,     # Minimum gyroradius for the pitch profile calculation
    'rprofmax': 8.0,     # Maximum gyroradius for the pitch profile calculation
    'pprofmin': 20.0,    # Minimum pitch for the gyroradius profile calculation
    'pprofmax': 90.0,    # Maximum pitch for the gyroradius profile calculation
    # Position of the FILD
    'rfild': 2.035,   # 2.196 for shot 32326, 2.186 for shot 32312
    'zfild': ss.dat.FILD[diag_ID-1]['z'],
    'alpha': ss.dat.FILD[diag_ID-1]['alpha'],
    'beta': ss.dat.FILD[diag_ID-1]['beta'],
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
# - Get the proper file name
filename = ss.vid.guess_filename(shot, ss.dat.FILD[diag_ID-1]['path'],
                                 ss.dat.FILD[diag_ID-1]['extension'])

# - open the video file:
vid = ss.vid.Video(filename, diag_ID=diag_ID)
# - read the frames:
tdummy = time()
print('Reading camera frames: ', shot, '...')
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)
print('Elapsed time [s]: ', time() - tdummy)
# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise and filter frames
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)

if apply_filter:
    vid.filter_frames(kind_of_filter, options_filter)
# -----------------------------------------------------------------------------
# --- Section 3: Remap
# -----------------------------------------------------------------------------
# - Initialize the calibration database object
database = ss.mapping.CalibrationDatabase(calibration_database)
# - Get the calibration for our shot
cal = database.get_calibration(shot, camera, 'PIX', diag_ID)
# - Remap frames:
vid.remap_loaded_frames(cal, shot, par)
# - Plot:
if plot_profiles_in_time:
    vid.plot_profiles_in_time()
# -----------------------------------------------------------------------------
# --- Section 4: Plot the frames
# -----------------------------------------------------------------------------

fig1, ax1 = plt.subplots(1, 2)
vid.plot_frame(t=tf1, strike_map='auto', ax=ax1[0])

vid.plot_frame(t=tf2, strike_map='auto', ax=ax1[1])

# Plot the remapped frames
cmap = ss.plt.Gamma_II()
fig2, ax2 = plt.subplots(1, 2)
it1 = np.argmin(abs(vid.remap_dat['tframes']-tf1))
it2 = np.argmin(abs(vid.remap_dat['tframes']-tf2))
ax_param = {'fontsize': 14, 'xlabel': 'Pitch [º]', 'ylabel': '$r_l [cm]$'}
c1 = ax2[0].contourf(vid.remap_dat['xaxis'], vid.remap_dat['yaxis'],
                     vid.remap_dat['frames'][:, :, it1].T, cmap=cmap)
plt.colorbar(c1, ax=ax2[0], label='Signal [counts/mm/º]')
ax2[0] = ss.plt.axis_beauty(ax2[0], ax_param)
c2 = ax2[1].contourf(vid.remap_dat['xaxis'], vid.remap_dat['yaxis'],
                     vid.remap_dat['frames'][:, :, it2].T, cmap=cmap)
plt.colorbar(c2, ax=ax2[1], label='Signal [counts/mm/º]')
ax2[1] = ss.plt.axis_beauty(ax2[1], ax_param)
plt.show()
