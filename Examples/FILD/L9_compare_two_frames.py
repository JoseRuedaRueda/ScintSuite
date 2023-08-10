"""
Remap and compare two frames

Lesson 9 from the FILD experimental analysis. Video files will be loaded,
noise will be subtracted, a median filter will be applied, two experimental
frames will be plotted and their remaps compared

Notes: just for simplicity, although we just want two frames, a large interval
of the video video will be loaded (to have the dark frames to subtract the
noise)

jose Rueda: jrrueda@us.es

Note: Written for version 0.2.6. Revised for version 0.7.3
"""
import ScintSuite.as ss
import matplotlib.pyplot as plt
import numpy as np
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 44732
diag_ID = 1  # 6 for rFILD (DLIF)
t1 = 0.15     # Initial time to be loaded, [s]
t2 = 0.35     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.1     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.4     # Final time to average the frames for noise subtraction [s]

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
tf1 = 0.24  # time of the first frame to be used [s]
tf2 = 0.28  # time of the second frame to be used [s]

# - Remapping options:
par = {
    'ymin': 1.2,      # Minimum gyroradius [in cm]
    'ymax': 16.5,     # Maximum gyroradius [in cm]
    'dy': 0.5,        # Interval of the gyroradius [in cm]
    'xmin': 20.0,     # Minimum pitch angle [in degrees]
    'xmax': 90.0,     # Maximum pitch angle [in degrees]
    'dx': 2.0,    # Pitch angle interval
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
    vid.subtract_noise(t1=tn1, t2=tn2)

if apply_filter:
    vid.filter_frames(kind_of_filter, options_filter)
# -----------------------------------------------------------------------------
# --- Section 3: Remap
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
# --- Section 4: Plot the frames
# -----------------------------------------------------------------------------
fig1, ax1 = plt.subplots(1, 2)
vid.plot_frame(t=tf1, strike_map='auto', ax=ax1[0])

vid.plot_frame(t=tf2, strike_map='auto', ax=ax1[1])

# Plot the remapped frames
cmap = ss.plt.Gamma_II()
fig2, ax2 = plt.subplots(1, 2)
it1 = np.argmin(abs(vid.remap_dat['t'].data-tf1))
it2 = np.argmin(abs(vid.remap_dat['t'].data-tf2))
ax_param = {'fontsize': 14, 'xlabel': 'Pitch [º]', 'ylabel': '$r_l [cm]$'}
c1 = ax2[0].contourf(vid.remap_dat['x'].data, vid.remap_dat['y'].data,
                     vid.remap_dat['frames'][:, :, it1].T, cmap=cmap)
plt.colorbar(c1, ax=ax2[0], label='Signal [counts/mm/º]')
ax2[0] = ss.plt.axis_beauty(ax2[0], ax_param)
c2 = ax2[1].contourf(vid.remap_dat['x'].data, vid.remap_dat['y'].data,
                     vid.remap_dat['frames'][:, :, it2].T, cmap=cmap)
plt.colorbar(c2, ax=ax2[1], label='Signal [counts/mm/º]')
ax2[1] = ss.plt.axis_beauty(ax2[1], ax_param)
plt.show()
