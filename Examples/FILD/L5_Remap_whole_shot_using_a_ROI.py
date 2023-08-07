"""
Remap video from FILD cameras using a ROI

Lesson 5 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise

jose Rueda: jrrueda@us.es

Note; Written for version 0.3.0. Revised for version 1.0.0
"""
import ScintSuite.as ss
import matplotlib.pyplot as plt
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 44732
diag_ID = 1  # 6 for rFILD
t1 = 0.1     # Initial time to be loaded, [s]
t2 = 0.9     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.24     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.28     # Final time to average the frames for noise subtraction [s]

# - Remapping options:
save_remap = False  # If true, the remap will be saved in a netCDF file
par = {
    'ymin': 1.2,      # Minimum gyroradius [in cm]
    'ymax': 16.5,     # Maximum gyroradius [in cm]
    'dy': 0.1,        # Interval of the gyroradius [in cm]
    'xmin': 20.0,     # Minimum pitch angle [in degrees]
    'xmax': 90.0,     # Maximum pitch angle [in degrees]
    'dx': 1.0,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 1}  # Precision for the strike map (1 is more than enough)

# Note, if the smap_folder variable is not present, the program will look for
# the strike maps in the path given by ss.paths.StrikeMaps
use_roi = True     # Flag to decide if we must use a ROI
t0 = 0.24        # time points to define the ROI for the remap
save_ROI = False   # Export the TT and the ROI used
# - Plotting options:
plot_profiles_in_time = True   # Plot the time evolution of pitch and r
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# Open the video file
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
# --- Section 3: Selection of the ROI for the remapping
# -----------------------------------------------------------------------------
if use_roi:
    # - Plot the frame
    ax_ref = vid.plot_frame(t=t0)
    fig_ref = plt.gcf()
    # - Define roi
    # Note: if you want the figure to re-appear after the selection of the roi,
    # call create roi with the option re_display=True
    roi = ss.tt.roipoly(fig_ref, ax_ref)
    # Create the mask
    mask = roi.getMask(vid.exp_dat['frames'][:, :, 0].squeeze())
else:
    mask = None
# -----------------------------------------------------------------------------
# --- Section 4: Remap
# -----------------------------------------------------------------------------
# Store the mask in the parameters dictionary for the remap
par['mask'] = mask
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

# -----------------------------------------------------------------------------
# --- Section 5: Export data
# -----------------------------------------------------------------------------
if save_ROI:
    print('Choose the name for the mask for the frame (select .nc!!!): ')
    ss.io.save_mask(mask, shot=shot)
if save_remap:
    vid.export_remap()
