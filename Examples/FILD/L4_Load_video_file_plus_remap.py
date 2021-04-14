"""
Remap video from FILD cameras

Lesson 4 from the FILD experimental analysis. Video files will be loaded,
possibility to substract noise and timetraces remap of the whole video

jose Rueda: jrrueda@us.es

Note; Written for version 0.3.0 Before running this script, please do:
plt.show(), if not, bug due to spyder 4.0 may arise
"""
import Lib as ss
import matplotlib.pyplot as plt
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 33127
diag_ID = 1  # 6 for rFILD (DLIF)
t1 = 0.5     # Initial time to be loaded, [s]
t2 = 4.8     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise substraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.5     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.6     # Final time to average the frames for noise subtraction [s]

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
# - TimeTrace options:
calculate_TT = True  # Wheter to calculate or not the TT
t0 = 0.4        # time points to define the ROI
save_TT = True   # Export the TT and the ROI used
plt_TT = True  # Plot the TT

# - Remapping options:
calibration_database = './Data/Calibrations/FILD/calibration_database.txt'
camera = ss.dat.FILD[diag_ID-1]['camera']
save_remap = True
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
    # Position of the FILD
    'rfild': 2.211,   # 2.196 for shot 32326, 2.186 for shot 32312
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
# --- Section 2: Substract the noise and filter frames
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)

if apply_filter:
    vid.filter_frames(kind_of_filter, options_filter)
# -----------------------------------------------------------------------------
# --- Section 3: Calculate the TT
# -----------------------------------------------------------------------------
if calculate_TT:
    # - Plot the frame
    ax_ref = vid.plot_frame(t=t0)
    fig_ref = plt.gcf()
    # - Define roi
    # Note: if you want the figure to re-appear after the selection of the roi,
    # call create roi with the option re_display=Ture
    fig_ref, roi = ss.tt.create_roi(fig_ref, re_display=True)
    # Create the mask
    mask = roi.get_mask(vid.exp_dat['frames'][:, :, 0].squeeze())
    # - Calculate the TimeTrace
    time_trace = ss.tt.TimeTrace(vid, mask)
    # - Save the timetraces and roi
    if save_TT:
        print('Choose the name for the TT file (select .txt!!!): ')
        time_trace.export_to_ascii()
        print('Choose the name for the mask file: ')
        ss.io.save_mask(mask)
    # - Plot if needed
    if plt_TT:
        time_trace.plot_single()

# -----------------------------------------------------------------------------
# --- Section 4: Remmap
# -----------------------------------------------------------------------------
# - Initialise the calibration database object
database = ss.mapping.CalibrationDatabase(calibration_database)
# - Get the calibration for our shot
cal = database.get_calibration(shot, camera, 'PIX', diag_ID)
# - Remap frames:
vid.remap_loaded_frames(cal, shot, par)
# - Plot:
if plot_profiles_in_time:
    vid.plot_profiles_in_time()
# - Export remapped data
if save_remap:
    vid.export_remap()
