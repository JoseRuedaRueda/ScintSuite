"""
Remap video from FILD cameras using a ROI

Lesson 5 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise

jose Rueda: jrrueda@us.es

Note; Written for version 0.3.0. Revised for version 0.7.2
"""
import Lib as ss
import matplotlib.pyplot as plt
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 32312
diag_ID = 1  # 6 for rFILD
t1 = 0.15     # Initial time to be loaded, [s]
t2 = 0.65     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.15     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.17     # Final time to average the frames for noise subtraction [s]

# - Remapping options:
calibration_database = ss.paths.ScintSuite \
    + '/Data/Calibrations/FILD/AUG/calibration_database.txt'
camera = ss.dat.FILD[diag_ID-1]['camera']
save_remap = True
par = {
    'rmin': 1.2,      # Minimum gyroradius [in cm]
    'rmax': 10.5,     # Maximum gyroradius [in cm]
    'dr': 0.05,        # Interval of the gyroradius [in cm]
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
use_roi = True      # Flag to decide if we must use a ROI
t0 = 0.35         # time points to define the ROI for the remap
save_ROI = True   # Export the TT and the ROI used
# - Plotting options:
plot_profiles_in_time = True   # Plot the time evolution of pitch and r
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - Get the proper file name
filename = ss.vid.guess_filename(shot, ss.dat.FILD[diag_ID-1]['path'],
                                 ss.dat.FILD[diag_ID-1]['extension'])

# - open the video file:
vid = ss.vid.FILDVideo(filename, diag_ID=diag_ID)
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
# --- Section 3: Read FILD logbook
# -----------------------------------------------------------------------------
FILD = ss.dat.FILD_logbook()
position = FILD.getPosition(shot, diag_ID)
orientation = FILD.getOrientation(shot, diag_ID)
# Add FILD positions to the remap options:
par['rfild'] = position['R']
par['zfild'] = position['z']
par['alpha'] = orientation['alpha']
par['beta'] = orientation['beta']
# -----------------------------------------------------------------------------
# --- Section 4: Selection of the ROI for the remapping
# -----------------------------------------------------------------------------
if use_roi:
    # - Plot the frame
    ax_ref = vid.plot_frame(t=t0)
    fig_ref = plt.gcf()
    # - Define roi
    # Note: if you want the figure to re-appear after the selection of the roi,
    # call create roi with the option re_display=Ture
    fig_ref, roi = ss.tt.create_roi(fig_ref, re_display=True)
    # Create the mask
    mask = roi.get_mask(vid.exp_dat['frames'][:, :, 0].squeeze())
else:
    mask = None
# -----------------------------------------------------------------------------
# --- Section 5: Remap
# -----------------------------------------------------------------------------
# - Initialize the calibration database object
database = ss.mapping.CalibrationDatabase(calibration_database)
# - Get the calibration for our shot
cal = database.get_calibration(shot, camera, 'PIX', diag_ID)
# - Remap frames:
vid.remap_loaded_frames(cal, shot, par, mask=mask)
# - Plot:
if plot_profiles_in_time:
    vid.plot_profiles_in_time()

# -----------------------------------------------------------------------------
# --- Section 6: Export data
# -----------------------------------------------------------------------------
if save_ROI:
    print('Choose the name for the mask for the frame (select .nc!!!): ')
    ss.io.save_mask(mask, shot=shot)
if save_remap:
    vid.export_remap()
