"""
Load a video and perform the remaping for each frame

Created as an example to use the routines without the graphical user interface

DISCLAIMER: This was created on the 09/12/2020. Since them several
improvements may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jose.rueda@ipp.mpg.de) by email

You should run paths.py before!!!

"""

# General python packages
import os
import numpy as np
import matplotlib.pyplot as plt
import Lib as ss

# ------------------------------------------------------------------------------
# Section 0: Settings
# Paths:
calibration_database = './Data/Calibrations/FILD/calibration_database.txt'
# calibration data
camera = 'CCD'
cal_type = 'PIX'
diag_ID = 4     # FILD Number
# Shot and time interval
t1 = 0.5     # Initial time to remap [in s]
t2 = 7.0       # Final time to remap [in s]
shot = 38595
# ROI options
troi = 4.05
save_ROI = True
# Video options
limitation = False  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources
# Noise substraction parameters:
sub_noise = True     # Just a flag to see if we substract noise or not
tn1 = 6.4     # Initial time to average the noise
tn2 = 7.0    # Initial time to average the noise
# remapping parameters
par = {
    'rmin': 2.0,      # Minimum gyroradius [in cm]
    'rmax': 10.0,     # Maximum gyroradius [in cm]
    'dr': 0.1,        # Interval of the gyroradius [in cm]
    'pmin': 20.0,     # Minimum pitch angle [in degrees]
    'pmax': 90.0,     # Maximum pitch angle [in degrees]
    'dp': 1.0,    # Pitch angle interval
    # Parameters for the pitch-gryroradius profiles
    'rprofmin': 2.0,     # Minimum gyroradius for the pitch profile calculation
    'rprofmax': 10.0,    # Maximum gyroradius for the pitch profile calculation
    'pprofmin': 20.0,    # Minimum pitch for the gyroradius profile calculation
    'pprofmax': 90.0,    # Maximum pitch for the gyroradius profile calculation
    # Position of the FILD
    'rfild': 2.035,   # 2.196 for shot 32326, 2.186 for shot 32312
    'zfild': -0.462,
    'alpha': 0.0,
    'beta': -12.0,
    # method for the interpolation
    'method': 2}
# Plotting options
p1 = False  # Plot a window with some sliders to see the evolution of the shot
p2 = True  # Plot the evolution of the profiles vs time
p3 = True  # Plot the timetraces (see section 3 for parameters)
# pEner = True  # Use energy instead of gyroradius to plot in p1
FS = 16     # Font size
# -----------------------------------------------------------------------------
# %% Section 1: Load calibration
# Innitialise the calibration database object
database = ss.mapping.CalibrationDatabase(calibration_database)
# Get the calibration for our shot
cal = database.get_calibration(shot, camera, cal_type, diag_ID)
# -----------------------------------------------------------------------------

# %% Section 2: Load video file and the necesary frames
# Prepare the name of the cin file to be loaded
dummy = str(shot)
file = ss.paths.PngFiles + 'FILD' + str(diag_ID) + '/'\
    + dummy[0:2] + '/' + dummy
# initialise the video object:
cin = ss.vid.Video(file, diag_ID=diag_ID)
# Load the frames we need:
it1 = np.argmin(abs(cin.timebase-t1))
it2 = np.argmin(abs(cin.timebase-t2))
cin.read_frame(np.arange(start=it1, stop=it2+1, step=1), limitation=limitation,
               limit=limit)
# Subtract the noise
if sub_noise:
    cin.subtract_noise(t1=tn1, t2=tn2)
# Select the ROI
frame_index = np.array([np.argmin(abs(cin.timebase-troi))])
fig_ref, ax_ref = cin.plot_frame(t=troi)
fig_ref, roi = ss.tt.create_roi(fig_ref)
frameROI = cin.read_frame(frame_index, internal=False)
mask = roi.get_mask(frameROI.squeeze())
# Remap all the loaded frames
cin.remap_loaded_frames(cal, shot, par, mask=mask)
# Save the used mask, if needed:
if save_ROI:
    ss.io.save_mask(mask, nframe=frame_index, shot=shot, frame=frameROI)

# -----------------------------------------------------------------------------
# %% Section 3: Plotting
# Basic built plots:
if p1:
    cin.plot_remaped_slider()
if p2:
    cin.plot_profiles_in_time()
    plt.show()
# Time Traces
if p3:
    gyr = cin.remap_dat['yaxis']
    signal_in_gyr = cin.remap_dat['sprofy']
    g_lim1 = 2.0
    g_lim2 = 4.0
    t1 = 6.4        # Base line correction of the timetraces
    t2 = 7.0
    flags12 = (gyr > g_lim1) * (gyr < g_lim2)
    trace12 = np.sum(signal_in_gyr[flags12, :], axis=0)

    tit = '#' + str(shot) + ' ' + 'FILD' + str(diag_ID)
    time = cin.timebase[it1:it2+1]
    # raw traces
    fig_tt, ax_tt = plt.subplots()
    label1 = str(g_lim1) + ' - ' + str(g_lim2) + ' cm'
    ax_tt.plot(time, trace12, linewidth=1.5, color='r', label=label1)
    plt.legend(fontsize=0.9*FS)
    options = {'fontsize': FS, 'ylabel': 'Counts', 'xlabel': 'Time [s]',
               'grid': 'both'}
    plt.title(tit)
    ss.plt.axis_beauty(ax_tt, options)

    # Corrected traces:
    flags = (time > t1) * (time < t2)
    trace12_corrected = trace12 - np.mean(trace12[flags])
    fig_tt_cor, ax_tt_cor = plt.subplots()
    ax_tt_cor.plot(time, trace12_corrected, linewidth=1.5, color='r',
                   label=label1)
    plt.legend(fontsize=0.9*FS)
    options = {'fontsize': FS, 'ylabel': 'Counts', 'xlabel': 'Time [s]',
               'grid': 'both'}
    ss.plt.axis_beauty(ax_tt_cor, options)
    plt.title(tit + ': base line corrected')
    plt.show()
