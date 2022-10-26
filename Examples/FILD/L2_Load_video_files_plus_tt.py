"""
Load video from FILD cameras

Lesson 2 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise and timetraces will be calculated

jose Rueda: jrrueda@us.es

Note; Written for version 0.3.0. Revised for version 1.0.0
"""
import Lib as ss
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 44732
diag_ID = 1  # 6 for rFILD
t1 = 0.20     # Initial time to be loaded, [s]
t2 = 3.5     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.20     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.25     # Final time to average the frames for noise subtraction [s]

# - TimeTrace options:
t0 = 0.30         # time points to define the ROI
save_TT = False   # Export the TT and the ROI used

# - Plotting options:
FS = 16        # FontSize for plotting
plt_TT = True  # Plot the TT
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
vid = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID, verbose=True)
# - read the frames:
print('Reading camera frames: ')
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)

# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)

# -----------------------------------------------------------------------------
# --- Section 3: Calculate the TT
# -----------------------------------------------------------------------------
# -- Old way, it should still work, but is complicating things for nothing
# # - Plot the frame
# ax_ref = vid.plot_frame(t=t0)
# fig_ref = plt.gcf()
# # - Define roi
# # Note: if you want the figure to re-appear after the selection of the roi,
# # call create roi with the option re_display=True
# roi = ss.tt.roipoly(fig_ref, ax_ref)
# # Create the mask
# mask = roi.getMask(vid.exp_dat['frames'][:, :, 0].squeeze())
# # Calculate the TimeTrace
# time_trace = ss.tt.TimeTrace(vid, mask)
# -- New direct way
time_trace, mask = vid.getTimeTrace(t=t0)
# -- Save the timetraces and roi
if save_TT:
    print('Choose the name for the TT file (select .txt!!!): ')
    time_trace.export_to_ascii()
    print('Choose the name for the mask file: ')
    ss.io.save_mask(mask)

# -----------------------------------------------------------------------------
# --- Section 4: Plotting
# -----------------------------------------------------------------------------
if plt_TT:
    time_trace.plot_single()

# By default the time_trace.plotsingle() plot the sum of counts on the roi, but
# mean and std are also calculated and can be plotted with it, just explore a
# bit the function

# -----------------------------------------------------------------------------
# --- Extra
# -----------------------------------------------------------------------------
# Fourier analysis of the trace can be easily done:
# time_trace.calculate_fft()
# time_trace.plot_fft()
#
# time_trace.calculate_spectrogram()
# time_trace.plot_spectrogram()
plt.show()
