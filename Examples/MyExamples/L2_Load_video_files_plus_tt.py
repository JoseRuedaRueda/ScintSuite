"""
Load video from FILD cameras

Lesson 2 from the FILD experimental analysis. Video files will be loaded,
possibility to substract noise and timetraces will be calculated

jose Rueda: jrrueda@us.es

Note; Written for version 0.1.8. Before running this script, please do:
plt.show(), if not, bug due to spyder 4.0 may arise
"""
import Lib as ss
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 38663
diag_ID = 6  # 6 for rFILD (DLIF)
t1 = 0.5     # Initial time to be loaded, [s]
t2 = 3.5     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise substraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.5     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.65     # Final time to average the frames for noise subtraction [s]

# - TimeTrace options:
t0 = 3.0         # time points to define the ROI
save_TT = False   # Export the TT and the ROI used

# - Plotting options:
FS = 16        # FontSize for plotting
plt_TT = True  # Plot the TT
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - Get the proper file name
filename = ss.vid.guess_filename(shot, ss.dat.FILD[diag_ID-1]['path'],
                                 ss.dat.FILD[diag_ID-1]['extension'])

# - open the video file:
vid = ss.vid.Video(filename, diag_ID=diag_ID)
# - read the frames:
print('Reading camera frames: ')
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)

# -----------------------------------------------------------------------------
# --- Section 2: Substract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)

# -----------------------------------------------------------------------------
# --- Section 3: Calculate the TT
# -----------------------------------------------------------------------------
# - Plot the frame
ax_ref = vid.plot_frame(t=t0)
fig_ref = ax_ref.figure
# - Define roi
# Note: if you want the figure to re-appear after the selection of the roi,
# call create roi with the option re_display=Ture
fig_ref, roi = ss.tt.create_roi(fig_ref, re_display=True)
# Create the mask
mask = roi.get_mask(vid.exp_dat['frames'][:, :, 0].squeeze())
# Calculate the TimeTrace
time_trace = ss.tt.TimeTrace(vid, mask)
# Save the timetraces and roi
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

# By default the TT.plotsingle() plot the summ of counts on the roi, but
# mean and std are also calculated and ca be ploted with it

# -----------------------------------------------------------------------------
# --- Extra
# -----------------------------------------------------------------------------
# Fourier analisis of the trace can be easily done:
# time_trace.calculate_fft()
# time_trace.plot_fft()
#
# time_trace.calculate_spectrogram()
# time_trace.plot_spectrogram()
