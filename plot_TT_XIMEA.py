"""
Load video from FILD cameras
"""
import ScintSuite as ss
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 77212
diag_ID = 1  # There is only one fild at TCV
apd_channels = [1]
# - Noise subtraction settings:
subtract_noise = False   # Flag to apply noise subtraction
tn1 = 0.     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.05     # Final time to average the frames for noise subtraction [s]
# - TimeTrace options:
t0 = 0.8       # time points to define the ROI
save_TT = False   # Export the TT and the ROI used
# - Plotting options:
FS = 16        # FontSize for plotting
plt_TT = True  # Plot the TT
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
vid = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID, verbose=True)
# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)
# -----------------------------------------------------------------------------
# --- Section 3: Calculate the TT
# -----------------------------------------------------------------------------
ax_ref = vid.plot_frame(t=t0)
fig_ref = plt.gcf()
# Define roi
# Note: if you want the figure to re-appear after the selection of the
# roi, call create roi with the option re_display=True
roi = ss.tt.roipoly(fig_ref, ax_ref, drawROI=True)
# Create the mask
mask = roi.getMask(vid.exp_dat['frames'][:, :, 0].squeeze())
# # Calculate the TimeTrace
time_trace = ss.tt.TimeTrace(vid, mask)
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