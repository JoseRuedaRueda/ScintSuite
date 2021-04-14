"""
Calculate the time trace.

Created as an example to use the library without the graphical user interface.
In this case we will calculate the timetrace of a whole shot

It is optimised for cin files. In this case the video is not loaded, we just
read the roi we want and integrate, so almost no RAM memory is consumed and you
can calculate the TT for the whole shot really fast

@section tt0 0: Settings
@section tt1 1: Reading video and selecting roi
@section tt2 2: Calculate and plot the time trace

Note; Written for version 0.3.0 Before running this script, please do:
plt.show(), if not, bug due to spyder 4.0 may arise
"""

# --- Importing packages
# 'General python packages'
import numpy as np
import matplotlib.pyplot as plt
# Suite packages
import Lib as ss

# -----------------------------------------------------------------------------
# Section 0: Settings
# -----------------------------------------------------------------------------
shot = 38626     # shot number
t0 = 3.0        # Reference time (to select the ROI) in s
# -----------------------------------------------------------------------------
# Section 1: Read the video file and create the roi
# -----------------------------------------------------------------------------
# --- Load the video
dummy = str(shot)
file = ss.vid.guess_filename(shot, ss.dat.FILD[0]['path'],
                             ss.dat.FILD[0]['extension'])

video = ss.vid.Video(file)
# --- Plot a frame to select the roi on it
# - Load the frame
frame_index = np.array([np.argmin(abs(video.timebase-t0))])
video.read_frame(frame_index)
# - Plot the frame
ax_ref = video.plot_frame(t=t0)
fig_ref = plt.gcf()
# - Define roi
# Note: if you want the figure to re-appear after the selection of the roi,
# call create roi with the option re_display=False
fig_ref, roi = ss.tt.create_roi(fig_ref)

# Create the mask
mask = roi.get_mask(video.exp_dat['frames'].squeeze())
# -----------------------------------------------------------------------------
# Section 2: Calculate and display the time traces
# time_trace = sstt.time_trace_cine(cin, mask, t1=0, t2=7.0)
time_trace = ss.tt.TimeTrace(video, mask, t1=0.0, t2=10.0)
# Plot the time trace
time_trace.plot_all()
# -----------------------------------------------------------------------------
# Section 3: Calculate and display the fft and spectrogram
time_trace.calculate_fft()
time_trace.plot_fft()

time_trace.calculate_spectrogram()
time_trace.plot_spectrogram()
