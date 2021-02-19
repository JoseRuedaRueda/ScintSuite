"""
Calculate the time trace.

Created as an example to use the library without the graphical user interface.
In this case we will calculate the timetrace of a whole shot

DISCLAIMER: This was created on the 10/12/2020. Since them several
improvement may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jrrueda@us.es) by email and he will update this 'tutorial'

@section tt0 0: Settings
@section tt1 1: Reading video and selecting roi
@section tt2 2: Calculate and plot the time trace

You must execute first the function paths.py!
"""

# --- Importing packages
# 'General python packages'
import numpy as np
# Suite packages
import Lib as ss

# -----------------------------------------------------------------------------
# Section 0: Settings
# -----------------------------------------------------------------------------
shot = 35953     # shot number
t0 = 1.25        # Reference time (to select the ROI) in s
fild_number = 4  # FILD number
# -----------------------------------------------------------------------------
# Section 1: Read the video file and create the roi
# -----------------------------------------------------------------------------
# --- Load the video
dummy = str(shot)
if fild_number == 1:
    file = ss.paths.CinFiles + dummy[0:2] + '/' + dummy + '_v710.cin'
else:
    file = ss.paths.PngFiles + 'FILD' + str(fild_number) + '/' +\
        dummy[0:2] + '/' + dummy


video = ss.vid.Video(file)
# --- Plot a frame to select the roi on it
# - Load the frame
frame_index = np.array([np.argmin(abs(video.timebase-t0))])
video.read_frame(frame_index)
# - Plot the frame
fig_ref, ax_ref = video.plot_frame(frame_index)
# - Define roi
# Note: if you want the figure to re-appear after the selection of the roi,
# call create roi with the option re_display=False
fig_ref, roi = ss.tt.create_roi(fig_ref)

# Create the mask
mask = roi.get_mask(video.exp_dat['frames'].squeeze())
# -----------------------------------------------------------------------------
# Section 2: Calculate and display the time traces
video.read_frame()
time_trace = ss.tt.TimeTrace(video, mask)
# Plot the time trace
time_trace.plot_all()
# -----------------------------------------------------------------------------
# Section 3: Calculate and display the fft and spectrogram
time_trace.calculate_fft()
time_trace.plot_fft()

time_trace.calculate_spectrogram()
time_trace.plot_spectrogram()
