"""
Calculate and compare the TimeTraces of the camera signal for different shots

Jose Rueda Rueda: jrrueda@us.es

It is not optimized for cin files, please do not proceed with this example to
compare cin files. Use please L13

Note: Done for version 0.5.3
"""

import Lib as ss
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------

shot = [39574, 39575, 39576, 39577]
diag_ID = 1             # FILD number
t0 = 2.62                # time (at the first shot) to define the roi

# -----------------------------------------------------------------------------
# --- Calculate and plot the TT
# -----------------------------------------------------------------------------

counter = 0
TT = []
for s in shot:
    # - Get the proper file name
    filename = ss.vid.guess_filename(s, ss.dat.FILD[diag_ID-1]['path'],
                                     ss.dat.FILD[diag_ID-1]['extension'])

    # - open the video file:
    vid = ss.vid.Video(filename, diag_ID=diag_ID)

    # - select the ROI:
    if counter == 0:
        frame_index = np.array([np.argmin(abs(vid.timebase-t0))])
        vid.read_frame(frame_index)
        # Plot the frame
        ax_ref = vid.plot_frame(t=t0)
        fig_ref = plt.gcf()
        # Define roi
        # Note: if you want the figure to re-appear after the selection of the
        # roi, call create roi with the option re_display=True
        fig_ref, roi = ss.tt.create_roi(fig_ref, re_display=True)
        # Create the mask
        mask = roi.get_mask(vid.exp_dat['frames'][:, :, 0].squeeze())
        fig, ax = plt.subplots()
    time_trace = ss.tt.TimeTrace(vid, mask, t1=0.0, t2=10.0)
    time_trace.plot_single(ax=ax, line_params={'label': '#' + str(s)})
    TT.append(time_trace)
    counter += 1
plt.legend()
fig.show()
print(counter, ' timetraces calculated.')
print('Thanks for using the ScintillatorSuite')
