"""
Calculate and compare the TimeTraces of the camera signal for different shots

Jose Rueda Rueda: jrrueda@us.es

It is not optimized for cin files, please do not proceed with this example to
compare cin files, because the whole video will be loaded and it would be a
waste of resources. Use please L13

Note: Done for version 0.5.3, reised for version 0.9.0
"""

import Lib as ss
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------

shot = [39612, 39613]
diag_ID = 4             # FILD number
t0 = 2.0                # time (at the first shot) to define the roi

# -----------------------------------------------------------------------------
# --- Calculate and plot the TT
# -----------------------------------------------------------------------------

counter = 0
TT = []
for s in shot:
    # - open the video file:
    vid = ss.vid.FILDVideo(shot=s, diag_ID=diag_ID)
    # - read the frames:
    print('Reading camera frames: ')
    vid.read_frame(t1=0.0, t2=10.0)
    # - select the ROI:
    if counter == 0:
        # Plot the frame
        ax_ref = vid.plot_frame(t=t0)
        fig_ref = plt.gcf()
        # Define roi
        # Note: if you want the figure to re-appear after the selection of the
        # roi, call create roi with the option re_display=True
        roi = ss.tt.roipoly(fig_ref, ax_ref)
        # Create the mask
        mask = roi.getMask(vid.exp_dat['frames'][:, :, 0].squeeze())
        fig, ax = plt.subplots()
    time_trace = ss.tt.TimeTrace(vid, mask)
    time_trace.plot_single(ax=ax, line_params={'label': '#' + str(s)})
    TT.append(time_trace)
    counter += 1
plt.legend()
fig.show()
print(counter, ' timetraces calculated.')
print('Thanks for using the ScintillatorSuite')
