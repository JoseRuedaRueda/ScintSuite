"""
Calculate and compare the TimeTraces of the camera and compare with heating

Jose Rueda Rueda: jrrueda@us.es

It is not optimized for cin files. You can used it, but it will be slow and
will load in memory and unnecesary amound of data

Under development, heating part still not included
Note: Done for version 0.5.3
"""

import Lib as ss
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------

shot = 39631
diag_ID = [1, 4, 5]     # FILD number
t0 = 4.0                # time (at the first shot) to define the roi
t1 = 0.1
t2 = 10.0
# -----------------------------------------------------------------------------
# --- Calculate and plot the TT
# -----------------------------------------------------------------------------
TT = []
for id in diag_ID:
    # - Get the proper file name
    filename = ss.vid.guess_filename(shot, ss.dat.FILD[id-1]['path'],
                                     ss.dat.FILD[id-1]['extension'])

    # - open the video file:
    vid = ss.vid.FILDVideo(filename, diag_ID=diag_ID)
    # - read the frames:
    print('Reading camera frames: ')
    vid.read_frame(t1=t1, t2=t2, limitation=False)
    # - select the ROI:
    # Plot the frame
    ax_ref = vid.plot_frame(t=t0)
    fig_ref = plt.gcf()
    # Define roi
    # Note: if you want the figure to re-appear after the selection of the
    # roi, call create roi with the option re_display=True
    fig_ref, roi = ss.tt.create_roi(fig_ref, re_display=True)
    # Create the mask
    mask = roi.get_mask(vid.exp_dat['frames'][:, :, 0].squeeze())
    time_trace = ss.tt.TimeTrace(vid, mask)
    plt.close('all')
    TT.append(time_trace)
# --- Plot the timetraces
fig, ax = plt.subplots()
for i in range(len(TT)):
    TT[i].plot_single(ax=ax, line_params={'label': 'FILD' + str(diag_ID[i])},
                      normalised=True, correct_baseline='ini')

fig.show()
plt.legend()
plt.title('#' + str(shot))

print('Thanks for using the ScintillatorSuite')
