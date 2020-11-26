"""
Calculate the time trace.

Created as an example to use the routines without the graphical user interface

DISCLAIMER: This was created on the 10/11/2020. Since them several
improvement may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jose.rueda@ipp.mpg.de) by email

You must execute first the function paths.py!

"""

# General python packages
import numpy as np
import matplotlib.pyplot as plt
import LibPlotting as ssplt
import LibVideoFiles as sscin
import LibTimeTraces as sstt

# ------------------------------------------------------------------------------
# Section 0: Settings
cin_file_name = '/p/IPP/AUG/rawfiles/FIT/34/34570_v710.cin'
reference_frame_name = './FILD_calibration/FILD_reference_800x600_03062019.png'
scintillator_name = './aug_fild1_scint.pl'
t0 = 2.16       # Reference time (to select the ROI) in s

# ------------------------------------------------------------------------------
# Section 1: Read the cin file and create the roi
cin = sscin.Video(cin_file_name)
dummy = np.array([np.argmin(abs(cin.timebase-t0))])
ref_frame = cin.read_frame(dummy)

# Create plot
fig_ref, ax_ref = plt.subplots()
ax_ref.imshow(ref_frame)

# Define roi
fig_ref, roi = sstt.create_roi(fig_ref)

# Create the mask
mask = roi.get_mask(ref_frame)
# -----------------------------------------------------------------------------
# Section 2: Calculate and display the time traces
# time_trace = sstt.time_trace_cine(cin, mask, t1=0, t2=7.0)
time_trace = sstt.TimeTrace(cin, mask, t1=0.0, t2=10.0)
# Plot the time trace
fig_tt, [ax_tt1, ax_tt2, ax_tt3] = plt.subplots(1, 3)
ax_tt1 = ssplt.p1D(ax_tt1, time_trace.time_base, time_trace.sum_of_roi,
                   {'linewidth': 2, 'color': 'r'})
ax_tt1 = ssplt.axis_beauty(ax_tt1, {'xlabel': 't [s]', 'ylabel': 'Counts',
                                    'grid': 'both'})
# plot the mean of the timetrace
ax_tt2 = ssplt.p1D(ax_tt2, time_trace.time_base, time_trace.mean_of_roi,
                   {'linewidth': 2, 'color': 'r'})
ax_tt2 = ssplt.axis_beauty(ax_tt2, {'xlabel': 't [s]', 'ylabel': 'Mean',
                                    'grid': 'both'})
# plot the std of the timetrace
ax_tt3 = ssplt.p1D(ax_tt3, time_trace.time_base, time_trace.std_of_roi,
                   {'linewidth': 2, 'color': 'r'})
ax_tt3 = ssplt.axis_beauty(ax_tt3, {'xlabel': 't [s]', 'ylabel': 'std',
                                    'grid': 'both'})
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------
# Section 3: Caculate and display the fft and spectrogram
time_trace.calculate_fft()
fig_fft, ax_fft = plt.subplots()
ax_fft.plot(time_trace.fft['faxis'], abs(time_trace.fft['data']))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.show()
fig_spec, ax_spec = plt.subplots()
time_trace.calculate_spectrogram(plot_flag=True)
