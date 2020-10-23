"""
This is a sample scrip which load a reference frame, load a video, calculate
and aling the strike maps and remap the scintillator signal.

Created as an example to use the routines without the graphical user interface

DISCLAIMER: This was created on the 21/10/2020. Since them several
improvement may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jose.rueda@ipp.mpg.de) by email

"""
import numpy as np
import LibPlotting as ssplt
import LibCinFiles as sscin
import LibMap as ssmap
import LibTimeTraces as sstt
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
# Section 0: Settings
cin_file_name = '/p/IPP/AUG/rawfiles/FIT/38/38020_v710.cin'
reference_frame_name = './FILD_calibration/FILD_reference_800x600_29062020.png'
scintillator_name = './aug_fild1_scint.pl'
t0 = 4.08       # Reference time (to select the ROI) in s

# ------------------------------------------------------------------------------
# Section 1: Read the cin file and create the roi
cin = sscin.Cin(cin_file_name)
dummy = np.array([np.argmin(abs(cin.timebase-t0))])
ref_frame = cin.read_frame(dummy)

# Create plot
fig_ref, ax_ref = plt.subplots()
ax_ref.imshow(ref_frame, vmax=100)

# Define roi
fig_ref, roi = sstt.create_roi(fig_ref)

# Create the mask
mask = roi.get_mask(ref_frame)
# ------------------------------------------------------------------------------
# Section 2: Calculate and display the time traces
time_trace = sstt.time_trace_cine(cin, mask, t1=0, t2=7.0)
# Plot the time trace
fig_tt, [ax_tt1,ax_tt2,ax_tt3] = plt.subplots(1, 3)
ax_tt1 = ssplt.p1D(ax_tt1, time_trace.time_base, time_trace.sum_of_roi,
            {'linewidth': 2, 'color': 'r'})
ax_tt1 = ssplt.axis_beauty(ax_tt1,{'xlabel': 't [s]', 'ylabel': 'Counts',
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

# ------------------------------------------------------------------------------
# Section 3: Orientate the scintillator
scint = ssmap.Scintillator(scintillator_name)

# plot the scintillator
fig_scint_FILDSIM, ax_scin_FILDSIM = plt.subplots()
scint.plot_real(ax_scin_FILDSIM)

# Plot the calibration frame
fig_call_frame, ax_call_frame = plt.subplots()
cal_frame = plt.imread(reference_frame_name)
cal_frame_tr = cal_frame[:,:,1].transpose()
cal_frame_inv = cal_frame_tr[::-1, ::-1]
ax_call_frame.imshow(cal_frame_inv)

