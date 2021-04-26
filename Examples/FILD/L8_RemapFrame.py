"""
Load a frame, load the calibration and remap the scintillator signal

Created as an example to use the routines without the graphical user interface

DISCLAIMER: This was created on the 21/11/2020. Since them several
improvements may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jrrueda@us.es) by email

DISCLAIMER 2: The script is quite slow because of the AUG routine which reads
the magnetic field... is not fault of the suite!
"""

# General python packages
import numpy as np
import matplotlib.pyplot as plt
import LibPlotting as ssplt
import LibVideoFiles as ssvid
import LibMap as ssmap


# -----------------------------------------------------------------------------
# Section 0: Settings
# -----------------------------------------------------------------------------
cin_file_name = '/p/IPP/AUG/rawfiles/FIT/32/32312_v710.cin'
calibration_database = './Data/Calibrations/FILD/calibration_database.txt'

strike_map = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILD_Strike_maps/' + \
    'AUG_map_-000.80000_008.10000_strike_map.dat'
shot = 32312
camera = 'PHANTOM'
cal_type = 'PIX'
diag_ID = 1     # FILD Number
# FILD position, in m, to calculate the magnetic field
rfild = 2.186
zfild = 0.32
t0 = 0.27
# -----------------------------------------------------------------------------
# --- Section 1: Load calibration
# -----------------------------------------------------------------------------
database = ssmap.CalibrationDatabase(calibration_database)
cal = database.get_calibration(shot, camera, cal_type, diag_ID)
# -----------------------------------------------------------------------------
# --- Section 2: Load the frame
# -----------------------------------------------------------------------------
# Load a frame
cin = ssvid.Video(cin_file_name)
dummy = np.array([np.argmin(abs(cin.timebase-t0))])
ref_frame = cin.read_frame(dummy, internal=False)
frame_shape = ref_frame.shape
fig_ref, ax_ref = plt.subplots()
ax_ref.imshow(ref_frame, origin='lower')
# %% Section 3: Load and remap strike map
smap = ssmap.StrikeMap(0, strike_map)
smap.calculate_pixel_coordinates(cal)
smap.interp_grid(ref_frame.shape, plot=False, method=2)
smap.plot_pix(ax_ref)
# -----------------------------------------------------------------------------
# --- Section 3: Remapping
# -----------------------------------------------------------------------------
# Perform the remapping with the default options
remaped, pitch, gyr = ssmap.remap(smap, ref_frame, delta_y=0.1)
# Plot the remapped frame
fig_remap, ax_remap = plt.subplots()
cmap = ssplt.Gamma_II()
a1 = plt.contourf(pitch, gyr, remaped.T, levels=20, cmap=cmap)

fig_remap.colorbar(a1, ax=ax_remap)

# -----------------------------------------------------------------------------
# --- Section 4: Calculation of the profiles
# -----------------------------------------------------------------------------
# Obtain a gyroradius profile
profile = ssmap.gyr_profile(remaped, pitch, 20.0, 90.0, verbose=True)

# Obtain a pitch profile
profile_pitch = ssmap.pitch_profile(remaped, gyr, 2.0, 8.0, verbose=True)

# Plot (adapted from
# https://stackoverflow.com/questions/10514315/
# how-to-add-a-second-x-axis-in-matplotlib)
fig, ax = plt.subplots(2)
ax[0].plot(gyr, profile)
ax[1].plot(pitch, profile_pitch)
plt.show()
