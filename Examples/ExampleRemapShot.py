"""
This is a sample scrip which load a reference frame, load a video, calculate
and aling the strike maps and remap the scintillator signal.

Created as an example to use the routines without the graphical user interface

DISCLAIMER: This was created on the 21/10/2020. Since them several
improvement may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jose.rueda@ipp.mpg.de) by email

"""

# General python packages
import numpy as np
import matplotlib.pyplot as plt
import LibPlotting as ssplt
import LibVideoFiles as ssvid
import LibMap as ssmap

# ------------------------------------------------------------------------------
# Section 0: Settings
cin_file_name = '/p/IPP/AUG/rawfiles/FIT/36/36358_v710.cin'
reference_frame_name = './FILD_calibration/FILD_reference_800x600_03062019.png'
calibration_database = './Calibrations/FILD/calibration_database.txt'
scintillator_name = '.temp_data/aug_fild1_scint.pl'
strike_map = '/afs/ipp-garching.mpg.de/home/j/jgq/PUBLIC/FILDSIM_strike_maps/FILD1_0.6MA_2.5T_strike_map.dat' 
shot = 36358
camera = 'PHANTOM'
cal_type = 'PIX'
diag_ID = 1     # FILD Number
# ------------------------------------------------------------------------------

# %% Section 1: Load calibration
database = ssmap.CalibrationDatabase(calibration_database)
cal = database.get_calibration(shot, camera, cal_type, diag_ID)
# ------------------------------------------------------------------------------

# %% Section 2: Load and remap strike map
smap = ssmap.StrikeMap(0, strike_map)
smap.calculate_pixel_coordinates(cal)
smap.interp_grid(np.array([800, 600]), plot=False, method=1)
# ------------------------------------------------------------------------------

# %% Section 3: test of the remapping algorithm
# Load a frame
cin = ssvid.Video(cin_file_name)
t0 = 2.50
dummy = np.array([np.argmin(abs(cin.timebase-t0))])
ref_frame = cin.read_frame(dummy)
# Perform the remaping with the default options
remaped, pitch, energy = ssmap.remap(smap, ref_frame)
# Plot the remapped frame
plt.contourf(pitch, energy, remaped.T)