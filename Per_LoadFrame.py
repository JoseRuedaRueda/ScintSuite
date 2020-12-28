"""
Load a video and perform the remaping for each frame

Created as an example to use the routines without the graphical user interface

DISCLAIMER: This was created on the 09/12/2020. Since them several
improvements may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jose.rueda@ipp.mpg.de) by email

You should run paths.py before!!!

"""

# General python packages
import os
import numpy as np
import matplotlib.pyplot as plt
import Lib as ss

# ------------------------------------------------------------------------------
# Section 0: Settings
# Paths:
calibration_database = './Calibrations/FILD/calibration_database.txt'
# calibration data
camera = 'PHANTOM'
cal_type = 'PIX'
diag_ID = 1     # FILD Number
# # Shot a time interval
# t1 = 0.25        # Initial time to remap [in s]
# t2 = 0.95        # Final time to remap [in s]
shot = 32312
frame = 4392
# Video options
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources
# remapping parameters
par = {
    'rmin': 1.2,      # Minimum gyroradius [in cm]
    'rmax': 10.5,     # Maximum gyroradius [in cm]
    'dr': 0.1,        # Interval of the gyroradius [in cm]
    'pmin': 20.0,     # Minimum pitch angle [in degrees]
    'pmax': 90.0,     # Maximum pitch angle [in degrees]
    'dp': 1.0,    # Pitch angle interval
    # Parameters for the pitch-gryroradius profiles
    'rprofmin': 2.0,     # Minimum gyroradius for the pitch profile calculation
    'rprofmax': 4.7,    # Maximum gyroradius for the pitch profile calculation
    'pprofmin': 20.0,    # Minimum pitch for the gyroradius profile calculation
    'pprofmax': 90.0,    # Maximum pitch for the gyroradius profile calculation
    # Position of the FILD
    'rfild': 2.186,   # 2.196 for shot 32326, 2.186 for shot 32312
    'zfild': 0.32,
    'alpha': 0.0,
    'beta': -12.0}
# -----------------------------------------------------------------------------
# %% Section 1: Load calibration
# Innitialise the calibration database object
database = ss.mapping.CalibrationDatabase(calibration_database)
# Get the calibration for our shot
cal = database.get_calibration(shot, camera, cal_type, diag_ID)
# -----------------------------------------------------------------------------

# %% Section 2: Load video file and the necesary frames
# Prepare the name of the cin file to be loaded
dummy = str(shot)
file = ss.paths.CinFiles + dummy[0:2] + '/' + dummy + '_v710.cin'
# initialise the video object:
cin = ss.vid.Video(file)
# Load the frames we need:
cin.read_frame(np.array([frame]), limitation=limitation,
               limit=limit)
# Remap all the loaded frames
# cin.remap_loaded_frames(cal, shot, par)
cin.plot_frame(frame)

plt.show()
