"""
Perform a tomographic inversion of a FILD frame

Done in 02/02/2021
"""

import Lib as ss
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# Paths:
calibration_database = './Data/Calibrations/FILD/calibration_database2.dat'
# As the strike points are needed and they are not included in the database,
# for the tomography one should manually select (for now) the strike map)
smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/tomography_strike_map.dat'
smap_points = \
    '/afs/ipp/home/r/ruejo/FILDSIM/results/tomography_strike_points.dat'
# calibration data
camera = 'PHANTOM'
cal_type = 'PIX'
diag_ID = 1     # FILD Number
# Shot a time interval
t1 = 0.185      # Initial time to load [in s]
t2 = 0.80       # Final time to load [in s]
ttomo = 0.27    # Time to perform the tomography
shot = 32312
# Video options
limitation = False  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources
# Noise substraction parameters:
sub_noise = True   # Just a flag to see if we subtract noise or not
tn1 = 0.185     # Initial time to average the noise
tn2 = 0.198    # Initial time to average the noise
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
    'rprofmax': 4.0,    # Maximum gyroradius for the pitch profile calculation
    'pprofmin': 55.0,    # Minimum pitch for the gyroradius profile calculation
    'pprofmax': 75.0,    # Maximum pitch for the gyroradius profile calculation
    # Position of the FILD
    'rfild': 2.186,   # 2.196 for shot 32326, 2.186 for shot 32312
    'zfild': 0.32,
    'alpha': 0.0,
    'beta': -12.0,
    # method for the interpolation
    'method': 2}
# Tomography parameters
scintillator_options = {
    'rmin': 2.0,
    'rmax': 6.0,
    'dr': 0.1,
    'pmin': 20.0,
    'pmax': 90.0,
    'dp': 1.0
}
pin_options = {
    'rmin': 2.0,
    'rmax': 6.0,
    'dr': 0.1,
    'pmin': 20.0,
    'pmax': 90.0,
    'dp': 1.0
}
# Plotting options
FS = 16     # Font size
# -----------------------------------------------------------------------------
# --- Section 1: Load calibration
# -----------------------------------------------------------------------------
# Initialize the calibration database object
database = ss.mapping.CalibrationDatabase(calibration_database)
# Get the calibration for our shot
cal = database.get_calibration(shot, camera, cal_type, diag_ID)
# -----------------------------------------------------------------------------
# --- Section 2: Load video file and the necessary frames
# -----------------------------------------------------------------------------
# Prepare the name of the .cin file to be loaded
dummy = str(shot)
file = ss.paths.CinFiles + dummy[0:2] + '/' + dummy + '_v710.cin'
# initialise the video object:
cin = ss.vid.Video(file)
# Load the frames we need, necesary to load some of them to subtract the noise
cin.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)
# Subtract the noise
if sub_noise:
    cin.subtract_noise(t1=tn1, t2=tn2)
# Load the desired frame
frame = cin.read_frame(t1=ttomo, internal=False)
# -----------------------------------------------------------------------------
# --- Section 3: Prepare the weight function
# -----------------------------------------------------------------------------
smap = ss.mapping.StrikeMap('FILD', smap_file)
smap.calculate_pixel_coordinates(cal)
smap.load_strike_points(smap_points)
smap.calculate_resolutions()
smap.interp_grid(frame.shape)
s1D, W2D, W4D, sg, pg = ss.tomo.prepare_X_y_FILD(frame, smap,
                                                 scintillator_options,
                                                 pin_options)
