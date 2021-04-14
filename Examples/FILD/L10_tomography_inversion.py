"""
Perform a tomographic inversion of a FILD frame.

Done in 07/04/2021

Note: efficiency is NOT included
"""

import Lib as ss
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - Paths:
calibration_database = './Data/Calibrations/FILD/calibration_database2.dat'
# As the strike points are needed and they are not included in the database,
# for the tomography one should manually select (for now) the strike map)
smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/tomography_strike_map.dat'
smap_points = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILDSIM//results/' +\
    'AUG_map_-002.10000_000.00000_strike_points.dat'

# - General options
diag_ID = 1     # FILD Number
shot = 33127    # shot number
t1 = 0.5      # Initial time to load [in s]
t2 = 4.8       # Final time to load [in s]
ttomo = 4.545    # Time to perform the tomography
# As you can see, I select a quite large interval, instead of just the frame,
# this is to have the black frames of the beggining and perform the noise
# subtraction. One more efficient way would be to load the first frames in
# another video object, and from there extract the 'noise frame' this is just
# some more lines of codes (the noise subtraction routines already gives as
# output the used frame and accept a external frame as input) but I wrote the
# code in the 'un-efficient way' because I thought it was a bit more easy to
# follow
# - Video options
limitation = False  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise substraction:
sub_noise = True   # Just a flag to see if we subtract noise or not
tn1 = 0.5     # Initial time to average the noise
tn2 = 0.6    # Initial time to average the noise

# - Filter options:
apply_filter = True  # Flag to apply filter to the frames
kind_of_filter = 'median'
options_filter = {
    'size': 3        # Size of the window to apply the filter
}
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
    'rfild': 2.211,   # 2.196 for shot 32326, 2.186 for shot 32312
    'zfild': ss.dat.FILD[diag_ID-1]['z'],
    'alpha': ss.dat.FILD[diag_ID-1]['alpha'],
    'beta': ss.dat.FILD[diag_ID-1]['beta'],
    # method for the interpolation
    'method': 2
}
# Tomography parameters
scintillator_options = {
    'rmin': 2.0,
    'rmax': 6.0,
    'dr': 0.2,
    'pmin': 20.0,
    'pmax': 90.0,
    'dp': 1.0
}
pin_options = {
    'rmin': 2.0,
    'rmax': 8.0,
    'dr': 0.2,
    'pmin': 20.0,
    'pmax': 80.0,
    'dp': 2.0
}
# Plotting options
FS = 16     # Font size
p1 = True   # Plot the MSE and r2 of the hyperparameter scan
p2 = True   # Plot the L curve and its derivative
# -----------------------------------------------------------------------------
# --- Section 1: Load calibration
# -----------------------------------------------------------------------------
# Initialize the calibration database object
database = ss.mapping.CalibrationDatabase(calibration_database)
# Get the calibration for our shot
cal = database.get_calibration(shot, ss.dat.FILD[diag_ID-1]['camera'],
                               'PIX', diag_ID)

# -----------------------------------------------------------------------------
# --- Section 2: Load video file and the necessary frames
# -----------------------------------------------------------------------------
# Prepare the name of the .cin file to be loaded

file = ss.vid.guess_filename(shot, ss.dat.FILD[diag_ID-1]['path'],
                             ss.dat.FILD[diag_ID-1]['extension'])
# initialise the video object:
cin = ss.vid.Video(file)
# Load the frames we need, it is necesary to load some of them to subtract the
# noise
cin.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)
# Subtract the noise
if sub_noise:
    cin.subtract_noise(t1=tn1, t2=tn2)
if apply_filter:
    cin.filter_frames(kind_of_filter, options_filter)
# Load the desired frame
iframe = np.argmin(abs(ttomo-cin.exp_dat['tframes']))
frame = cin.exp_dat['frames'][:, :, iframe].squeeze()

# -----------------------------------------------------------------------------
# --- Section 3: Prepare the weight function
# -----------------------------------------------------------------------------
# Calculate resolutions
smap = ss.mapping.StrikeMap('FILD', smap_file)
smap.calculate_pixel_coordinates(cal)
smap.load_strike_points(smap_points)
smap.calculate_resolutions()
# Prepare the grid for the remap
smap.interp_grid(frame.shape)
# Prepare the weight function and the signal
s1D, W2D, W4D, sg, pg, remap = ss.tomo.prepare_X_y_FILD(frame, smap,
                                                        scintillator_options,
                                                        pin_options)
# Normalise to speed up inversion
smax = s1D.max()
Wmax = W2D.max()
s1Dnorm = s1D / smax
W2Dnorm = W2D / Wmax
# -----------------------------------------------------------------------------
# --- Section 4: Perform the inversion
# -----------------------------------------------------------------------------
# Ridge (0th Tiko...) is also implemented, but there is no option in the build
# library to perform a Ridge regression forcing the coefficients to be
# possitive, but for elastic net yes. therefore we will use elastic net with
# l1 parameter of 0.05, the minimum allowed by the python library. This will be
# almost equivalent to Ridge (as the lasso penalty is small) but we could
# benefit from the quick implementation so the calculation can be made in
# seconds
# beta, MSE, r2, alpha = ss.tomo.Elastic_net_scan(
#     W2Dnorm, s1Dnorm, 1e-8, 1000, n_alpha=40, log_spaced=True, plot=p1,
#     l1_ratio=0.05, positive=True, max_iter=4000)


first_scan = ss.tomo.Ridge_scan(W2Dnorm, s1Dnorm, 1e-8, 1000, n_alpha=500,
                                plot=p1)
# - Calculate the L-curve to choose the best value
L_curve_fit, opt_res = ss.tomo.L_curve_fit(first_scan['norm'],
                                           first_scan['residual'], plot=p2)

# - Perform a fine scan around this value:
verbose = True
if verbose:
    print('The approximate optimum residual is: ', opt_res)
    print('Proceed with a finer scan')
opt_alpha = first_scan['alpha'][np.argmin(abs(first_scan['residual']-opt_res))]
second_scan = ss.tomo.Ridge_scan(W2Dnorm, s1Dnorm, opt_alpha/50, opt_alpha*50,
                                 n_alpha=500, plot=p1)

# -----------------------------------------------------------------------------
# --- Section 5: Representation
# -----------------------------------------------------------------------------
nalpha = len(second_scan['alpha'])
inversions = np.zeros((pg['nr'], pg['np'], nalpha))

for i in range(nalpha):
    inversions[:, :, i] =\
        np.reshape(second_scan['beta'][:, i] * Wmax, (pg['nr'], pg['np']))
# --- Prepare the dictionary for the GUI:
data = {
    'frame': frame,
    'remap': remap,
    'tomoFrames': inversions,
    'alpha': second_scan['alpha'],
    'norm': second_scan['norm'],
    'residual': second_scan['residual'],
    'MSE': second_scan['MSE'],
    'pg': pg,
    'sg': sg
}
