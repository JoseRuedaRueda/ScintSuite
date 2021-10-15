"""
Perform a tomographic inversion of a FILD frame.

Consider only the radius, ie, average over a pitch. In this way we gain more
statistics and the inversion matrix is much more smaller, so we can compute it
much faster

Done in 02/09/2021

Note: efficiency is included, but with default parameters, D in Tg green
"""

import Lib as ss
import numpy as np
import tkinter as tk

# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - Paths:
calibration_database = ss.paths.ScintSuite \
    + '/Data/Calibrations/FILD/calibration_database.txt'
# As the strike points are needed and they are not included in the database,
# for the tomography one should manually select (for now) the strike map)
smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/09_tomo_strike_map.dat'
# The strike points are supposed to be saved in the same folder and with the
# same run id

# - General options
diag_ID = 1     # FILD Number
shot = 39612    # shot number
t1 = 0.15      # Initial time to load [in s]
t2 = 0.4       # Final time to load [in s]
ttomo = 0.346    # Time to perform the tomography
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
tn1 = 0.15     # Initial time to average the noise
tn2 = 0.19    # Initial time to average the noise

# - Filter options:
apply_filter = False  # Flag to apply filter to the frames
kind_of_filter = 'median'
options_filter = {
    'size': 2        # Size of the window to apply the filter
}

# Tomography parameters
scintillator_options = {
    'rmin': 1.8,
    'rmax': 7.0,
    'dr': 0.05,
    'pmin': 40.0,
    'pmax': 60.0,
    'dp': 0.5
}
pin_options = {
    'rmin': 1.8,
    'rmax': 7.0,
    'dr': 0.05,
    'pmin': 40.0,
    'pmax': 60.0,
    'dp': 0.5
}
diag_params = {
    'p_method': 'Gauss',
    'g_method': 'Gauss'
}
size_filter = 3
alpha_max = 1e1
alpha_min = 1e-3
nalpha1 = 300  # Number of points for the first hyperparameters scan
nalpha2 = 1  # Number of points for the second hyperparameters scan
# Plotting options
FS = 16     # Font size
Ridge = True
l1 = 0.33
MC_markers = 600  # Markers for the MC remap
efficiency = True
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
smap.load_strike_points()
smap.calculate_resolutions(diag_params=diag_params)
# Prepare the grid for the remap
grid_params = {
    'ymin': scintillator_options['rmin'],
    'ymax': scintillator_options['rmax'],
    'dy': scintillator_options['dr'],
    'xmin': scintillator_options['pmin'],
    'xmax': scintillator_options['pmax'],
    'dx': scintillator_options['dp']
}
smap.interp_grid(frame.shape, grid_params=grid_params, MC_number=MC_markers)
if efficiency:
    eff = ss.scintcharact.ScintillatorEfficiency()
else:
    eff = None
# Prepare the weight function and the signal
s1D, W2D, W4D, sg, pg, remap = \
    ss.tomo.prepare_X_y_FILD(frame, smap, scintillator_options, pin_options,
                             efficiency=eff,
                             filter_option={'size': size_filter},
                             LIMIT_REGION_FCOL=False,
                             only_gyroradius=True)
# Normalise to speed up inversion
smax = s1D.max()
Wmax = W2D.max()
s1Dnorm = s1D / smax
W2Dnorm = W2D / Wmax
# -----------------------------------------------------------------------------
# --- Section 4: Perform the inversion
# -----------------------------------------------------------------------------
# Ridge (0th Tiko...) and Elastic Net (Ridge and LASSSO combination) are also
# available
if Ridge:
    first_scan, fig_first_scan = \
        ss.tomo.nnRidge_scan(W2Dnorm, s1Dnorm, alpha_min, alpha_max,
                             n_alpha=nalpha1, plot=False)
else:
    first_scan, fig_first_scan = \
        ss.tomo.Elastic_net_scan(W2Dnorm, s1Dnorm, alpha_min, alpha_max,
                                 n_alpha=nalpha1, l1_ratio=l1, plot=False)

# -----------------------------------------------------------------------------
# --- Section 5: Representation
# -----------------------------------------------------------------------------
inversions1 = np.zeros((pg['nr'], pg['np'], nalpha1))

profile_pinhole1 = np.zeros((pg['nr'], nalpha1))

frames1 = np.zeros((sg['nr'], sg['np'], nalpha1))

profiles1 = np.zeros((sg['nr'], nalpha1))

for i in range(nalpha1):
    inversions1[:, :, i] =\
        np.zeros((pg['nr'], pg['np']))
    frames1[:, :, i] = np.zeros((sg['nr'], sg['np']))
    profiles1[:, i] = W2D @ first_scan['beta'][:, i]
    profile_pinhole1[:, i] = first_scan['beta'][:, i]

# --- Prepare the dictionary for the GUI:
data1 = {
    'frame': frame,
    'remap': remap,
    'tomoFrames': inversions1,
    'forwardFrames': frames1,
    'alpha': first_scan['alpha'],
    'norm': first_scan['norm'],
    'residual': first_scan['residual'],
    'MSE': first_scan['MSE'],
    'pg': pg,
    'sg': sg,
    'profiles': profiles1,
    'profiles_pinhole': profile_pinhole1
}

# --- open the gui:
root = tk.Tk()
ss.GUI.ApplicationShowTomography(root, data1)
root.mainloop()
root.destroy()
