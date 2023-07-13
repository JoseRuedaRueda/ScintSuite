"""
Perform a tomographic inversion of a FILD frame.

Done in 07/08/2021

Revised for version 0.8.0
"""

import ScintSuite.as ss
import numpy as np
import tkinter as tk

# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - Paths:
# As the strike points are needed and they are not included in the database,
# for the tomography one should manually select (for now) the strike map)
smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/' \
    + 'tomography_new_geometry2_strike_map.dat'    # Change this
# - General options
diag_ID = 1     # FILD Number
shot = 39612    # shot number
t1 = 0.15      # Initial time to load [in s]
t2 = 0.4       # Final time to load [in s]
ttomo = 0.38    # Time to perform the tomography
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
    'rmin': 1.2,
    'rmax': 8.0,
    'dr': 0.1,
    'pmin': 20.0,
    'pmax': 85.0,
    'dp': 1.0
}
pin_options = {
    'rmin': 1.2,
    'rmax': 8.0,
    'dr': 0.1,
    'pmin': 20.0,
    'pmax': 85.0,
    'dp': 1.0
}
size_filter = 0  # Size of the median filter to apply to the remap frame
alpha_max = 1e2
alpha_min = 1e-5
nalpha1 = 10  # Number of points for the first hyperparameters scan
# Plotting options
FS = 16     # Font size
Ridge = True
l1 = 1.
MC_markers = 300  # Markers for the MC remap

# -----------------------------------------------------------------------------
# --- Section 1: Load video file and the necessary frames
# -----------------------------------------------------------------------------
# Prepare the name of the .cin file to be loaded
# initialise the video object:
cin = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID)
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
# --- Section 2: Prepare the weight function
# -----------------------------------------------------------------------------
# Calculate resolutions
smap = ss.mapping.StrikeMap('FILD', smap_file)
smap.calculate_pixel_coordinates(cin.CameraCalibration)
smap.load_strike_points()
diag_params = {
    'p_method': 'Gauss',
    'g_method': 'Gauss'
}
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
eff = ss.scintcharact.ScintillatorEfficiency()
# Prepare the weight function and the signal
s1D, W2D, W4D, sg, pg, remap = \
    ss.tomo.prepare_X_y_FILD(frame, smap, scintillator_options,
                             pin_options, efficiency=eff,
                             filter_option={'size': size_filter},
                             LIMIT_REGION_FCOL=False)
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
        ss.tomo.Ridge_scan(W2Dnorm, s1Dnorm, alpha_min, alpha_max,
                           n_alpha=nalpha1, plot=True)
else:
    first_scan, fig_first_scan = \
        ss.tomo.Elastic_net_scan(W2Dnorm, s1Dnorm, alpha_min, alpha_max,
                                 n_alpha=nalpha1, l1_ratio=l1)

# -----------------------------------------------------------------------------
# --- Section 5: Representation
# -----------------------------------------------------------------------------
inversions1 = np.zeros((pg['nr'], pg['np'], nalpha1))

profile_pinhole1 = np.zeros((pg['nr'], nalpha1))

frames1 = np.zeros((sg['nr'], sg['np'], nalpha1))

profiles1 = np.zeros((sg['nr'], nalpha1))

for i in range(nalpha1):
    inversions1[:, :, i] =\
        np.reshape(first_scan['beta'][:, i] * Wmax, (pg['nr'], pg['np']))
    frames1[:, :, i] = \
        np.reshape(W2D @ first_scan['beta'][:, i], (sg['nr'], sg['np']))
    profiles1[:, i] = np.sum(frames1[:, :, i].squeeze(), axis=1)
    profile_pinhole1[:, i] = np.sum(inversions1[:, :, i].squeeze(), axis=1)

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
