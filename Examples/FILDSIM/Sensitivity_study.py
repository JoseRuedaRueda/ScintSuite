"""
Generate a synthetic remap and perform a tomography on it

Jose Rueda Rueda : jrrueda@us.es

Input distribution in the pingole will be considered as a series of gaussians
centered in r_mu[i], P_mu[i], with sigmas r_si[i], p_si[i] and relative
amplitude amp[i].

Note: Please change (in your local copy on MyRoutines!!!) lines starting ending
with # -#

DISCLAIMER: This was created for version 0.5, it is possible that some
function has been changed and the script does not work at all now.
(or we have a more fancy and quick way to do this stuff). If this happens,
contact Jose Rueda (jrrueda@us.es) by email and he will update this 'tutorial'
"""

import ScintSuite.as ss
import numpy as np
import tkinter as tk
# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
# -- Strike maps and points files
smap_file = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILDSIM/results/' +\
    'AUG_map_-000.60000_007.50000_strike_map.dat'                          # -#
spoints_file = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILDSIM/results/' +\
    'AUG_map_-000.60000_007.50000_strike_points.dat'                       # -#

# -- Models to generate the synthetic signal
synthetic_parameter = {
    'g_method': 'sGauss',  # Methods, sGauss means skewed Gaussian
    'p_method': 'Gauss'    # Gauss means standard Gaussian
}
eff_in_synthetic_signal = True  # Flag to include scint yield in the signal
# -- Grid for the synthetic signal
synthetic_grid = {
    'rmin': 1.5,
    'rmax': 10.0,
    'dr': 0.01,
    'pmin': 40.0,
    'pmax': 90.0,
    'dp': 1.0,
}
# -- Models for the tomography reconstruction
tomography_parameter = {
    'g_method': 'sGauss',  # Methods, sGauss means skewed Gaussian
    'p_method': 'Gauss'    # Gauss means standard Gaussian
}
eff_in_tomography = True  # Flag to include scint yield in the tomography
# -- Grid for the tomography reconstruction
# For the scintillator, the same grid than the synthetic signal will be used,
# for the pinhole, the following:
pinhole_grid = {
    'rmin': 1.5,
    'rmax': 10.0,
    'dr': 0.01,
    'pmin': 40.0,
    'pmax': 90.0,
    'dp': 1.0,
}
inversion_method = 'nnRidge'  # Ridge, nnRidge or ElasticNet are available
alpha_max = 1e3               # Maximum value of the alpha hyperparameter
alpha_min = 1e-4              # Minimum value of the alpha hyperparameter
nalpha = 15                   # Number of hyperparameters in the scan
# -- Peaks modelling
r_mu = [1.952,  2.41,  3.4]
r_si = [0.025, 0.025, 0.025]

p_mu = [65.0, 65.0, 65.0]
p_si = [5.0,  5.0,  5.0]

ampl = [0.4,  0.6,  1.0]
# -- Save options
path_to_save = '/afs/ipp/home/r/ruejo/ScintSuite/MyRoutines/nnRidge'       # -#
# -- Print options
p1 = True  # Plot the shyntetic signals
# -----------------------------------------------------------------------------
# --- Load the strike map & efficiency
# -----------------------------------------------------------------------------
smap = ss.mapping.StrikeMap(0, smap_file)
smap.load_strike_points(spoints_file, verbose=True)
eff = ss.scintcharact.ScintillatorEfficiency()
# -----------------------------------------------------------------------------
# --- Generate the synthetic signal
# -----------------------------------------------------------------------------
smap.calculate_resolutions(diag_params=synthetic_parameter)

synthetic_signals = []
if eff_in_synthetic_signal:
    efficiency = eff
else:
    efficiency = None
for i in range(len(r_mu)):
    input = ss.fildsim.gaussian_input_distribution(r_mu[i], r_si[i],
                                                   p_mu[i], p_si[i])
    signal = \
        ss.fildsim.synthetic_signal_remap(input, smap,
                                          efficiency=efficiency,
                                          **synthetic_grid)
    synthetic_signals.append(signal.copy())
    del signal
    del input
# Add signals coming from the different peaks
synthetic_frame = np.zeros(synthetic_signals[0]['signal'].shape)

for i in range(len(r_mu)):
    synthetic_frame += ampl[i] * synthetic_signals[i]['signal']

# Verbose some information
print('----------------')
print('Nuber of r in the scintillator grid: ',
      synthetic_signals[0]['gyroradius'].size)
print('Nuber of pitches in the scintillator grid: ',
      synthetic_signals[0]['pitch'].size)
print('Shape of the scintillator signal: ',
      synthetic_signals[0]['signal'].shape)
print('----------------')
if p1:
    ss.fildsim.plot_synthetic_signal(synthetic_signals[0]['gyroradius'],
                                     synthetic_signals[0]['pitch'],
                                     synthetic_frame)

# -----------------------------------------------------------------------------
# --- Prepare the weight function and the signal for the tomography
# -----------------------------------------------------------------------------
if eff_in_tomography:
    efficiency = eff
else:
    efficiency = None
# Prepare the weight function and the signal
s1D, W2D, W4D, sg, pg, remap = ss.tomo.prepare_X_y_FILD(synthetic_frame, smap,
                                                        synthetic_grid,
                                                        pinhole_grid,
                                                        efficiency=efficiency,
                                                        is_remap=True,
                                                        LIMIT_REGION_FCOL=False)
# Normalise to speed up inversion
# Normalise to speed up inversion
smax = s1D.max()
Wmax = W2D.max()
s1Dnorm = s1D / smax
W2Dnorm = W2D / Wmax

# -----------------------------------------------------------------------------
# --- Perform the tomography
# -----------------------------------------------------------------------------
if inversion_method == 'Ridge':
    scan, fig_scan = ss.tomo.Ridge_scan(W2Dnorm, s1Dnorm, alpha_min,
                                        alpha_max, n_alpha=nalpha,
                                        plot=True, folder_to_save=path_to_save)
elif inversion_method == 'nnRidge':
    scan, fig_scan = ss.tomo.nnRidge_scan(W2Dnorm, s1Dnorm, alpha_min,
                                          alpha_max, n_alpha=nalpha,
                                          plot=True,
                                          folder_to_save=path_to_save)
else:
    raise Exception('Method not understood')

# -----------------------------------------------------------------------------
# --- Calculate the scintillator distribution of these pinhole distributions
# -----------------------------------------------------------------------------
inversions = np.zeros((pg['nr'], pg['np'], nalpha))   # Pinhole distro
profile_pinhole = np.zeros((pg['nr'], nalpha))        # Pinhole profile
frames = np.zeros((sg['nr'], sg['np'], nalpha))       # Modeled scintillator
profiles = np.zeros((sg['nr'], nalpha))               # Scintillator profile

for i in range(nalpha):
    inversions[:, :, i] =\
        np.reshape(scan['beta'][:, i] * Wmax, (pg['nr'], pg['np']))
    frames[:, :, i] = \
        np.reshape(W2D @ scan['beta'][:, i], (sg['nr'], sg['np']))
    profiles[:, i] = np.sum(frames[:, :, i].squeeze(), axis=1)
    profile_pinhole[:, i] = np.sum(inversions[:, :, i].squeeze(), axis=1)

# -----------------------------------------------------------------------------
# --- Represent the results
# -----------------------------------------------------------------------------
# Note: In the GUI, there is a plot of the camera frame, this is though for the
# case of tomography performed with real experimental data. Please just ignore
# this plot
data = {
    'frame': synthetic_frame,
    'remap': remap,
    'tomoFrames': inversions,
    'forwardFrames': frames,
    'alpha': scan['alpha'],
    'norm': scan['norm'],
    'residual': scan['residual'],
    'MSE': scan['MSE'],
    'pg': pg,
    'sg': sg,
    'profiles': profiles,
    'profiles_pinhole': profile_pinhole
}
# --- open the gui:
root = tk.Tk()
ss.GUI.ApplicationShowTomography(root, data)
root.mainloop()
root.destroy()
