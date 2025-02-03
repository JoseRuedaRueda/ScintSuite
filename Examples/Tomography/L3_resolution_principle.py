"""
Perform a tomographic inversion of a synthetic FILD frame 
using algebraic techniques and the resolution principle as stopping criteria.

"""

import os
import ScintSuite as ss
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import ScintSuite._Tomography._synthetic_signal as synthetic_signal


# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - Paths:
homeDir = '/home/marjimcom/ScintSuite/'

WFfile = homeDir + 'MyRoutines/W_39573_2.000.nc'

# - Paths for outputs:
outputPath = 'MyRoutines/tomography/results/test'

# Path for remap file
remapFile = '~/ScintSuite/MyRoutines/remap47132_-010to450ms_centr_rL.nc'
ttomo = 0.28    # Time to perform the tomography

# Resolution maps paths
resolution_pitch_file = homeDir + \
    'MyRoutines/tomography/results/W_39573_2.000/resolution_map_pitch_20_09_100.nc'
resolution_gyro_file = homeDir + \
    'MyRoutines/tomography/results/W_39573_2.000/resolution_map_gyro_13_09_100.nc'

# Setting frame to be used: synthetic or experimental
frame_type = 'synthetic'

# Parameters for the synthetic signal
seed = 0
mu_gyro = [3.1, 4.3, 5.4]
power = [0.1, 0.2, 0.7]
sigma_gyro = 0.01
mu_pitch = [55, 55, 55]
sigma_pitch = 7
noise_level = 0.1
background_level = 0.01
window = [45, 65, 3, 9]

# - Tomography parameters
# Setting the number of maximum iterations
# iters = np.array([10,25])
iters = 10
# Flag to use the resolution principle
resolution = True

# Setting algorithm to be used: 'descent', 'kaczmarz' or 'cimmino'
inverter = 'descent'

# Setting values for damping 
damp = 0.1
relaxParam = 1

# Setting plotting parameters
cmap = ss.plt.Gamma_II()

# -----------------------------------------------------------------------------
# --- Section 1: Prepare the weight function and frame
# -----------------------------------------------------------------------------
WF = xr.load_dataarray(WFfile)

# Generate the synthetic signal or read frame and initialize tomography object
if frame_type == 'synthetic':
    x_synthetic, frame_synthetic = synthetic_signal.create_synthetic_signal(WF, 
                                                mu_gyro, mu_pitch,
                                                power, sigma_gyro, 
                                                sigma_pitch, noise_level,
                                                background_level,
                                                seed=seed)
    tomo = ss.tomography(WF,frame_synthetic)

else:
    remap = xr.load_dataset(remapFile)
    frame_original =remap.frames.sel(t=ttomo, method='nearest')
    frame = frame_original.squeeze().interp(x=WF.xs, y=WF.ys)
    tomo = ss.tomography(WF,frame)



# -----------------------------------------------------------------------------
# --- Section 2: Perform the tomography
# -----------------------------------------------------------------------------

# Load resolution maps
resolution_pitch = xr.load_dataarray(resolution_pitch_file)
resolution_gyro = xr.load_dataarray(resolution_gyro_file)

# Pinhole size
n = WF.shape[2]*WF.shape[3]


x0 = np.zeros(n)
if 'descent' == inverter:
    tomo.coordinate_descent_solve(x0, 
                                  window = window, damp = damp, 
                                  relaxParam = relaxParam, 
                                  pitch_map = resolution_pitch,
                                  gyro_map = resolution_gyro, 
                                  resolution = resolution)


# -----------------------------------------------------------------------------
# --- Section 3: Plot results
# -----------------------------------------------------------------------------

# best_alpha = len(iters)-1

fig, axs = plt.subplots(1, 3,  figsize=(20, 5))

# Plot x_synthetic
x_synthetic.plot(ax = axs[0], cmap=cmap)
axs[0].set_title('x_synthetic')
axs[0].set_xlabel('Gyroradius')
axs[0].set_ylabel('pitch')

# Plot y_synthetic
frame_synthetic.plot(ax = axs[1], cmap=cmap)
axs[1].set_title('y_synthetic')
axs[1].set_xlabel('Gyroradius')
axs[1].set_ylabel('pitch')

# Plot inversion kacmarz
tomo.inversion[inverter].F.isel(alpha = 0 ).plot(ax = axs[2], 
                                                                cmap=cmap)
axs[2].set_title('Iterative algorithm with resolution principle')


plt.subplots_adjust(wspace=0.6)
plt.savefig(os.path.join(outputPath,'full_result.png')) 
