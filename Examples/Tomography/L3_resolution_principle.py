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

WFfile = '~/ScintSuite/MyRoutines/W_47132_0.300_new_p46_ext.nc'

# - Paths for outputs:
outputPath = 'MyRoutines/tomography/results/test'

remapFile = '~/ScintSuite/MyRoutines/remap47132_-010to450ms_centr_rL.nc'
ttomo = 0.30    # Time to perform the tomography

# Resolution maps paths
resolution_pitch_file = homeDir + \
    'MyRoutines/tomography/results/ECPD2025_paper_p46/resolution_map_by2_pitch70.nc'
resolution_gyro_file = homeDir + \
    'MyRoutines/tomography/results/ECPD2025_paper_p46/resolution_map_by2_gyro70.nc'

# Setting frame to be used: synthetic or experimental
frame_type = 'synthetic'

# Parameters for the synthetic signal
seed = 0
mu_gyro = [12.3, 8.8, 7.2]
power = [0.7, 0.2, 0.1]
sigma_gyro = [0.45, 0.45, 0.45]
mu_pitch = [60, 60, 60]
sigma_pitch = [0.1, 0.1, 0.1]
noise_level = 0.1
background_level = 0.01
window = [59.0, 61.0, 6.5, 15.0]

# - Tomography parameters

# Flag to use the resolution principle
# If resolution is set to True, the algorithm will stop when the resolution principle
# is no longer satisfied. However, to prevent calculations from taking too long, the number
# of iterations will be limited by a control number of iterations set by default. This
# number can be changed in the variable control_iters
resolution = True

# Setting algorithm to be used: 'descent', 'kaczmarz' or 'cimmino'
inverter = 'kaczmarz'

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
                                  resolution = resolution,
                                  peak_amp=0.10, 
                                  control_iters=150)
    
if 'kaczmarz' == inverter:
    tomo.kaczmarz_solve(x0, 
                        window = window, damp = damp, 
                        relaxParam = relaxParam, 
                        pitch_map = resolution_pitch,
                        gyro_map = resolution_gyro, 
                        resolution = resolution,
                        peak_amp=0.10, 
                        control_iters=150)
    
if 'cimmino' == inverter:
    tomo.cimmino_solve(x0, 
                       window = window, damp = damp, 
                       relaxParam = relaxParam, 
                       pitch_map = resolution_pitch,
                       gyro_map = resolution_gyro, 
                       resolution = resolution,
                       peak_amp=0.10, 
                       control_iters=150)


# Tomo object by default normalizes the WF and the frame. If this is the case, 
# then the output needs to be denormalized

norm = tomo.norms['s']/tomo.norms['W']
xHat = tomo.inversion[inverter].F.isel(alpha = -1)*norm

# -----------------------------------------------------------------------------
# --- Section 3: Plot results for the synthetic case
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

# Plot inversion
xHat.plot(ax = axs[2], cmap=cmap)
axs[2].set_title('Iterative algorithm with resolution principle')


plt.subplots_adjust(wspace=0.6)
plt.savefig(os.path.join(outputPath,'resolution_principle.png')) 
