"""
Perform a tomographic inversion of a synthetic FILD frame 
using algebraic techniques.

"""

import os
import ScintSuite as ss
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import ScintSuite._Tomography._synthetic_signal as synthetic_signal
plt.ion()

# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - Paths:
WFfile = '~/ScintSuite/MyRoutines/W_39573_2.000.nc'

# - Paths for outputs:
outputPath = 'MyRoutines/tomography/results/test'

# Setting frame to be used: synthetic or experimental
frame_type = 'synthetic'

# Path for remap file
remapFile = '~/ScintSuite/MyRoutines/remap47132_-010to450ms_centr_rL.nc'
ttomo = 0.28    # Time to perform the tomography

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
iters = np.array([1, 10, 30, 50, 100])
# Setting algorithms to be used
inverters = ['kaczmarz','descent','cimmino']

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
    tomo = ss.tomography(WF,frame_synthetic, normalise=False)

else:
    remap = xr.load_dataset(remapFile)
    frame_original =remap.frames.sel(t=ttomo, method='nearest')
    frame = frame_original.squeeze().interp(x=WF.xs, y=WF.ys)
    tomo = ss.tomography(WF,frame)



# -----------------------------------------------------------------------------
# --- Section 2: Perform the tomography
# -----------------------------------------------------------------------------

x0 = np.zeros(tomo.s1D.shape)
if 'cimmino' in inverters: 
    tomo.cimmino_solve(x0, iters, window = window, 
                       damp = damp, relaxParam = relaxParam)


x0 = np.zeros(tomo.s1D.shape)
if 'descent' in inverters:
    tomo.coordinate_descent_solve(x0, iters, window = window, damp = damp, 
                                  relaxParam = relaxParam)


x0 = np.zeros(tomo.s1D.shape)
if 'kaczmarz' in inverters:
    tomo.kaczmarz_solve(x0, iters, window = window, 
                        damp = damp, relaxParam = relaxParam)

# -----------------------------------------------------------------------------
# --- Section 3: Plot metrics
# -----------------------------------------------------------------------------

tomo.plot_computational_time(inverters=inverters)
plt.savefig(os.path.join(outputPath,'computation_time.png'))

tomo.plot_MSE_error(inverters=inverters)
plt.savefig(os.path.join(outputPath,'MSE.png')) 

tomo.plot_synthetic_error(x_syntheticXR=x_synthetic, inverters=inverters)
plt.savefig(os.path.join(outputPath,'error.png')) 


# -----------------------------------------------------------------------------
# --- Section 4: Plot results
# -----------------------------------------------------------------------------

best_alpha_kacmarz = len(iters)-1
best_alpha_descent = len(iters)-1
best_alpha_cimmino = len(iters)-1

fig, axs = plt.subplots(1, 5,  figsize=(20, 5))

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
tomo.inversion['descent'].F.isel(alpha = best_alpha_kacmarz).plot(ax = axs[2], 
                                                                  cmap=cmap)
axs[2].set_title('Descent algorithm')

# Plot inversion
tomo.inversion['kaczmarz'].F.isel(alpha = best_alpha_descent).plot(ax = axs[3],
                                                                   cmap=cmap)
axs[3].set_title('Kaczmarz algorithm')

# Plot inversion
tomo.inversion['cimmino'].F.isel(alpha = best_alpha_cimmino).plot(ax = axs[4],
                                                                   cmap=cmap)
axs[4].set_title('Cimmino algorithm')

plt.subplots_adjust(wspace=0.6)
plt.savefig(os.path.join(outputPath,'full_result.png')) 
