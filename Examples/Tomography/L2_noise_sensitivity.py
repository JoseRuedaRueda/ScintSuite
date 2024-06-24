import matplotlib.pyplot as plt
import numpy as np
import os
import ScintSuite as ss
import ScintSuite._Tomography._noise_sensitivity as noise
import xarray as xr

# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - Paths:
WFfile = '~/Old_ScintSuite/W_39573_2.000.nc'
# - Paths for outputs:
outputPath = '/home/marjimcom/ScintSuite/MyRoutines/tomography/results/W_39573_2.000'

# Setting the number of maximum iterations
maxiter = 10
# Setting algorithm to be used. Pick just one algebraic algorithm:
# 'descent', 'kaczmarz' or 'cimmino'
inverter = 'descent'
# Window
window = [40, 80, 3, 8]
# Noise levels
noise_levels = [0.01, 0.03, 0.09, 0.12, 0.15]
# -----------------------------------------------------------------------------
# --- Section 1: Prepare the weight function and frame
# -----------------------------------------------------------------------------
WF = xr.load_dataarray(WFfile)

# -----------------------------------------------------------------------------
# --- Section 2: Calculate the fidelity map
# -----------------------------------------------------------------------------
fidelity_map = noise.fidelity_map(WF, inverter, window, maxiter, noise_levels)

fig, ax = plt.subplots()
fidelity_map.plot(ax = ax)
ax.set_xlabel('Gyroscalar')
ax.set_ylabel('Pitch')
ax.set_title('Fidelity map')
# plt.show()
plt.savefig(os.path.join(outputPath,'fidelity_map.png')) 

# -----------------------------------------------------------------------------
# --- Section 3: Study of noise sensitivity
# -----------------------------------------------------------------------------

noise_sensitivity = noise.noise_sensitivity(WF, inverter, window, 
                                            maxiter, noise_levels)

fig, ax = plt.subplots()
noise_sensitivity.plot(ax = ax)
ax.set_xlabel('Gyroscalar')
ax.set_ylabel('Pitch')
ax.set_title('Noise sensitivity')
# plt.show()
plt.savefig(os.path.join(outputPath,'noise_sensitivity_gyro.png'))

