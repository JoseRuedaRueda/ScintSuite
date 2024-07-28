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
WFfile = '~/ScintSuite/MyRoutines/W_47132_0.280_doldby2.nc'
# - Paths for outputs:
outputPath = '/home/marjimcom/ScintSuite/MyRoutines/tomography/results/W_47132_0.280_doldby2/'

# Setting the number of maximum iterations
maxiter = 100
# Setting algorithm to be used. Pick just one algebraic algorithm:
# 'descent', 'kaczmarz' or 'cimmino'
inverter = 'descent'
# Window
window = [35, 75, 3, 17]
# Noise levels
noise_levels = [0.01, 0.03, 0.09, 0.10, 0.15]
max_noise = 0.35
# Setting plotting parameters
cmap = ss.plt.Gamma_II()
# -----------------------------------------------------------------------------
# --- Section 1: Prepare the weight function and frame
# -----------------------------------------------------------------------------
WF = xr.load_dataarray(WFfile)

# -----------------------------------------------------------------------------
# --- Section 2: Calculate the fidelity map
# -----------------------------------------------------------------------------
fidelity_map = noise.fidelity_map(WF, inverter, window, maxiter, noise_levels[-1])

fig, ax = plt.subplots()
fidelity_map.plot(ax = ax, cmap=cmap)
ax.set_xlabel('Gyroscalar')
ax.set_ylabel('Pitch')
ax.set_title('Fidelity map')
# plt.show()

fidelity_map.to_netcdf(os.path.join(outputPath, 'fidelity_map.nc'))
plt.savefig(os.path.join(outputPath,'fidelity_map.png')) 

