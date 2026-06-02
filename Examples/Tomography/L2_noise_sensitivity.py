"""
Study noise effect on algebraic tomographic techniques. This lecture calculates 
the fidelity map for the algebraic algorithm 'descent' and saves the results in 
a netCDF file and a png image.

"""
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
outputPath = '/home/marjimcom/ScintSuite/MyRoutines/tomography/results/test/'

# Setting the number of maximum iterations
maxiter = 50

# domain: Domain of the grid where you want to calculate the fidelity map
# minimum pitch, maximum pitch, minimum gyroscalar, maximum gyroscalar
domain = [35, 70, 7, 16]

# Window: Space to project the reconstruction to avoid artifacts
# minimum pitch, maximum pitch, minimum gyroscalar, maximum gyroscalar
window = [35, 75, 7, 18]

# Signal noise level
noise_level = 0.1

# Background noise level
background_level = 0.1

# Setting algorithm to be used. Pick just one algebraic algorithm:
# 'descent', 'kaczmarz' or 'cimmino'
inverter = 'descent'

# Error to use. Two options: 'relativel2' or 'snr'
error_metric = 'relativel2'

# Setting the resolution principle as stopping criteria.
# This is not recommended for the fidelity map as it might not stop the 
# reconstruction of a single delta. In any case, the number of iterations 
# will be limited by a control number of iterations set by default to 150.
resolution = False

# Setting plotting parameters
cmap = ss.plt.Gamma_II()
# -----------------------------------------------------------------------------
# --- Section 1: Prepare the weight function and frame
# -----------------------------------------------------------------------------
WF = xr.load_dataarray(WFfile)

# -----------------------------------------------------------------------------
# --- Section 2: Calculate the fidelity map
# -----------------------------------------------------------------------------
fidelity_map = noise.fidelity_map(domain, WF, inverter, window, \
                                   maxiter, noise_level, background_level,
                                   resolution = resolution, 
                                   error_metric = error_metric)

fig, ax = plt.subplots()
fidelity_map.plot(ax = ax, cmap=cmap)
ax.set_xlabel('Gyroscalar')
ax.set_ylabel('Pitch')
ax.set_title('Fidelity map')
# plt.show()

fidelity_map.to_netcdf(os.path.join(outputPath, 'fidelity_map_descent.nc'))
plt.savefig(os.path.join(outputPath, 'fidelity_map_descent.png')) 

