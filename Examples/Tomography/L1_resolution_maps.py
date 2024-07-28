import matplotlib.pyplot as plt
import numpy as np
import os
import ScintSuite as ss
import ScintSuite._Tomography._resolution_principle as resP
import xarray as xr


# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - Paths:
WFfile = '~/ScintSuite/MyRoutines/W_39573_2.000.nc'
# - Paths for outputs:
outputPath = '/home/marjimcom/ScintSuite/MyRoutines/tomography/results/W_39573_2.000/'

# - Resolution maps parameters
# Type of map to calculate: 'pitch' or 'gyro'.
map_type = 'pitch'
# Setting the number of maximum iterations
maxiter = 100
# Setting algorithm to be used. Pick just one algebraic algorithm:
# 'descent', 'kaczmarz' or 'cimmino'
inverter = 'descent'
# Window
window = [40, 80, 3, 8]
# Setting plotting parameters
cmap = ss.plt.Gamma_II()
# -----------------------------------------------------------------------------
# --- Section 1: Prepare the weight function and frame
# -----------------------------------------------------------------------------
WF = xr.load_dataarray(WFfile)

# -----------------------------------------------------------------------------
# --- Section 2: Perform the tomography
# -----------------------------------------------------------------------------
resolution = resP.calculate_resolution_map(WF, inverter, window,
                                            maxiter, map_type)

savePath = outputPath +'resolution_map_'+ map_type + '_' + str(maxiter) 

resolution.to_netcdf(os.path.join(savePath + '.nc'))

fig, ax = plt.subplots()
resolution.plot(ax = ax, cmap=cmap)
ax.set_xlabel('Gyroscalar')
ax.set_ylabel('Pitch')
ax.set_title(map_type + ' resolution map')
plt.savefig(os.path.join(savePath +'.png')) 




