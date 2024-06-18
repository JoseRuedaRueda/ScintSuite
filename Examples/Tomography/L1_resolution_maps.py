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
outputPath = 'MyRoutines/tomography/results/W_47132_0.280'

# - Resolution maps parameters
# Type of map to calculate: 'pitch' or 'gyro'.
map_type = 'gyro'
# Setting the number of maximum iterations
maxiter = 10
# Setting algorithm to be used. Pick just one algebraic algorithm:
# 'descent', 'kaczmarz' or 'cimmino'
inverters = 'descent'
# Window
window = [50, 55, 4, 4.5]
# -----------------------------------------------------------------------------
# --- Section 1: Prepare the weight function and frame
# -----------------------------------------------------------------------------
WF = xr.load_dataarray(WFfile)

# -----------------------------------------------------------------------------
# --- Section 2: Perform the tomography
# -----------------------------------------------------------------------------
resolution = resP.calculate_resolution_map(WF, inverters, window,
                                            maxiter, map_type)

fig, ax = plt.subplots()
resolution.plot(ax = ax)
ax.set_xlabel('Gyroscalar')
ax.set_ylabel('GyroScalar')
ax.set_title('Gyro resolution map')

plt.savefig(os.path.join(outputPath,'resolution_map_gyro.png')) 




