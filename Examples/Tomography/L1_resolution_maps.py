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

# domain: Domain of the grid where you want to calculate the resolution map
# minimum pitch, maximum pitch, minimum gyroscalar, maximum gyroscalar
domain = [50, 70, 7, 16]

# Window: Space to project the reconstruction to avoid artifacts
window = [50, 75, 7, 18]

# Type of map to calculate: 'pitch' or 'gyro'.
map_type = 'gyro'

# Setting the number of maximum iterations
maxiter = 50

# Setting algorithm to be used. Pick just one algebraic algorithm:
# 'descent', 'kaczmarz' or 'cimmino'
inverter = 'descent'


# Setting plotting parameters
cmap = ss.plt.Gamma_II()
# -----------------------------------------------------------------------------
# --- Section 1: Prepare the weight function and frame
# -----------------------------------------------------------------------------
WF = xr.load_dataarray(WFfile)

# -----------------------------------------------------------------------------
# --- Section 2: Load smap and fit parameters
# -----------------------------------------------------------------------------
# Path to the smap file
smapFile = '/home/marjimcom/ScintSuite/MyRoutines/W_-8.75_-8.08_MU_FILD.map'

# Loading smap
smap = ss.smap.Fsmap(file=smapFile)
smap.load_strike_points()

# Load fit parameters
smap.build_parameters_xarray()
fits = smap._resolutions['fit_xarrays']


# -----------------------------------------------------------------------------
# --- Section 3: Calculate resolution map
# -----------------------------------------------------------------------------
resolution = resP.calculate_resolution_map(WF, inverter, domain,
                                            maxiter, fits, window = window,
                                              map_type = map_type)

savePath = outputPath +'resolution_map_'+ map_type + str(maxiter)
resolution.to_netcdf(os.path.join(savePath + '.nc'))

fig, ax = plt.subplots()
resolution.plot(ax = ax, cmap=cmap)
ax.set_xlabel('Gyroscalar')
ax.set_ylabel('Pitch')
ax.set_title(map_type + ' resolution map')
plt.savefig(os.path.join(savePath +'.png')) 




