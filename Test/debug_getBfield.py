import numpy as np
import pyuda
import netCDF4 as nc
import ScintSuite._Video._NetCDF4files as ncdf
import matplotlib.pyplot as plt
from rich import inspect
import ScintSuite as ss

## import os and disable file locking
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' 

## activate the client
client = pyuda.Client()

path = "$MAST_DATA/054/54277/LATEST/xfx054277.nc"
shot = 54277



## righto.
## args passed
shot = shot
Rin = 1.39
zin = 0.5
time = 0.3

## alt args; they're all multivalued this time
time = np.array([0.21, 0.22, 0.23])
Rin = np.array([1.39, 1.40, 1.41, 1.41])
zin = np.array([-0.1, 0.0, 0.1, 0.1])

## if time is a list or array, pass it through
## otherwise it is a float/single valued; then make it an array
if isinstance(time, (list, np.ndarray)):
        pass
else:  # it should be just a number
    time = np.array([time])

if not isinstance(Rin, np.ndarray):
    Rin = np.array([Rin])
if not isinstance(zin, np.ndarray):
    zin = np.array([zin])

br = np.zeros((time.shape[0], Rin.shape[0]))
bz = np.zeros((time.shape[0], Rin.shape[0]))
bp = np.zeros((time.shape[0], Rin.shape[0]))
bt = np.zeros((time.shape[0], Rin.shape[0]))

t = client.get('/epm/time', shot).data
r = client.get('/epm/output/profiles2d/r', shot).data
z = client.get('/epm/output/profiles2d/z', shot).data
br_all = client.get('/epm/output/profiles2d/br', shot).data
bz_all = client.get('/epm/output/profiles2d/bz', shot).data
bpol_all = client.get('/epm/output/profiles2d/bpol', shot).data
bphi_all = client.get('/epm/output/profiles2d/bphi', shot).data

idxt = np.abs(t[None, :] - time[:, None]).argmin(axis=-1)
idxr = np.abs(r[None, :] - Rin[:, None]).argmin(axis=-1)
idxz = np.abs(z[None, :] - zin[:, None]).argmin(axis=-1)

br = br_all[idxt, ...][:, idxr, idxz]
bz = bz_all[idxt, ...][:, idxr, idxz]
bp = bpol_all[idxt, ...][:, idxr, idxz]
bt = bphi_all[idxt, ...][:, idxr, idxz]

## method 1
for ii in range(len(time)):
    idxt = (np.abs(t-time[ii])).argmin()
    for jj in range(len(Rin)):
        idxr = (np.abs(r-Rin[jj])).argmin()
        idxz = (np.abs(z-zin[jj])).argmin() 
        br[ii, jj] = client.get('/epm/output/profiles2d/br', shot).data[idxt,idxr,idxz]
        bz[ii, jj] = client.get('/epm/output/profiles2d/bz', shot).data[idxt,idxr,idxz]
        bp[ii, jj] = client.get('/epm/output/profiles2d/bpol', shot).data[idxt,idxr,idxz]
        bt[ii, jj] = client.get('/epm/output/profiles2d/bphi', shot).data[idxt,idxr,idxz]



