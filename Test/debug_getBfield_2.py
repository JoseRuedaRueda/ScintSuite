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

