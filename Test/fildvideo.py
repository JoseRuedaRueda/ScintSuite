### a test scipt to debug video access on UDA
## import essential modules
import numpy as np
import pyuda
import netCDF4 as nc
import ScintSuite._Video._NetCDF4files as ncdf
import matplotlib.pyplot as plt
from rich import inspect


## import os and disable file locking
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' 

## activate the client
client = pyuda.Client()

path = "$MAST_DATA/054/54277/LATEST/xfx054277.nc"

## directories list 
directories_list = client.list_archive_directories("$MAST_DATA/054/54277/")

## file list
file_list = client.list_archive_files("$MAST_DATA/054/54277/LATEST")

## signal names 
signal_names = client.list_file_signals(path)

## using actual signal path seems to be successful..
client.get("/xfx/video", path)
## or this also works (equivalent)
client.get("/xfx/video", 54287)


vvideo = ncdf.client.get("/xfx/video",path)
vtime = ncdf.client.get("/xfx/time",path)
vfps = ncdf.client.get("/devices/fps",path)
vexposure = ncdf.client.get("/devices/exposure",path)
vrfild = ncdf.client.get("/devices/RFILD",path)
vfildangle = ncdf.client.get("/devices/FILDangle",path)
vanaloguegain = ncdf.client.get("/devices/analoggain",path)
vdigitalgain = ncdf.client.get("/devices/diggain",path)