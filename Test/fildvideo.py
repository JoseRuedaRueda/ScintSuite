### a test scipt to debug video access on UDA
## import essential modules
import numpy as np
import pyuda
import netCDF4 as nc
from rich import inspect

## import os and disable file locking
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' 

## activate the client
client = pyuda.Client()

path = "$MAST_DATA/054/54287/LATEST/xfx054287.nc"

## directories list 
directories_list = client.list_archive_directories("$MAST_DATA/054/54287/")

## file list
file_list = client.list_archive_files("$MAST_DATA/054/54287/LATEST")

## signal names 
signal_names = client.list_file_signals(path)

## using actual signal path seems to be successful..
client.get("/xfx/video", path)
## or this also works (equivalent)
client.get("/xfx/video", 54287)