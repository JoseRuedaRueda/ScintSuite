"""
Routines to read the h5 files for the D3D cameras.

Written by Jose Rueda: jruedaru@uci.edu
"""
import os
import sys
import xarray as xr
import logging
import numpy as np
logger = logging.getLogger('ScintSuite.Video')

# ------------------------------------------------------------------------------
# %% Metadata reading
# ------------------------------------------------------------------------------
def read_data(filename: str,):
    """
    Read the metadata from the h5 file.

    Jose Rueda Rueda: jruedaru@uci.edu
    
    :param filename: path to the file
    :param verbose: print the metadata
    
    :return metadata: dictionary with the metadata
    """
    dataset = xr.open_dataset(filename)
    # get the image shape
    imageheader = {
        'biWidth': dataset['image'].shape[1],
        'biHeight': dataset['image'].shape[2],
        'framesDtype': dataset['image'].dtype}
    # Get the video metadata
    header = {
        'ImageCount': dataset['image'].shape[0],
    }
    # Try to get the insetion, only for FILD
    try:
        header['Insertion'] = dataset['bellowPosition'].values
    except KeyError:
        pass
    settings = {
        'ShutterNs': dataset['integration_time']*1.0e9,
    }
    # Possible bytes per pixels for the camera
    BPP = {'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64}
    try:
        settings['RealBPP'] = BPP[imageheader['framesDtype'].name]
        text = 'In the H5 there is no info about the real BitesPerPixel'\
            + ' used in the camera. Assumed that the BPP coincides with'\
            + ' the byte size of the variable!!!'
        logger.warning(text)
    except KeyError:
        raise Exception('Expected uint8,16,32,64 in the frames')
    time_base = dataset['time'].values
    # Close the dataset
    dataset.close()
    return header, imageheader, settings, time_base


def read_frame(vid, frames_number, limitation: bool = None, 
               limit: int = 2048):
    """
    Read frames from a .cin file

    Jose Rueda Rueda: jrrueda@us.es

    :param  vid: Video Object with the file information. See the video
        object of the BasicVideoObject.py file
    :param  frames_number: np array with the frame numbers to load
    :param  limitation: maximum size allowed to the output variable,
    in Mbytes, to avoid overloading the memory trying to load the whole
    video of 100 Gb
    :param  limit: bool flag to decide if we apply the limitation of we
    operate in mode: YOLO

    :return M: 3D numpy array with the frames M[px,py,nframes]
    """
    # check the size of the output, assumig that we load them in the worst 
    # case scenario, float 64
    size = vid.imageheader['biWidth']*vid.imageheader['biHeight']*8/1024/1024
    # Open de dataset:
    dataset = xr.open_dataset(vid.file)
    if frames_number is None:
        # We load all
        sizeToLoad = size * vid.header['ImageCount']
        if limitation and sizeToLoad > limit:
            raise Exception('The size of the file is too big to be loaded')
        # If we are inside the limit, load the video
        M = dataset['image'].transpose('y','x','time').values
    else:
        # We load only the frames in the list
        sizeToLoad = size * len(frames_number)
        if limitation and sizeToLoad > limit:
            raise Exception('The size of the file is too big to be loaded')
        # If we are inside the limit, load the video
        M = dataset['image'].isel(time=frames_number).transpose(
            'y','x','time').values
    
    # Close the dataset
    dataset.close()
    return M