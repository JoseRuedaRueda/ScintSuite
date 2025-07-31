"""
Routines to reading TCV style matlab files with video data from FILD cameras
Written by Anton Jansen van Vuuren: anton.jansenvanvuuren@epfl.ch
"""

import scipy.io as scio
import numpy as np
import h5py
import xarray as xr

def read_file(filename: str):
    '''
    Function to .mat files of the XIMEA camera
    :param  filename: full path pointing to the .mat file
    '''

    def get_ds_dictionaries(name, node): ###Method for reading older .mat files into a structure
        fullname = node.name
        if isinstance(node, h5py.Dataset):
        # node is a dataset
            #print(f'Dataset: {fullname}; adding to dictionary')
            ds_dict[fullname] = np.array(node[()])
            #print('ds_dict size', len(ds_dict))
        #else:
        #
        # node is a group
        #    print(f'Group: {fullname}; skipping')  

    #try:
    with h5py.File(filename,'r') as h5f:
        ds_dict = {}  
        #print ('**Walking Datasets to get dictionaries**\n')
        h5f.visititems(get_ds_dictionaries)
        print('\nDONE')
        #print('ds_dict size', len(ds_dict))
    #except:
    #    ds_dict = scio.loadmat(filename)

    return ds_dict


def read_frame(video_object, frames_number=None, limitation: bool = True,
            limit: int = 2048, verbose: bool = True):
    pass

    frames_number = np.array(frames_number)
    M = video_object.exp_dat['original_frames'][ :, :, frames_number]

    return M
