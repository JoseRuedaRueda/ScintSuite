"""
Routines to reading TCV style matlab files with video data from FILD cameras

Written by Anton Jansen van Vuuren: anton.jansenvanvuuren@epfl.ch

"""

import scipy.io
import numpy as np
import h5py


def read_file(filename: str):
    '''
    Function to .mat files of the XIMEA camera
    

    :param  filename: full path pointing to the .mat file
    '''
    #filename = '/videodata/pcfild002/data/fild002/74718.mat'
    

    ###Method for reading TCV .mat files into a 
    def get_ds_dictionaries(name, node):
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
        
    with h5py.File(filename,'r') as h5f:
        ds_dict = {}  
        #print ('**Walking Datasets to get dictionaries**\n')
        h5f.visititems(get_ds_dictionaries)
        print('\nDONE')
        #print('ds_dict size', len(ds_dict))


    return ds_dict


