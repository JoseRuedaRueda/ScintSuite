##Convert a single file with all XY points gathered in inkscape for the APD channels
import numpy as np

channels = np.arange(130)
n_paths_inkscape = 37   #number of xy points per inkscape object

apd = np.loadtxt('all_channels.txt', delimiter="\t")
apd[:, 1] = 864 - apd[:, 1]
for ch in channels:
    fname = 'ch_%i.txt'%(ch+1)
    
    np.savetxt(fname, 
               apd[ch*n_paths_inkscape : ch*n_paths_inkscape + n_paths_inkscape,:] )