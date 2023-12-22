##Convert a single file with all XY points gathered in inkscape for the APD channels
import numpy as np

channels = np.arange(130)+1
n_paths_inkscape = 37   #number of xy points per inkscape object

apd = np.loadtxt('all_channels.txt', delimiter="\t")

for ch in channels:
    fname = 'ch_%i.txt'%ch
    np.savetxt(fname, 
               apd[ch*n_paths_inkscape : ch*n_paths_inkscape + n_paths_inkscape, 
                   ch*n_paths_inkscape : ch*n_paths_inkscape + n_paths_inkscape] )