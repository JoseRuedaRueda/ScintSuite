"""
In this example the new video filters speed is compared to the old ones

Alex Reyner Viñolas: areyner@us.es

Note: Done for version 2.0
"""

import ScintSuite as ss

import matplotlib
import matplotlib.pyplot as plt

import scipy.ndimage as spnd

import numpy as np
import xarray as xr
import copy
import pickle
import time as time
from tabulate import tabulate
plt.ion()

## REMAP FILD video and compare filter speed
shot = 41256
diag_ID = 1 # FILD manipulator number
new_vid = True
t1, t2 = 0, 0.3
limit, limitation = 2048, True # To avoid overloading the resources
subtract_noise = False   # Flag to apply noise subtraction
tn1, tn2 = 0.05, 0.35
apply_filter = True  # Flag to apply filter to the 
kind_of_filter = ['median', 'gaussian']
options_filter = {'size':2, 'sigma':3}

vid = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID)
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)
""" Substract noise """
if subtract_noise:
    frame = vid.subtract_noise(t1=tn1, t2=tn2, speed_flag = True) #from BVO

cvid = copy.deepcopy(vid) # copy of the video to apply old and new filters in the same conditions

""" Filter frames new"""
if apply_filter:
    for filter in kind_of_filter:
        time1=time.time()
        vid.filter_frames(method = filter, options = options_filter,
                            speed_flag = True)
        time2=time.time()
        elapsed = time2-time1
        globals()[f'new_{filter}_time'] = elapsed
        # print(f' --- NEW {filter} method duration:', elapsed)

""" Filter frames old"""
if apply_filter:
    for filter in kind_of_filter:
        time1=time.time()
        vid.filter_frames(method = filter, options = options_filter)
        time2=time.time()
        elapsed = time2-time1
        globals()[f'old_{filter}_time'] = elapsed
        # print(f' --- OLD {filter} method duration:', elapsed)

data = [['Median', f'{new_median_time:.2f} s', f'{old_median_time:.2f} s'],
        ['Gaussian', f'{new_gaussian_time:.2f} s', f'{old_gaussian_time:.2f} s'],
        ]
headers = ['Method', 'New', 'Old']
print(tabulate(data, headers=headers, tablefmt='grid', 
               colalign=('center', 'center', 'center')))

## REMAP video with new and old method
par = {
    'ymin': 0.6,      # Minimum gyroradius [in cm]
    'ymax': 10,     # Maximum gyroradius [in cm]
    'dy': 0.1,        # Interval of the gyroradius [in cm]
    'xmin': 20.0,     # Minimum pitch angle [in degrees]
    'xmax': 90.0,     # Maximum pitch angle [in degrees]
    'dx': 1,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 1, # Precision for the strike map (1 is more than enough)
    'allIn': 2,
    'remap_method': 'forward_warping_simple',
    }   
time1=time.time()
old_remap = copy.deepcopy(vid)
old_remap.remap_loaded_frames(par)
time2=time.time()
elapsed_old = time2-time1

par = {
    'ymin': 0.6,      # Minimum gyroradius [in cm]
    'ymax': 10,     # Maximum gyroradius [in cm]
    'dy': 0.1,        # Interval of the gyroradius [in cm]
    'xmin': 20.0,     # Minimum pitch angle [in degrees]
    'xmax': 90.0,     # Maximum pitch angle [in degrees]
    'dx': 1,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 1, # Precision for the strike map (1 is more than enough)
    'allIn': 2,
    'remap_method': 'forward_warping_simple',
    'speed_flag': True
    }   
time1=time.time()
new_remap = copy.deepcopy(vid)
new_remap.remap_loaded_frames(par)
time2=time.time()
elapsed_new = time2-time1

print(f'Remapping time: NEW {elapsed_new:.2f} s vs. OLD {elapsed_old:.2f} s')

plt.close('all')
fig, ax = plt.subplots(1,3, figsize=(20,5),sharex=True, sharey=True)
old = old_remap.remap_dat.frames.sel(t=1.420, method='nearest')
old.T.plot.imshow(ax=ax[1])
new = new_remap.remap_dat.frames.sel(t=1.420, method='nearest')
new.T.plot.imshow(ax=ax[0])
(new-old).T.plot.imshow(ax=ax[2])
ax[0].set_title(f'Old method ({elapsed_old:.2f} s)')
ax[1].set_title(f'Vectorized ({elapsed_new:.2f} s)')
ax[2].set_title('Difference')
plt.tight_layout()
plt.show()