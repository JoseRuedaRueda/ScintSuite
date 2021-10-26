"""
BEP spectra plotting.

This reads directly from the database the data from the BEP shotfile and plots
the raw signal (uncalibrated signal) for a set of given shots.s
"""

import Lib.BEP.LibBEP as rbep
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# --- Configuration variables.
# -----------------------------------------------------------------------------
shot_list = (38023, 34436, 32177)  # List of shots to plot.
shot_time_list = ( (5.0, 5.1),     # For each shot, in order, the time range.
                   (5.0, 5.1),
                   (3.0, 3.1)
                 )

signal_name = ('Sigma', 'Mixed', 'Pi')

# -----------------------------------------------------------------------------
# --- Checking the inputs.
# -----------------------------------------------------------------------------
if len(shot_list) != len(shot_time_list):
    raise Exception('The number of shots and time windows has to be the same!')

# -----------------------------------------------------------------------------
# --- Reading the list of shots.
# -----------------------------------------------------------------------------
shotdata = dict()

for ishot, shotno in enumerate(shot_list):
    shotdata[shotno] = rbep.BEPfromSF(shotnumber=shotno, 
                                      time=shot_time_list[ishot])
    
# -----------------------------------------------------------------------------
# --- Creating the plots.
# -----------------------------------------------------------------------------
fig1, ax1 = plt.subplots(5, 3)   # Spectra plot (5 LOS, sigma/pi/mixed).
fig2, ax2 = plt.subplots(1)      # Neon lamp.


for ii in np.arange(5, dtype=int):
    for jj in np.arange(3,  dtype=int):
        for ishot, shotno in enumerate(shot_list):
            losname='BEP-%d-%d'%(ii+1, jj+1)
            if shotno >= 36900:
                if jj == 0:
                    losname='BEP-%d-%d'%(ii+1, 3)
                elif jj == 2:
                    losname='BEP-%d-%d'%(ii+1, 1)
            
            opts  = { 'label': '%05d - %s'%(shotno, losname)}
            ax1[ii, jj]=rbep.plotBEP_fromSF(shotdata[shotno], losname=losname, 
                                            ax=ax1[ii, jj],
                                            line_options=opts)
        ax1[ii, jj].legend()    

for ishot, shotno in enumerate(shot_list):
    opts  = { 'label': '%05d - Neon'%shotno}
    ax2 = rbep.plotBEP_fromSF(shotdata[shotno], losname='neon', ax=ax2,
                              line_options=opts)
    
# -----------------------------------------------------------------------------
# --- Setting up the labels.
# -----------------------------------------------------------------------------
for ii in np.arange(5, dtype = int):
    ax1[ii, 0].set_ylabel('Counts')

for ii in np.arange(3, dtype = int):
    ax1[-1, ii].set_xlabel('Pixel numbers')
    ax1[0, ii].set_title(signal_name[ii])
    
ax2.set_xlabel('Pixel number')
ax2.set_ylabel('Counts [uncorrected]')
ax2.set_title('Neon channel')
ax2.legend()

plt.show()