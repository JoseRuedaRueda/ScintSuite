"""
BEP spectra plotting.

This reads using Ralph Dux's bepget code the calibrated BEP signal and plots
it for a number of inputs shots.
"""

import Lib.BEP.libBEP as rbep
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
    shotdata[shotno] = rbep.readBEP(shotnumber=shotno, 
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
            ax1[ii, jj]=rbep.plotBEP(shotdata[shotno], losname=losname,
                                     line_options={'label': '#%05d'%shotno}, 
                                     ax=ax1[ii, jj])
        ax1[ii, jj].legend()    

for ishot, shotno in enumerate(shot_list):
    ax2 = rbep.plotBEP(shotdata[shotno], losname='neon',
                       line_options={'label':'#%05d'%shotno}, ax=ax2)
            
# -----------------------------------------------------------------------------
# --- Setting up the labels.
# -----------------------------------------------------------------------------
for ii in np.arange(5, dtype = int):
    ax1[ii, 0].set_ylabel('Counts')

for ii in np.arange(3, dtype = int):
    ax1[-1, ii].set_xlabel('Wavelength [nm]')
    ax1[0, ii].set_title(signal_name[ii])

plt.show()