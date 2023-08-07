"""
Plot the shape evolution

It plot the evolution of the separatrix, elongation, triangularity, etc

Jose Rueda: jrrueda@us.es
"""

import sys
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib')
from sf2equ_20200525 import EQU
import mapeq_20200507 as meq
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import ScintSuite.as ss
# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
shot = 39612

t_initial = 0.5   # in seconds
t_final = 3.2     # in seconds
dt = 0.2          # space between time points for the separatrix

t_mean_ini = 2.0    # initial time to take the mean value
t_mean_final = 3.0  # final time to take the mean value, triangularity
#                   # elongation and so on will be compared with the mean value
#                   # in this interval
diag = 'EQH'        # Diag for equilibrium stuff

cmap = ss.plt.Gamma_II()
# -----------------------------------------------------------------------------
# --- Load and plot the separatrix
# -----------------------------------------------------------------------------
# --- Get the separatrix
ntime = int((t_final - t_initial) / dt) + 1
time = np.linspace(t_initial, t_final, ntime)
equ = EQU(shot, diag=diag)
r_sep, z_sep = meq.rho2rz(equ, 1.0, t_in=time, coord_in='rho_pol')
# --- Open the figure
fig, ax = plt.subplots()
ss.plt.plot_vessel(shot=shot, ax=ax)

for i in range(ntime):
    ax.plot(r_sep[i][0], z_sep[i][0],
            color=cmap(time[i] / (t_final - t_initial)))
ax.set_aspect('equal')

norm = mpl.colors.Normalize(vmin=t_initial, vmax=t_final)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm)
fig.show()

# -----------------------------------------------------------------------------
# --- Load and plot the other parameters
# -----------------------------------------------------------------------------
# --- Load data:
data = ss.dat.get_shot_basics(shot, diag=diag)
it1 = np.argmin(np.abs(data['time'] - t_mean_ini))
it2 = np.argmin(np.abs(data['time'] - t_mean_final))
# --- Plot absolute data
fig2, ax2 = plt.subplots(3, sharex=True)
# Volume
ax2[0].plot(data['time'], data['Vol'])
ax2[0].set_ylabel('Volume [$m^{-3}$]')

# Elongation
ax2[1].plot(data['time'], data['k'])
ax2[1].set_ylabel('Elongation')

# triangularity
ax2[2].plot(data['time'], data['delRuntn'])
ax2[2].set_ylabel('Triangularity')
ax2[2].set_xlabel('T [s]')
ax2[2].set_xlim(t_initial, t_final)
fig2.show()

# --- Plot relative data
fig3, ax3 = plt.subplots(3, sharex=True)
# Volume
mean = data['Vol'][it1:it2].mean()
y = (data['Vol'] - mean) / mean * 100.
ax3[0].plot(data['time'], y)
ax3[0].set_ylim(0.9 * y.min(), 1.1 * y.max())
ax3[0].set_ylabel('$\\Delta V$ [%]')

# Elongation
mean = data['k'][it1:it2].mean()
y = (data['k'] - mean) / mean * 100.
ax3[1].plot(data['time'], y)
ax3[1].set_ylim(0.9 * y.min(), 1.1 * y.max())
ax3[1].set_ylabel('$\\Delta Elongation$ [%]')

# triangularity
mean = data['delRuntn'][it1:it2].mean()
y = (data['delRuntn'] - mean) / mean * 100.
ax3[2].plot(data['time'], y)
ax3[2].set_ylabel('$\\Delta Trian$ [%]')
ax3[2].set_xlabel('T [s]')
ax3[2].set_xlim(t_initial, t_final)
ax3[1].set_ylim(0.9 * y.min(), 1.1 * y.max())
fig3.show()
