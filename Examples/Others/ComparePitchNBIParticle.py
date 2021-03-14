"""
Particle pitch comparative

This script is done to show one of the little extra features of the suite,
given a radial position and a pitch, it calculate the pitch of the particle at
other radial position (assuming conservation of magnetic moment and a field
going like 1/R) and compare with the pitch profile of NBI. This allows to have
a quite view of were the losses are coming from
"""
# -----------------------------------------------------------------------------
# --- Section 0: Modules import
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import Lib as ss
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
R0 = 2.182      # Radial position from where we know the pitch
P0 = 45.        # Pitch angle at that radial position
nNBI = 8        # Number of the NBI we want to use to compare
t = 0.25        # Time in seconds to calculate the pitch profile
shot = 32312    # Shot to calculate the pitch profile
# -----------------------------------------------------------------------------
# --- Section 1: Core
# -----------------------------------------------------------------------------
# - Define the vector were we will evaluate the pitch
R = np.linspace(1.1, 2.2, 50)
# - Calculate the pitch at those positions, first pass to v_par/v
P1 = np.cos(P0 * np.pi / 180.)
P = ss.extra.pitch_at_other_place(R0, P1, R)
P = 180. * np.arccos(P) / np.pi
# - Load the NBI pitch profile
nbi = ss.dat.NBI(nNBI)
nbi.calc_pitch_profile(shot, t, rmin=1.1, deg=True)
# - Plot the resluts
fig, ax = plt.subplots()
ax.plot(R, P, 'k', linewidth=1.5, label='Particle')
ax.plot(nbi.pitch_profile['R'], nbi.pitch_profile['pitch'], 'r', linewidth=1.5,
        label='NBI#' + str(nNBI))
param = {'xlabel': 'R [m]', 'ylabel': '$\\lambda [{}^o]$', 'fontsize': 14,
         'grid': 'both'}
ax = ss.plt.axis_beauty(ax, param)
plt.legend()
plt.show()
