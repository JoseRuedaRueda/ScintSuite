"""
Load and plot the magnetic field and the separatrix
"""
import numpy as np
import ScintSuite as ss
import matplotlib.pyplot as plt
plt.ion()
# ------------------------------------------------------------------------------
# %% Settings
# ------------------------------------------------------------------------------
shot = 198952
time = 2.0

# ------------------------------------------------------------------------------
# %% Load the equilibrium
# ------------------------------------------------------------------------------
R = np.linspace(1.05, 2.60)
z = np.linspace(-1.1, 1.1, 53)
RR, ZZ = np.meshgrid(R, z)
br,bz, bt, bp = ss.dat.get_mag_field(shot, RR, ZZ, time)
rsep, zsep = ss.dat.get_separatrix(shot, time)
rmag, zmag = ss.dat.get_magnetic_axis(shot, time)
# ------------------------------------------------------------------------------
# %% Plot
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3)
im1 = ax[0].contourf(RR.T, ZZ.T, br.T, cmap='bwr', vmin=-.4, vmax=.4)
plt.colorbar(im1)
im2 = ax[1].contourf(RR.T, ZZ.T, bz.T, cmap='bwr', vmin=-.4, vmax=.4)
plt.colorbar(im2)
im3 = ax[2].contourf(RR.T, ZZ.T, bt.T, cmap='plasma')
plt.colorbar(im3)
names = ['Br', 'Bz', 'Bt']
for i in range(3):
    ax[i].plot(rsep, zsep, '-k')
    ax[i].plot(rmag, zmag, 'x')
    ax[i].set_aspect('equal')
    ax[i].set_xlim(1.05, 2.60)
    ax[i].set_ylim(-1.2, 1.2)
