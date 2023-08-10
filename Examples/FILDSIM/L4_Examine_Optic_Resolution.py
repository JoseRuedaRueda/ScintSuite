"""
Lecture 2 of the FILDSIM course

Examine the effect of the optic resolution

This example will add the defocusing effect of the optics modelled as a
Gaussian and them compare with the original resolution

Note:
    A resample will be done, do not use this with simulations with extremelly
    high number of points or you will kill the system

Jose Rueda Rueda: jrrueda@us.es
"""
import ScintSuite.as ss
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.interpolate as interp

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/Optic_resolution_strike_map.dat'
p0 = 50.0  # Focus pitch [º] Resolution will be calculated plot for this pitch
r0 = 3.3   # focus rl [cm] Resolution will be plot along for radius
experiment = 0.4
sigma_optics = np.linspace(0.00001, 0.10, 5)
n_resample = 100
# -----------------------------------------------------------------------------
# --- Load the strike points and calculate the resolutions
# -----------------------------------------------------------------------------
smap = ss.mapping.StrikeMap(0, smap_file)  # 0 indicated FILD map
smap.load_strike_points()
smap.calculate_mapping_interpolators(k=2)
sigmas = np.zeros(sigma_optics.size)
iy = smap.strike_points.header['info']['y']['i']
iz = smap.strike_points.header['info']['z']['i']
irl = smap.strike_points.header['info']['remap_rl']['i']
ig = np.argmin(np.abs(smap.unique_gyroradius - r0))
ip = np.argmin(np.abs(smap.unique_pitch - p0))
npoints, ncolums = smap.strike_points.data[ip, ig].shape
FILDSIM_sigma = smap.strike_points.data[ip, ig][:, irl].std()
f = smap.map_interpolators['Gyroradius']
for j in range(sigma_optics.size):
    # Resample data
    data = np.zeros((n_resample * npoints, 4))
    for i in range(npoints):
        y_point = smap.strike_points.data[ip, ig][i, iy]
        z_point = smap.strike_points.data[ip, ig][i, iz]
        y_new = \
            5 * sigma_optics[j] * (0.5 - np.random.rand(n_resample)) + y_point
        z_new = \
            5 * sigma_optics[j] * (0.5 - np.random.rand(n_resample)) + z_point
        d2 = (y_new - y_point)**2 + (z_new - z_point)**2
        weight = np.exp(-d2/2/sigma_optics[j])
        data[(i*n_resample):((i+1)*n_resample), 0] = y_new
        data[(i*n_resample):((i+1)*n_resample), 1] = z_new
        data[(i*n_resample):((i+1)*n_resample), 2] = weight
    # Remap de data
    rl = f.ev(data[:, 0], data[:, 1])
    average = np.average(rl, weights=data[:, 2])
    variance = np.average((rl-average)**2, weights=data[:, 2])
    sigmas[j] = math.sqrt(variance)

# -----------------------------------------------------------------------------
# --- Plot the comparison
# -----------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(sigma_optics, sigmas)
ax.set_xlabel('Optic Sigma [cm]')
ax.set_ylabel('$\\sigma_r [cm]$')
ax.plot([sigma_optics[0], sigma_optics[-1]], [experiment, experiment], '--r')
# ax.plot([sigma_optics[0], sigma_optics[-1]],
#         [FILDSIM_sigma, FILDSIM_sigma], '--k')

fig.show()
