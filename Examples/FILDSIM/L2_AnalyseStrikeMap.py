"""
Lecture 2 of the FILDSIM course

Analyse the strike map (resolutions, sigmas, etc)

Jose Rueda Rueda: jrrueda@us.es
"""
import Lib as ss
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/why1_strike_map.dat'
diag_params = {
        'p_method': 'Gauss',  # Gauss = Gaussian shape
        'g_method': 'Gauss'   # sGauss = SkewedGaussian Shape
}
p0 = 50.0  # Focus pitch [º] Resolution will be plot along this pitch
r0 = 3.0   # focus rl [cm] Resolution will be plot along this radius
# -----------------------------------------------------------------------------
# --- Load the strike points and calculate the resolutions
# -----------------------------------------------------------------------------
# Note: The strike point file is supposed to be in the same folder that the
# strike map
smap = ss.mapping.StrikeMap(0, smap_file)  # 0 indicated FILD map
smap.load_strike_points()
smap.calculate_resolutions(diag_params=diag_params)
smap.plot_resolutions(nlev=50)
# -----------------------------------------------------------------------------
# --- Compare with the strike points
# -----------------------------------------------------------------------------
npitch = smap.strike_points.header['npitch']
ngyr = smap.strike_points.header['ngyr']

# Calculate the std of the markers in pitch
sigma_along_pitch = np.zeros(npitch)
ig = np.argmin(np.abs(smap.unique_gyroradius - r0))
rtrue = smap.unique_gyroradius[ig]
icolum = smap.strike_points.header['info']['remap_rl']['i']
for i in range(npitch):
    sigma_along_pitch[i] = smap.strike_points.data[i, ig][:, icolum].std()
# Calculate the std of the markers in rl
sigma_along_gyr = np.zeros(ngyr)
ip = np.argmin(np.abs(smap.unique_pitch - p0))
ptrue = smap.unique_pitch[ip]
for i in range(ngyr):
    sigma_along_gyr[i] = smap.strike_points.data[ip, i][:, icolum].std()

# Get the sigmas from the interpolation
rl = np.linspace(smap.unique_gyroradius[0], smap.unique_gyroradius[-1])
pitch = np.linspace(smap.unique_pitch[0], smap.unique_pitch[-1])
sigma_along_gyr_inter = \
    smap.interpolators['gyroradius']['sigma'](rl, ptrue * np.ones(rl.size))
sigma_along_pitch_inter = \
    smap.interpolators['gyroradius']['sigma'](rtrue * np.ones(pitch.size), pitch)

# -----------------------------------------------------------------------------
# --- Plot the comparison
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2)

# Sigma rl along r
ax[0].plot(smap.unique_gyroradius, sigma_along_gyr, 'ob')
ax[0].plot(rl, sigma_along_gyr_inter, 'k')
ax[0].set_xlabel('Gyroradius [cm]')
ax[0].set_ylabel('$\\sigma_ {r} [cm]$')
# Sigma rl along pitch
ax[1].plot(smap.unique_pitch, sigma_along_pitch, 'ob')
ax[1].plot(pitch, sigma_along_pitch_inter, 'k')
ax[1].set_xlabel('Pitch [deg]')
ax[1].set_ylabel('$\\sigma_ {r} [cm]$')
fig.show()
