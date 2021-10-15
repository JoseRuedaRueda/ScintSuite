"""
Lecture 2 of the FILDSIM course

Compare resolutions with geometric variable

This example will load several FILDSIM simulations to compare its resolution
at a given value of gyroradius pitch

Jose Rueda Rueda: jrrueda@us.es
"""
import Lib as ss
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
simulations = {
    'files': [
        '/afs/ipp/home/r/ruejo/FILDSIM/results/10_width_strike_map.dat',
        '/afs/ipp/home/r/ruejo/FILDSIM/results/9_width_strike_map.dat',
        '/afs/ipp/home/r/ruejo/FILDSIM/results/8_width_strike_map.dat',
        '/afs/ipp/home/r/ruejo/FILDSIM/results/7_width_strike_map.dat',
        '/afs/ipp/home/r/ruejo/FILDSIM/results/6_width_strike_map.dat',
        '/afs/ipp/home/r/ruejo/FILDSIM/results/5_width_strike_map.dat',
    ],
    'width': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
}
p0 = 50.0  # Focus pitch [º] Resolution will be plot along this pitch
r0 = 3.3   # focus rl [cm] Resolution will be plot along this radius
experiment = 0.4
# -----------------------------------------------------------------------------
# --- Load the strike points and calculate the resolutions
# -----------------------------------------------------------------------------
nsims = len(simulations['files'])
sigmas = np.zeros(nsims)
for i in range(nsims):
    smap = ss.mapping.StrikeMap(0, simulations['files'][i])
    # Note: The strike point file is supposed to be in the same folder that the
    # strike map
    smap.load_strike_points()
    ig = np.argmin(np.abs(smap.unique_gyroradius - r0))
    ip = np.argmin(np.abs(smap.unique_pitch - p0))
    ptrue = smap.unique_pitch[ip]
    icolum = smap.strike_points.header['info']['remap_rl']['i']
    rtrue = smap.unique_gyroradius[ig]
    sigmas[i] = smap.strike_points.data[ip, ig][:, icolum].std()
    plt.hist(smap.strike_points.data[ip, ig][:, icolum])
    plt.ginput(1, 0.1)

# -----------------------------------------------------------------------------
# --- Plot the comparison
# -----------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(simulations['width'], sigmas)
ax.set_xlabel('Pinhole Width [cm]')
ax.set_ylabel('$\\sigma_r [cm]$')
ax.plot([simulations['width'][0], simulations['width'][-1]],
        [experiment, experiment], '--r')
