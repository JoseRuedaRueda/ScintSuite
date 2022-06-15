"""
Calculate TAE frequency, really basic approximation

Done for version 0.5.1
"""
import Lib as ss
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as interp
import os


# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
shot = 39612
path_to_pyspec = \
    '/afs/ipp/home/r/ruejo/AUG_39612_diag:_B31_sig:_B31-14.npz_FILES'
# Set to None to ignore pysepct data

rho_to_overplot = [0.20, 0.35, 0.6]

# -----------------------------------------------------------------------------
# --- Get profiles
# -----------------------------------------------------------------------------
ne = ss.dat.get_ne_ida(shot)

data = ss.dat.get_shot_basics(shot)
# -----------------------------------------------------------------------------
# --- Calculate Plot / TAE frequency
# -----------------------------------------------------------------------------
B_interp = interp(data['bttime'], np.abs(data['bt0']))
B = B_interp(ne['time'])
R = interp(data['time'], np.abs(data['Rmag']))(ne['time'])
f = np.zeros(ne['data'].shape)  # Nt, Nrho
q = ss.dat.get_q_profile(shot)
for i in range(ne['data'].shape[0]):
    # Find the closer q profile
    it = np.argmin(np.abs(q['time'] - ne['time'][i]))
    # Interpolate it in the density rho
    flags = q['rhop'][it, :] < 0.98
    qq = interp(q['rhop'][it, flags], np.abs(q['data'][it, flags]),
                fill_value='extrapolate')(ne['rhop'])
    f[i, :] = B[i] / np.sqrt(ne['data'][i, :]) / R[i] / qq / 4.0 / np.pi
f *= 1 / np.sqrt(4*np.pi*1.e-7 * 2. * ss.par.amu2kg)
f = f.T
# Contour plot
fig1, ax1 = plt.subplots()
contf = ax1.contourf(ne['time'], ne['rhop'], f, 50, vmin=80000, vmax=200000)
plt.colorbar(contf, ax=ax1)

# -----------------------------------------------------------------------------
# --- Load and Plot PySpecView spectra
# -----------------------------------------------------------------------------
if path_to_pyspec is not None:
    # - tvec
    file = os.path.join(path_to_pyspec, 'tvec.npy')
    tvec = np.load(file)
    # - fvec
    file = os.path.join(path_to_pyspec, 'fvec.npy')
    fvec = np.load(file)
    # - fvec
    file = os.path.join(path_to_pyspec, 'spect.npy')
    spect = np.load(file)
    # - Plot
    fig3, ax3 = plt.subplots()
    ax3.imshow(spect, extent=[tvec[0], tvec[-1], fvec[0], fvec[-1]],
               aspect='auto', origin='lower', cmap=ss.plt.Gamma_II(),
               label='__nolegend__')
    for r in rho_to_overplot:
        ir = np.argmin(abs(ne['rhop'] - r))
        y = sp.savgol_filter(f[ir, :], window_length=5, polyorder=3)
        ax3.plot(ne['time'], y, label='$\\rho = $' + str(round(r, 2)))
    ax3.legend()
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Frequency [Hz]')
plt.show()
