"""
Calculate BAE frequency

Done for version 0.5.1
"""
import ScintSuite.as ss
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as interp
import os
# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
shot = 39612
R0 = 1.67
Ti_equal_Te = True
path_to_pyspec = \
    '/afs/ipp/home/r/ruejo/AUG_39612_diag:_B31_sig:_B31-14.npz_FILES'
# Set to None to ignore pysepct data

rho_to_overplot = [0.20, 0.35, 0.6]

# -----------------------------------------------------------------------------
# --- Get profiles
# -----------------------------------------------------------------------------
Te = ss.dat.get_Te(shot)
if not Ti_equal_Te:
    Ti = ss.dat.get_Ti_cxrs(shot)

# -----------------------------------------------------------------------------
# --- Calculate Plot / BAE frequency
# -----------------------------------------------------------------------------
if not Ti_equal_Te:
    # We will interpolate all to Ti rho and time. As IDA has a ms timebase,
    # we will ust take the closer point in time of IDA and then interpolate in
    # rho Elongaton nor q profile corrections where applied
    freq = np.zeros(Ti['fit']['data'].shape)
    for it in range(Ti['fit']['time'].size):
        iit = np.argmin(abs(Te['time'] - Ti['fit']['time'][it]))
        te_interp = interp(Te['rhop'], Te['data'][iit, :])
        freq[it, :] = te_interp(Ti['fit']['rhop']) + 7/4 * Ti['fit']['data'][it, :]

    freq /= 2 * np.pi**2 * R0**2 * 2 * ss.par.mp_kg / 1.609e-19
    freq = np.sqrt(freq.T)
    # Contour plot
    fig1, ax1 = plt.subplots()
    contf = ax1.contourf(Ti['fit']['time'], Ti['fit']['rhop'], freq, 50)
    plt.colorbar(contf, ax=ax1)
    # Line plots
    fig2, ax2 = plt.subplots()
    for rho in np.linspace(0.05, 0.95, 10):
        ir = np.argmin(abs(Ti['fit']['rhop'] - rho))
        y = sp.savgol_filter(freq[ir, :], window_length=5, polyorder=3)
        ax2.plot(Te['time'], y, label='$\\rho = $' + str(rho))
    plt.legend()
else:
    # We will interpolate all to Ti rho and time. As IDA has a ms timebase,
    # we will ust take the closer point in time of IDA and then interpolate in
    # rho Elongaton nor q profile corrections where applied
    freq = 11/4 * Te['data']
    freq /= 2 * np.pi**2 * R0**2 * 2 * ss.par.mp_kg / 1.609e-19
    freq = np.sqrt(freq.T)
    # Contour plot
    fig1, ax1 = plt.subplots()
    contf = ax1.contourf(Te['time'], Te['rhop'], freq, 50)
    plt.colorbar(contf, ax=ax1)
    # Line plots
    fig2, ax2 = plt.subplots()
    for rho in np.linspace(0.05, 0.95, 10):
        ir = np.argmin(abs(Te['rhop'] - rho))
        y = sp.savgol_filter(freq[ir, :], window_length=5, polyorder=3)
        ax2.plot(Te['time'], y, label='$\\rho = $' + str(round(rho, 2)))
    plt.legend()

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
        ir = np.argmin(abs(Te['rhop'] - r))
        y = sp.savgol_filter(freq[ir, :], window_length=5, polyorder=3)
        ax3.plot(Te['time'], y, label='$\\rho = $' + str(round(r, 2)))
    ax3.legend()
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Frequency [Hz]')
plt.show()
