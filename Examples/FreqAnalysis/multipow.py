"""
Multi-pow analysis.

This script makes the cross-power spectral density calculation of the ECE and
the magnetics.

Pablo Oyola - pablo.oyola@ipp.mpg.de
revised by Jose Rueda Rueda for version 1.2.2
"""
import warnings
import matplotlib

import Lib as ss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from tqdm import tqdm
from Lib._FrequencyAnalysis import stft, sfft, get_nfft, myCPSD, stft2


# -----------------------------------------------------------------------------
# %% Scripts parameter definition.
# -----------------------------------------------------------------------------
# Shot data and timing.
shotnumber = 41090
coilNumber = 14
tBegin = 3.4
tEnd   = 3.5

# FFT options.
#                        # For the window type, go to:
windowType = 'hann'      # https://docs.scipy.org/doc/scipy/reference/
#                        # generated/scipy.signal.get_window.html
freqLims = np.array((80.0, 120.0))  # Frequency limits.
freq1d = np.array((6.0, 8.0))  # Frequency limits for the plot in 1D.
specType = 'stft'  # Spectogram type:
#                  # -> Short-Time Fourier Transform in frequency (sfft)
#                  # -> Short-Time Fourier Transform in time (stft)
resolution = int(1000)
timeResolution = 0.80  # Frequency resolution.
cmap = matplotlib.cm.plasma  # Colormap

# Diagnostic for the electron gradient reading.
diag_Te = 'IDA'
exp_Te  = 'AUGD'
ed_Te   = 0

# Plotting flags:
plot_spectrograms   = True # Plot the spectrogram for MHI and ECE and the
#                          # sample CPSD for cross-checking.
plot_Te_sample      = False   # Plots the Te data as taken from the RMD.
plot_profiles       = False    # Plot the rhopol vs. f spectrogram.
plot_vessel_flag    = False    # Plots the vessel and the separatrix
#                              # with the ECE and pick up coil position.

# Plotting options.
ece_rhop_plot = 0.90  # rho_pol of the ECE to plot.
spec_abstype = 'linear'  # linear, sqrt or log
spec_abstype_cpsd = 'linear'  # This is the same as above, but for the CPSD.

# -----------------------------------------------------------------------------
# %% Reading the data from the database.
# -----------------------------------------------------------------------------
mhi = ss.dat.get_magnetics(shotnumber, coilNumber, timeWindow=[tBegin, tEnd])
ece = ss.dat.get_ECE(shotnumber, timeWindow=[tBegin, tEnd], fast=True)

# -----------------------------------------------------------------------------
# --- Magnetic spectrogram -  Calculation
# -----------------------------------------------------------------------------
# This assumes that the data is uniformly taken in time.
dt = mhi['time'][1] - mhi['time'][0]  # For the sampling time.

nfft = get_nfft(timeResolution, specType, len(mhi['time']), windowType, dt)
nfft = int(nfft)

print('Computing the magnetics spectrogram...')
warnings.filterwarnings('ignore', category=DeprecationWarning)
if specType == 'stft':
    Sxx, freqs, times = stft(mhi['time']-tEnd,  mhi['data'], nfft,
                             window=windowType,
                             tmin=tBegin-tEnd, tmax=0.00,
                             fmin=freqLims[0]*1000.0,
                             fmax=freqLims[-1]*1000.0,
                             pass_DC=False, complex_spectrum=True)
elif specType == 'sfft':
    Sxx, freqs, times = sfft(mhi['time']-tEnd,  mhi['data'], nfft,
                             window=windowType,
                             tmin=tBegin-tEnd, tmax=0.00,
                             fmin=freqLims[0]*1000.0,
                             fmax=freqLims[-1]*1000.0,
                             pass_DC=False, complex_spectrum=True)
elif specType == 'stft2':
    Sxx, freqs, times = stft2(mhi['time']-tEnd,  mhi['data'], nfft,
                              window=windowType,
                              pass_DC=False, complex_spectrum=True,
                              resolution=resolution)
    f0, f1 = np.searchsorted(freqs, freqLims.copy()*1000.0)
    freqs = freqs[f0:f1]
    Sxx = Sxx[:, f0:f1]

mhi['fft'] = {
                'freq': freqs.copy()/1000.0,
                'time': times+tEnd,
                'spec': Sxx.T
             }
# The MHI data is related to the time variation of the magnetic field.
# In Fourier space, the derivative is Bdot ~ \omega · B
Bdot = mhi['fft']['spec'].copy()
for jj in np.arange(Bdot.shape[1]):
    Bdot[1:, jj] = Bdot[1:, jj]/(mhi['fft']['freq'][1:]*1000.0)

#mhi['fft']['freq'] = mhi['fft']['freq'][1:]
mhi['fft']['B'] = Bdot/Bdot.max()
# Applying the correction.
if 'phase_corr' in mhi:
    print('Applying phase-correction to the spectrogram.')
    mhi['fft']['B'] *= np.tile(np.exp(1j*mhi['phase_corr']['interp'](freqs/1000.)),
                               (len(times), 1)).T
    mhi['fft']['spec'] *= np.tile(np.exp(1j*mhi['phase_corr']['interp'](freqs/1000.)),
                               (len(times), 1)).T
del Bdot
#%% -----------------------------------------------------------------------------
# --- ECE spectrogram - Calculation
# -----------------------------------------------------------------------------
dt = ece['time'][1] - ece['time'][0]

print('Building spectrograms for the ECE...')

nfft = get_nfft(timeResolution, specType, len(ece['time']), windowType, dt)
nfft = int(nfft)

#del Sxx
for ii in tqdm(np.arange(ece['Trad_norm'].shape[1])):
    if specType == 'stft':
        Syy, freqs, times = stft(ece['time']-tEnd,
                                 ece['Trad_norm'][:, ii].T.astype(np.single),
                                 nfft,
                                 window=windowType,
                                 tmin=tBegin-tEnd, tmax=0.0,
                                 fmin=freqLims[0]*1000.0,
                                 fmax=freqLims[1]*1000.0,
                                 pass_DC=False, complex_spectrum=True)
    elif specType == 'sfft':
        Syy, freqs, times = sfft(ece['time']-tEnd,
                                 ece['Trad_norm'][:, ii].T,
                                 nfft,
                                 window=windowType,
                                 tmin=tBegin-tEnd, tmax=0.0,
                                 fmin=freqLims[0]*1000.0,
                                 fmax=freqLims[1]*1000.0,
                                 pass_DC=False, complex_spectrum=True)
    elif specType == 'stft2':
        Syy, freqs, times = stft2(ece['time']-tEnd,
                                  ece['Trad_norm'][:, ii].T,
                                  nfft,
                                  window=windowType,
                                  pass_DC=False, complex_spectrum=True,
                                  resolution=resolution)
        f0, f1 = np.searchsorted(freqs, freqLims.copy()*1000.0)
        freqs = freqs[f0:f1]
        Syy = Syy[:, f0:f1]
        Syy /= Syy.max()
    if ii == 0:
        Sxx = Syy
    else:
        Sxx = np.dstack((Sxx, Syy))
    del Syy

ece['fft'] = {
                'freq': freqs.copy()/1000.0,
                'time': times.copy()+tEnd,
                'spec': Sxx
              }

# Correct the spectra with the dTe_dr (derivative wrt the major radius!)
ece = ss.dat.correctShineThroughECE(ece, diag=diag_Te,
                                    exp=exp_Te, edition=ed_Te)
warnings.filterwarnings('default')


#%% -----------------------------------------------------------------------------
# --- Spectrogram plotting.
# -----------------------------------------------------------------------------
if plot_spectrograms:
    plt.ion()

    # --- Plotting the Magnetic spectrogram.
    fig, ax = plt.subplots(nrows=2, ncols=2)
    mhiplot = np.abs(mhi['fft']['spec'])
    if spec_abstype == 'linear':
        im1 = ax[0][0].imshow(mhiplot, origin='lower',
                              extent=(mhi['fft']['time'][0],
                                      mhi['fft']['time'][-1],
                                      mhi['fft']['freq'][0],
                                      mhi['fft']['freq'][-1]),
                              aspect='auto', interpolation='nearest',
                              cmap=cmap)
    elif spec_abstype == 'log':
        im1 = ax[0][0].imshow(mhiplot, origin='lower',
                              extent=(mhi['fft']['time'][0],
                                      mhi['fft']['time'][-1],
                                      mhi['fft']['freq'][0],
                                      mhi['fft']['freq'][-1]),
                              aspect='auto', interpolation='nearest',
                              norm=colors.LogNorm(mhiplot.min(), mhiplot.max()),
                              cmap=cmap)
    elif spec_abstype == 'sqrt':
        im1 = ax[0][0].imshow(mhiplot, origin='lower',
                              extent=(mhi['fft']['time'][0],
                                      mhi['fft']['time'][-1],
                                      mhi['fft']['freq'][0],
                                      mhi['fft']['freq'][-1]),
                              aspect='auto', interpolation='nearest',
                              cmap=cmap, norm=colors.PowerNorm(gamma=0.50))

    ax[0][0].set_title('B-coil:' + str(coilNumber))
    ax[0][0].set_xlabel('Time [s]')
    ax[0][0].set_ylabel('Frequency [kHz]')
    fig.colorbar(im1, ax=ax[0][0])

    # Plotting the ECE.
    # Find the nearest channel.
    nchann = (np.abs(ece['rhop'] - ece_rhop_plot)).argmin()

    print('Plotting TradA:'+str(ece['channels'][nchann]))
    print('rho_pol = '+str(ece['rhop'][nchann]))

    # --- Plotting ECE sample channel.
    eceplot = np.abs(ece['fft']['spec'][:, :, nchann]).T

    if spec_abstype == 'linear':
        im2 = ax[0][1].imshow(eceplot, origin='lower',
                              extent=(ece['fft']['time'][0],
                                      ece['fft']['time'][-1],
                                      ece['fft']['freq'][0],
                                      ece['fft']['freq'][-1]),
                              aspect='auto', interpolation='nearest',
                              cmap=cmap)
    elif spec_abstype == 'log':
        im2 = ax[0][1].imshow(eceplot, origin='lower',
                              extent=(ece['fft']['time'][0],
                                      ece['fft']['time'][-1],
                                      ece['fft']['freq'][0],
                                      ece['fft']['freq'][-1]),
                              aspect='auto', interpolation='nearest',
                              norm=colors.LogNorm(eceplot.min(),
                                                  eceplot.max()),
                              cmap=cmap)
    elif spec_abstype == 'sqrt':
        im2 = ax[0][1].imshow(eceplot, origin='lower',
                              extent=(ece['fft']['time'][0],
                                      ece['fft']['time'][-1],
                                      ece['fft']['freq'][0],
                                      ece['fft']['freq'][-1]),
                              aspect='auto', interpolation='nearest',
                              cmap=cmap,
                              norm=colors.PowerNorm(gamma=0.50))

    ax[0][1].set_title('TradA:'+str(ece['channels'][nchann]) +
                       ' - $\\rho_{pol}$ = ' + str(ece['rhop'][nchann]))
    ax[0][1].set_xlabel('Time [s]')
    fig.colorbar(im2, ax=ax[0][1])

    # --- Plotting the correlation matrix in (freq, time).
    t, freq, A = myCPSD(mhi['fft']['B'],
                        ece['fft']['spec'][:, :, nchann].T,
                        mhi['fft']['time'], mhi['fft']['freq'],
                        ece['fft']['time'], ece['fft']['freq'])

    xcor_plot = np.abs(A)

    if spec_abstype == 'linear':
        im3 = ax[1][0].imshow(xcor_plot, origin='lower',
                              extent=(t[0], t[-1], freq[0], freq[-1]),
                              aspect='auto', interpolation='nearest',
                              cmap=cmap)
    elif spec_abstype == 'log':
        im3 = ax[1][0].imshow(xcor_plot, origin='lower',
                              extent=(t[0], t[-1], freq[0], freq[-1]),
                              aspect='auto', interpolation='nearest',
                              norm=colors.LogNorm(eceplot.min(), eceplot.max()),
                              cmap=cmap)
    elif spec_abstype == 'sqrt':
        im3 = ax[1][0].imshow(xcor_plot, origin='lower',
                              extent=(t[0], t[-1], freq[0], freq[-1]),
                              aspect='auto', interpolation='nearest',
                              cmap=cmap, norm=colors.PowerNorm(gamma=0.50))

    ax[1][0].set_title('Cross-correlation')
    ax[1][0].set_xlabel('Time [s]')
    fig.colorbar(im3, ax=ax[1][0])

    # --- Plotting correlation phase.
    xcor_phase = np.angle(A)
    im4 = ax[1][1].imshow(xcor_phase, origin='lower',
                          extent=(t[0], t[-1], freq[0], freq[-1]),
                          aspect='auto', interpolation='nearest', cmap=cmap,
                          vmin=-np.pi, vmax=np.pi)

    ax[1][1].set_title('Cross-correlation (Phase)')
    ax[1][1].set_xlabel('Time [s]')
    fig.colorbar(im4, ax=ax[1][1])
    plt.tight_layout()
    del Sxx
    del xcor_phase
    del A
    del xcor_plot
    del eceplot
    del mhiplot

#%% -----------------------------------------------------------------------------
# --- ECE data plotting.
# -----------------------------------------------------------------------------
if plot_Te_sample:
    fig1, ax1 = plt.subplots(nrows=1, ncols=2)

    if len(ece['time']) > 256:
        downsample = len(ece['time'])//256
    else:
        downsample = 1

    ss.plt.plot2D_ECE(ece, rType='channels',
                      downsample=downsample, ax=ax1[0],
                      cmap=cmap, fig=fig1, which='norm')
    ss.plt.plot2D_ECE(ece, rType='rho_pol',
                      downsample=downsample, ax=ax1[1],
                      cmap=cmap, fig=fig1, which='norm')

    plt.tight_layout()

#%% -----------------------------------------------------------------------------
# --- Computing the radial correlation.
# -----------------------------------------------------------------------------
for ii in tqdm(np.arange(ece['rhop'].shape[0])):
    A = myCPSD(mhi['fft']['spec'],
               ece['fft']['spec'][:, :, ii].T,
               mhi['fft']['time'], mhi['fft']['freq'],
               ece['fft']['time'], ece['fft']['freq'])[2]

    if ii == 0:
        Sxx = A
    else:
        Sxx = np.dstack((Sxx, A))

ece['xrel'] = {
    'time': mhi['fft']['time'],
    'freq': mhi['fft']['freq'],
    'rho':  ece['rhop'],
    'data': Sxx,
    'data2D': np.sqrt(np.sum(np.abs(Sxx)**2.0, axis=1)),  # [freq, rho_pol]
    'desc': 'Cross-power spectral density ECE - \
             MHI for all channels',
    'short': '$\\delta B_r \\ast \\delta T_e$'
}

# -----------------------------------------------------------------------------
# --- Plotting the cross-correlation.
# -----------------------------------------------------------------------------
if plot_profiles:
    fig3, ax3 = plt.subplots(nrows=1, ncols=2)

    opts = {
        'cmap': cmap,
        'shading': 'flat',
        'antialiased': True
    }

    if spec_abstype_cpsd == 'log':
        opts['norm'] = colors.LogNorm(ece['xrel']['data2D'].min(),
                                      ece['xrel']['data2D'].max())
    elif spec_abstype_cpsd == 'sqrt':
        opts['norm'] = colors.PowerNorm(gamma=0.50)

    im5 = ax3[0].pcolormesh(ece['xrel']['rho'], ece['xrel']['freq'],
                            ece['xrel']['data2D'], **opts)

    ax3[0].set_xlim([0, 1.0])
    ax3[0].set_title('Cross-correlation vs. rhopol')
    ax3[0].set_xlabel('$\\rho_{pol}$')
    ax3[0].set_ylabel('Frequency [kHz]')
    fig3.colorbar(im3, ax=ax3[0])


    # --- Plotting the 1D profile.
    f0 = (np.abs(ece['xrel']['freq'] - freq1d[0])).argmin()
    f1 = (np.abs(ece['xrel']['freq'] - freq1d[-1])).argmin()
    ece_1D = np.sqrt(np.sum(ece['xrel']['data2D'][f0:f1, :]**2, axis=0))

    rhoplot, eceplot1d = zip(*sorted(zip(ece['xrel']['rho'], ece_1D)))

    rhoplot = np.array(rhoplot)
    eceplot1d = np.array(eceplot1d)
    ax3[1].plot(rhoplot, eceplot1d/eceplot1d.max(), 'r*-',
                label='Cross-correlation ECE-MHI')
    ax3[1].set_title('f $\\in$ [%.1f, %.1f] kHz'%(ece['xrel']['freq'][f0],
                                                  ece['xrel']['freq'][f1]))


    # --- Plotting the normalized electron temperature gradient.
    ax3[1].plot(ece['fft']['dTe_base'],
                ece['fft']['dTe']/ece['fft']['dTe'].max(), 'b-',
                label='Electron temperature gradient.')

    ax3[1].set_xlabel('$\\rho_{pol}$')
    ax3[1].set_ylabel('Amplitude [au]')

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# --- Plotting the vessel and the ECE positions.
# -----------------------------------------------------------------------------
if plot_vessel_flag:
    fig4, ax4 = plt.subplots(1)

    ax4 = ss.plt.plot_vessel(projection='pol', shot=shotnumber, ax=ax4)

    # Getting the flux surfaces.
    ax4 = ss.plt.plot_flux_surfaces(shotnumber, (tBegin+tEnd)/2.0,
                                    ax=ax4)


    im6 = plt.scatter(x=ece['r'],  y=ece['z'], c=ece['channels'],
                      label='ECE positions', cmap='hsv', alpha=0.75)

    cbar = plt.colorbar(im6, ax=ax4)

    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('ECE Channel number', rotation=270)
    ax4.axis('equal')

    ax4.plot(mhi['R'], mhi['z'], 'k.', markersize=12,
             label='Magnetic coil')

    plt.legend()
    plt.tight_layout()
    plt.show()
