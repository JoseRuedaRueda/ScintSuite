"""
Plotting a magnetic pick-up spectrogram.

This example shows an example on how to use the frequency tracker as
implemented in the LibFrequencyAnalysis.py. It also provides,
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
import numpy as np
import Lib as ss
import Lib.LibFrequencyAnalysis as lf

# -----------------------------------------------------------------------------
# --- Scripts parameter definition.
# -----------------------------------------------------------------------------
# Shot data and timing.
shotnumber = 34430
tBegin     = 5.60
tEnd       = 5.90
coilNumber = 31

# FFT options.
                        # For the window type, go to:
windowType = 'hann'     # https://docs.scipy.org/doc/scipy/reference/
                        # generated/scipy.signal.get_window.html
freqLimit = np.array((1.0, 350.0)) # Frequency limits.
specType = 'stft' # Spectogram type:
                  # -> Short-Time Fourier Transform in frequency (sfft)
                  # -> Short-Time Fourier Transform in time (stft)
resolution = int(1000)
timeResolution = 0.70 # Time resolution.
cmap = matplotlib.cm.plasma # Colormap
spec_abstype = 'log' # linear, sqrt or log


# -----------------------------------------------------------------------------
# --- Reading the data from the database.
# -----------------------------------------------------------------------------
if (not ('mhi' in locals())) and (not('mhi' in globals())):
    mhi = ss.dat.get_magnetics(shotnumber, coilNumber,
                               timeWindow=[tBegin, tEnd])

# -----------------------------------------------------------------------------
# --- Magnetic spectrogram -  Calculation
# -----------------------------------------------------------------------------
# This assumes that the data is uniformly taken in time.
dt = mhi['time'][1] - mhi['time'][0] # For the sampling time.

nfft = int(lf.get_nfft(timeResolution, specType,
                       len(mhi['time']), windowType, dt))

print('Computing the magnetics spectrogram...')
warnings.filterwarnings('ignore', category=DeprecationWarning)
if specType == 'stft':
    Sxx, freqs, times = lf.stft(mhi['time'],  mhi['data'], nfft,
                             window=windowType,
                             pass_DC=False, complex_spectrum=True,
                             resolution=resolution)
elif specType == 'sfft':
    Sxx, freqs, times = lf.sfft(mhi['time']-tEnd,  mhi['data'], nfft,
                             window=windowType,
                             tmin=tBegin-tEnd, tmax=0.00,
                             fmin=freqLimit[0]*1000.0,
                             fmax=freqLimit[-1]*1000.0,
                             pass_DC=False, complex_spectrum=True,
                             resolution=resolution)

elif specType == 'stft2':
    Sxx, freqs, times = lf.stft2(mhi['time']-tEnd,  mhi['data'], nfft,
                                 window=windowType,
                                 pass_DC=False, complex_spectrum=True,
                                 resolution=resolution)
mhi['fft'] = {
                'freq': freqs/1000.0,
                'time': times+tEnd,
                'spec': Sxx
             }
# The MHI data is related to the time variation of the magnetic field.
# In Fourier space, the derivative is Bdot ~ \omega · B
Bdot = mhi['fft']['spec'].T
f0, f1 = mhi['fft']['freq'].searchsorted(freqLimit)

warnings.filterwarnings('ignore', category=RuntimeWarning)
for jj in np.arange(Bdot.shape[1]):
    Bdot[:, jj] = Bdot[:, jj]/(mhi['fft']['freq']*1000.0)

Bdot[0, :] = 0.0j
mhi['fft']['B'] = Bdot

# Applying the correction.
if 'phase_corr' in mhi:
    print('Applying phase-correction to the spectrogram.')
    mhi['fft']['B'] *= np.tile(np.exp(1j*mhi['phase_corr']['interp'](freqs)),
                               (len(times), 1)).T
warnings.filterwarnings('default')
del Bdot

# -----------------------------------------------------------------------------
# --- Spectrogram plotting.
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1)
mhiplot = np.abs(mhi['fft']['B'])[f0:f1, :]


if spec_abstype == 'linear' or spec_abstype == 'lin':
     im1 = ax.imshow(mhiplot, origin='lower',
                     extent=(mhi['fft']['time'][0],
                             mhi['fft']['time'][-1],
                             mhi['fft']['freq'][f0],
                             mhi['fft']['freq'][f1]),
                     aspect='auto', interpolation='nearest', cmap=cmap)
elif spec_abstype == 'log':
     im1 = ax.imshow(mhiplot, origin='lower',
                     extent=(mhi['fft']['time'][0],
                             mhi['fft']['time'][-1],
                             mhi['fft']['freq'][f0],
                             mhi['fft']['freq'][f1]),
                     aspect='auto', interpolation='nearest',
                     norm=colors.LogNorm(mhiplot.min(), mhiplot.max()),
                     cmap = cmap)
elif spec_abstype == 'sqrt':
    im1 = ax.imshow(mhiplot, origin='lower',
                     extent=(mhi['fft']['time'][0],
                             mhi['fft']['time'][-1],
                             mhi['fft']['freq'][f0],
                             mhi['fft']['freq'][f1]),
                     aspect='auto', interpolation='nearest', cmap=cmap,
                     norm=colors.PowerNorm(gamma=0.50))

ax.set_title('B-coil:' + str(coilNumber))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Frequency [kHz]')
plt.show()
