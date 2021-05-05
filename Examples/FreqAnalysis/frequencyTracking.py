"""
Frequency tracking example.

This example shows an example on how to use the frequency tracker as 
implemented in the LibFrequencyAnalysis.py. It also provides, 
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
import numpy as np
import Lib as ss
import LibFrequencyAnalysis as lf
from LibPlotting import p1D_shaded_error as plot_error_band
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# --- Scripts parameter definition.
# -----------------------------------------------------------------------------
# Shot data and timing.
shotnumber = 38663
tBegin     = 1.60
tEnd       = 1.90
coilNumber = 31

# FFT options.
                        # For the window type, go to:
windowType = 'hann'     # https://docs.scipy.org/doc/scipy/reference/
                        # generated/scipy.signal.get_window.html
freqLimit = np.array((2.0, 14.0)) # Frequency limits.
specType = 'stft2' # Spectogram type:
                  # -> Short-Time Fourier Transform in frequency (sfft)
                  # -> Short-Time Fourier Transform in time (stft)
resolution = int(2000)
timeResolution = 0.70 # Time resolution.
cmap = matplotlib.cm.plasma # Colormap
spec_abstype = 'lin' # linear, sqrt or log
sigma = 1.0 # When filtering the spectrogram, this decides how many widths to
            # take.

# -----------------------------------------------------------------------------
# --- Configuration of the tracker. 
# -----------------------------------------------------------------------------
peakOpts = { 'prominence': 0.05, # This sets the relative height to consider a 
                                 # peak. This needs to be varied in case the 
                                 # frequency path is not the one with highest
                                 # amplitude. An array like (0.4, 0.5) would 
                                 # choose the appropriate range.
             'rel_height': 0.50  # Relative height (to the peak) at which the
                                 # height is measured.
           }

timeConnection = 3.0e-3 # Allow for peaks to be connected within this time
                        # range.
freqThreshold = 10.0 # Jumps above this frequency are disregarded.

verbose = False # Write to output the process of peak search.

limsFromGUI = False # Retrieve the starting point and other from the GUI.
timeMin = 1.601
freqmin = 7.0
freqmax = 14.0
freqOrigin = 10.0

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
                     spect='auto', interpolation='nearest', cmap=cmap,
                     norm=colors.PowerNorm(gamma=0.50))
    
ax.set_title('B-coil:' + str(coilNumber))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Frequency [kHz]')
plt.show()

# -----------------------------------------------------------------------------
# --- Asking the user for starting and ending points in the spectrum.
# -----------------------------------------------------------------------------
if limsFromGUI:
    print('Click on the spectogram to select and starting time point and')
    print('frequency.')
    a = plt.ginput(timeout=0)
    
    timeMin = a[0][0]
    freqOrigin = a[0][1]
    
    ax.plot(timeMin, freqOrigin, 'bo', label='Origin')
    plt.show()
    print('Click to select the highest frequency by clicking on the spectrogram')
    
    (_, freqmax) = plt.ginput(timeout=0)[0]
    
    ax.plot([timeMin, mhi['fft']['time'][-1]], [freqmax, freqmax], 'k--', 
            label='Maximum frequency.')
    
    plt.show()
    print('Click to select the lowest frequency by clicking on the spectrogram')
    (_, freqmin) = plt.ginput(timeout=0)[0]
        
    ax.plot([timeMin, mhi['fft']['time'][-1]], [freqmin, freqmin], 'k--', 
            label='Minmum frequency.')
    
    plt.show()
# -----------------------------------------------------------------------------
# --- Calling the frequency tracking.
# -----------------------------------------------------------------------------
time = mhi['fft']['time']
freq = mhi['fft']['freq']
data = np.abs(mhi['fft']['B'])

origin = freqOrigin

timeLims = [timeMin, tEnd]
freqLims = [freqmin, freqmax]

trk, ax = lf.trackFrequency(time=time, freq=freq, spec=data, 
                            origin=origin, plot=True, 
                            peak_opts=peakOpts, 
                            graph_TimeConnect=timeConnection, 
                            freqLims=freqLims, timeLims=timeLims, 
                            freqThr=freqThreshold, ax=ax, fig=fig)



# -----------------------------------------------------------------------------
# --- Plotting the frequency curve.
# -----------------------------------------------------------------------------
lineOpts = { 'color': 'w'}
ax=plot_error_band(ax=ax, x=trk['track']['time'],
                   y=trk['track']['freq'], color='w', 
                   u_up=trk['track']['width']/2.0, 
                   alpha=0.2, line=True, line_param=lineOpts)



# -----------------------------------------------------------------------------
# --- Plotting the amplitude of the spectogram.
# -----------------------------------------------------------------------------
fig2, ax2 = plt.subplots(nrows=2)

ax2[0].plot(trk['track']['time'], trk['track']['Atot'], 'r-',
         label='Amplitude at max.')


# -----------------------------------------------------------------------------
# --- Filtering
# -----------------------------------------------------------------------------
y_up = trk['track']['width']/2.0*sigma + trk['track']['freq']
y_down = -trk['track']['width']/2.0*sigma + trk['track']['freq']

y_up_fun = interp1d(trk['track']['time'], y_up, kind='linear',
                    bounds_error=False, fill_value='extrapolate',
                    assume_sorted=True)

y_down_fun = interp1d(trk['track']['time'], y_down, kind='linear',
                    bounds_error=False, fill_value='extrapolate',
                    assume_sorted=True)

y_center_fun = interp1d(trk['track']['time'], trk['track']['freq'], 
                        kind='linear', bounds_error=False, 
                        fill_value='extrapolate', assume_sorted=True)
s_center_fun = interp1d(trk['track']['time'], sigma*trk['track']['width']/2.772, 
                        kind='linear', bounds_error=False, 
                        fill_value='extrapolate', assume_sorted=True)
time2, freq2 = np.meshgrid(time, freq)
mask = (freq2 < y_up_fun(time2)) & (freq2 > y_down_fun(time2))

new_B = mhi['fft']['spec']
mhi_new = { 'fft': { 'mask': mask,
                     'time': time,
                     'freq': freq,
                     'B': new_B
                   }
          }

# --- Using a Gaussian filter around the central frequency.
filter_gauss = np.exp(-(freq2-y_center_fun(time2))**2.0/\
                      (2.0*(s_center_fun(time2)**2.0))).T
    
mhi_new['fft']['B'] *= filter_gauss
mhi_new['fft']['B']  = mhi_new['fft']['B'].T
mhiplot = np.abs(mhi_new['fft']['B'])[f0:f1, :]
if spec_abstype == 'linear' or spec_abstype == 'lin':
     im1 = ax2[1].imshow(mhiplot, origin='lower',
                     extent=(mhi_new['fft']['time'][0],
                             mhi_new['fft']['time'][-1],
                             mhi_new['fft']['freq'][f0],
                             mhi_new['fft']['freq'][f1]),
                     aspect='auto', interpolation='nearest', cmap=cmap)
elif spec_abstype == 'log':
     im1 = ax2[1].imshow(mhiplot, origin='lower',
                     extent=(mhi_new['fft']['time'][0], 
                             mhi_new['fft']['time'][-1],
                             mhi_new['fft']['freq'][0], 
                             mhi_new['fft']['freq'][-1]),
                     aspect='auto', interpolation='nearest', 
                     norm=colors.LogNorm(mhiplot.min(), mhiplot.max()),
                     cmap = cmap)
elif spec_abstype == 'sqrt':
    im1 = ax2[1].imshow(mhiplot, origin='lower',
                     extent=(mhi_new['fft']['time'][0],
                             mhi_new['fft']['time'][-1],
                             mhi_new['fft']['freq'][0], 
                             mhi_new['fft']['freq'][-1]),
                     spect='auto', interpolation='nearest', cmap=cmap,
                     norm=colors.PowerNorm(gamma=0.50))

plt.show()

# -----------------------------------------------------------------------------
# --- Inverting the signal back.
# -----------------------------------------------------------------------------
# Friendly note: it may be that the offset of the original signal does not
# match with the reconstructed signal, as the filter probably took away
# the continuous part.

times = mhi_new['fft']['time']
freqs = mhi_new['fft']['freq'].copy()
Sxx   = mhi_new['fft']['B']
dt = mhi['time'][1] - mhi['time'][0]
fs = 1.0/dt

itime, filter_data = lf.istft2(times, freqs*1000.0, Sxx, nfft=nfft, 
                               tRes=timeResolution, 
                               nyqFreq=freqs[-1]*1000.0, 
                               window=windowType, 
                               resolution=resolution, 
                               fs=fs, 
                               ntime=len(mhi['time']))

fig3, ax3 = plt.subplots(1)
ax3.plot(mhi['time'], mhi['data'], 'r-', linewidth=2.0, 
         label='Original signal')
ax3.plot(itime, filter_data, 'b-', linewidth=1.5, label='Reconstructed mode')

plt.show()