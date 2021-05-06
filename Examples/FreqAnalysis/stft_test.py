#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:49:04 2021

@author: poyo
"""

import Lib as ss
import LibFrequencyAnalysis as lf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import matplotlib.colors as colors
from scipy import signal

# --- Script configuration.
freq0 = 250.0
freq1 = 50.0
freq2 = 300.0

omega0 = freq0*2.0*np.pi
omega1 = freq1*2.0*np.pi
omega2 = freq2*2.0*np.pi
ampNoise = 0.20
# FFT options.
                        # For the window type, go to:
windowType = 'hamming'     # https://docs.scipy.org/doc/scipy/reference/
                        # generated/scipy.signal.get_window.html
specType = 'stft' # Spectogram type:
                  # -> Short-Time Fourier Transform in frequency (sfft)
                  # -> Short-Time Fourier Transform in time (stft)
resolution = int(1000)
timeResolution = 0.70 # Time resolution.
cmap = matplotlib.cm.plasma # Colormap

# -----------------------
# --- Generating some dummy data:

time = np.linspace(start=5.0, stop=7.5, num=2000, dtype=np.single)
data = np.cos(omega2*time)*np.exp(time)
data += np.random.normal(scale=ampNoise, size=time.shape)

fig1, ax1 = plt.subplots(1)
ax1.plot(time, data, 'r-', label='Sample Data')

plt.show()

# --- Generating the spectrogram.
# This assumes that the data is uniformly taken in time.
dt = time[1] - time[0] # For the sampling time.

nfft = int(lf.get_nfft(timeResolution, 'stft', 
                       len(time), windowType, dt))

Sxx, freqs, times = lf.stft2(time,  data, nfft,
                            window=windowType,
                            pass_DC=True, complex_spectrum=True,
                            resolution=resolution)

# --- Filter away all frequencies different from 250kHz

fig, ax = plt.subplots(1)
im1 = ax.imshow(np.abs(Sxx.T), origin='lower',
                extent=(times[0],times[-1],freqs[0],freqs[-1]),
                aspect='auto', interpolation='nearest', cmap=cmap)
ax.set_title('Total spectrum')

f0 = np.abs(freqs - freq2*0.80).argmin()
f1 = np.abs(freqs - freq2*1.1).argmin()

Sxx[:, 0:f0] = 0.0
Sxx[:, f1:]  = 0.0


fig, ax = plt.subplots(1)
im1 = ax.imshow(np.abs(Sxx.T), origin='lower',
                extent=(times[0],times[-1],freqs[0],freqs[-1]),
                aspect='auto', interpolation='nearest', cmap=cmap)
ax.set_title('Filtered spectrum')
plt.show()

# --- Inverting the STFT.
fs = 1.0/dt
norm = np.abs(Sxx).max()
time2, data2 = lf.istft2(times, freqs, Sxx, nfft=nfft, tRes=timeResolution, 
                         nyqFreq=freqs[-1], window=windowType, 
                         resolution=resolution, fs=fs, ntime=len(time))

ax1.plot(time2, data2, 'b--')
ax1.plot(time, data, 'r-')