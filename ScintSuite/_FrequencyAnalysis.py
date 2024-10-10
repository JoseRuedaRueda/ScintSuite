"""
Routines to analyse a time signal in the frequency domain

Include band signal and other filtres aimed to reduce the noise
"""
import numpy as np
import xarray as xr
import heapq
import logging
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as colorMap
import scipy
from scipy.signal import get_window, istft
from scipy.fftpack import fftfreq, rfft, ifft, fftshift
from multiprocessing import cpu_count
from scipy.fftpack import next_fast_len
from scipy.interpolate import interp2d
from collections import defaultdict
from ScintSuite._Plotting import p1D_shaded_error as plot_error_band
from ScintSuite._SideFunctions import smooth
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)
except ModuleNotFoundError:
    print('Only partial support for fft')
    print('Install pyfftw for full support')
logger = logging.getLogger('ScintSuite.Freq')
# -----------------------------------------------------------------------------
# --- Band filters
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- Fourier analysis. Taken from pyspecview
# -----------------------------------------------------------------------------
def sfft(tvec, x, nfft, resolution=1000, window='hann', fmin=0, fmax=np.infty,
         tmin=-np.infty, tmax=np.infty, pass_DC=True, complex_spectrum=False):
    """
    Short time Fourier Tranform. in the frequency domain done along 1. axis!

    Taken from pyspecview from Giovanni Tardini

    :param  tvec: array with the time basis
    :param  x: array with the data points
    :param  nfft: length of the SFFT window
    :param  resolution: horizontal resolution of the spectrogram
    :param  window: window type to calculate the spectrogram
    :param  fmin: Minimum frequency for the spectra (in units of tvec**-1)
    :param  fmax: Maximum frequency for the spectra (in units of tvec**-1)
    :param  tmin: Minimum time for the spectra (in units of tvec)
    :param  tmax: Maximum time for the spectra (in units of tvec)
    :param  pass_DC: remove constant background
    :param  complex_spectrum: To return a complex or real spectra
    """
    # --- Check inputs types
    if x.dtype in [np.cdouble, np.csingle]:
        raise Exception('Complex signals are not supported yet')
    # --- Take the necessary time interval
    f_nfft = nfft
    nfft = next_fast_len(int(nfft))

    iimin = tvec.searchsorted(tmin)
    iimax = tvec.searchsorted(tmax)
    

    tmin = tvec[iimin]
    
    tmax = tvec[iimax-1]
        
    dt = (tmax-tmin)/(iimax-iimin)

    fmax = min(fmax, 1./dt/2)

    nt = next_fast_len(len(x))

    x = x[iimin:iimax]

    # --- Check if some points of signal are complex (still not implemented)
    complex_sig = x.dtype == np.csingle

    if not complex_sig:
        sig_dtype = 'single'
        nf = nt//2+1
    else:
        sig_dtype = 'csingle'
        nf = nt

    nalign = pyfftw.simd_alignment
    # input signal
    sig = pyfftw.n_byte_align_empty(x.shape[1:]+(nt,), nalign, dtype=sig_dtype)
    # output signalsignal
    out = pyfftw.n_byte_align_empty(x.shape[1:]+(nf,), nalign,
                                    dtype=np.complex64)
    fft_forward = pyfftw.FFTW(sig, out, direction='FFTW_FORWARD',
                              flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'],
                              axes=(-1,), threads=cpu_count()//2)

    # input signal
    out_step = pyfftw.n_byte_align_empty(x.shape[1:] + (nfft,), nalign,
                                         dtype=np.complex64)
    # output signal
    in_step = pyfftw.n_byte_align_empty(x.shape[1:] + (nfft,),
                                        nalign, dtype=np.complex64)
    fft_backward = pyfftw.FFTW(in_step, out_step, direction='FFTW_BACKWARD',
                               flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'],
                               axes=(-1,), threads=cpu_count()//2)

    sig[..., :len(x)] = x
    sig[..., :len(x)] -= x.mean(0).T
    sig[..., len(x):] = 0

    fft_forward()

    del sig

    ifmin = int(max(nf*fmin*(2*dt), -nf))
    ifmax = int(min(nf*fmax*(2*dt), nf))

    fft_step = max(1, (ifmax-ifmin)//resolution//2)
    fvec = np.arange(ifmin, ifmax, fft_step)/dt/2./nf
    ntvec = (len(x)*nfft//2)//nf
    tvec = np.linspace(tmin, len(x)*tmax/nf, ntvec, endpoint=False)

    dtype = np.complex64 if complex_spectrum else np.single
    spec = np.empty((ntvec, len(fvec)) + x.shape[1:], dtype=dtype)

    if window == 'gauss':
        window = window, f_nfft/8.

    win = get_window(window, nfft, fftbins=False).astype('single')

    for i, ii in enumerate(range(ifmin, ifmax, fft_step)):

        L = min(ii+nfft//2, nf) - max(0, ii-nfft//2)
        in_step[..., :L] = out[..., max(0, ii-nfft//2):min(ii+nfft//2, nf)]
        in_step[..., L:] = 0
        in_step[..., :L] *= win[max(0, ii-nfft//2)-ii+nfft//2:nfft//2-ii
                                + min(nf, ii+nfft//2)]

        # main step FFT
        fft_backward()

        if complex_spectrum:
            spec[:, i] = out_step.T[:ntvec]
        else:
            spec[:, i] = np.abs(out_step.T[:ntvec])

    if not pass_DC:
        spec = spec[:, 1:]
        fvec = fvec[1:]

    return spec, fvec, tvec


def stft(tvec, x, nfft, resolution=1000, window='gauss', fmin=-np.infty,
         fmax=np.infty, tmin=-np.infty, tmax=np.infty, pass_DC=True,
         complex_spectrum=False):
    """
    Short time Fourier Tranform. in time domain

    x - real or complex, transformation done along first axis

    Taken from pyspecview from Giovanni Tardini

    :param  tvec: array with the time basis
    :param  x: array with the data points
    :param  nfft: length of the SFFT window
    :param  resolution: horizontal resolution of the spectrogram
    :param  window: window type to calculate the spectrogram
    :param  fmin: Minimum frequency for the spectra (in units of tvec**-1)
    :param  fmax: Maximum frequency for the spectra (in units of tvec**-1)
    :param  tmin: Minimum time for the spectra (in units of tvec)
    :param  tmax: Maximum time for the spectra (in units of tvec)
    :param  pass_DC: remove constant background
    :param  complex_spectrum: To return a complex or real spectra
    """
    if x.dtype in [np.double, np.cdouble]:
        raise Exception('use dtype single or csingle')

    complex_sig = x.dtype == np.csingle

    iimin, iimax = tvec.searchsorted((tmin, tmax))
    iimax -= 1
    n = iimax-iimin
    dt = (tvec[iimax]-tvec[iimin])/n

    f_nfft = nfft

    nfft = next_fast_len(int(nfft))
    fvec = fftfreq(nfft, dt)
    if not complex_sig:
        fvec = fvec[:nfft//2]
        sig_dtype = 'single'
    else:
        fvec = fftshift(fvec)
        sig_dtype = 'csingle'

    ifmin, ifmax = fvec.searchsorted([fmin, fmax])
    ifmax = np.minimum(ifmax+1, len(fvec))

    # input signal
    nalign = pyfftw.simd_alignment
    sig = pyfftw.n_byte_align_empty(x.shape[1:]+(nfft,), nalign,
                                    dtype=sig_dtype)
    # output signal
    nf = nfft if complex_sig else nfft//2+1
    out = pyfftw.n_byte_align_empty(x.shape[1:]+(nf,),
                                    nalign, dtype=np.complex64)

    fft_forward = pyfftw.FFTW(sig, out, direction='FFTW_FORWARD',
                              flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'],
                              axes=(-1,), threads=cpu_count()//2)

    fft_step = max(1, n//resolution)
    dtype = np.complex64 if complex_spectrum else np.single
    spec = np.empty((int(n/fft_step), ifmax-ifmin) + x.shape[1:],
                    dtype=dtype)
    win = None

    for i in range(int(n//fft_step)):
        imin = int(max(0, iimin+np.floor(i*fft_step-nfft//2)))
        imax = int(min(len(x), iimin + np.floor(i*fft_step+nfft//2)))

        if np.size(win) != imax-imin:
            if window == 'gauss':
                win = get_window((window, f_nfft*(imax-imin)//8//nfft),
                                 imax-imin, fftbins=False)
            else:
                win = get_window(window, imax-imin, fftbins=False)

            win = win.astype('single')

        # implicit conversion from int (or anything else) to single!
        if pass_DC:
            sig[..., :imax-imin] = x[imin:imax].T
        else:
            sig[..., :imax-imin] = x[imin:imax].T
            sig[..., :imax-imin] -= np.mean(x[imin:imax], 0)[None].T

        sig[..., imax-imin:] = 0
        sig[..., :imax-imin] *= win

        # the main step, FFT
        fft_forward()

        # prepare output spectrum
        if complex_sig:
            # to get monotonously increasing frequency
            ind_neg = slice(min(nfft, ifmin+(nfft+1)//2),
                            min(ifmax+(nfft+1)//2, nfft))
            ind_pos = slice(max(0, ifmin-(nfft)//2),
                            max(ifmax-(nfft)//2, 0))
            nneg = ind_neg.stop - ind_neg.start
            # npos = ind_pos.stop - ind_pos.start
            if complex_spectrum:
                spec[i, :nneg] = out.T[ind_neg]
                spec[i, nneg:] = out.T[ind_pos]
            else:  # compute abs(fft)
                np.abs(out.T[ind_neg], out=spec[i, :nneg])
                np.abs(out.T[ind_pos], out=spec[i, nneg:])
        else:
            if complex_spectrum:
                spec[i] = out.T[ifmin:ifmax]
            else:
                np.abs(out.T[ifmin:ifmax], out=spec[i])  # compute abs(fft)
    fout = fvec[ifmin:ifmax]
    tout = tvec[iimin:iimax:fft_step][:int(n//fft_step)]
    return spec, fout, tout


def stft2(tvec, x, nfft, resolution=1000, window='hann', pass_DC=True,
          complex_spectrum=True):
    """
    Short-Time Fourier Transform - Wrapper to the scipy.signal implementation.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  tvec: time vector where the signal 'x' is defined.
    :param  x: time-dependent signal to build the spectrogram.
    :param  resolution: time resolution to build the spectogram.
    :param  window: type of windowing to apply.
    :param  pass_DC: sets the amplitude (and phase) to 0 for the first element
    in the frequency array returned (i.e., f = 0).
    :param  complex_spectrum: returns the complex_spectrum. True by default.
    :return data: bidimensional array with the spectrogram.
    :return freq: frequency array.
    :return time: time array.
    """

    nt = len(tvec)
    fs = 1.0/(tvec[1] - tvec[0])

    # --- Checking the inputs.
    if nt < 1:
        raise Exception('The time vector must be 1D array.')

    if x.shape != (tvec.size,):
        raise Exception('The signal must have the dimensions time')

    # --- Translating the inputs into the inputs for the scipy implementation.
    nperseg = next_fast_len(nfft)
    noverlap = nperseg - max(1, int(nt//resolution))

    freq, time, data = signal.stft(x, fs=fs, window=window, nperseg=nperseg,
                                   noverlap=noverlap, nfft=nperseg,
                                   return_onesided=True, padded=False,
                                   boundary='even')

    time += tvec[0]
    data = data.T

    if(not pass_DC):
        data[0, :] = 0.0j

    if(not complex_spectrum):
        data = np.abs(data)

    return data, freq, time


def istft2(tvec: float, fvec: float, x, tRes: float, nyqFreq: float,
           nfft: int, window='gauss', resolution: int = 1000,
           tmin: float = None, tmax: float = None, fs: float = 1.0,
           ntime: int = None):
    """
    Wrapper to scipy.signal.istft.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  tvec: Time vector where the spectrogram is built.
    :param  fvec: Frequency vector where the spectrogram is built.
    :param  tRes: frequency resolution originally used to build the spectrogram.
    :param  nyqFreq: original Nyquist frequency.
    :param  window: window to use.
    :param  resolution: time resolution used.
    :param  tmin: Minimal time to use in the reconstruction.
    :param  tmax: Maximal time to use in the reconstruction.
    """
    nf = len(fvec)
    nt = len(tvec)

    # --- Checking the inputs.
    if nt < 1:
        raise Exception('The time vector must be 1D array.')
    if nf < 1:
        raise Exception('The time vector must be 1D array.')

    if x.shape == (fvec.size, tvec.size):
        x = x.T

    if x.shape != (tvec.size, fvec.size):
        raise Exception('The signal must have the dimensions time x frequency')

    # Transform it into a complex signal.
    if x.dtype in [np.single, np.double]:
        x = np.asarray(x, dtype=np.cdouble)

    if tmin is None:
        tmin = tvec[0]
    if tmax is None:
        tmax = tvec[-1]

    if ntime is None:
        ntime = len(tvec)

    # Searching for the initial and ending time points.
    iimin, iimax = tvec.searchsorted((tmin, tmax))
    iimax -= 1
    n = iimax-iimin

    # --- Filtering in time the signal:
    sig_copy = x.copy()[iimin:iimax, :]

    # --- Adding some padding for f = 0 to fmin.
    fmin = fvec[0]
    fmax = fvec[-1]
    fs_spec = (fvec[1] - fvec[0])

    if(fmin != 0):
        padding_size = int(fmin/fs_spec)
        padding = np.zeros((n, padding_size), dtype=x.dtype)
        print('Adding %d points of padding up to fmin = %.3f' % (padding_size,
                                                                 fmin))

        # --- Attaching the padding.
        sig_copy = np.hstack((padding, sig_copy))
        del padding_size
        del padding

    if(fmax != nyqFreq):

        padding_size = int((fmax-nyqFreq)/fs_spec)
        padding = np.zeros((n, padding_size), dtype=x.dtype)
        print('Adding %d points of padding from fmax = %.3f' % (padding_size,
                                                                fmax))

        # --- Attaching the padding.
        sig_copy = np.hstack((sig_copy, padding))
        del padding_size
        del padding

    # --- Translating the inputs into the inputs for the scipy implementation.
    nperseg = nfft
    noverlap = nperseg - max(1, int(ntime//resolution))

    time, data = istft(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, nfft=nperseg, input_onesided=True,
                       boundary=True, time_axis=0, freq_axis=1)

    time += tvec[0]

    return time, data


def get_nfft(tau0, specType, nt, windowType, dt):
    """
    Getting the number of FFT required to reach the resolution.

    by Giovanni Tardini

    :param  x: Time resolution parameter
    :param  specType: kind of spectrum.
    :param  nt: Number of time points.
    :param  windowType: kind of window applied.
    :param  dt: time step.

    :return nfft: number of points to overlap.
    """
    if specType in ('sparse', 'frequency', 'stfft', 'sfft'):
        tau0 = 1.20 - tau0

    tau = nt**tau0 * dt
    nfft = tau/dt if windowType == 'gauss' else 2**np.ceil(np.log2(tau/dt))

    return nfft


# -----------------------------------------------------------------------------
# --- Cross-power density calculation.
# -----------------------------------------------------------------------------
def myCPSD(sig1, sig2, time1, freq1, time2, freq2):
    """
    Computing the cross-power spectral density for two complex input signals.

    The spectrograms of the input signals may be defined onto two different
    time-frequency basis. The second one will interpolated along the first.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  sig1: complex matrix. Spectrogram of the 1st signal.
    :param  sig2: complex matrix. Spectrogram of the 2nd signal.
    :param  time1: Time basis of the first signal. Also time output.
    :param  freq1: Frequency basis of the first signal. Also the freq. output.
    :param  time2: Time basis onto which the 2nd signal is defined.
    :param  freq2: Frequency basis of the 2nd signal is defined.
    """

    # --- We interpolate the signal 2 into the time and frequency basis of the
    # first.
    sig2_fun_r  = interp2d(time2, freq2, sig2.real, kind='linear',
                           bounds_error=False)
    sig2_fun_i  = interp2d(time2, freq2, sig2.imag, kind='linear',
                           bounds_error=False)
    # The -j is because we need to multiply by the complex conjugate
    sig2_on1  = sig2_fun_r(time1, freq1) - 1j*sig2_fun_i(time1, freq1)

    # Computing the element-wise matrix product:
    return time1, freq1, (sig1*sig2_on1)

# -----------------------------------------------------------------------------
# --- Tracking algorithm
# -----------------------------------------------------------------------------
def trackFrequency(time: float, freq: float, spec: float, origin: float,
                   target: float = None, freqLims: float = None,
                   timeLims: float = None, tOverlap: float = None,
                   graph_TimeConnect: float = 0.0, freqThr: float = np.inf,
                   k_exp: float = 4.0, kt_exp: float = 1.0,
                   peak_opts: dict = {},
                   costFunction=None, peakFilterFnc=None,
                   smooth: bool = True, smooth_opts: dict = {},
                   verbose: bool = True, plot: bool = True,
                   plotOpts: dict = {}, lineOpts: dict = {}, ax=None,
                   fig=None):
    """
    This function will try to follow a frequency in a spectrogram based on
    peak detection algorithm + Dijsktra's algorithm.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  time: array with the time points (equispaced)
    :param  freq: arraq with the frequency points (equispaced)
    :param  spec: spectrogram 2D array.
    :param  origin: starting frequency to start the search. The initial time
    point will be taken from the initial time point in the array 'time' or,
    whenever provided, from the 'timeLims'.
    :param  target: if provided, the algorithm will look for the shortest path
    until arriving this target frequency. It forces the code to go through all
    the nodes. Generally, unrecommended.
    :param  freqLims: frequency windows to look for the modes. If None, all the
    frequency window is used instead.
    :param  timeLims: time windows to look for the peaks. This can be useful for
    ELM synchronization. If None, the whole time window is used.
    :param  costFunction: callable function to provided two nodes dictionaries,
    and must provide a POSITIVE value for any kind of input that allows to
    measure the cost of jumping between two nodes. If not provided, a standard
    function is always provided ($\\Delta f^$)
    :param  peakFilterFnc: function to filter peaks. It must admit as inputs
    the frequency axis, the smoothed data for each time slice, the peak
    position array and the peak properties.
    :param  tOverlap: sum over time when looking for peaks. If None, the sum
    is set to dt.
    :param  smooth: sets if the data is smoothed before searching for peaks.
    By default, it is set to True.
    :param  smooth_opts: dictionary containing the options for the sav-golay
    filter. If not provided, windows_length = 5 and the polynomial order is set
    to 3.
    :param  peak_opts: dictionary containing the options for the peak searching.
        @see{scipy.signal.find_peaks}
    :param  verbose: writes into the console the partial results.
    :param  plot: plots to a new axis the results from the frequency tracking.
    :param  plotOpts: dictionary with options to plot the spectrogram. The
    'plasma' colormap is used as deafault is none is provided.
    :param  graph_TimeConnect: Time window in which the connection of the nodes.
    If None is provided, then it is taken to be 5*dt, i.e., only the 5 closest
    time slices are connected among themselves.
    :param  freqThr: above the threshold, the routine will disconnect the nodes
    with a frequency jump above this.
    :param  k_exp: exponent to weight the frequency contribution to the jump.
    Only useful, when cost function is not provided as input.
    :param  lineOpts: plotting options for the frequency tracker curve. Only
    used whenever the plot flat is True. If not provided, color is set to
    white.
    """
    dt = time[1] - time[0]
    df = freq[1] - freq[0]
    nfreq = len(freq)
    freqMean = np.mean(freq)

    # --- Checking the inputs.
    if costFunction is not None:
        if not callable(costFunction):
            raise Exception('The cost function must be a callable function')

    if peakFilterFnc is not None:
        if not callable(peakFilterFnc):
            raise Exception('The peak filtering function must be callable')

    if spec.ndim != 2:
        raise Exception('The input spectrogram must be a 2D object.')

    if time.ndim != 1:
        raise Exception('The time input must be 1D array.')

    if freq.ndim != 1:
        raise Exception('The time input must be 1D array.')

    if (spec.shape[0] == freq.size) and (spec.shape[1] == time.size):
        spec = spec.T
    elif (spec.shape[0] != time.size) or (spec.shape[1] != freq.size):
        raise Exception('The input spectrogram mismatch in size with axis')

    # Checking the plotting options.
    if 'cmap' not in plotOpts:
        plotOpts['cmap'] = colorMap.plasma
    if 'shading' not in plotOpts:
        plotOpts['shading'] = 'flat'
    if 'antialiased' not in plotOpts:
        plotOpts['antialiased'] = True

    # The time overlap cannot be smaller than the time step.
    if (tOverlap is None) or (tOverlap < dt):
        tOverlap = dt

    # Computing the number of time points to make the time average.
    nOverlap = int(np.floor(tOverlap/dt))
    if verbose:
        print('Overlapping %d points for average'%nOverlap)

    # Checking the windows timing.
    if timeLims is not None:
        if len(timeLims)%2 != 0:
            timeLims.append(time[-1])

        if timeLims[0] < time[0]:
            timeLims[0] = time[0]
        if timeLims[-1] > time[-1]:
            timeLims[-1] = time[-1]
    else:
        timeLims = [time[0], time[-1]]

    if freqLims is None:
        freqLims = [freq[0], freq[-1]]

    f0, f1 = freq.searchsorted(freqLims)
    spec2 = spec.copy()
    spec2 = spec2[:, f0:f1]
    freq2 = freq[f0:f1]

    # Checking the time overlap
    if graph_TimeConnect < dt:
        graph_TimeConnect = 10.0*dt

    nGraphConne = int(np.floor(graph_TimeConnect/dt))
    if verbose:
        print('Connecting nodes in graph separated by %d time-slices'%\
              nGraphConne)

    # --- Checking the filtering data.
    if 'window_length' not in smooth_opts:
        smooth_opts['window_length'] = 5
    elif smooth_opts['window_length'] % 2 == 0:
        print('The smoothing window must be odd in points. Increasing by 1')
        smooth_opts['window_length'] += 1

    if 'polyorder' not in smooth_opts:
        smooth_opts['polyorder'] = 3

    # --- Checking the peaking finding data.
    if 'prominence' not in peak_opts:
        peak_opts['prominence'] = 0.50 # 66.7%
    if 'width' not in peak_opts:
        peak_opts['width'] = (None, None)

    if 'rel_height' not in peak_opts:
        peak_opts['rel_height'] = 0.50 # By default set to FWHM

    # --- Creating a Graph object to store the collection of vertices.
    dgraph = Graph()

    # --- Looking for the peaks in the spectrum.
    # The peaks in the spectrum work as the vertex of the Graph used to
    # track the frequency.
    nwindows = np.floor(len(time)%2)
    if verbose:
        print('There are %d time windows to parse'%nwindows)

    # Peak data as lists, se we can just append.
    peak_data = { 'peak_idx': defaultdict(list),
                  'prominences': defaultdict(list),
                  'width': defaultdict(list),
                  'time': defaultdict(list),
                  'freq': defaultdict(list),
                  'spec_val': defaultdict(list),
                  'spec_norm': defaultdict(list),
                  'timeList': defaultdict(list)
                }

    # This will help us mapping the node ID to the peak time-frequency.
    peak_map = defaultdict(list)
    kk = int(0) # Index running for the time ordered list.
    for ii in range(nwindows):
        t0 = np.abs(time - timeLims[2*ii]).argmin()
        t1 = np.abs(time - timeLims[2*ii + 1]).argmin()

        nTimes_slices = int((t1-t0+1)/nOverlap)
        for jj in range(nTimes_slices):
            t0_avg = t0 + jj*nOverlap
            t1_avg = np.minimum(t0+(jj+1)*nOverlap, t1+1)

            data = np.mean(spec2[t0_avg:t1_avg, :], axis=0)
            data -= np.min(data)
            data /= np.max(data)

            if smooth:
                try:
                    data = signal.savgol_filter(data, **smooth_opts)
                except:
                    print('Data at t = %d non-smoothed!'%t0_avg)

            peaks, props = signal.find_peaks(data, **peak_opts)
            props['widths'], _, _,_ = signal.peak_widths(data, peaks,
                                     rel_height=peak_opts['rel_height'])

            # If after everything, there are not any peaks, just go to the
            # next slice.
            if len(peaks) == 0:
                continue

            # External filtering function.
            if peakFilterFnc is not None:
                flags = peakFilterFnc(freq2, data, peaks, props)
                peaks = peaks[flags]
                for ikey in props.keys():
                    props[ikey] = props[ikey][flags]

            # If after everything, there are not any peaks, just go to the
            # next slice.
            if len(peaks) == 0:
                continue

            # Adding the peak data to the list.
            peak_data['peak_idx'][kk] = peaks
            peak_data['width'][kk] = props['widths']*df
            peak_data['freq'][kk] = freq2[peaks]
            peak_data['spec_val'][kk]= np.mean(spec2[t0_avg:t1_avg, peaks],
                                               axis=0)
            peak_data['spec_norm'][kk]  = data[peaks]
            peak_data['prominences'][kk] = props['prominences']

            time_idx = int(np.floor((t0_avg+t1_avg)/2.0)) * \
                       np.ones(len(peaks), dtype=int)
            peak_data['time'][kk] = time[time_idx]
            peak_data['timeList'][kk] = kk*nfreq*np.ones(len(time_idx)) +\
                                        peak_data['peak_idx'][kk]

            # Adding the vertex to the list in the graph class.
            for inode,node in enumerate(peak_data['timeList'][kk]):
                dgraph.add_vertex(node)
                peak_map[node] = (peak_data['time'][kk][inode],
                                  peak_data['freq'][kk][inode],
                                  peak_data['width'][kk][inode],
                                  peak_data['spec_val'][kk][inode],
                                  peak_data['spec_norm'][kk][inode])
            kk += 1

    if verbose:
        print('Found %d peaks!'%len(peak_map))

    if len(peak_map) == 0:
        raise RuntimeError('No peaks found in the spectrum. Tweak the inputs')

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.pcolormesh(time, freq2, spec2.T, **plotOpts)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [kHz]')

        # Plotting the peaks
        for ii in peak_map:
            ax.plot(peak_map[ii][0], peak_map[ii][1], 'r.')

    ntime_peaks = kk - 1
    if verbose:
        print('#time slices = %d'%ntime_peaks)
    # --- Generating the graph: connecting the vertex
    # The graph will conect the timepoints with the next timepoints peaks only
    for ii in range(ntime_peaks):
        # Loop over the starting nodes.
        for jj, frm in enumerate(peak_data['timeList'][ii]):
            # Loop over the next nodes: we will connect every node all the
            # nodes that are lying after it, given a maximum time distance.
            stop_conn = int(np.minimum(ii+nGraphConne, ntime_peaks+1))
            for itime in range(ii+1, stop_conn):
                # Loop over the time slices after the current one.
                for kk, to in enumerate(peak_data['timeList'][itime]):
                    # Loop over all the nodes in timeslice 'itime'
                    peak_prv = { 'time': peak_data['time'][ii][jj],
                                 'freq': peak_data['freq'][ii][jj],
                                 'width': peak_data['width'][ii][jj],
                                 'spec_val': peak_data['spec_val'][ii][jj],
                                 'spec_norm': peak_data['spec_norm'][ii][jj]
                               }
                    peak_nxt = { 'time': peak_data['time'][itime][kk],
                                 'freq': peak_data['freq'][itime][kk],
                                 'width': peak_data['width'][itime][kk],
                                 'spec_val': peak_data['spec_val'][itime][kk],
                                 'spec_norm': peak_data['spec_norm'][itime][kk]
                               }
                    if costFunction is None:
                        deltaF = np.abs(peak_prv['freq'] - peak_nxt['freq'])
                        deltaA = np.abs(peak_prv['spec_norm'] - \
                                        peak_nxt['spec_norm'])
                        deltaTime = np.abs(peak_prv['time'] - peak_nxt['time'])

                        sigmaF = np.abs(peak_prv['width'])
                        # sigmaA = 1.0
                        sigmaTime = dt

                        cost = (deltaF/freqMean)**k_exp * \
                               (deltaTime/graph_TimeConnect)**kt_exp
                        if (deltaF > freqThr):
                             cost *= (np.inf)
                    else:
                        cost = costFunction(prv=peak_prv, nxt=peak_nxt,
                                            origin=False, target=False)

                    dgraph.add_edge(frm=frm, to=to, cost=cost, forceAdd=False)

    # --- Adding the origin vertex in the graph.
    dgraph.add_vertex('origin')
    peak_origin = { 'time': timeLims[0],
                    'freq': origin,
                    'width': 0.0,
                    'spec_val': interp2d(time, freq2, spec2.T)\
                                (timeLims[0], origin),
                    'spec_norm': 0.0,
                }

    peak_map['origin'] = (peak_origin['time'],
                          peak_origin['freq'],
                          peak_origin['width'],
                          peak_origin['spec_val'],
                          peak_origin['spec_norm'])
    for jj in range(0, nGraphConne):
        for ii, to in enumerate(peak_data['timeList'][jj]):
            peak_nxt = { 'time': peak_data['time'][jj][ii],
                         'freq': peak_data['freq'][jj][ii],
                         'width': peak_data['width'][jj][ii],
                         'spec_val': peak_data['spec_val'][jj][ii],
                         'spec_norm': peak_data['spec_norm'][jj][ii],
                       }
            if costFunction is None:
                deltaF = np.abs(peak_origin['freq'] - peak_nxt['freq'])
                sigmaF = np.abs(peak_nxt['width'])
                deltaTime = np.abs(peak_origin['time'] - peak_nxt['freq'])

                cost = (deltaF/freqMean)**k_exp* \
                               (deltaTime/graph_TimeConnect)**kt_exp
                if (deltaF > freqThr):
                     cost *= (np.inf)
            else:
                cost = costFunction(prv=peak_origin, nxt=peak_nxt,
                                    origin=True, target=False)
            dgraph.add_edge(frm='origin', to=to, cost=cost, forceAdd=False)

    # --- Checking for the final vertex:
    if target is not None:
        dgraph.add_vertex('target')
        peak_target = { 'time': [],
                        'freq': target,
                        'width': [],
                        'spec_val': [],
                        'spec_norm': [],
                    }
        for ii, frm in enumerate(peak_data['timeList'][-1]):
            peak_prv = { 'time': peak_data['time'][-1][ii],
                         'freq': peak_data['freq'][-1][ii],
                         'width': peak_data['width'][-1][ii],
                         'spec_val': peak_data['spec_val'][-1][ii],
                         'spec_norm': peak_data['spec_norm'][-1][ii],
                       }
            if costFunction is None:
                deltaF = np.abs(peak_prv['freq'] - peak_target['freq'])
                sigmaF = np.abs(peak_prv['width'])

                cost = deltaF/sigmaF
            else:
                cost = costFunction(prv=peak_prv, nxt=peak_target,
                                    origin=False, target=True)
            if cost.size == 0:
                print('Adding target')
                print(ii, frm)
                print(deltaF, sigmaF)
            dgraph.add_edge(frm=frm, to='target', cost=cost, forceAdd=False)

    # --- Using Dijsktra.
    # The Dijsktra implemented in this library uses one source-one target.
    # The origin is the typical input, but the ending point is free and to
    # be determined by the algorithm.
    dgraph.Dijsktra(dgraph.get_vertex('origin'), verbose=False)

    # --- Getting the final point:
    path = list()
    if target:
        path.append('target')
        graph_shortest(dgraph.get_vertex('target'), path)
    else:
        # Loop over all the endings to see which is the target vertex.
        done_flag = False
        for jj in np.arange(start=ntime_peaks, stop=0, step=-1, dtype=int):
            # Test all the time slices, until finding one where one of the
            # with a node connected to the frequency.
            v = np.zeros((len(peak_data['timeList'][jj]),))
            if len(v) == 0:
                continue
            for ii, node in enumerate(peak_data['timeList'][jj]):
                v[ii] = dgraph.get_vertex(node=node).distance
            if np.any(v != np.inf):
                imin = v.argmin()
                distmin = v[imin]
                path.append(peak_data['timeList'][jj][imin])
                graph_shortest(dgraph.get_vertex(peak_data['timeList']\
                                                 [jj][imin]), path)

            if 'origin' in path:
                done_flag = True
                break

        if not done_flag:
            print('Be careful, the algorithm could not\
                   find the origin in the path backwards')
            distmin = np.inf
    # --- Translating the path into the curve (t, freq)
    timecurve = list()
    freqcurve = list()
    ampcurve_norm  = list()
    ampcurve_total = list()
    widths_curve   = list()
    for ii in path:
        timecurve.append(peak_map[ii][0])
        freqcurve.append(peak_map[ii][1])
        ampcurve_norm.append(peak_map[ii][2])
        ampcurve_total.append(peak_map[ii][3])
        widths_curve.append(peak_map[ii][4])

    output = { 'track': { 'time': np.flip(np.array(timecurve)),
                          'freq': np.flip(np.array(freqcurve)),
                          'Anorm': np.flip(np.array(ampcurve_norm)),
                          'Atot': np.flip(np.array(ampcurve_total)),
                          'width': np.flip(np.array(widths_curve))
                        },
              'peak_data': peak_data,
              'peak_map': peak_map,
              'path_by_graph': path.reverse(),
              'cost': distmin
             }

    # --- Print the curve.
    if plot:
        if 'color' not in lineOpts:
            lineOpts['color'] = 'w'
        ax = plot_error_band(ax=ax, x=output['track']['time'],
                             y=output['track']['freq'], color='w',
                             u_up=output['track']['width']/2.0,
                             alpha=0.2, line=True, line_param=lineOpts)
    del dgraph

    return output, ax

# ----------------------------------------------------------------------------
# --- Graph and vertex classes for shortest path algorithm.
# ----------------------------------------------------------------------------
class Vertex:
    """
    Vertex class. A vertex is an element of a Graph. This class contains
    the information of the neighbouring vertices and its distances.

    Taken from:
    https://www.bogotobogo.com/
    """
    def __init__(self, node):
        """
        Initializes the Vertex class with a given node identification.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  node: ID of the node.
        """
        self.id = node
        self.adjacent = defaultdict(list)
        # Set distance to infinity for all nodes
        self.distance = np.inf
        # Mark all nodes unvisited
        self.visited = False
        # Predecessor
        self.previous = None

    def __del__(self):
        """
        Destructor of the vertex content.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.id = 0
        self.adjacent = defaultdict(list)
        self.distance = np.inf
        self.visited = False
        self.previous = None

    def __lt__(self, b):
        """
        Compares the distance to the origin of two vertices.
        """
        return (self.distance < b.distance)

    def __le__(self, b):
        """
        Compares the distance to the origin of two vertices.
        """
        return (self.distance <= b.distance)

    def add_neighbor(self, neighbor, weight: float = 0.0):
        """
        Add a neighbour to the current vertex and adds its corresponding
        weight.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  neighbor: identification of the neighbour (a string, number,...)
        :param  weight: the weight to go from one point to the next. Only
        non-negative values. 0 implies direct connection, Inf means no
        connection at all. In the latter, it will not be added to the list.
        """
        if weight.size == 0:
            raise Exception('The weight must be a number for neighbour= '+\
                            str(neighbor.id))

        if weight < 0.0:
            raise Exception('Weigths must be non-negative numbers!')
        elif weight == np.inf:
            return

        self.adjacent[neighbor] = weight

    @property
    def connections(self):
        """
        Returns the connections names as stored in the dictionary adjacent.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        return self.adjacent.keys()

    def get_weight(self, neighbor):
        """
        Returns the weigth for a given pair self->neighbor edge.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  neigbor: name or identification of the neighbour whose weight
        is wanted to be known.
        """
        return self.adjacent[neighbor]

    def __str__(self):
        """
        Converts the vertex relations to a string output. It will improve the
        debugging.
        """
        return str(self.id) + ' adjacent: ' +  \
               str([x.id for x in self.adjacent])


class Graph:
    """
    The Graph class contains a set of vertices and their connections.

    Taken from: https://www.bogotobogo.com/

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """

    def __init__(self):
        """
        Initializes the vertex class with no nodes.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.vert_dict = defaultdict(list)
        self.nVertices = int(0)

    def __iter__(self):
        """
        Defines the iterator over the vertices.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        return iter(self.vert_dict.values())

    def __del__(self):
        """
        Destructor of the class. This will call the destructor of all the
        vertices contained.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        for ii in self.vert_dict:
            del ii

        self.nVertices = 0

    def add_vertex(self, node_id):
        """
        Add a new vertex to the list that will be identified with the ID
        node_id, that must be unique.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  node_id: identificator of the node.
        :return new_vertex: Vertex class created.
        """

        # --- Check that there are no collisions.
        if node_id in self.vert_dict:
            raise Exception('The vertex ID is repeated!')

        # --- Creating the vertex.
        new_vertex = Vertex(node_id)
        self.vert_dict[node_id] = new_vertex  # Adding to the list.

        self.nVertices += 1

        return new_vertex

    def get_vertex(self, node):
        """
        Get the vertex associated with the ID node.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  node: node identificator to get the Vertex class.
        """
        if node in self.vert_dict:
            return self.vert_dict[node]
        else:
            return None

    def add_edge(self, frm, to, cost: float=0.0, forceAdd: bool=True):
        """
        Add a connection between two vertices with a cost value of cost. It can
        be forced to add the two vertices if they did not exist before.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  frm: node-id of the starting node.
        :param  to:  node-id of the ending node.
        :param  cost: weighting of the node-node connection.
        :param  forceAdd: force to add the two new vertices 'frm'&'to' into the
        list. By default it will add them.
        """

        if frm not in self.vert_dict:
            if forceAdd:
                self.add_vertex(frm)
            else:
                raise Exception('Node %s not available in the list'%frm)

        if to not in self.vert_dict:
            if forceAdd:
                self.add_vertex(to)
            else:
                raise Exception('Node %s not available in the list'%to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], weight=cost)

    def get_vertices(self):
        """
        Getting the vertices ids.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """

        return self.vert_dict.keys()

    def Dijsktra(self, start, verbose: bool=True):
        """
        Dijsktra algorithm to look for the shortest path in a Graph.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  agraph: graph class containing the collection of vertices.
        :param  start: starting vertex.
        """

        # Set the distance to the origin of the starting vertex to 0
        start.distance = 0.0

        # Put tuple pair into the priority queue.
        unvisited_queue = [(v.distance, v) \
                           for v in iter(self.vert_dict.values())]
        heapq.heapify(unvisited_queue)

        while len(unvisited_queue):
            # Pops a vertex with the smallest distance.
            uv = heapq.heappop(unvisited_queue)
            current = uv[1]
            current.visited = True

            for nxt in current.adjacent:
                if nxt.visited:
                    continue

                new_dist = current.distance + current.get_weight(nxt)

                if new_dist < nxt.distance:
                    nxt.distance = new_dist
                    nxt.previous = current
                    if verbose:
                        print('Updated: current = '+ str(current.id)+'\n' + \
                                       'next = '    + str(nxt.id)   + '\n' + \
                                       'new_dist = ' +str(nxt.distance)+ '\n')

                else:
                    if verbose:
                        print('Non-updated: current = '+ str(current.id)+'\n'+\
                              'next = '     + str(nxt.id)     + '\n' + \
                                'new_dist = ' + str(nxt.distance) + '\n')

            # Rebuild heap:
            # 1. Pop every item.
            while len(unvisited_queue):
                heapq.heappop(unvisited_queue)

            # 2. Put all vertices not visited into the queue.
            unvisited_queue = [(v.distance, v) \
                               for v in iter(self.vert_dict.values()) \
                               if not v.visited]
            heapq.heapify(unvisited_queue)

def graph_shortest(v, path):
    """
    This searches for the path starting by the vertex 'v' and go backwards.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  v: ending vertex.
    :param  path: vertex path back to the origin. A collection of the IDs.
    """
    if v.previous:
        path.append(v.previous.id)
        graph_shortest(v.previous, path)

    return


