"""
Routines to analyse a time signal in the frequency domain

Include banpass signal and other filtres aimed to reduce the noise
"""
import numpy as np
import pyfftw
from scipy.signal import get_window
from scipy.fftpack import fftfreq, rfft, ifft, fftshift
from multiprocessing import cpu_count
from scipy.fftpack import next_fast_len
from scipy.interpolate import interp2d

# -----------------------------------------------------------------------------
# --- Band filters
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- Fourier analysis. Taken from pyspecview
# -----------------------------------------------------------------------------
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(30)


def sfft(tvec, x, nfft, resolution=1000, window='hann', fmin=0, fmax=np.infty,
         tmin=-np.infty, tmax=np.infty, pass_DC=True, complex_spectrum=False):
    """
    Short time Fourier Tranform. in the frequency domain done along 1. axis!

    Taken from pyspeckview from Giovanni Tardini

    @param tvec: array with the time basis
    @param x: array with the data points
    @param nfft: length of the SFFT window
    @param resolution: horizontal resolution of the spectrogram
    @param window: window type to calculate the spectrogram
    @param fmin: Minimum frequency for the spectra (in units of tvec**-1)
    @param fmax: Maximum frequency for the spectra (in units of tvec**-1)
    @param tmin: Minimum time for the spectra (in units of tvec)
    @param tmax: Maximum time for the spectra (in units of tvec)
    @param pass_DC: remove constant background
    @param complex_spectrum: To return a complex or real spectra
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


def stft(tvec, signal, nfft, resolution=1000, window='gauss', fmin=-np.infty,
         fmax=np.infty, tmin=-np.infty, tmax=np.infty, pass_DC=True,
         complex_spectrum=False):
    """
    Short time Fourier Tranform. in time domain

    x - real or complex, transformation done along first axis

    Taken from pyspeckview from Giovanni Tardini

    @param tvec: array with the time basis
    @param x: array with the data points
    @param nfft: length of the SFFT window
    @param resolution: horizontal resolution of the spectrogram
    @param window: window type to calculate the spectrogram
    @param fmin: Minimum frequency for the spectra (in units of tvec**-1)
    @param fmax: Maximum frequency for the spectra (in units of tvec**-1)
    @param tmin: Minimum time for the spectra (in units of tvec)
    @param tmax: Maximum time for the spectra (in units of tvec)
    @param pass_DC: remove constant background
    @param complex_spectrum: To return a complex or real spectra
    """
    if signal.dtype in [np.double, np.cdouble]:
        raise Exception('use dtype single or csingle')

    complex_sig = signal.dtype == np.csingle

    iimin, iimax = tvec.searchsorted((tmin, tmax))  # BUG SLOW for large signal
    iimax -= 1
    n = iimax-iimin
    # BUG will be wrong for not equally spaced time vectors
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
    sig = pyfftw.n_byte_align_empty(signal.shape[1:]+(nfft,), nalign,
                                    dtype=sig_dtype)
    # output signal
    nf = nfft if complex_sig else nfft//2+1
    out = pyfftw.n_byte_align_empty(signal.shape[1:]+(nf,),
                                    nalign, dtype=np.complex64)

    fft_forward = pyfftw.FFTW(sig, out, direction='FFTW_FORWARD',
                              flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'],
                              axes=(-1,), threads=cpu_count()//2)

    fft_step = max(1, n//resolution)
    dtype = np.complex64 if complex_spectrum else np.single
    spec = np.empty((int(n/fft_step), ifmax-ifmin) + signal.shape[1:],
                    dtype=dtype)
    win = None

    for i in range(int(n//fft_step)):
        imin = int(max(0, iimin+np.floor(i*fft_step-nfft//2)))
        imax = int(min(len(signal), iimin + np.floor(i*fft_step+nfft//2)))

        if np.size(win) != imax-imin:
            if window == 'gauss':
                win = get_window((window, f_nfft*(imax-imin)//8//nfft),
                                 imax-imin, fftbins=False)
            else:
                win = get_window(window, imax-imin, fftbins=False)

            win = win.astype('single')

        # implicit conversion from int (or anything else) to single!
        if pass_DC:
            sig[..., :imax-imin] = signal[imin:imax].T
        else:
            sig[..., :imax-imin] = signal[imin:imax].T
            sig[..., :imax-imin] -= np.mean(signal[imin:imax], 0)[None].T

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
# -----------------------------------------------------------------------------
# --- Noise filtering
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# --- Cross-power density calculation.
# -----------------------------------------------------------------------------
def get_nfft(tau0, specType, nt, windowType, dt):
    """
    Getting the number of FFT required to reach the resolution.
    
    by Giovanni Tardini
    
    @param x: Time resolution parameter
    @param specType: kind of spectrum.
    @param nt: Number of time points.
    @param windowType: kind of window applied.
    @param dt: time step.
    @return nfft: number of points to overlap.
    """
    if specType in ('sparse', 'frequency', 'stfft', 'sfft'):
        tau0 = 1.20 - tau0
    
    tau = nt**tau0 * dt
    nfft = tau/dt if windowType == 'gauss' else 2**np.ceil(np.log2(tau/dt))
    
    return nfft


def myCPSD(sig1, sig2, time1, freq1, time2, freq2):
    """
    Computing the cross-power spectral density for two complex input signals.
    The spectrograms of the input signals may be defined onto two different 
    time-frequency basis. The second one will interpolated along the first.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param sig1: complex matrix. Spectrogram of the 1st signal.
    @param sig2: complex matrix. Spectrogram of the 2nd signal.
    @param time1: Time basis of the first signal. Also time output.
    @param freq1: Frequency basis of the first signal. Also the freq. output.
    @param time2: Time basis onto which the 2nd signal is defined. 
    @param freq2: Frequency basis of the 2nd signal is defined. 
    """
    
    #--- We interpolate the signal 2 into the time and frequency basis of the 
    ## first.
    sig2_fun_r  = interp2d(time2, freq2, sig2.real, kind='linear',
                           bounds_error=False)
    sig2_fun_i  = interp2d(time2, freq2, sig2.imag, kind='linear',
                           bounds_error=False)
    sig2_on1  = sig2_fun_r(time1, freq1) + 1j*sig2_fun_i(time1, freq1)
    
    # Computing the element-wise matrix product:
    return time1, freq1, (sig1*sig2_on1)



