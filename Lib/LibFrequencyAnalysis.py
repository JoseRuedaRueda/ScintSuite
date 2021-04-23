"""
Routines to analyse a time signal in the frequency domain

Include bandpass signal and other filtres aimed to reduce the noise
"""
import numpy as np
import pyfftw
import heapq
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as colorMap
from scipy.signal import get_window
from scipy.fftpack import fftfreq, rfft, ifft, fftshift
from multiprocessing import cpu_count
from scipy.fftpack import next_fast_len
from scipy.interpolate import interp2d
from collections import defaultdict
from LibPlotting import p1D_shaded_error as plot_error_band

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
# -----------------------------------------------------------------------------
# --- Noise filtering
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# --- Cross-power density calculation.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# --- Tracking algorithm
# -----------------------------------------------------------------------------
def trackFrequency(time: float, freq: float, spec: float, origin: float, 
                   target: float=None, freqLims: float=None, 
                   timeLims: float=None,
                   costFunction=None, peakFilterFnc=None,
                   tOverlap: float=None, smooth: bool=True,
                   smooth_data: dict={}, peak_opts: dict={}, 
                   verbose: bool=True, plotandwait: bool=True, 
                   plotOpts: dict = {}):
    """
    This function will try to follow a frequency in a spectrogram based on
    peak detection algorithm + Dijsktra's algorithm. To make this function 
    more flexible, a particular limit in frequency and time can be used. In 
    particular, masked arrays can (should?) be used as inputs and outputs.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param time: array with the time points (equispaced)
    @param freq: arraq with the frequency points (equispaced)
    @param spec: spectrogram 2D array. Mask arrays can be used (and will
    make the work even easier)
    @param freqLims: frequency windows to look for the modes. If None, all the
    frequency window is used instead.
    @param timeLims: time windows to look for the peaks. This can be useful for
    ELM synchronization. If None, the whole time window is used.
    @param costFunction: callable function to provided two nodes (time, freq,
    spec) gives the cost (only positive) that will be used to weight the path
    maker. If None, a standard w = Delta_frequency**2 is used to avoid
    frequency jumps.
    @param peakFilterFnc: function to filter peaks. It must admit 
    (time, freq, spec) and (peaks, width) as inputs to filter them. If None,
    all the peaks found are used.
    @param tOverlap: sum over time when looking for peaks. If None, the sum
    is set to dt.
    """
    dt = time[1] - time[0]
    nfreq = len(freq)
    ntime = len(time)
    
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
    else:
        timeLims = [time[0], time[-1]]
        
    if freqLims is None:
        freqLims = [freq[0], freq[-1]]
        
    # --- Checking the filtering data.
    if 'window_length' not in smooth_data:
        smooth_data['window_length'] = 5
    elif smooth_data['window_length'] % 2 == 0:
        print('The smoothing window must be odd in points. Increasing by 1')
        smooth_data['window_length'] += 1
    
    if 'polyorder' not in smooth_data:
        smooth_data['polyorder'] = 3
        
    # --- Checking the peaking finding data.
    if 'prominence' not in peak_opts:
        peak_opts['prominence'] = 1.0 - 1.0/np.exp(1.0) # 66.7%
    if 'width' not in peak_opts:
        peak_opts['width'] = (None, None)
        
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
    
    if verbose:
        print('Shape of spec = '+str(spec.shape))
        print('Shape of time = '+str(time.shape))
        print('Shape of freq = '+str(freq.shape))
    # This will help us mapping the node ID to the peak time-frequency.
    peak_map = defaultdict(list)
    kk = 0 # Index running for the time ordered list.
    for ii in np.arange(nwindows, dtype=int):
        t0 = np.abs(time - timeLims[2*ii]).argmin()
        t1 = np.abs(time - timeLims[2*ii + 1]).argmin()
        
        nTimes_slices = int((t1-t0+1)/nOverlap)
        for jj in np.arange(nTimes_slices, dtype=int):
            t0_avg = t0 + jj*nOverlap
            t1_avg = np.minimum(t0+(jj+1)*nOverlap, t1+1)
            
            data = np.mean(spec[t0_avg:t1_avg, :], axis=0)
            data -= np.min(data)
            data /= np.max(data)
            
            if smooth:
                try:
                    data = signal.savgol_filter(data, **smooth_data)
                except:
                    print('Data at t = %d non-smoothed!'%t0_avg)
            
            peaks, props = signal.find_peaks(data, **peak_opts)
            
            # Filtering the peaks, according to their frequency.
            flags = np.logical_and(freq[peaks] >= freqLims[0], 
                                   freq[peaks] <= freqLims[-1])
            
            peaks = peaks[flags]
            for ikey in props.keys():
                props[ikey] = props[ikey][flags]
            
            # If after everything, there are not any peaks, just go to the
            # next slice.
            if len(peaks) == 0:
                continue
                
            # External filtering function.
            if peakFilterFnc is not None:
                flags = peakFilterFnc(freq, data, peaks, props)
                peaks = peaks[flags]
                for ikey in props.keys():
                    props[ikey] = props[ikey][flags]
            
            # If after everything, there are not any peaks, just go to the
            # next slice.
            if len(peaks) == 0:
                continue
                    
            # Adding the peak data to the list.
            peak_data['peak_idx'][kk] = peaks
            peak_data['width'][kk] = props['width_heights']
            peak_data['freq'][kk] = freq[peaks]
            peak_data['spec_val'][kk]= np.mean(spec[t0_avg:t1_avg, peaks],
                                               axis=0)
            peak_data['spec_norm'][kk]  = data[peaks]
            peak_data['prominences'][kk] = props['prominences']
            
            time_idx = int(np.floor((t0_avg+t1_avg)/2.0)) * \
                       np.ones(len(peaks), dtype=int)
            peak_data['time'][kk] = time[time_idx]
            peak_data['timeList'][kk] = kk*nfreq*np.ones(len(time_idx)) + \
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
        
    if plotandwait:
        fig, ax = plt.subplots(1)
        im1 = ax.pcolormesh(time, freq, spec.T, **plotOpts)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [kHz]')
        fig.colorbar(im1, ax=ax)
        
        # Plotting the peaks
        for ii in peak_map:
            ax.plot(peak_map[ii][0], peak_map[ii][1], 'r.')
        
        print('Check the input and tell me: 0 -> finish here\n')
        print('                          else -> keep going\n')
        bb = int(input())
        
        if bb == 0:
            output = { 
              'peak_data': peak_data,
              'peak_map': peak_map
             }
            print('bye-bye')
    
            del dgraph
            return output
    
    ntime_peaks = kk - 1
    # --- Generating the graph.
    # The graph will conect the timepoints with the next timepoints peaks only
    for ii in np.arange(ntime_peaks, dtype=int):
        for jj, frm in enumerate(peak_data['timeList'][ii]):
            for kk, to in enumerate(peak_data['timeList'][ii + 1]):
                peak_prv = { 'time': peak_data['time'][ii][jj],
                             'freq': peak_data['freq'][ii][jj],
                             'width': peak_data['width'][ii][jj],
                             'spec_val': peak_data['spec_val'][ii][jj],
                             'spec_norm': peak_data['spec_norm'][ii][jj],
                           }
                peak_nxt = { 'time': peak_data['time'][ii+1][kk],
                             'freq': peak_data['freq'][ii+1][kk],
                             'width': peak_data['width'][ii+1][kk],
                             'spec_val': peak_data['spec_val'][ii+1][kk],
                             'spec_norm': peak_data['spec_norm'][ii+1][kk],
                           }
                if costFunction is None:
                    deltaF = np.abs(peak_prv['freq'] - peak_nxt['freq'])
                    # deltaA = np.abs(peak_prv['spec_norm'] - \
                    #                 peak_nxt['spec_norm'])
                    sigmaF = np.sqrt(peak_prv['width']**2.0 + 
                                     peak_nxt['width']**2.0)
                    
                    cost = deltaF**2.0
                else:
                    cost = costFunction(prv=peak_prv, nxt=peak_nxt,
                                        origin=False, target=False)
                        
                dgraph.add_edge(frm=frm, to=to, cost=cost, forceAdd=False)
    
    # --- Adding the origin vertex in the graph.
    dgraph.add_vertex('origin')
    peak_origin = { 'time': [],
                    'freq': origin,
                    'width': [],
                    'spec_val': [],
                    'spec_norm': [],
                }
    for ii, to in enumerate(peak_data['timeList'][0]):
        peak_nxt = { 'time': peak_data['time'][0][ii],
                     'freq': peak_data['freq'][0][ii],
                     'width': peak_data['width'][0][ii],
                     'spec_val': peak_data['spec_val'][0][ii],
                     'spec_norm': peak_data['spec_norm'][0][ii],
                   }
        if costFunction is None:
            deltaF = np.abs(peak_origin['freq'] - peak_nxt['freq'])
            cost = deltaF
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
                cost = deltaF
            else:
                cost = costFunction(prv=peak_prv, nxt=peak_target,
                                    origin=False, target=True)
        
            dgraph.add_edge(frm=frm, to='target', cost=cost, forceAdd=False)
            
    # --- Using Dijsktra.
    # The Dijsktra implemented in this library uses one source-one target.
    # The origin is the typical input, but the ending point is free and to
    # be determined by the algorithm. 
    dgraph.Dijsktra(dgraph.get_vertex('origin'), verbose=False)
    
    # --- Getting the final point:
    path = list()
    if target:
        graph_shortest(dgraph.get_vertex('target'), path)    
    else:
        # Loop over all the endings to see which is the target vertex.
        v = np.zeros((len(peak_data['timeList'][ntime_peaks]),))
        for ii, node in enumerate(peak_data['timeList'][ntime_peaks]):
            v[ii] = dgraph.get_vertex(node=node).distance
        
        imin = v.argmin()
        distmin = v[imin]
        graph_shortest(dgraph.get_vertex(peak_data['timeList'][ntime_peaks][imin]), 
                       path)   
        
    # --- Translating the path into the curve (t, freq)
    timecurve = list()
    freqcurve = list()
    ampcurve_norm  = list()
    ampcurve_total = list()
    widths_curve   = list()
    for ii in path:
        if ii == 'origin':
            break
        timecurve.append(peak_map[ii][0])
        freqcurve.append(peak_map[ii][1])
        ampcurve_norm.append(peak_map[ii][2])
        ampcurve_total.append(peak_map[ii][3])
        widths_curve.append(peak_map[ii][4])
        
    output = { 'track': { 'time': np.array(timecurve),
                          'freq': np.array(freqcurve),
                          'Anorm': np.array(ampcurve_norm),
                          'Atot': np.array(ampcurve_total),
                          'width': np.array(widths_curve)
                        },
              'peak_data': peak_data,
              'peak_map': peak_map,
              'path_by_graph': path,
              'cost': distmin
             }
    
    # --- Print the curve.
    if plotandwait:
        ax=plot_error_band(ax=ax, x=output['track']['time'],
                           y=output['track']['freq'], color='b', 
                           u_up=output['track']['width']/2.0, 
                           alpha=0.2, line=True)
    del dgraph
    
    return output
        
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
        
        @param node: ID of the node.
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

    def add_neighbor(self, neighbor, weight: float=0.0):
        """
        Add a neighbour to the current vertex and adds its corresponding 
        weight.
        
        Pablo Oyola - pablo.oyola@ipp.mpg.de
        
        @param neighbor: identification of the neighbour (a string, number,...)
        @param weight: the weight to go from one point to the next. Only
        non-negative values. 0 implies direct connection, Inf means no
        connection at all. In the latter, it will not be added to the list.
        """
        if weight < 0.0:
            raise Exception('Weigths must be non-negative numbers!')
        elif weight == np.inf:
            print('The neighbour'+str(neighbor)+' is not added to the list\n')
            print('because its cost is Infinity')
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
        
        @param neigbor: name or identification of the neighbour whose weight
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
        
        @param node_id: identificator of the node.
        @return new_vertex: Vertex class created.
        """
        
        # --- Check that there are no collisions.
        if node_id in self.vert_dict:
            raise Exception('The vertex ID is repeated!')
            
        # --- Creating the vertex.
        new_vertex = Vertex(node_id)
        self.vert_dict[node_id] = new_vertex # Adding to the list.
        
        self.nVertices += 1
        
        return new_vertex
            
    def get_vertex(self, node):
        """
        Get the vertex associated with the ID node.
        
        Pablo Oyola - pablo.oyola@ipp.mpg.de
        
        @param node: node identificator to get the Vertex class.
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
        
        @param frm: node-id of the starting node.
        @param to:  node-id of the ending node.
        @param cost: weighting of the node-node connection.
        @param forceAdd: force to add the two new vertices 'frm'&'to' into the
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
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], weight=cost)
        
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
        
        @param agraph: graph class containing the collection of vertices.
        @param start: starting vertex.
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
                            
                # else:
                #     if verbose:
                #         print('Non-updated: current = ' + str(current.id)+'\n'+\
                #               'next = '     + str(nxt.id)     + '\n' + \
                #                'new_dist = ' + str(nxt.distance) + '\n')
                            
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
    
    @param v: ending vertex.
    @param path: vertex path back to the origin. A collection of the IDs.
    """
    if v.previous:
        path.append(v.previous.id)
        graph_shortest(v.previous, path)
    
    return