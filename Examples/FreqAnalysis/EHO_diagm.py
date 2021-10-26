"""
EHOs tracker.

This example is based upon the combination of several libraries: magnetic
spectrogram reader + frequency tracking in the spectrogram looking for modes
that are increasing in toroidal mode number 'n' separated by the vtor.

This script starts using one single magnetic pick-up coil as feed for 
searching the first EHO frequency, typically lying in the frequency range
(1, 7) kHz. Once the first signal is found and accepted, upper harmonics are
searched using the f0 + n*vtor.
"""

#import matplotlib
import Lib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as colormaps
import warnings
import numpy as np
import Lib.LibFrequencyAnalysis as lf
from Lib.LibData.AUG.Profiles import get_diag_freq as diagCorr
from LibPlotting import p1D_shaded_error as plot_error_band
from scipy.interpolate import interp1d


#%% Script startup
# -----------------------------------------------------------------------------
# --- Script configuration.
# -----------------------------------------------------------------------------
shotnumber = 37289   # Shot number to analyze.
tBegin     = 1.55    # Starting point of the time window.
tEnd       = 1.65    # Ending point of the time window.
coilNumber = 1       # Ballooning coil number (B31-XX)

# Toroidal rotation diagnostics.
vtor_diag = 'CXRS'
vtor_exp  = 'AUGD'
vtor_edition = 0
dR_shift = { 'CMZ': 0.0e-3,   # Radial shift.
           }

rhopol_min = 0.70   # Mininum point to take the rotation velocity.
rhopol_max = 1.00   # Maximum point to take the rotation velocity.
rhopol_mode= 0.995   # A guess where the mode is :) ?
sigma_vtor = 0.80   # Width to apply to the toroidal rotation.

# FFT options.
                        # For the window type, go to:
windowType = 'hann'     # https://docs.scipy.org/doc/scipy/reference/
                        # generated/scipy.signal.get_window.html

freqLimit = np.array((1.0, 40.0)) # Frequency limits.
specType = 'stft2'  # Spectogram type:
                    # -> Short-Time Fourier Transform in frequency (sfft)
                    # -> Short-Time Fourier Transform in time (stft)
                    # -> Short-Time Fourier Transform in time (scipy version)
resolution = 1000
timeResolution = 0.80 # Time resolution.
cmap = colormaps.plasma # Colormap
spec_abstype = 'lin' # linear, sqrt or log

# -----------------------------------------------------------------------------
# --- Configuration of the tracker. 
# -----------------------------------------------------------------------------
peakOpts = { 'prominence': 0.10, # This sets the relative height to consider a 
                                 # peak. This needs to be varied in case the 
                                 # frequency path is not the one with highest
                                 # amplitude. An array like (0.4, 0.5) would 
                                 # choose the appropriate range.
             'rel_height': 0.50  # Relative height (to the peak) at which the
                                 # height is measured.
           }

timeConnection = 5.0e-4  # Allow for peaks to be connected within this time
                         # range. This is given in [s]
freqThreshold = 10.0 # Jumps above this frequency are disregarded.

verbose = False # Write to output the process of peak search.

limsFromGUI = True # Retrieve the starting point and other from the GUI.
timeMin = 1.601
freqmin = 6.0
freqmax = 8.0
freqOrigin = 7.0
sigma = 1.0
search_high_n = False

#%% Reading magnetics and making spectrogram.
# -----------------------------------------------------------------------------
mhi = Lib.dat.get_magnetics(shotnumber, coilNumber, timeWindow=[tBegin, tEnd])

# Getting the parameters for the spectrogram.
dt = mhi['time'][1] - mhi['time'][0] # For the sampling time.
nfft = int(lf.get_nfft(timeResolution, specType, 
           len(mhi['time']), windowType, dt))

warnings.filterwarnings('ignore', category=DeprecationWarning)
if specType != 'stft2':
    raise Exception('The methods for the spectrogram must be stft-scipy to\
                    preserve the amplitude!')
                    
Sxx, freqs, times = lf.stft2(mhi['time']-tEnd,  mhi['data'], nfft, 
                             window=windowType,
                             pass_DC=False, complex_spectrum=True,
                             resolution=resolution)
mhi['fft'] = {
                'freq': freqs/1000.0,
                'time': times+tEnd, 
                'spec': Sxx
             }

Bdot = mhi['fft']['spec'].T

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

f0, f1 = mhi['fft']['freq'].searchsorted(freqLimit)

#%% Spectrogram plotting.
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

#%% Asking the user for starting and ending points in the spectrum.
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
    
    plt.ginput(timeout=0.1) # This forces the rendering.
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
lineOpts = { 'color': 'w', 'linestyle': '--'}
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


#%% Filtering
# -----------------------------------------------------------------------------
y_center_fun = interp1d(trk['track']['time'], trk['track']['freq'], 
                        kind='linear', bounds_error=False, 
                        fill_value='extrapolate', assume_sorted=True)
s_center_fun = interp1d(trk['track']['time'], 
                        sigma*trk['track']['width']/2.772, 
                        kind='linear', bounds_error=False, 
                        fill_value='extrapolate', assume_sorted=True)
time2, freq2 = np.meshgrid(time, freq)

new_B = mhi['fft']['spec'].copy()
mhi_new = { 'fft': {
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

#%%  Loading the rotation velocity.
# -----------------------------------------------------------------------------
vtor = Lib.dat.get_tor_rotation(shotnumber=shotnumber, time=[tBegin, tEnd],
                                diag=vtor_diag, exp=vtor_exp, 
                                edition=vtor_edition,
                                smooth_factor = 150.0, rhop0=0.2,
                                rhop1=1.10, dr=dR_shift)

# We use as a feed for the next toroidal mode number search the toroidal 
# rotation velocity at the edge. We make an interpolation in time.

# The CXRS shall be treated differently.
if vtor_diag == 'CXRS':
    vtor_time = vtor['fit']['time'].copy()[1:]
    vtor_data = vtor['fit']['data'].copy()[1:, :]
    vtor_rhop = vtor['fit']['rhop'].copy()
else:
    vtor_time = vtor['time']
    vtor_data = vtor['data']
    vtor_rhop = vtor['rhop']
    
# Getting the colors for plotting.
cmap_line = colormaps.tab20(np.linspace(0, 1.0, len(vtor_time)))
    
fig5, ax5 = plt.subplots(1)
for ii, t in enumerate(vtor_time):
    ax5.plot(vtor_rhop, vtor_data[ii, :].copy()/(2.0*np.pi*1e3), 
             label='t = %.3f s'%t,
             color=cmap_line[ii])
    if vtor_diag == 'CXRS':
        for jj in np.arange(len(vtor['raw']['rhopol'])):
            ax5.plot(vtor['raw']['rhopol'][jj][ii, :],
                     vtor['raw']['data'][jj][ii, :].copy()/(2.0*np.pi*1e3), 
                     '.', color=cmap_line[ii])
            
ax5.set_xlabel('$\\rho_{pol}$')
ax5.set_ylabel('$f_{tor}$ [kHz]')
ax5.set_title('#%05d - Toroidal rotation'%shotnumber)

vtor_data = vtor_data/(2.0*np.pi*1e3)

# We create an interpolator able to ignore the NaN in the 2D data.
omega_tor_fun = interp1d(vtor_rhop, np.nanmean(vtor_data, axis=0),
                         kind='linear', fill_value='extrapolate',
                         assume_sorted=True)

somega_tor_fun = interp1d(vtor_rhop, np.nanstd(vtor_data, axis=0),
                         kind='linear', fill_value='extrapolate',
                         assume_sorted=True)

#%% Computing the diamagnetic contribution.
diagmag_corr = diagCorr(shotnumber=shotnumber, tBegin=tBegin, tEnd=tEnd)


fig10, ax10 = plt.subplots(1)
ax10.plot(diagmag_corr['rhop'], diagmag_corr['fdiag'])


vdiag_data = diagmag_corr['fdiag']
flags = np.isinf(vdiag_data)
vdiag_data[flags] = np.nan

# With this diamagnetic correction, we modify the toroidal rotation velocity.
omega_diagm_fun = interp1d(diagmag_corr['rhop'], 
                           np.nanmean(vdiag_data, axis=-1),
                           kind='linear',
                           assume_sorted=True, 
                           fill_value='extrapolate',
                           bounds_error=False)

somega_diagm_fun = interp1d(diagmag_corr['rhop'],
                           np.nanstd(vdiag_data, axis=-1),
                           kind='linear',
                           assume_sorted=True, 
                           fill_value='extrapolate',
                           bounds_error=False)


omega_tot_fun  = lambda x: - omega_tor_fun(x) + omega_diagm_fun(x)
somega_tot_fun = lambda x: np.sqrt(somega_diagm_fun(x)**2.0 + \
                                   somega_tor_fun(x)**2.0)

#%% Searching for the next EHO frequency.
# -----------------------------------------------------------------------------
time = mhi['fft']['time']
freq = mhi['fft']['freq']
data = np.abs(mhi['fft']['B'])

# Shifting the origin to the new origin.
origin = y_center_fun(timeMin) + omega_tot_fun(rhopol_mode)
freqMax_new = origin + np.abs(freqLims[1] - freqLims[0])
freqMin_new = freqLims[1]

if freqMax_new < freqMin_new:
    tmp = freqMax_new
    freqMax_new = freqMin_new
    freqMin_new = tmp
    del tmp
print('New frequency limits = [%.2f, %.2f] kHz'%(freqMin_new, 
                                             freqMax_new))
print('New search origin f0 = %.2f kHz'%origin)

timeLims = [timeMin, tEnd]
freqLims = [freqMin_new, freqMax_new]

trk2, ax = lf.trackFrequency(time=time, freq=freq, spec=data, 
                            origin=origin, plot=True, 
                            peak_opts=peakOpts, 
                            graph_TimeConnect=timeConnection, 
                            freqLims=freqLims, timeLims=timeLims, 
                            freqThr=freqThreshold, ax=ax, fig=fig)

plt.show()

#%% Computing the frequency difference and locating the position.
y_fun_n1 = interp1d(trk['track']['time'][1:], trk['track']['freq'][1:], 
                    bounds_error=False, assume_sorted=True, kind='cubic')
y_fun_n2 = interp1d(trk2['track']['time'][1:], trk2['track']['freq'][1:], 
                    bounds_error=False, assume_sorted=True, kind='cubic')

s_fun_n1 = interp1d(trk['track']['time'][1:], trk['track']['width'][1:]/2.772, 
                    bounds_error=False, assume_sorted=True, kind='cubic')

s_fun_n2 = interp1d(trk2['track']['time'][1:], trk2['track']['width'][1:]/2.772, 
                    bounds_error=False, assume_sorted=True, kind='cubic')


time4dy_eval = np.linspace(trk['track']['time'][1], trk['track']['time'][-1])
dy_n1_2 = y_fun_n2(time4dy_eval) - y_fun_n1(time4dy_eval)
sy_n1_2 = np.sqrt(s_fun_n1(time4dy_eval)**2.0 + s_fun_n2(time4dy_eval)**2.0)


lineOpts = { 'color': 'b', 'label': 'n=1 vs. n=2'}
fig7, ax7 = plt.subplots(1)
ax7 = plot_error_band(x=time4dy_eval, y=dy_n1_2, color='b', u_up=sy_n1_2, 
                      alpha=0.1, line=True, line_param=lineOpts, ax=ax7)

ax7.set_xlabel('Time [s]')
ax7.set_ylabel('Frequency difference [kHz]')

rhop_mesh = np.linspace(start=0.80, stop=1.00,
                        num=512)
rhop_central = np.zeros(time4dy_eval.shape)
rhop_lb = np.zeros(time4dy_eval.shape)
rhop_ub = np.zeros(time4dy_eval.shape)

# For each time slice, we get the frequency.
vt_tmp = omega_tot_fun(rhop_mesh)
dvt_tmp = omega_tot_fun(rhop_mesh)
for ii, t in enumerate(time4dy_eval):
    # Finding the central point:
    idx = np.abs(vt_tmp -  dy_n1_2[ii]).argmin()
    rhop_central[ii] = rhop_mesh[idx]
    
    # Finding the lower bound.
    idx = np.abs(vt_tmp -  (dy_n1_2[ii] - sy_n1_2[ii])).argmin()
    rhop_lb[ii] = rhop_mesh[idx]
    
    # Finding the lower bound.
    idx = np.abs(vt_tmp -  (dy_n1_2[ii] + sy_n1_2[ii])).argmin()
    rhop_ub[ii] = rhop_mesh[idx]
    
fig8, ax8 = plt.subplots(1)
error_up = np.abs(rhop_lb[1:] - rhop_central[1:] )
error_dw = np.abs(rhop_ub[1:]  - rhop_central[1:] )
ax8 = plot_error_band(x=time4dy_eval[1:] , y=rhop_central[1:], 
                      color='b', u_up=error_up, 
                      u_down=error_dw, alpha=0.2, line=True, 
                      line_param=lineOpts, ax=ax8)

ax8.set_xlabel('Time [s]')
ax8.set_ylabel('$\\rho_{pol}$')
plt.show()
plt.ginput(timeout=0.01)

#%% Higher modes! n= 1, 3
# If you get to this part it means the magnetic pick-up coils has more modes
# you can analyze!
if search_high_n:
    time = mhi['fft']['time']
    freq = mhi['fft']['freq']
    data = np.abs(mhi['fft']['B'])
    
    
    # The feed for the frequency:
    origin_n2_3 = origin + dy_n1_2[1]  # We add on top the first frequency
    freqMax_new23 = origin_n2_3 + np.abs(freqMax_new - freqMin_new)/2.0
    freqMin_new23 = freqMax_new
    
    if freqMax_new < freqMin_new:
        tmp = freqMax_new
        freqMax_new = freqMin_new
        freqMin_new = tmp
        del tmp
    print('New frequency limits = [%.2f, %.2f] kHz'%(freqMin_new23, 
                                                 freqMax_new23))
    print('New search origin f0 = %.2f kHz'%origin_n2_3)
    
    timeLims = [timeMin, tEnd]
    freqLims = [freqMin_new23, freqMax_new23]
    
    trk3, ax = lf.trackFrequency(time=time, freq=freq, spec=data, 
                                 origin=origin_n2_3, plot=True, 
                                 peak_opts=peakOpts, 
                                 graph_TimeConnect=timeConnection, 
                                 freqLims=freqLims, timeLims=timeLims, 
                                 freqThr=freqThreshold, ax=ax, fig=fig)
    
    # Computing the frequency difference between the two tracks.
    y_fun_n3 = interp1d(trk3['track']['time'][1:], trk3['track']['freq'][1:], 
                        bounds_error=False, assume_sorted=True, kind='cubic')
    
    s_fun_n3 = interp1d(trk3['track']['time'][1:], trk3['track']['width'][1:]/2.772, 
                        bounds_error=False, assume_sorted=True, kind='cubic')
    
    dy_n1_3 = y_fun_n3(time4dy_eval) - y_fun_n1(time4dy_eval)
    sy_n1_3 = np.sqrt(s_fun_n1(time4dy_eval)**2.0 + s_fun_n3(time4dy_eval)**2.0)
    
    # Plotting the frequency difference:
    lineOpts = { 'color': 'r', 'label': 'n=1 vs. n=3 (div. by 2)'}
    ax7 = plot_error_band(x=time4dy_eval, y=dy_n1_3/2.0, color='r', u_up=sy_n1_3, 
                          alpha=0.1, line=True, line_param=lineOpts, ax=ax7)
    
    rhop_central13 = np.zeros(time4dy_eval.shape)
    rhop_lb13 = np.zeros(time4dy_eval.shape)
    rhop_ub13 = np.zeros(time4dy_eval.shape)
    vt_tmp = 2*omega_tot_fun(rhop_mesh)
    for ii, t in enumerate(time4dy_eval):
        # Finding the central point:
        idx = np.abs(vt_tmp -  dy_n1_3[ii]).argmin()
        rhop_central13[ii] = rhop_mesh[idx]
        
        # Finding the lower bound.
        idx = np.abs(vt_tmp -  (dy_n1_3[ii] - sy_n1_3[ii])).argmin()
        rhop_lb13[ii] = rhop_mesh[idx]
        
        # Finding the lower bound.
        idx = np.abs(vt_tmp -  (dy_n1_3[ii] + sy_n1_3[ii])).argmin()
        rhop_ub13[ii] = rhop_mesh[idx]
    
    error_up = np.abs(rhop_lb13[1:] - rhop_central13[1:] )
    error_dw = np.abs(rhop_ub13[1:]  - rhop_central13[1:] )
    ax8 = plot_error_band(x=time4dy_eval[1:] , y=rhop_central13[1:], 
                          color='r', u_up=error_up, 
                          u_down=error_dw, alpha=0.1, line=True, 
                          line_param=lineOpts, ax=ax8)


    #%% Higher modes! n = 1, 4
    # If you get to this part it means the magnetic pick-up coils has more modes
    # you can analyze!
    time = mhi['fft']['time']
    freq = mhi['fft']['freq']
    data = np.abs(mhi['fft']['B'])
    
    
    # The feed for the frequency:
    origin_n3_4 = origin_n2_3 + dy_n1_2[1]  # We add on top the first frequency
    freqMax_new34 = origin_n3_4 + np.abs(freqMax_new23 - freqMin_new23)/2.0
    freqMin_new34 = freqMax_new23
    
    if freqMax_new34 < freqMin_new34:
        tmp = freqMax_new34
        freqMax_new34 = freqMin_new34
        freqMin_new34 = tmp
        del tmp
    print('New frequency limits = [%.2f, %.2f] kHz'%(freqMin_new34, 
                                                 freqMax_new34))
    print('New search origin f0 = %.2f kHz'%origin_n3_4)
    
    timeLims = [timeMin, tEnd]
    freqLims = [freqMin_new34, freqMax_new34]
    
    trk4, ax = lf.trackFrequency(time=time, freq=freq, spec=data, 
                                 origin=origin_n3_4, plot=True, 
                                 peak_opts=peakOpts, 
                                 graph_TimeConnect=timeConnection, 
                                 freqLims=freqLims, timeLims=timeLims, 
                                 freqThr=freqThreshold, ax=ax, fig=fig)
    
    # Computing the frequency difference between the two tracks.
    y_fun_n4 = interp1d(trk4['track']['time'][1:], trk4['track']['freq'][1:], 
                        bounds_error=False, assume_sorted=True, kind='cubic')
    
    s_fun_n4 = interp1d(trk4['track']['time'][1:], trk4['track']['width'][1:]/2.772, 
                        bounds_error=False, assume_sorted=True, kind='cubic')
    
    dy_n1_4 = y_fun_n4(time4dy_eval) - y_fun_n1(time4dy_eval)
    sy_n1_4 = np.sqrt(s_fun_n1(time4dy_eval)**2.0 + s_fun_n4(time4dy_eval)**2.0)
    
    # Plotting the frequency difference:
    lineOpts = { 'color': 'g', 'label': 'n=1 vs. n=4 (div. by 3)'}
    ax7 = plot_error_band(x=time4dy_eval, y=dy_n1_4/3.0, color='g', u_up=sy_n1_4, 
                          alpha=0.1, line=True, line_param=lineOpts, ax=ax7)
    
    rhop_central14 = np.zeros(time4dy_eval.shape)
    rhop_lb14 = np.zeros(time4dy_eval.shape)
    rhop_ub14 = np.zeros(time4dy_eval.shape)
    vt_tmp = 3*omega_tot_fun(rhop_mesh)
    for ii, t in enumerate(time4dy_eval):
        # Finding the central point:
        idx = np.abs(vt_tmp -  dy_n1_4[ii]).argmin()
        rhop_central14[ii] = rhop_mesh[idx]
        
        # Finding the lower bound.
        idx = np.abs(vt_tmp -  (dy_n1_4[ii] - sy_n1_4[ii])).argmin()
        rhop_lb14[ii] = rhop_mesh[idx]
        
        # Finding the lower bound.
        idx = np.abs(vt_tmp -  (dy_n1_4[ii] + sy_n1_4[ii])).argmin()
        rhop_ub14[ii] = rhop_mesh[idx]
    
    error_up = np.abs(rhop_lb14[1:] - rhop_central14[1:] )
    error_dw = np.abs(rhop_ub14[1:]  - rhop_central14[1:] )
    ax8 = plot_error_band(x=time4dy_eval[1:] , y=rhop_central14[1:], 
                          color='g', u_up=error_up, 
                          u_down=error_dw, alpha=0.1, line=True, 
                          line_param=lineOpts, ax=ax8)

    #%% Higher modes! n = 1, 5
    # If you get to this part it means the magnetic pick-up coils has more modes
    # you can analyze!
    time = mhi['fft']['time']
    freq = mhi['fft']['freq']
    data = np.abs(mhi['fft']['B'])
    
    
    # The feed for the frequency:
    origin_n4_5 = origin_n3_4 + dy_n1_2[1]  # We add on top the first frequency
    freqMax_new45 = origin_n4_5 + np.abs(freqMax_new34 - freqMin_new34)/2.0
    freqMin_new45 = freqMax_new34
    
    if freqMax_new45 < freqMin_new45:
        tmp = freqMax_new45
        freqMax_new45 = freqMin_new45
        freqMin_new45 = tmp
        del tmp
    print('New frequency limits = [%.2f, %.2f] kHz'%(freqMin_new45, 
                                                 freqMax_new45))
    print('New search origin f0 = %.2f kHz'%origin_n4_5)
    
    timeLims = [timeMin, tEnd]
    freqLims = [freqMin_new45, freqMax_new45]
    
    trk5, ax = lf.trackFrequency(time=time, freq=freq, spec=data, 
                                 origin=origin_n4_5, plot=True, 
                                 peak_opts=peakOpts, 
                                 graph_TimeConnect=timeConnection, 
                                 freqLims=freqLims, timeLims=timeLims, 
                                 freqThr=freqThreshold, ax=ax, fig=fig)
    
    # Computing the frequency difference between the two tracks.
    y_fun_n5 = interp1d(trk5['track']['time'][1:], trk5['track']['freq'][1:], 
                        bounds_error=False, assume_sorted=True, kind='cubic')
    
    s_fun_n5 = interp1d(trk5['track']['time'][1:], trk5['track']['width'][1:]/2.772, 
                        bounds_error=False, assume_sorted=True, kind='cubic')
    
    dy_n1_5 = y_fun_n5(time4dy_eval) - y_fun_n1(time4dy_eval)
    sy_n1_5 = np.sqrt(s_fun_n1(time4dy_eval)**2.0 + s_fun_n5(time4dy_eval)**2.0)
    
    # Plotting the frequency difference:
    lineOpts = { 'color': 'k', 'label': 'n=1 vs. n=5 (div. by 4)'}
    ax7 = plot_error_band(x=time4dy_eval, y=dy_n1_5/4.0, color='k', u_up=sy_n1_4, 
                          alpha=0.1, line=True, line_param=lineOpts, ax=ax7)
    
    rhop_central15 = np.zeros(time4dy_eval.shape)
    rhop_lb15 = np.zeros(time4dy_eval.shape)
    rhop_ub15 = np.zeros(time4dy_eval.shape)
    vt_tmp = 4*omega_tot_fun(rhop_mesh)
    for ii, t in enumerate(time4dy_eval):
        # Finding the central point:
        idx = np.abs(vt_tmp -  dy_n1_5[ii]).argmin()
        rhop_central15[ii] = rhop_mesh[idx]
        
        # Finding the lower bound.
        idx = np.abs(vt_tmp -  (dy_n1_5[ii] - sy_n1_5[ii])).argmin()
        rhop_lb15[ii] = rhop_mesh[idx]
        
        # Finding the lower bound.
        idx = np.abs(vt_tmp -  (dy_n1_5[ii] + sy_n1_5[ii])).argmin()
        rhop_ub15[ii] = rhop_mesh[idx]
    
    error_up = np.abs(rhop_lb15[1:] - rhop_central15[1:] )
    error_dw = np.abs(rhop_ub15[1:]  - rhop_central15[1:] )
    ax8 = plot_error_band(x=time4dy_eval[1:] , y=rhop_central15[1:], 
                          color='k', u_up=error_up, 
                          u_down=error_dw, alpha=0.1, line=True, 
                          line_param=lineOpts, ax=ax8)
    plt.show()