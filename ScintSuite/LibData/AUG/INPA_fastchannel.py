"""
Routines to load and work with INPA fast channels (pmt or shutter).
Also contains the routines for the INPA
(update 02/10/2026)

Alex Reyner Viñolas: areyner@us.es
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import ScintSuite.LibData.AUG.Misc as misc
from .Misc import to_dict_with_metadata
xr.set_options(keep_attrs=True)
import aug_sfutils as sf
import os


def get_INPA_pmt(shot, diag='npi', channels=np.arange(1,8+1), 
                 xArrayOutput = True):
    NPI = sf.SFREAD(shot, diag)
    open = NPI('ShutOpen')
    closed = NPI('ShutClos')
    ucoil = NPI('UCoil')
    uimon = NPI('UImon')
    ts = np.array(NPI.gettimebase('ShutOpen'), dtype='f4')
    inpa = xr.Dataset(
        {'open':(['ts'],open,{'long_name':'Open', 
                        'units':'V', 
                        'signal':'ShutOpen'}),
        'closed':(['ts'],closed,{'long_name':'Closed', 
                        'units':'V', 
                        'signal':'ShutClos'}),
        'ucoil':(['ts'],ucoil,{'long_name':'Ucoil', 
                        'units':'V', 
                        'signal':'Ucoil'}),
        'uimon':(['ts'],uimon,{'long_name':'UImon', 
                        'units':'V', 
                        'signal':'UImon'}),
        },
        coords = {'ts':ts}
        )
    t = np.array(NPI.gettimebase('PMT_C01'), dtype='f4')
    pmts = xr.DataArray(dims={'t','channel',},
                    coords={'channel':channels,'t':t,})
    pmts = pmts.transpose('channel','t')
    for ch in channels:
        name = 'PMT_C'+str(f'{ch:02}')
        try:
            data = NPI(name)
            pmts[ch-1] = data
        except:
            print(name+' not found')

    pmts.coords['t'].attrs['long_name'] = 'Time'
    pmts.coords['t'].attrs['units'] = 's'  
    inpa.coords['ts'].attrs['long_name'] = 'Time'
    inpa.coords['ts'].attrs['units'] = 's' 
    inpa['pmt'] = pmts
    inpa['pmt'].attrs['long_name'] = 'INPA pmt'
    inpa['pmt'].attrs['units'] = 'a.u.'
    inpa.attrs['shot'] = shot
    inpa.attrs['diag'] = diag
    inpa['sample_rate'] = t[1]-t[0]

    if xArrayOutput: return inpa
    else: return to_dict_with_metadata(inpa)

def plot_INPA_pmt(inpa, roll=100, ignore_channel = 6, **kwargs):
    '''
    Plots all channels for inpa pmt
    '''
    xr.set_options(keep_attrs=True)
    data = inpa.pmt.coarsen(t=100, boundary='trim').mean().sel(t=slice(0,10))
    data = data - data.min('t')
    noise = data.sel(t=slice(0,1)).std('t')
    data = data/noise

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    ax = ax.ravel()
    for i in range(2):
        channels = (np.arange(4)+1) + i*4

        if ignore_channel is not None:
            channels = channels[channels != ignore_channel]
        j=0
        for ch in (channels):
            signal = data.sel(channel=(ch))
            mean = signal.rolling(t=roll, center=True).mean()
            std  = signal.rolling(t=roll, center=True).std()
            c = ['blue','red','orange','limegreen']
            ax[i].fill_between(mean.t, mean - std, mean + std, alpha=0.2, color=c[j])
            mean.plot(ax=ax[i], color=c[j], label=f'Ch {ch}', **kwargs)
            j+=1
        ax[i].legend(loc='upper right')
        ax[i].set_title('')
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(1e-1, 1e1))

    plt.tight_layout()

def plot_INPA_shutter(inpa, ax=None, **kwargs):
    '''
    Plots inpa shutter information
    '''
    if ax == None: 
        fig, ax = plt.subplots(4,1,figsize=(8,8), sharex=True)
        # return fig, ax
    
    inpa.open.plot(ax=ax[0], **kwargs)
    inpa.closed.plot(ax=ax[1], **kwargs)
    inpa.ucoil.plot(ax=ax[2], **kwargs)
    inpa.uimon.plot(ax=ax[3], **kwargs)
    for axs in ax:
        axs.set_xlabel('')
    ax[0].set_ylim([0,6])
    ax[1].set_ylim([0,6])
    ax[2].set_ylim([0,3])
    ax[3].set_ylim([0,3])
    ax[-1].set_xlabel('Time [s]')
    # fig.align_ylabels()
    plt.tight_layout()