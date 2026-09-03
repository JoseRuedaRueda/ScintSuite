"""
Routines to load and work with FILD fast channels (pmt or apd).
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


def get_FILD1_pmt(shot, xArrayOutput = True):
    if shot >= 30000:
        FHX = sf.SFREAD(shot, 'FHC')
        mult = 1.00000000000000000
    else:
        FHX = sf.SFREAD(shot, 'FHA')
        mult = 0.10000000000000000

    slist = [s for s in FHX.getlist() if 'FI' in s][:20]
    t = FHX.gettimebase(slist[0])
    fild = xr.DataArray(dims={'t','channel',},
                        coords={'t':t,'channel':np.arange(len(slist))+1})
    fild = fild.transpose('t','channel')
    for c, name in zip(fild.channel.values, slist):
        try:
            fild.loc[:, c] = FHX(name)
        except:
            print('No '+name)

    fild.coords['t'].attrs['long_name'] = 'Time'
    fild.coords['t'].attrs['units'] = 's'
    fild.attrs['shot'] = shot

    if xArrayOutput: return fild
    else: return to_dict_with_metadata(fild)

def get_FILD4_apd(shot, fps, APD_dir='/shares/departments/AUG/users/juaord/',
                  plot = False, xArrayOutput = True):
    '''
    Docstring for get_FILD4_APD
    
    :param  shot: shot number
    :param  fs: fps of the camera (to be standarized)
    :param  APD_dir: path to where the files are stored
    :param  plot: immediately plot the timetraces
    '''
    data_dir = os.path.join(APD_dir, str(shot))
    apd = xr.DataArray(dims={'t','channel',},
                        coords={'t':np.arange(fps*10)/fps,
                                'channel':np.arange(32)+1})
    apd = apd.transpose('channel','t')
    for i in range(32):
        filepath = os.path.join(data_dir, f'Channel{i:02d}.dat')
        if not os.path.exists(filepath):
            continue
        data = np.fromfile(filepath, dtype=np.int16)
        apd.loc[i+1] = data[:int(fps*10)]

    apd.coords['t'].attrs['long_name'] = 'Time'
    apd.coords['t'].attrs['units'] = 's'
    apd.coords['channel'].attrs['long_name'] = 'Channel'
    apd.attrs['shot'] = shot
    apd.attrs['diag'] = 'APD4'
    apd.attrs['fps'] = str(fps)

    if plot: plot_FILD4_apd(apd.coarsen(t=1000, boundary='trim').mean())
    if xArrayOutput: return apd
    else: return to_dict_with_metadata(apd)

def plot_FILD1_pmt(pmt, roll=100, **kwargs):
    '''
    Plots all channels for fild1 pmt
    '''
    xr.set_options(keep_attrs=True)
    data = pmt.coarsen(t=100, boundary='trim').mean().sel(t=slice(0,10))

    data = data - data.min('t')
    noise = data.sel(t=slice(0,1)).std('t')
    data = data/noise

    fig, ax = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    ax = ax.ravel()
    for i in range(4):
        channels = (np.arange(5)+1) + i*5
        j=0
        for ch in (channels):
            c = ['blue','red','orange','limegreen','fuchsia']
            signal = data.sel(channel=(ch))
            mean = signal.rolling(t=roll, center=True).mean()
            std  = signal.rolling(t=roll, center=True).std()
            ax[i].fill_between(mean.t, mean - std, mean + std, alpha=0.2, color=c[j])
            mean.plot(ax=ax[i], color=c[j], label=f'Ch {ch}', **kwargs)
            j+=1
        ax[i].legend(loc='upper right')
        ax[i].set_title('')
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(1e-1, 1e1))

    plt.tight_layout()

def plot_FILD4_apd(apd, roll=100, **kwargs):
    '''
    Plots all channels for an APD shotfile
    '''
    apd = apd - apd.sel(t=slice(8,9)).mean('t')
    # APDnoise = apd.sel(t=slice(1,2)).std('t')
    # apd = apd/APDnoise

    fig, ax = plt.subplots(2, 4, figsize=(21, 10), sharex=True, sharey=True)
    ax = ax.ravel()
    for i in range(8):
        channels = (np.arange(i*4, min(i*4 + 4, 33))+1)
        j=0
        for ch in (channels):
            c = ['blue','red','orange','limegreen']
            signal = apd.sel(channel=(ch))
            mean = signal.rolling(t=roll, center=True).mean()
            std  = signal.rolling(t=roll, center=True).std()
            ax[i].fill_between(mean.t, mean - std, mean + std, alpha=0.2, color=c[j])
            signal.rolling(t=roll, center=True).mean()\
                .plot(ax=ax[i], label=f'Ch {ch}', lw=2, **kwargs)
            j+=1
        ax[i].legend(loc='upper right')
        ax[i].set_title('')
    plt.tight_layout()


def get_VRT_filds(shot):
    fild_vrt = xr.Dataset()
    try:
        fil1 = misc.get_VRT_roi(shot, camera='07MEM', name='HotSpot', quantity='raw',)
        fild_vrt['fild1'] = fil1
    except: pass
    try:
        fil2 = misc.get_VRT_roi(shot, camera='06Bul2', name='FILD_HS', quantity='raw',)
        fild_vrt['fild2'] = fil2
    except: pass
    try:
        fil4 = misc.get_VRT_roi(shot, camera='07Eod1', name='FILD4_HS', quantity='raw',)
        fild_vrt['fild4'] = fil4
    except: pass
    try:
        fil5 = misc.get_VRT_roi(shot, camera='07Eod1', name='FILD5_HS', quantity='raw',)
        fild_vrt['fild5'] = fil5
    except: pass
    return fild_vrt

def plot_VRT_filds(shot):
    fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    try:
        misc.plot_VRT_roi(shot, ax=ax[0], c='mediumblue', label = 'FILD1',
                 camera='07MEM', name='HotSpot', quantity='raw',)
    except: pass
    try:
        misc.plot_VRT_roi(shot, ax=ax[1], c='red', label = 'FILD2',
                 camera='06Bul2', name='FILD_HS', quantity='raw',)
    except: pass
    try:
        misc.plot_VRT_roi(shot, ax=ax[2], c='orange', label = 'FILD4',
                 camera='07Eod1', name='FILD4_HS', quantity='raw',)
    except: pass
    try:
        misc.plot_VRT_roi(shot, ax=ax[3], c='limegreen', label = 'FILD5',
                 camera='07Eod1', name='FILD5_HS', quantity='raw',)
    except: pass

    for axs in ax:
        axs.set_xlabel('')
        axs.set_ylabel('')
        axs.set_ylim([0,None])
        axs.legend()
    ax[-1].set_xlabel('Time [s]')
    fig.supylabel('FILD OH time traces [counts]')
    plt.tight_layout()