"""
Other routines.

This library will contain routines that because of its nature do not belong to
a particular category.
This library contains:

- ELM and other edge quantities
- Power anh heating
- MPs
- other diagnostics from AUG

"""

import numpy as np
import xarray as xr
import aug_sfutils as sf
import ScintSuite.errors as errors
import ScintSuite.LibData.AUG.DiagParam as params

from ScintSuite._Paths import Path
import ScintSuite.errors as errors
import xarray as xr
xr.set_options(keep_attrs=True)
from datetime import datetime
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xml.etree.ElementTree as et

import os
pa = Path()

# -----------------------------------------------------------------------------
# --- Return function (update 02/10/2026)
# -----------------------------------------------------------------------------
def _safe_get(func, shot, xArrayOutput: bool = True):
    try: return func(shot, xArrayOutput=xArrayOutput)
    except: return None

def _clean_val(v):
    '''
    Converts objects to values
    '''
    if hasattr(v, "item") and np.ndim(v) == 0:
        return v.item()
    return v

def to_dict_with_metadata(obj):
    '''
    Converts Dataset or DataArray into a dictionary

    Keeps local and global atributes
    '''
    output = {}
    for k, v in obj.attrs.items(): output[k] = _clean_val(v)

    if isinstance(obj, xr.Dataset): # If its a Dataset
        for var_name, var_da in obj.data_vars.items():
            output[var_name] = var_da.values
            for attr_k, attr_v in var_da.attrs.items():
                output[f"{var_name}_{attr_k}"] = _clean_val(attr_v)
    else: # If its a DataArray
        output["data"] = obj.values

    for coord_name, coord_da in obj.coords.items():
        output[coord_name] = coord_da.values
        for attr_k, attr_v in coord_da.attrs.items():
            output[f"{coord_name}_{attr_k}"] = _clean_val(attr_v)

    return output

##
# -----------------------------------------------------------------------------
# --- GENERIC SIGNAL RETRIEVING.
# -----------------------------------------------------------------------------
def get_signal_generic(shot: int, diag: str, signame: str, exp: str = 'AUGD',
                       edition: int = 0, tBegin: float = None,
                       tEnd: float = None):
    """
    Function that generically retrieves a signal from the database in AUG.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: shotnumber of the shotfile to read.
    :param  diag: diagnostic name.
    :param  signame: signal name.
    :param  exp: experiment where the shotfile is stored. Default to AUGD.
    :param  edition: edition of the shotfile to open. If 0, the last closed
    edition is opened.
    :param  tBegin: initial time point to read.
    :param  tEnd: final time point to read.
    """

    # Reading the second diagnostic data.
    sfo = sf.SFREAD(shot, diag, edition=edition, experiment=exp)

    if not sfo.status:
        raise errors.DatabaseError('The signal data cannot be read for #%05d:%s:%s(%d)'
                        % (shot, diag, signame, edition))

    data = sfo(name=signame)
    if data is None:
        raise errors.DatabaseError('Cannot find signal %s' % signame)
    time = sfo.gettimebase(signame)

    if tBegin is None:
        t0 = 0
    else:
        t0 = np.abs(time - tBegin).argmin()

    if tBegin is None:
        t1 = len(time)
    else:
        t1 = np.abs(time - tEnd).argmin()

    data = np.array(data[t0:t1, ...], dtype=float)
    time = np.array(time[t0:t1, ...], dtype=float)

    return time, data


# -----------------------------------------------------------------------------
# --- SIGNAL OF FAST CHANNELS.
# -----------------------------------------------------------------------------
def get_fast_channel(diag: str, diag_number: int, channels, shot: int,
                     ed: int = 0, exp: str = 'AUGD'):
    """
    Get the signal for the fast channels (PMT, APD)

    Jose Rueda Rueda: jrrueda@us.es

    :param  diag: diagnostic: 'FILD' or 'INPA'
    :param  diag_number: 1-5
    :param  channels: channel number we want, or arry with channels
    :param  shot: shot file to be opened
    """
    # Check inputs:
    suported_diag = ['FILD', 'INPA']
    if diag not in suported_diag:
        raise errors.NotValidInput('No understood diagnostic')

    # Load diagnostic names:
    if diag.lower() == 'fild':
        if (diag_number > 5) or (diag_number < 1):
            print('You requested: ', diag_number)
            raise errors.NotValidInput('Wrong fild number')
        info = params.FILD[diag_number - 1]
        diag_name = info['diag']
        signal_prefix = info['channel']
        nch = info['nch']
    elif diag.lower() == 'inpa':
        if diag_number != 1:
            print('You requested: ', diag_number)
            raise errors.NotValidInput('Wrong INPA number')
        info = params.INPA[diag_number - 1]
        diag_name = info['diag']
        signal_prefix = info['channel']
        nch = info['nch']


    # Look which channels we need to load:
    try:    # If we received a numpy array, all is fine
        nch_to_load = channels.size
        if nch_to_load == 1:
            # To solve the bug that just one channel is passed but as a
            # component of a numpy array
            ch = np.array([channels]).flatten()
        else:
            ch = channels
    except AttributeError:  # If not, we need to create it
        ch = np.array([channels]).flatten()
        nch_to_load = ch.size
        if channels == None:
            nch_to_load = nch
            ch = np.arange(nch)+1 #all channels

    # Open the shot file
    fast = sf.SFREAD(diag_name, shot, ed=ed, exp=exp)
    dummy_name = signal_prefix + "{0:02}".format(ch[0])
    time = np.array(fast.gettimebase(dummy_name))
    data = []
    for ic in range(nch):
        real_channel = ic + 1
        if real_channel in ch:
            name_channel = signal_prefix + "{0:02}".format(real_channel)
            channel_dat = np.array(fast(name_channel))
            data.append(channel_dat[:time.size])
        else:
            pass
            # data.append(None)
    print('Number of requested channels: ', nch_to_load)
    return {'time': time, 'data': data, 'channels': ch}


# -----------------------------------------------------------------------------
# --- ELMs and edge (update 10/08/2026)
# -----------------------------------------------------------------------------
def get_ELM(shot, plot = False, xArrayOutput: bool = True):
    '''
    Returns an xarray with ELM related quantities

    Alex Reyner Vñolas

    :returns duration, enery lost, Wmhd and an on/off mask
    '''

    ELM = sf.SFREAD(shot,'ELM')
    beg = np.array(ELM('t_begELM'), dtype='f4')
    end = np.array(ELM('t_endELM'), dtype='f4')
    dur = np.array(ELM('dt_ELM'), dtype='f4')
    t_max = np.array(ELM('t_maxELM'), dtype='f4')
    t_elm = np.array(ELM.gettimebase('ELMENER'), dtype='f4')

    t = np.arange(0,10,0.1e-5)
    start_idx = np.searchsorted(t, beg, side='left')
    end_idx = np.searchsorted(t, end, side='left')
    indices = np.arange(len(t))

    event_mask = np.any((start_idx[:, None] <= indices) & \
                        (indices < end_idx[:, None]), axis=0)
    obj = xr.Dataset(
        data_vars={
            'duration': (('t_elm',), dur*1e3, 
                         {'long_name': '$ELM_{\\Delta t}$', 'units': 'ms'}),
            'Eloss': (('t_elm',), np.array(ELM('ELMENER'), dtype='f4'), 
                      {'long_name': '$E_{lost}$', 'units': 'J'}),
            'Wmhd': (('t_elm',), np.array(ELM('Wmhd'), dtype='f4'), 
                     {'long_name': '$E_{MHD}$', 'units': 'J'}),
            'elm_mask': (('t',), event_mask, 
                         {'long_name': 'ELM Mask'})
        },
        coords={
            't_max': ('t_max', t_max, {'long_name': 'Peak', 'units': 's'}),
            't_elm': ('t_elm', t_elm, {'long_name': 'Time', 'units': 's'}),
            't': ('t', t, {'long_name': 'Time', 'units': 's'})
        },
        attrs={'shot': shot, 'diag': 'ELM'}
    )

    if plot:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        obj.Wmhd.plot(ax=ax)
        obj.duration.plot(ax=ax2, color='r', alpha=0.5)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_Ipol(shot, plot = False, xArrayOutput: bool = True):
    MAC = sf.SFREAD(shot, 'MAC')
    inner = np.array(MAC('Ipolsoli')/1000)
    outer = np.array(MAC('Ipolsola')/1000)
    t = np.array(MAC.gettimebase('Ipolsola'), dtype='f4')
    obj = xr.Dataset(
        {'Ipolsoli':(['t'], inner, {'long_name':'$Div._{curr}$', 'units':'kA', 
                                    'signal':'Ipolsoli'}),
         'Ipolsola':(['t'], outer, {'long_name':'$Div._{curr}$', 'units':'kA', 
                                    'signal':'Ipolsola'}),
        },
        coords = {'t': ('t', t, {'long_name': 'Time', 'units': 's'})},
        attrs={'shot': shot, 'diag': 'MAC'}
    )
    if plot:
        fig, ax = plt.subplots()
        obj.Ipolsoli.plot(ax=ax)
        obj.Ipolsola.plot(ax=ax)
    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_Raus(shot, plot = False, xArrayOutput: bool = True):
    FPG = sf.SFREAD(shot, 'FPG')
    name = 'Raus'
    R = np.array(FPG(name), dtype='f4')
    t = np.array(FPG.gettimebase(name), dtype='f4')
    obj = xr.DataArray(R*100.0 - 215.0, dims=['t'],
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'})},
        attrs={'units': 'cm', 'long_name': '$R_{aus}$', 
               'shot': shot, 'diag': 'FPG', 'signal': name}
    )
    if plot:
        fig, ax = plt.subplots()
        obj.plot(ax=ax)
    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)


# -----------------------------------------------------------------------------
# --- HEATING (update 10/08/2026)
# -----------------------------------------------------------------------------
def get_NBI(shot, plot = False, xArrayOutput: bool = True):
    '''
    Returns an xarray with NBI power and parameters from NIS shotfile

    Alex Reyner Viñolas: areyner@us.es
    '''

    NIS = sf.SFREAD(shot, 'NIS')
    name = 'PNIQ'
    pNBI = NIS(name) / 1.0e6
    tNBI = np.array(NIS.gettimebase(name), dtype='f4')

    obj = xr.Dataset(
        coords={'t': ('t', tNBI, {'long_name': 'Time', 'units': 's'}), 
                'b': ('b', np.arange(1, 9), {'long_name': 'Beam'}),
            },
        attrs={'shot': shot, 'diag': 'NIS'}
    )

    power_data = np.moveaxis(pNBI, 0, -1).reshape(8, -1)
    obj['power'] = (('b', 't'), power_data, 
                    {'signal': name, 'long_name': '$P_{NB}$', 'units': 'MW'}
                )
    obj['total'] = obj.power.sum('b')
    
    ion = np.concatenate([NIS('INJ1')['MQ'],NIS('INJ2')['MQ']], dtype='f4')
    ene = np.concatenate([NIS('INJ1')['UEXQSOLL'],NIS('INJ2')['UEXQSOLL']], dtype='f4')
    obj['ion'] = (('b',), ion, {'long_name': 'Ion', 'units': 'A'})
    obj['ene'] = (('b',), ene, {'long_name': 'Inj. energy', 'units': 'keV'})
    # nbi stat: 1=H, 2=D, 3=T
    obj['stat'] = obj['ion'].where(obj['power'] >= 0.2)
    obj['stat'].attrs.update({'long_name': 'Active species', 'units': 'Z'})

    if plot:
        discrete_map = mcolors.ListedColormap(['gold','red','orchid'])
        norm_discrete = mcolors.BoundaryNorm([0.5,1.5,2.5,3.5], discrete_map.N)
        fig, ax = plt.subplots()
        ax2=ax.twinx()
        obj.total.rolling(t=3).mean().plot(ax=ax, label='$P_{NB}$', color='k',)
        im = obj.stat.plot.imshow(ax=ax2, alpha=0.5, 
                                  cmap=discrete_map, norm=norm_discrete, 
                                  add_colorbar=False, zorder=1)
        ax2.set_yticks([1,2,3,4,5,6,7,8])
        ax2.tick_params(axis='y', labelsize=12)
        for beam in np.arange(8):
            ax2.axhline(y=beam+0.5,c='grey',lw=1,ls='--',alpha=0.5)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_ICRH(shot, coupled = False, plot = False, xArrayOutput: bool = True):
    '''
    Returns an xarray with coupled ICRH power parameters

    Alex Reyner Viñolas: areyner@us.es
    '''

    ICP = sf.SFREAD(shot, 'ICP')
    name = 'PICRFc' if coupled else 'PICRN'
    pICRH = np.array(ICP(name), dtype='f4') / 1.0e6
    tICRH = np.array(ICP.gettimebase(name), dtype='f4')

    obj = xr.Dataset(
        coords={'t': ('t', tICRH, {'long_name': 'Time', 'units': 's'}),
                'antenna': ('antenna', [1, 2, 3, 4], {'long_name': 'Antenna'}),
                },
        attrs={'shot': shot, 'diag': 'ICP'}
    )
    obj['total'] = (('t',), pICRH, 
                    {'signal': name, 'long_name': '$P_{IC}$', 'units': 'MW'}
                )
    pnet_data =\
        np.array([ICP(f'pnet{ant+1}') for ant in range(4)], dtype='f4') / 1.0e6
    obj['power'] = (('antenna', 't'), pnet_data, 
                    {'long_name': '$P_{IC}$', 'units': 'MW'}
                )
    fICRH = ICP.getparset(pset='Frequenz')
    freqs = np.array(list(fICRH.values()), dtype='f4') / 1e6
    obj['freq'] = (('antenna',), freqs, 
                   {'long_name': 'IC Freq.', 'units': 'MHz'}
                )
    obj['stat'] = obj['freq'].where(obj['power'] >= 0.2)

    if plot:
        discrete_map = mcolors.ListedColormap(['cornflowerblue','navy',])
        norm_discrete = mcolors.BoundaryNorm([0,40,100], discrete_map.N)
        fig, ax = plt.subplots()
        ax2=ax.twinx()
        obj.total.rolling(t=3).mean().plot(ax=ax, label='$P_{NB}$', color='k',)
        im = obj.stat.plot.imshow(ax=ax2, alpha=0.5, 
                                  cmap=discrete_map, norm=norm_discrete, 
                                  add_colorbar=False, zorder=1)
        ax2.set_yticks([1,2,3,4])
        ax2.tick_params(axis='y', labelsize=12)
        for antena in np.arange(4):
            ax2.axhline(y=antena+0.5,c='grey',lw=1,ls='--',alpha=0.5)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_ECRH(shot, plot = False, xArrayOutput: bool = True):
    '''
    Returns an xarray with ECS power and parameters from NIS shotfile
    
    Alex Reyner Viñolas: areyner@us.es
    '''

    ECS = sf.SFREAD(shot, 'ECS')
    name = 'PECRH'
    pECRH = np.array(ECS(name), dtype='f4') / 1.0e6
    tECRH = np.array(ECS.gettimebase(name), dtype='f4')


    obj = xr.Dataset(
        coords={'t': ('t', tECRH, {'long_name': 'Time', 'units': 's'}), 
                'gyrotron': ('gyrotron', np.arange(1, 9), {'long_name': 'Gyrotron'}),
        },
        attrs={'shot': shot, 'diag': 'ECS'}
    )
    obj['total'] = (('t',), pECRH, 
                    {'signal': name, 'long_name': '$P_{EC}$', 'units': 'MW'}
                )

    pecs_list = []
    freq_list = []
    for box in range(2):
        sy = box + 1
        for source in range(4):
            gy = source + 1
            sig_name = f'PG{gy}' if sy == 1 else f'PG{gy}N'
            pecs_list.append(np.array(ECS(sig_name), dtype='f4') / 1.0e6)
            freq_list.append(np.array(ECS(f'P_sy{sy}_g{gy}')['gyr_freq'], dtype='f4') / 1.0e9)
    obj['power'] = (('gyrotron', 't'), np.array(pecs_list, dtype='f4'), 
                    {'long_name': '$P_{EC}$', 'units': 'MW'}
                )
    obj['freq'] = (('gyrotron',), np.array(freq_list, dtype='f4'), 
                   {'long_name': 'EC Freq.','units': 'GHz'}
                )

    tANGLE = np.array(ECS('T-C'), dtype='f4')
    tor_data = np.array([ECS(f'phtr-G{g+1}') for g in range(8)], dtype='f4')
    pol_data = np.array([ECS(f'thpl-G{g+1}') for g in range(8)], dtype='f4')
    tor_angle = xr.DataArray(tor_data, dims=['gyrotron', 't'], 
                             coords={'gyrotron': obj['gyrotron'], 't': tANGLE}, 
                             attrs={'long_name': 'Toroidal ang.', 'units': 'º'}
                        )
    pol_angle = xr.DataArray(pol_data, dims=['gyrotron', 't'], 
                             coords={'gyrotron': obj['gyrotron'], 't': tANGLE}, 
                             attrs={'long_name': 'Poloidal ang.', 'units': 'º'}
                        )
    interp_kwargs = {'fill_value': 'extrapolate'}
    obj['tor_ang'] = tor_angle.interp(t=obj['t'], method='linear', kwargs=interp_kwargs)
    obj['pol_ang'] = pol_angle.interp(t=obj['t'], method='linear', kwargs=interp_kwargs)

    obj['stat'] = obj['freq'].where(obj['power'] >= 0.2)

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        discrete_map = mcolors.ListedColormap(['lightgreen','darkgreen',])
        norm_discrete = mcolors.BoundaryNorm([90,120,160], discrete_map.N)
        ax2=ax.twinx()
        obj.total.rolling(t=3).mean().plot(ax=ax, label='$P_{NB}$', color='k',)
        im = obj.stat.plot.imshow(ax=ax2, alpha=0.5, 
                                  cmap=discrete_map, norm=norm_discrete, 
                                  add_colorbar=False, zorder=1)
        ax2.set_yticks([1,2,3,4,5,6,7,8])
        ax2.tick_params(axis='y', labelsize=12)
        for gyrotron in np.arange(8):
            ax2.axhline(y=gyrotron+0.5,c='grey',lw=1,ls='--',alpha=0.5)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_heating(shot, xArrayOutput: bool = True):
    '''
    Wrap to obtain the external heating
    '''
    nbi = _safe_get(get_NBI, shot, xArrayOutput=xArrayOutput)
    icrh = _safe_get(get_ICRH, shot, xArrayOutput=xArrayOutput)
    ecrh = _safe_get(get_ECRH, shot, xArrayOutput=xArrayOutput)

    return nbi, icrh, ecrh

def get_Prad(shot, plot = False, xArrayOutput: bool = True):
    """
    Return the total radiated power from the BPD shotfile in AUG.

    Alex Reyner Viñolas: areyner@us.es

    :param  shot: shotnumber
    """

    BPD = sf.SFREAD(shot, 'BPD')
    name = 'Pradtot'
    pBPD = np.array(BPD(name), dtype='f4') / 1.0e6
    tBPD = np.array(BPD.gettimebase(name), dtype='f4')

    obj = xr.DataArray(pBPD, dims=['t'], 
        coords={'t': ('t', tBPD, {'long_name': 'Time', 'units': 's'})},
        attrs={'units': 'MW', 'long_name': '$P_{rad}$', 
               'shot': shot, 'diag': 'BPD', 'signal': name}
    )

    if plot:
        fig, ax = plt.subplots()
        obj.plot(ax=ax)

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)


# -----------------------------------------------------------------------------
# --- MPs (update 10/08/2026)
# -----------------------------------------------------------------------------
def _count_zero_crossings_cyclic(vec):
    if vec.size == 0 or np.all(np.isnan(vec)):
        ans = np.nan
    else:
        vec_centered = vec - np.nanmean(vec)
        signs = np.sign(vec_centered)
        signs[signs == 0] = 1
        crossings = signs[:-1] * signs[1:] < 0
        cyclic_cross = signs[-1] * signs[0] < 0
        ans = np.sum(crossings) + cyclic_cross
    return ans

def compute_phase(signal, n):
    x = np.linspace(0, n*(2*np.pi), 8, endpoint=False)  # sampling points
    mask = ~np.isnan(signal)
    if np.sum(mask) < 2:
        return np.nan
    s = signal[mask]
    x_masked = x[mask]

    # Project onto sine and cosine
    a = np.sum(s * np.cos(x_masked))
    b = np.sum(s * np.sin(x_masked))

    # Phase in degrees (0-360)
    phase = np.degrees(np.arctan2(-b, a))  # arctan2(sin, cos)
    phase = (phase + 90) % 360  # shift to 0-360
    return phase

def get_MP(shot, diag='SSV', unit = 'I', coars = 10, phase = True, 
           plot = False, xArrayOutput: bool = True):
    '''
    Gets the currents in the RMP coils

    Alex Reyner Viñolas: areyner@us.es

    :param  diag: SVV or MAW, depending on the user. SVV has the base data
    :param  unit: for MAW, I or U
    :param  phase: add the diferential and toroidal phases
    '''

    if unit != 'I' or 'U':
        unit = 'I'

    diags_to_try = ['MAW', 'SSV'] if diag == 'MAW' else ['SSV', 'MAW']
    errors = {}
    for d in diags_to_try:
        try:
            if d == 'SSV': obj = get_MP_SSV(shot)
            elif d == 'MAW': obj = get_MP_MAW(shot, unit)
            break
        except Exception as e:
            errors[d] = e

    # individual coils
    for p in ['u', 'l']:
        for i, n in enumerate(obj.coil.values):
            name = f'B{p}{n}'
            obj[name] = (['t'], obj[f'B{p}'].isel(coil=i).values, 
                            {'signal': name, 'long_name': f'Coil {name}', 
                            'units': 'kA',},
            )
            folder_path = pa.ScintSuite + '/ScintSuite/LibData/AUG/Bcoils'
            file_path = os.path.join(folder_path, name+'.txt')
            R, z, phi = np.loadtxt(file_path, unpack=True, dtype='f4')
            obj[name].attrs['R'] = R
            obj[name].attrs['Z'] = z
            obj[name].attrs['phi'] = phi
    
    obj = obj.coarsen(t=coars,boundary='trim').mean()
    rmp_on = np.abs(obj['Bu']).sum('coil') > 0.2
    obj = obj.where(rmp_on, np.nan)

    if phase:
        Bu = obj['Bu']
        Bl = obj['Bl']
        try:
            num_osc = [_count_zero_crossings_cyclic(Bu.sel(t=ti).values)//2 for ti in obj.t]
            u_osc = xr.DataArray(num_osc, dims='t', coords={'t': obj.t})
            n_u = u_osc.max()

            num_osc = [_count_zero_crossings_cyclic(Bl.sel(t=ti).values)//2 for ti in obj.t]
            l_osc = xr.DataArray(num_osc, dims='t', coords={'t': obj.t})
            n_l = l_osc.max()

            u_phi = xr.DataArray(
                [compute_phase(Bu[:,t].values, n_u) for t in range(len(Bu.t))],
                dims=('t',), coords={'t': Bu.t})
            l_phi = xr.DataArray(
                [compute_phase(Bl[:,t].values, n_l) for t in range(len(Bl.t))],
                dims=('t',), coords={'t': Bl.t})
            
            obj['phi_u'] = u_phi
            obj['phi_l'] = l_phi
            obj['phi0'] = l_phi
            delta = (u_phi - l_phi)
            obj['dphi'] = (delta) % 360

        except:
            print('Not able to extract phases')
            pass  

    if plot:
        fig, ax = plt.subplots(sharex=True)
        bot = obj['Bl']
        top = obj['Bu'].assign_coords(coil=obj.coil+9)
        dummy = xr.concat([bot,top], dim='coil')
        dummy = dummy.interp(coil=np.arange(17)+1,method='nearest')
        dummy.loc[9,:] = np.nan
        dummy.attrs['long_name'] = 'Current'
        dummy.attrs['units'] = 'kA' 
        dummy.plot.imshow(ax=ax, ylim=[0,18], cmap='coolwarm_r', 
                          add_colorbar=True, center=0, vmax=1,)
        ax.set_ylabel('')
        ax.set_yticks([1,8,10,17])
        ax.set_yticklabels(['Bl1','Bl8','Bu1','Bu8'])
        ax.tick_params(axis='y', labelsize=12)
    
    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_MP_MAW(shot, unit = 'I'):
    if unit != 'I' or 'U':
        unit = 'I'
    MAW = sf.SFREAD(shot, 'MAW')
    t = np.array(MAW.gettimebase('IBl1'), dtype='f4')
    coils = np.arange(1,9)

    obj = xr.Dataset(coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}), 
                             'coil': ('coil', coils, {'long_name': 'Coil Number'}),},
                     attrs={'shot': shot, 'diag': 'MAW'}
    )
    for p in ['u', 'l']:
        data_list = []
        for n in coils:
            name = f'B{p}{n}'
            sig = np.array(MAW(unit + name), dtype='f4') / 1000.0
            data_list.append(sig)
        arr_2d = np.array(data_list, dtype='f4')
        obj[f'B{p}'] = (['coil', 't'], arr_2d, 
                        {'long_name': f'Current', 'units': 'kA'},
                    )

    return obj

def get_MP_SSV(shot):

    SSV = sf.SFREAD(shot, 'SSV')
    t = np.array(SSV.gettimebase('Iact1'), dtype='f4')
    number = np.arange(1, 17)
    coils = np.arange(1,9)
    obj = xr.Dataset(coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}), 
                             'coil': ('coil', coils, {'long_name': 'Coil Number'}),},
                     attrs={'shot': shot, 'diag': 'SSV'}
    )
    bu_signals = []
    bl_signals = []
    for n in number:
        name = f'Iact{n}'
        signal = np.array(SSV(name), dtype='f4') / 1000.0
        obj[name] = (['t'], signal, 
                     {'long_name': f'Current', 'units': 'kA',
                      'signal': name})
        if n <= 8: bu_signals.append(signal)
        else: bl_signals.append(signal)
        # Requested currents
        name = f'Ireq{n}'
        req = np.array(SSV(name), dtype='f4') / 1000.0
        obj[name] = (['t'], req, 
                     {'long_name': f'Requested', 'units': 'kA',
                      'signal': name})
        
    obj['Bu'] = (['coil', 't'], np.array(bu_signals, dtype='f4'), 
                 {'long_name': 'Upper currents', 'units': 'kA'})
    obj['Bl'] = (['coil', 't'], np.array(bl_signals, dtype='f4'), 
                 {'long_name': 'Lower currents', 'units': 'kA'}
                 )
    return obj

# -----------------------------------------------------------------------------
# --- OTHER USEFUL SIGNALS. (update 10/08/2026)
# -----------------------------------------------------------------------------
def get_neutrons(shot, plot = False, xArrayOutput: bool = True):
    """
    Gets neutron flux during shot and total

    Alex Reyner Viñolas: areyner@us.es
    """
    ENR = sf.SFREAD(shot, 'ENR')
    t = np.array(ENR.gettimebase('NRATE'), dtype='f4')
    signal_name = 'NRATE_II' if 'NRATE_II' in ENR.getlist() else 'NRATE'

    obj = xr.Dataset(
        data_vars = {'rate': (('t',), np.array(ENR(signal_name), dtype='f4'), 
                              {'long_name': 'Rate', 'units': 'neutrons/s', 
                               'signal': signal_name}), 
                     'err': (('t',), np.array(ENR('NRATEERR'), dtype='f4'), 
                             {'long_name': 'Error', 'units': '', 
                              'signal': 'NRATEERR'})
        },
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'})},
        attrs={'shot': shot, 'diag': 'ENR'}
    )
    obj.attrs['total'] = float(obj['rate'].integrate('t'))

    if plot:
        fig, ax = plt.subplots()
        obj.rate.plot(ax=ax)

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_neutron_history(shot_0: int, shot_1: int, plot = True, 
                        xArrayOutput: bool = True):
    """
    Compute for all the shot in the range of shots given, the evolution of the
    neutron emitted.

    Pablo Oyola - poyola@us.es
    Alex Reyner Viñolas: areyner@us.es

    :param shot_0: starting shot.
    :param shot_1: ending shot to return.
    """

    shots = np.arange(shot_0, shot_1 + 1) if shot_0 != shot_1 else np.array([shot_0])
    totals = []
    for shot in shots:
        try:
            neut = get_neutrons(shot)
            totals.append(neut.attrs['total'] if hasattr(neut, 'attrs') and 'total' in neut.attrs else neut.total)
        except Exception:
            totals.append(np.nan)
    obj = xr.DataArray(np.array(totals, dtype='f4'), dims=['shot'],
        coords={'shot': ('shot', shots, {'long_name': 'Shot'})},
        attrs={'long_name': 'Total neutrons', 'units': 'neutrons'        }
    )

    if plot:
        fig, ax = plt.subplots()
        obj.plot(ax=ax)

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)
    
def get_NPA(shot, xArrayOutput: bool = True):
    """
    Gets NPA data with H2D ratio

    Alex Reyner Viñolas: areyner@us.es
    """
    
    CXF = sf.SFREAD(shot, 'CXF')
    species = ['RH','LH','RD','LD']
    t = np.array(CXF.gettimebase('RfluxH'), dtype='f4')
    obj = xr.Dataset(
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'})},
        attrs={'shot': shot, 'diag': 'CXF'}
    )

    for sp in species:
        name = f'{sp[0]}flux{sp[1]}'
        elabel = f'{sp[0]}energy{sp[1]}'
        try:
            data = np.array(CXF(name), dtype='f4')
            e_data = np.array(CXF.getareabase(name), dtype='f4').flatten()
            obj.coords[elabel] = (elabel, e_data, {'long_name': 'Energy', 'units': 'keV'})

            if data.shape == (len(t), len(e_data)): dims = ('t', elabel)
            elif data.shape == (len(e_data), len(t)): dims = (elabel, 't')
            else:
                print(f'Warning: strange shape for {name}: {data.shape}')
                continue
            data_cleaned = np.clip(data, 0, None)
            obj[name] = (dims, data_cleaned, 
                        {'long_name': f'{sp[0]} {sp[1]} flux', 'units': 'a.u.', 
                        'signal': name}
                    )
        except Exception as err:
            print(f'nop ({name}): {err}')

    h2d_data = np.array(CXF('H2HD'), dtype='f4')
    obj['H2HD'] = (('t',), h2d_data, {'long_name': 'H ratio', 'units': ''})

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_IDA(shot, xArrayOutput: bool = True):
    IDA = sf.SFREAD(shot, 'IDA')
    obj = xr.Dataset(attrs={'shot': shot, 'diag': 'IDA'})

    for signal in ['Te','Te_up','Te_lo','Te_unc',
                   'ne','ne_up','ne_lo','ne_unc',
                   'pe','pe_up','pe_lo','pe_unc']:
        try:
            data = IDA(signal)
            area = IDA.getareabase(signal)
            t = np.array(IDA.gettimebase(signal), dtype='f4')
            label = 'rhop'
            dummy = xr.DataArray(data,
                                dims={'t', label,}, 
                                coords={'t':t, label:area[:,0],})
            dummy = dummy.transpose(label,'t')
            dummy.coords['rhop'].attrs['long_name'] = '$\\rho_{pol}$'
            dummy.attrs['long_name'] = signal          
            if 'Te' in signal:
                dummy = dummy/1000
                dummy.attrs['units'] = 'keV'
            elif 'ne' in signal:
                dummy.attrs['units'] = '1/m³'
            elif 'pe' in signal:
                dummy.attrs['units'] = 'Pa'
            obj[signal] = dummy
        except:
            continue
    
    for signal in IDA.getlist_by_type('SignalGroup'):
        if 'dcn_' not in signal: continue
        try:
            data = IDA(signal)
            area = IDA.getareabase(signal)
            t = np.array(IDA.gettimebase(signal), dtype='f4')
            dummy = xr.DataArray(data, 
                                dims={'x_dcn', 'mmm', 't', }, 
                                coords={'x_dcn':[1,2,3,4,5,6,7,8],
                                        'mmm':[1,2,3,4,5,6,7,8,9,10,11],
                                        't':t,})
            dummy.coords['x_dcn'].attrs['long_name'] = 'line'
            dummy.attrs['long_name'] = signal          
            obj[signal] = dummy
        except:
            continue


    obj.coords['t'].attrs['long_name'] = 'Time'
    obj.coords['t'].attrs['units'] = 's'

    if xArrayOutput: return obj
    else: return to_dict_with_metadata(obj)

def get_VRT_roi(shot, camera = '06Bul2', name = 'ICRHS2', 
                quantity = 'temperature', plot = True, 
                xArrayOutput: bool = True):
    class calibrated_camera:
        def __init__(self,configXML,args):
            self.args=args       
            parxml=configXML.find('./*[@name="obj:Parameterizations"]')
            objxml=parxml.find('OBJECTLIST/OBJECT')
            self.camtype=camtype=objxml.attrib['name'].replace('Camera','')
            self.gain=gain=int(objxml.find('.//USAGE[@name="p:Gain"]/PARAMETER').attrib['value'])
            if camtype=='Jai':
                shutter=int(objxml.find('.//USAGE[@name="p:ShutterPreset"]/PARAMETER').attrib['value'])
                self.texp=texp={0:0.00833333333333,1:0.004,2:0.002,3:0.001,4:0.0005,5:0.00025,6:0.000125,7:0.0001,8:6.66666666667e-05,9:3.33333333333e-05}[shutter]
            else:
                shutter=int(objxml.find('.//USAGE[@name="p:Shutter"]/PARAMETER').attrib['value'])
                self.texp=texp=shutter*1e-6
            fnview = "/shares/departments/AUG/users/vida/projects/camera_description/views/view_%s.xml" % args.camera
            viewxml=et.parse(fnview).getroot()
            self.viewparameters=vp={}
            for m in viewxml.findall("modification"):
                if int(m.attrib['pulse'])<args.shot:
                    for k,v in m.attrib.items():vp[k]=v
            fnobj = "/shares/departments/AUG/users/vida/projects/camera_description/objectives/objective_%s.xml" % vp['objectiveid']
            objxml=et.parse(fnobj).getroot()
            if  vp['aperture'].lower()!='none':
                aperture_ring=float(vp['aperture'])            # this is the aperture ring position for the given discharge. 0:minimal opening (typically 1%), 1:maximal opening (typically 100%)
                art=self.aring_vs_atrans=np.array([[float(e.attrib['value']),float(e.attrib['transmission'])] for e in objxml.findall('aperture')])
                self.a=np.interp(aperture_ring,art[:,0],art[:,1]) # this is the aperture of the objective lens for the given discharge
            else:
                self.a=1.0
                
            calroot = None
            if args.quantity in ['radiance','temperature']:
                calroot = et.parse(args.pcal+'/'+args.fncal).getroot()
                self.a_cal = float(calroot.attrib["a"])      # this is the aperture (transmission) of the objective lens on the day of calibration
                #t_cal = float([x.attrib["t_exp"] for x in calroot.find("parameterdescription") if x.attrib["SH"]==shutter][0])
                self.LeCS = float(calroot.find("weighted_radiance_source").attrib["LeCS"])
                coeffs = [c for c in calroot.findall("setting") if int(c.attrib["GA"])==gain][0]
                self.c1w=float(coeffs.attrib["c1w"])
                self.c1b=float(coeffs.attrib["c1b"])
                self.c0w=float(coeffs.attrib["c0w"])
                self.c0b=float(coeffs.attrib["c0b"])
                self.T_vs_rad=np.array([[float(e.attrib['T']),float(e.attrib['LeW'])] for e in calroot.findall('radiance_table/entry')])
        def getlabel(self):
            return {'raw':'raw signal [counts]','radiance':'(weighted) radiance [$Wm^{-2}sr^{-1}$]','temperature':'$T$ [K]'}[self.args.quantity]
        def raw2q(self,I):
            if self.args.quantity == 'raw':
                return I
            else:
                I_w = self.c1w*self.texp + self.c0w
                I_b = self.c1b*self.texp + self.c0b          
                L = (self.LeCS*self.a_cal/self.a) * (I-I_b)/(I_w-I_b)
                if self.args.quantity == 'radiance':
                    return L
                if self.args.quantity == 'temperature':
                    return np.interp(L,self.T_vs_rad[:,1],self.T_vs_rad[:,0])
    class Args:
        shot = 43178
        shottype = "s"
        quantity = "raw"
        pcal = "/shares/departments/AUG/users/vida/projects/camera_description/calibration/simple/"
        fncal = "01Eod_calibration.xml"

    args = Args()
    args.shot = shot
    args.camera = camera
    args.quantity = quantity
    if args.shottype == "t":
        fn = "/shares/experiments/aug-rawfiles/VRT/Test/%02i/T%i/T%i_%s.meta.xml"\
              % (args.shot//1000, args.shot, args.shot, camera)
    elif args.shottype == "s":
        fn = "/shares/experiments/aug-rawfiles/VRT/%02i/S%i/S%i_%s.meta.xml"\
              % (args.shot//1000, args.shot, args.shot, camera)
    metaroot = et.parse(fn).getroot()
    ts6 = int(metaroot.attrib["ts6"], 0)
    configXML = metaroot.find("config").find("AP_CONF[@type='config']")
    interlocksXML=metaroot.find("config").find("AP_CONF[@type='interlocks']")
    comparatorsXML=interlocksXML.findall('USAGE[@name="obj:Comparators"]/OBJECTLIST/OBJECT')
    messagesXML=metaroot.find("config").find("AP_CONF[@type='messages']")
    evaluatorsXML=configXML.findall('USAGE[@name="obj:Evaluators"]/OBJECTLIST/OBJECT')
    signalsXML=metaroot.findall("signals/signal")
    object = calibrated_camera(configXML, args)
    def get_thresholds(sn):
        th=[]
        for cXML in comparatorsXML:
            if cXML.find('USAGE[@name="p:FloatInputSignals"]/SIGNALVECTOR').attrib['members']==sn:
                nm=cXML.find('USAGE[@name="o:Scaling"]/SIGNAL').attrib['name']
                message=''
                for mXML in messagesXML.findall('USAGE[@name="obj:Executors"]/OBJECTLIST/OBJECT'):
                    if nm==mXML.find('USAGE[@name="p:InputSignal"]/SIGNAL').attrib['name']:
                        message=mXML.attrib['class']
                th+=[[float(cXML.find('USAGE[@name="p:ParameterValues"]/PARAMETER').attrib['value']),message]]
        
        return th
    sn = 'rts:Dia/VRT-%s/Evaluator/%s/max.val' % (args.camera, name)
    ths=get_thresholds(sn)

    for i, evaluator in enumerate(evaluatorsXML):
        if name.lower() in evaluator.attrib['name'].lower():
            tv=np.array([[(int(t.attrib["timestamp"])-ts6)*1e-9,
                          float(t.attrib['max'])] for t in signalsXML[i].findall('value')])
            t = np.array(tv[:, 0], dtype='f4')
            data = np.array(object.raw2q(tv[:, 1]), dtype='f4')

            cam_roi = xr.DataArray(data, dims=['t'],
                coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'})}, 
                attrs={'long_name': object.getlabel(), 'shot': shot,
                       'camera': camera, 'name': name, 'threshold': ths[0][0], 
                       'info': ths[0][1]}
            )
    try:
        if plot:
            fig, ax = plt.subplots()
            cam_roi.plot(ax=ax)
        if xArrayOutput: return cam_roi
        else: return to_dict_with_metadata(cam_roi)
    except:
        print('Wrong camera or ROI')
        return 0


