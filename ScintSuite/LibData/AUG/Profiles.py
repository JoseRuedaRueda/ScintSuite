"""
Originally written by Pablo Oyola (pablo.oyola@ipp.mpg.de). Reimplemented by 
Jose Rueda (jrrueda@us.es) and Alex Reyner (areyner@us.es)

Routines to interact with the AUG database. Contains:

- electron quantities (density and temperature)
- ion quantities (temperature an rotation)
- IDA, IDI, CXRS, interferometer (DCN, DCK) and Thompson (VTA) imports
- ECE and Zeff
"""

import numpy as np
import xarray as xr
import aug_sfutils as sf

from typing import Optional
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d, UnivariateSpline

from ScintSuite.LibData.AUG.Equilibrium import get_rho, get_shot_basics
from ScintSuite._Paths import Path
pa = Path()

from ._common import safe_sfread, safe_equ, _clean_val, to_dict_with_metadata

xr.set_options(keep_attrs=True)
import logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# --- Electron density and temperature profiles (update 19/10/2026)
# -----------------------------------------------------------------------------
def get_ne_core(shot, diag = 'VTA', xArrayOutput: bool = True):
    """
    Wrap to get AUG core electron density
    Supports IDA and VTA (Thompson) and DCX (line integrated)

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  diag: diagnostic from which 'ne' will extracted (default VTA)
    :param  xArrayOutput: get the output in xarray format or dictionary
    """
    try:
        if diag == 'VTA':
            vta = get_VTA(shot)
            out =  vta[['ne', 'ne_unc']].mean('channel')
        elif diag == 'IDA':
            ida = get_IDA(shot)
            out =  ida[['ne', 'ne_unc']].sel(rho=0, method='nearest')
        elif diag in ['DCK', 'DCN']:
            out = get_DCX(shot, diag)
    except Exception: return
    return out if xArrayOutput else to_dict_with_metadata(out)

def get_ne(shot, xArrayOutput: bool = True, **kwargs):
    """
    Wrap to get AUG electron density profile.
    Supports IDA

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  xArrayOutput: get the output in xarray format or dictionary
    """
    try:
        ida = get_IDA(shot, **kwargs)
        out =  ida[['ne', 'ne_unc']]
    except Exception: return
    return out if xArrayOutput else to_dict_with_metadata(out)

def get_Te_core(shot, diag = 'VTA', xArrayOutput: bool = True):
    """
    Wrap to get AUG core electron temperature.
    Supports IDA and VTA (Thompson)

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  diag: diagnostic from which 'Te' will extracted (default VTA)
    :param  xArrayOutput: get the output in xarray format or dictionary
    """
    try:
        if diag == 'VTA':
            vta = get_VTA(shot)
            out =  vta[['Te', 'Te_unc']].mean('channel')
        elif diag == 'IDA':
            ida = get_IDA(shot)
            out =  ida[['Te', 'Te_unc']].sel(rho=0, method='nearest')
    except Exception: return
    return out if xArrayOutput else to_dict_with_metadata(out)

def get_Te(shot, xArrayOutput: bool = True, **kwargs):
    """
    Wrap to get AUG electron temperature profile
    Supports IDA

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  xArrayOutput: get the output in xarray format or dictionary
    """
    try:
        ida = get_IDA(shot, **kwargs)
        out =  ida[['Te', 'Te_unc']]
    except Exception: return
    return out if xArrayOutput else to_dict_with_metadata(out)


# -----------------------------------------------------------------------------
# --- Ion temperature (update 19/10/2026)
# -----------------------------------------------------------------------------
def get_Ti(shot, diag = 'IDI', xArrayOutput: bool = True, **kwargs):
    """
    Retrieves from the database the ion temperature.

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  diag: diagnostic from which 'Ti' will extracted.
    :param  xArrayOutput: get the output in xarray format or dictionary
    """
    try:
        if diag == 'IDI':
            idi = get_IDI(shot, **kwargs)
            out =  idi[['Ti', 'Ti_unc']]
        else:
            cxrs = get_CXRS(shot, diag, **kwargs).isel(line=0)
            out = cxrs[['Ti', 'Ti_unc']]
    except Exception: return
    return out if xArrayOutput else to_dict_with_metadata(out)

def get_Ti_core(shot: int, diag = 'IDI', xArrayOutput: bool = True):
    """"
    Wrapper to all the routines to read the ion core temperature.
    Supports CXRS and IDI

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  diag: diagnostic from which 'Ti' will extracted.
    :param  xArrayOutput: get the output in xarray format or dictionary
    """
    try:
        if diag == 'IDI':
            idi = get_IDI(shot).sel(rho_ti=0, method='nearest')
            out = idi[['Ti', 'Ti_unc']]
        else:
            cxrs = get_CXRS(shot, diag).isel(line=0)
            out = cxrs[['Ti', 'Ti_unc']]
    except Exception: return
    
    return out if xArrayOutput else to_dict_with_metadata(out)

def get_Ti_CXRS_new(shot, time: float = None, tavg: float = 0.05, 
                    nrho: int = 1000, rhop0: float = 0, rhop1: float = 1.1, 
                    smooth_factor: float = 100,
                    plot = False, xArrayOutput: bool = True):
    """
    Retrieves from the CXRS the ion temperature profile.

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es
    
    :param  shot: shotnumber to read.
    :param  time: time window to get the data.
    :param  tavg: averaging time in seconds. 50 ms by default.
    :param  nrho: number of points in rho_pol to calculate the smoothed profile
    :param  rhop0, rhop1: rhop limits
    :param  smooth_factor: smoothing factor to send to the UnivariateSpline
    class to perform the smoothing regression.
    :param  dr: correction in the radial direction. Can be just a number or a
    dict containing a correction for each diagnostic
    :param  dz: correction in the z direction
    """

    datasets = []
    diags = ['CMZ', 'CEZ', 'CUZ', 'COZ']
    for diag in diags:
        ds = get_CXRS(shot, diag)
        if ds is not None:
            datasets.append(ds)
    if not datasets:
        return
    ds_full = xr.concat(datasets, dim='line').sortby('R')
    ds_full.attrs['diag'] = 'CXRS'

    # --- Checking the time input.
    if time is None:
        time = np.array((0, 3600.0))  # Dummy limits.
    else:
        time = np.atleast_1d(time)
        # For a single time point, we choose a time window of the size
        # of the averaging window.
        if len(time) == 1:
            time = np.array([time.item() - tavg/2, time.item() + tavg/2])
        # For more than two points, we get the extreme ones.
        elif len(time) > 2:
            time = np.array([time.min(), time.max()])

    # --- Mean over the time window and transforming line -> rho
    ds_win = ds_full.sel(t=slice(*time))
    nt = ds_win.t.size

    data = ds_win.mean('t')
    data['Ti_unc'] = np.sqrt((ds_win['Ti_unc']**2).sum('t'))/nt
    data = data.swap_dims({'line': 'rho'}).sortby('rho')

    # --- Fitting the profile
    rho_vals = data['rho'].values
    ti_vals = data['Ti'].values
    err_vals = data['Ti_unc'].values
    ti_fitted, rhop_out = _fit_profile(rho=rho_vals, data=ti_vals, err=err_vals,
                                       nrho=nrho, rho0=rhop0, rho1=rhop1, 
                                       smooth_factor=smooth_factor)

    # Add fit to dataset
    data['Ti_fit'] = (('rhop',), ti_fitted, {'signal': 'Ti_fit', 
                                             'long_name': r'$T_i$', 
                                             'units': 'keV',})
    data = data.assign_coords(rhop=rhop_out)
    data['rhop'].attrs['long_name'] = r'$\rho_{pol}$'

    if plot:
        fig, ax = plt.subplots()
        plt.errorbar(data['rho'].values, data['Ti'].values, 
                     yerr=data['Ti_unc'].values,
                     fmt='o', capsize=3, linestyle='none')
        data.Ti_fit.plot()
        ax.set_xlim(data.rhop[0], data.rhop[-1])
        ax.set_ylim([0,None])

    if xArrayOutput: return data
    else: return to_dict_with_metadata(ds_full)

get_Ti_CXRS_profile = get_Ti_CXRS_new

def get_Ti_CXRS_old(shotnumber: int, time: float = None,
                exp: str = 'AUGD', edition: int = 0,
                tavg: float = 50.0, nrho: int = 200,
                smooth_factor: float = 500.0,
                rhop0: float = None, rhop1: float = None,
                dr=None, dz=None):
    """
    Read the ion temperature from CXRS diagnostics

    Jose Rueda: jrrueda@us.es

    Copied from Pablo routine of vtor

    :param  shotnumber: shotnumber to read.
    :param  time: time window to get the data. If None, all the available times
    are read.
    :param  exp: experiment under which the shotfile is stored.
    :param  edition: edition of the shotfile to read
    :param  tavg: averaging time in miliseconds. 50 ms by default.
    :param  nrho: number of points in rho_pol to calculate the smoothed profile.
    :param  smooth_factor: smoothing factor to send to the UnivariateSpline
    class to perform the smoothing regression.
    :param  dr: correction in the radial direction. Can be just a number or a
    dict containing a correction for each diagnostic
    :param  dz: correction in the z direction
    """
    text = 'This CXRS fit does not properly describe separatrix behaviour!'
    logger.warning('17: %s' % text)

    diags = ('CMZ', 'CEZ', 'CUZ', 'COZ',)
    signals = ('Ti_c', 'Ti_c', 'Ti_c', 'Ti_c', )
    error_signals = ('err_Ti_c', 'err_Ti_c', 'err_Ti_c', 'err_Ti_c')
    tavg *= 1.0e-3

    # --- Checking the inputs consistency.
    if dr is None:
        dr_corr_flag = False
    else:
        dr_corr_flag = True

    if dz is None:
        dz_corr_flag = False
    else:
        dz_corr_flag = True

    # --- Opening the shotfiles.
    nshotfiles = 0
    sfo = list()
    for ii in diags:
        sf_aux = sf.SFREAD(shotnumber, ii, exp=exp, edition=edition)
        if not sf_aux.status:
            print('Cannot open %s for shot #%05d\n' % (ii, shotnumber))
            continue

        nshotfiles += 1
        sfo.append(sf_aux)

    if nshotfiles == 0:
        raise Exception('Ti not available!')

    # --- Checking the time input.
    if time is None:
        time = np.array((0, 3600.0))  # Dummy limits.
    else:
        time = np.atleast_1d(time)

        # For a single time point, we choose a time window of the size
        # of the averaging window.
        if len(time) == 1:
            time = np.array([time - tavg/2.0, time + tavg/2.0])
        # For more than two points, we get the extreme ones.
        elif len(time) > 2:
            time = np.array([time.min(), time.max()])

    # --- Reading the shotfiles.
    Ti = list()
    timebase = list()
    rhopol = list()
    Ti_err = list()
    dt = list()
    for ii in np.arange(len(sfo), dtype=int):
        Ti_data = np.array(sfo[ii](signals[ii])).squeeze()
        zaux = np.array(sfo[ii](name='z')).squeeze()
        Raux = np.array(sfo[ii](name='R')).squeeze()
        err_aux = np.array(sfo[ii](name=error_signals[ii])).squeeze()
        time_aux = np.array(sfo[ii].gettimebase(signals[ii]))

        if Ti_data.size == 0:
            nshotfiles -= 1
            continue

        if len(sfo) == 0:
            raise Exception('Ti not available!')

        if ii == 0:
            Ti_aux = Ti_data

            # Cut only the interesting time window.
            t0, t1 = time_aux.searchsorted(time)

            if t1 == t0:
                t1 += 1
            if not isinstance(t0, np.int64):
                t0 = t0[0]
                t1 = t1[0]
            time_aux = time_aux[t0:t1]

            if len(time_aux) == 0:
                nshotfiles -= 1
                continue

            if len(sfo) == 0:
                raise Exception('Ti not available!')

            # Some channels are broken. For those R = 0, and we can easily
            # take them out.
            flags = Raux > 1.0
            R = Raux[flags]
            z = zaux[flags]

            # Adding the dR and dZ corrections into the diagnostic
            if dr_corr_flag:
                if isinstance(dr, dict):
                    if diags[ii] in dr:
                        R += dr[diags[ii]]
                elif isinstance(dr, float):
                    R += dr

            if dz_corr_flag:
                if isinstance(dz, dict):
                    if diags[ii] in dr:
                        z += dz[diags[ii]]
                elif isinstance(dz, float):
                    z += dz

            Ti_aux = Ti_aux[t0:t1, flags]
            err_aux = err_aux[t0:t1, flags]

            # Appending to the diagnostic list.
            Ti.append(Ti_aux)
            timebase.append(time_aux)
            Ti_err.append(err_aux)

            # Transforming (R, z) into rhopol.
            rhopol_aux = get_rho(shot=shotnumber, Rin=R, zin=z,
                                 time=time_aux)

            rhopol.append(rhopol_aux)

            if time_aux.size > 1:
                dt.append(time_aux[1]-time_aux[0])
            else:
                dt.append(tavg)
            del R
            del z
            del rhopol_aux
            del Ti_aux
        else:
            # Getting the time window.
            Ti_aux = Ti_data
            t0, t1 = time_aux.searchsorted(time)
            if t0 == 1:
                t1 += 1
            if not isinstance(t0, np.int64):
                t0 = t0[0]
                t1 = t1[0]
            time_aux = time_aux[t0:t1]
            if len(time_aux) == 0:
                nshotfiles -= 1
                continue

            if len(sfo) == 0:
                raise Exception('Ti not available!')

            # If the major radius is zero, that channel should be taken
            # away.
            flags = Raux > 1.0
            R = Raux[flags]
            z = zaux[flags]

            # Adding the dR and dZ corrections into the diagnostic
            if dr_corr_flag:
                if isinstance(dr, dict):
                    if diags[ii] in dr:
                        R += dr[diags[ii]]
                elif isinstance(dr, float):
                    R += dr

            if dz_corr_flag:
                if isinstance(dz, dict):
                    if diags[ii] in dr:
                        z += dz[diags[ii]]
                elif isinstance(dz, float):
                    z += dz

            # Transforming (R, z) into rhopol.
            rhopol_aux = get_rho(shot=shotnumber, Rin=R, zin=z,
                                 time=time_aux)

            rhopol.append(rhopol_aux)
            Ti_aux = Ti_aux[t0:t1, flags]
            err_aux = err_aux[t0:t1, flags]

            # Appending to the diagnostic list.
            Ti.append(Ti_aux)
            timebase.append(time_aux)
            Ti_err.append(err_aux)
            dt.append(time_aux[1]-time_aux[0])

            del R
            del z
            del time_aux
            del rhopol_aux
            del Ti_aux
        del Raux
        del zaux
        del Ti_data
        del err_aux

    # --- Transforming R -> rhopol.
    output = {
        'diags': diags,
        'raw': {
            'data': Ti,
            'rhopol': rhopol,
            'time': timebase,
            'err': Ti_err
        }
    }

    # --- Fitting the profiles.
    if nshotfiles > 1:
        tBegin = np.concatenate(timebase).min()
        tEnd = np.concatenate(timebase).max()
        if rhop0 is None:
            rhop0 = np.array([x.min() for x in rhopol]).min()
        if rhop1 is None:
            rhop1 = np.array([x.max() for x in rhopol]).max()
    else:
        tBegin = np.array(timebase).min()
        tEnd = np.array(timebase).max()
        if rhop0 is None:
            rhop0 = np.array(rhopol).min()
        if rhop1 is None:
            rhop1 = np.array(rhopol).max()

    dt = max(dt)
    tavg = max(tavg, dt)
    nwindows = max(1, int((tEnd - tBegin)/tavg))

    time_out = np.linspace(tBegin, tEnd, nwindows)
    rhop_out = np.linspace(rhop0, rhop1, num=nrho)
    Ti_out = np.zeros((time_out.size, rhop_out.size))
    for iwin in np.arange(nwindows, dtype=int):
        data = list()
        rhop = list()
        weight = list()
        # Appending to a list all the data points within the time range
        # for all diagnostics.
        for idiags in np.arange(nshotfiles, dtype=int):
            time0 = float(iwin) * tavg + tBegin
            time1 = time0 + tavg

            t0, t1 = timebase[idiags].searchsorted((time0, time1))

            data.append(Ti[idiags][t0:t1, :].flatten())
            rhop.append(rhopol[idiags][t0:t1, :].flatten())
            weight.append(1.0/Ti_err[idiags][t0:t1, :].flatten())

        # Using the smoothing spline.
        data = np.asarray(np.concatenate(data)).flatten()
        err = np.asarray(np.concatenate(weight)).flatten()
        rhop = np.asarray(np.concatenate(rhop)).flatten()

        sorted_index = np.argsort(rhop)
        rhop = rhop[sorted_index]
        data = data[sorted_index]
        err = err[sorted_index]

        flags_err = (err == np.inf) | (err == 0.0) | (data == 0.0)
        rhop = rhop[~flags_err]
        err = err[~flags_err]
        data = data[~flags_err]

        if len(data) < 8:
            time_out[iwin] = np.nan
            Ti_out[iwin, :] = np.nan
            continue

        # Creating smoothing spline
        splineFun = UnivariateSpline(x=rhop, y=data, w=err, s=smooth_factor,
                                     ext=0.0)
        Ti_out[iwin, :] = splineFun(rhop_out)
        rhop_local_min = rhop.min()
        rhop_local_max = rhop.max()
        flags = (rhop_out < rhop_local_min) | (rhop_out > rhop_local_max)
        Ti_out[iwin, flags] = np.nan
        del splineFun
        del data
        del rhop
        del weight
        del flags

    flags = np.isnan(time_out)

    output['fit'] = {
        'rhop': rhop_out,
        'data': Ti_out[~flags, :],
        'time': time_out[~flags]
    }

    return output


# -----------------------------------------------------------------------------
# --- Toroidal rotation velocity (update 19/10/2026)
# -----------------------------------------------------------------------------
def get_tor_rotation(shot, diag = 'IDI', xArrayOutput: bool = True, **kwargs):
    """
    Retrieves from the database the toroidal velocity velocity (omega_tor).
    To get the linear velocity (i.e., vtor) multiply by the major radius.

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  diag: either IDI or any CXRS
    :param  xArrayOutput: get the output in xarray format or dictionary
    """
    try:
        if diag == 'IDI':
            idi = get_IDI(shot, **kwargs).sel(rho_ti=0, method='nearest')
            out = idi[['vt', 'vt_unc']]
        else:
            cxrs = get_CXRS(shot, diag, **kwargs).isel(line=0)
            out = cxrs[['vt', 'vt_unc']]
    except Exception: return
    return out if xArrayOutput else to_dict_with_metadata(out)

def get_tor_rotation_CXRS_new(shot, time: float = None, tavg: float = 0.05, 
                    nrho: int = 1000, rhop0: float = 0, rhop1: float = 1.1, 
                    smooth_factor: float = 100,
                    plot = False, xArrayOutput: bool = True):
    """
    Retrieves from the CXRS the toroidal velocity profile (omega_tor).
    To get the linear velocity (i.e., vtor) multiply by the major radius.

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shotnumber to read.
    :param  time: time window to get the data.
    :param  tavg: averaging time in seconds. 50 ms by default.
    :param  nrho: number of points in rho_pol to calculate the smoothed profile
    :param  rhop0, rhop1: rhop limits
    :param  smooth_factor: smoothing factor to send to the UnivariateSpline
    class to perform the smoothing regression.
    :param  dr: correction in the radial direction. Can be just a number or a
    dict containing a correction for each diagnostic
    :param  dz: correction in the z direction
    """

    datasets = []
    diags = ['CMZ', 'CEZ', 'CUZ', 'COZ']
    for diag in diags:
        ds = get_CXRS(shot, diag)
        if ds is not None:
            datasets.append(ds)
    if not datasets:
        return
    ds_full = xr.concat(datasets, dim='line').sortby('R')
    ds_full.attrs['diag'] = 'CXRS'

    # --- Checking the time input.
    if time is None:
        time = np.array((0, 3600.0))  # Dummy limits.
    else:
        time = np.atleast_1d(time)
        # For a single time point, we choose a time window of the size
        # of the averaging window.
        if len(time) == 1:
            time = np.array([time.item() - tavg/2, time.item() + tavg/2])
        # For more than two points, we get the extreme ones.
        elif len(time) > 2:
            time = np.array([time.min(), time.max()])

    # --- Mean over the time window and transforming line -> rho
    ds_win = ds_full.sel(t=slice(*time))
    nt = ds_win.t.size

    data = ds_win.mean('t')
    data['vt_unc'] = np.sqrt((ds_win['vt_unc']**2).sum('t'))/nt
    data = data.swap_dims({'line': 'rho'}).sortby('rho')

    # --- Fitting the profile
    rho_vals = data['rho'].values
    vt_vals = data['vt'].values
    err_vals = data['vt_unc'].values
    ti_fitted, rhop_out = _fit_profile(rho=rho_vals, data=vt_vals, err=err_vals,
                                       nrho=nrho, rho0=rhop0, rho1=rhop1, 
                                       smooth_factor=smooth_factor)

    # Add fit to dataset
    data['vt_fit'] = (('rhop',), ti_fitted, {'signal': 'vt_fit', 
                                             'long_name': r'$\omega$', 
                                             'units': 'rad/s',})
    data = data.assign_coords(rhop=rhop_out)
    data['rhop'].attrs['long_name'] = r'$\rho_{pol}$'

    if plot:
        fig, ax = plt.subplots()
        plt.errorbar(data['rho'].values, data['vt'].values, 
                     yerr=data['vt_unc'].values,
                     fmt='o', capsize=3, linestyle='none')
        data.vt_fit.plot()
        ax.set_xlim(data.rhop[0], data.rhop[-1])
        ax.set_ylim([0,None])

    if xArrayOutput: return data
    else: return to_dict_with_metadata(ds_full)

get_tor_rotation_CXRS_profile = get_tor_rotation_CXRS_new

def get_tor_rotation_cxrs_old(shotnumber: int, time: float = None,
                          exp: str = 'AUGD', edition: int = 0,
                          tavg: float = 2.0, nrho: int = 200,
                          smooth_factor: float = 500.0,
                          rhop0: float = None, rhop1: float = None,
                          dr=None, dz=None):
    """
    Reads from several diagnostics containing information about the toroidal
    rotation velocity.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: shotnumber to read.
    :param  time: time window to get the data. If None, all the available times
    are read.
    :param  exp: experiment under which the shotfile is stored.
    :param  edition: edition of the shotfile to read
    :param  tavg: averaging time in miliseconds. 50 ms by default.
    :param  nrho: number of points in rho_pol to calculate the smoothed profile.
    :param  smooth_factor: smoothing factor to send to the UnivariateSpline
    class to perform the smoothing regression.
    """

    diags = ('CMZ', 'CEZ', 'CUZ', 'COZ',)
    signals = ('vr_c', 'vr_c', 'vr_c', 'vr_c', )
    error_signals = ('err_vr_c', 'err_vr_c', 'err_vr_c', 'err_vr_c')
    tavg *= 1.0e-3

    # --- Checking the inputs consistency.
    if dr is None:
        dr_corr_flag = False
    else:
        dr_corr_flag = True

    if dz is None:
        dz_corr_flag = False
    else:
        dz_corr_flag = True

    # --- Opening the shotfiles.
    nshotfiles = 0
    sfo = list()
    for ii in diags:
        sf_aux = sf.SFREAD(shotnumber, ii, exp=exp, edition=edition)
        if not sf_aux.status:
            print('Cannot open %s for shot #%05d\n' % (ii, shotnumber))
            continue

        nshotfiles += 1
        sfo.append(sf_aux)

    if nshotfiles == 0:
        raise Exception('Toroidal rotation velocity not available!')

    # --- Checking the time input.
    if time is None:
        time = np.array((0, 3600.0))  # Dummy limits.
    else:
        time = np.atleast_1d(time)

        # For a single time point, we choose a time window of the size
        # of the averaging window.
        if len(time) == 1:
            time = np.array([time - tavg/2.0, time + tavg/2.0])
        # For more than two points, we get the extreme ones.
        elif len(time) > 2:
            time = np.array([time.min(), time.max()])

    # --- Reading the shotfiles.
    vtor = list()
    timebase = list()
    rhopol = list()
    vt_err = list()
    dt = list()
    for ii in range(len(sfo)):
        vtor_data = np.array(sfo[ii](signals[ii])).squeeze()
        zaux = np.array(sfo[ii](name='z')).squeeze()
        Raux = np.array(sfo[ii](name='R')).squeeze()
        err_aux = np.array(sfo[ii](name=error_signals[ii])).squeeze()
        time_aux = np.array(sfo[ii].gettimebase(signals[ii]))

        if vtor_data.size == 0:
            nshotfiles -= 1
            continue

        if len(sfo) == 0:
            raise Exception('Toroidal rotation velocity not available!')
        if ii == 0:
            # Cut only the interesting time window.
            t0, t1 = time_aux.searchsorted(time)

            if t1 == t0:
                t1 += 1
            if not isinstance(t0, np.int64):
                t0 = t0[0]
                t1 = t1[0]
            time_aux = time_aux[t0:t1]

            if len(time_aux) == 0:
                nshotfiles -= 1
                continue
            if len(sfo) == 0:
                raise Exception('Toroidal rotation velocity not available!')

            # Some channels are broken. For those R = 0, and we can easily
            # take them out.
            flags = Raux > 1.0
            R = Raux[flags]
            z = zaux[flags]
            # Adding the dR and dZ corrections into the diagnostic
            if dr_corr_flag:
                if isinstance(dr, dict):
                    if diags[ii] in dr:
                        R += dr[diags[ii]]
                elif isinstance(dr, float):
                    R += dr

            if dz_corr_flag:
                if isinstance(dz, dict):
                    if diags[ii] in dr:
                        z += dz[diags[ii]]
                elif isinstance(dz, float):
                    z += dz

            # Getting the rotation velocity (rad/s)
            vtor_aux = vtor_data[t0:t1, flags]/R
            err_aux = err_aux[t0:t1, flags]/R

            # If the velocity is zero, we take it away.
            flags_nan = (vtor_aux == 0.0) & (err_aux == 0.0)
            err_aux[flags_nan] = np.inf

            # Appending to the diagnostic list.
            vtor.append(vtor_aux)
            timebase.append(time_aux)
            vt_err.append(err_aux)

            # Transforming (R, z) into rhopol.
            rhopol_aux = get_rho(shot=shotnumber, Rin=R, zin=z,
                                 time=time_aux)

            rhopol.append(rhopol_aux)

            dt.append(time_aux[1]-time_aux[0])
            del R
            del z
            del rhopol_aux
            del vtor_aux
        else:
            # Cut only the interesting time window.
            t0, t1 = time_aux.searchsorted(time)

            if t1 == t0:
                t1 += 1
            if not isinstance(t0, np.int64):
                t0 = t0[0]
                t1 = t1[0]
            time_aux = time_aux[t0:t1]

            if len(time_aux) == 0:
                nshotfiles -= 1
                continue
            if len(sfo) == 0:
                raise Exception('Toroidal rotation velocity not available!')

            # If the major radius is zero, that channel should be taken
            # away.
            flags = Raux > 1.0
            R = Raux[flags]
            z = zaux[flags]

            # Adding the dR and dZ corrections into the diagnostic
            if dr_corr_flag:
                if isinstance(dr, dict):
                    if diags[ii] in dr:
                        R += dr[diags[ii]]
                elif isinstance(dr, float):
                    R += dr

            if dz_corr_flag:
                if isinstance(dz, dict):
                    if diags[ii] in dr:
                        z += dz[diags[ii]]
                elif isinstance(dz, float):
                    z += dz

            # Going to angular rotation velocity.
            vtor_aux = vtor_data[t0:t1, flags]/R
            err_aux  = err_aux[t0:t1, flags]/R

            # Transforming (R, z) into rhopol.
            rhopol_aux = get_rho(shot=shotnumber, Rin=R, zin=z,
                                 time=time_aux)

            # If there is some 0.0 rotation velocity, we remove it by
            # setting it to NaN.
            flags_nan = vtor_aux == 0.0
            flags_nan = (vtor_aux == 0.0) & (err_aux == 0.0)
            err_aux[flags_nan] = np.inf

            # Adding to the list.
            vtor.append(vtor_aux)
            rhopol.append(rhopol_aux)
            timebase.append(time_aux)
            vt_err.append(err_aux)
            dt.append(time_aux[1]-time_aux[0])

            del R
            del z
            del time_aux
            del rhopol_aux
            del vtor_aux
        del Raux
        del zaux
        del vtor_data
        del err_aux

    # --- Transforming R -> rhopol.
    output = {'diags': diags,
              'raw': {
                   'data': vtor,
                   'rhopol': rhopol,
                   'time': timebase,
                   'err': vt_err
              }
    }

    # --- Fitting the profiles.
    if nshotfiles > 1:
        tBegin = np.concatenate(timebase).min()
        tEnd = np.concatenate(timebase).max()
        if rhop0 is None:
            rhop0 = np.array([x.min() for x in rhopol]).min()
        if rhop1 is None:
            rhop1 = np.array([x.max() for x in rhopol]).max()
    else:
        tBegin = np.array(timebase).min()
        tEnd = np.array(timebase).max()
        if rhop0 is None:
            rhop0 = np.array(rhopol).min()
        if rhop1 is None:
            rhop1 = np.array(rhopol).max()

    dt     = max(dt)
    tavg   = max(tavg, dt)
    nwindows = max(1, int((tEnd - tBegin)/tavg))

    time_out = np.linspace(tBegin, tEnd, nwindows)
    rhop_out = np.linspace(rhop0, rhop1, num=nrho)
    vtor_out = np.zeros((time_out.size, rhop_out.size))
    for iwin in range(nwindows):
        data = list()
        rhop = list()
        weight = list()
        # Appending to a list all the data points within the time range
        # for all diagnostics.
        for idiags in range(nshotfiles):
            time0 = float(iwin) * tavg + tBegin
            time1 = time0 + tavg

            t0, t1 = timebase[idiags].searchsorted((time0, time1))
            # t1 += 1

            data.append(vtor[idiags][t0:t1, :].flatten())
            rhop.append(rhopol[idiags][t0:t1, :].flatten())
            weight.append(1.0/vt_err[idiags][t0:t1, :].flatten())

        # Using the smoothing spline.
        data = np.asarray(np.concatenate(data)).flatten()
        err  = np.asarray(np.concatenate(weight)).flatten()
        rhop = np.asarray(np.concatenate(rhop)).flatten()

        sorted_index = np.argsort(rhop)

        rhop = rhop[sorted_index]
        data = data[sorted_index]
        err  = err[sorted_index]

        flags_err = (err == np.inf) | (err == 0.0) | (data == 0.0)
        rhop = rhop[~flags_err]
        err = err[~flags_err]
        data = data[~flags_err]

        if len(data) < 8:
            time_out[iwin] = np.nan
            vtor_out[iwin, :] = np.nan
            continue

        # Creating smoothing spline
        splineFun = UnivariateSpline(x=rhop, y=data, w=err, s=smooth_factor,
                                     ext=0.0)
        vtor_out[iwin, :] = splineFun(rhop_out)
        rhop_local_min = rhop.min()
        rhop_local_max = rhop.max()
        flags = (rhop_out < rhop_local_min) | (rhop_out > rhop_local_max)
        vtor_out[iwin, flags] = np.nan
        del splineFun
        del data
        del rhop
        del weight
        del flags

    flags = np.isnan(time_out)

    output['fit'] = {
        'rhop': rhop_out,
        'data': vtor_out[~flags, :],
        'time': time_out[~flags]
    }

    return output

def get_diag_freq(shotnumber: int, tBegin: float, tEnd: float,
                  equ_diag: dict = None, prof_diag: dict = None):
    """
    Computes the diamagnetic Doppler correction in the large aspect-ratio
    tokamak assuming that this is evaluated in a resonant surface such that
    q = m/n, for a provided 'n'.

    This approximation is quite conservative and shall only be used as an
    approximation and only within the confined region (i.e., rhopol < 2)

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: shotnumber to get the diamagnetic drift.
    :param  tBegin: initial point of the time window to use.
    :param  tEnd: ending point of the time window to use.
    :param  equ_diag: diagnostic for the magnetic equilibrium. If None, EQH is
    used.
    :param  prof_diag: diagnostic to retrieve the pressure and density profiles.
    If None, IDA is chosen by default.
    """

    # --- Checking the diagnostics.
    if prof_diag is None:
        prof_diag = { 'diag': 'IDA',
                      'edition': 0,
                      'exp': 'AUGD'
                    }
    if equ_diag is None:
        equ_diag = { 'diag': 'EQH',
                      'edition': 0,
                      'exp': 'AUGD'
                    }

    # --- Getting the position of the edge and the magnetic axis.
    shotbasics = get_shot_basics(shotnumber=shotnumber,
                                 diag=equ_diag['diag'],
                                 exp=equ_diag['exp'],
                                 edition=equ_diag['edition'],
                                 time=(tBegin, tEnd))

    Raus = shotbasics['Raus']
    Raxis = shotbasics['Rmag']
    zaxis = shotbasics['Zmag']

    # Getting the poloidal flux matrix:
    equ = sf.EQU(shotnumber, diag=equ_diag['diag'],
                      exp=equ_diag['exp'], ed=equ_diag['edition'])

    Rpfm = equ.Rmesh
    zpfm = equ.Zmesh
    time = equ.time
    t0, t1 = np.searchsorted(time, (tBegin, tEnd))
    equ.read_pfm()
    pfm = equ.pfm[:, :, t0:t1].copy()
    nt = pfm.shape[2]
    time = time[t0:t1]

    # Getting the PFL at the axis and at the separatrix.

    psi_ax = np.zeros(Raus.shape)
    psi_ed = np.zeros(Raus.shape)

    # --- Getting the poloidal flux at the separatrix.
    for itime in range(nt):
        pfm_interp = interp2d(Rpfm, zpfm, pfm[:, :, itime].T, kind='linear')

        psi_ax[itime] = pfm_interp(Raxis[itime], zaxis[itime])
        psi_ed[itime] = pfm_interp(Raus[itime], zaxis[itime])

    # --- This is the normalization to get the rho_pol
    psinorm = psi_ed - psi_ax

    # --- Getting the electron pressure and density
    ne = get_ne(shotnumber=shotnumber, time=time, **prof_diag)

    Te = get_Te(shotnumber=shotnumber, time=time, **prof_diag)

    prs_e = ne['data'] * Te['data']
    rhop = ne['rhop']

    psinorm = interp1d(shotbasics['time'], psinorm, kind='linear')(time)

    drhop = rhop[1] - rhop[0]
    dprs_e = np.gradient(prs_e, drhop, axis=0) # Pressure gradient.

    # --- Getting the omega_tor
    omega_star = -dprs_e/(ne['data']*(2.0*np.pi *1.0e3))
    omega_star /= np.tile(ne['rhop'], (omega_star.shape[1], 1)).T
    omega_star /= np.tile(psinorm, (omega_star.shape[0], 1))

    output = {
               'fdiag': omega_star,
               'rhop': rhop,
               'time': time
             }

    return output


# -----------------------------------------------------------------------------
# --- IDA, IDI and other group diagnostic (update 19/10/2026)
# -----------------------------------------------------------------------------
def get_IDA(shot, plot = False, xArrayOutput: bool = True, **kwargs):
    """
    Returns some of the IDA quantities. IDA is not available in all shots

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  plot: quick plot of the data
    :param  xArrayOutput: get the output in xarray format or dictionary
    """

    IDA = safe_sfread('IDA', shot, **kwargs)
    if IDA is None:
        return None
    t = np.array(IDA.gettimebase('Te'), dtype='f4')
    rho_grid = np.array(IDA.getareabase('Te'), dtype='f4')

    obj = xr.Dataset(
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}),
                'rho': ('rho', rho_grid[:,0], {'long_name': r'$\rho_{pol}$'}),
                },
        attrs={'shot': shot, 'diag': 'IDA'}
    )

    for signal in ['Te', 'Te_up', 'Te_lo', 'Te_unc', 'dTe_r',
                   'ne', 'ne_up', 'ne_lo', 'ne_unc', 'dne_r',
                   'pe', 'pe_up', 'pe_lo', 'pe_unc', 'dpe_r']:
        try:
            data = np.array(IDA(signal), dtype='f4')       
            if 'Te' in signal:
                data /= 1e3
                units = 'keV'
                long_name = r'$T_e$'
            elif 'ne' in signal:
                data /= 1e19
                units = r'$10^{19}$ /m^3'
                long_name = r'$n_e$'
            elif 'pe' in signal:
                data = data
                units = 'Pa'
                long_name = r'$p_e$'
            if '_r' in signal:
                long_name = long_name + 'grad.'
                units = ''
            if 'unc' in signal:
                long_name = r'$\Delta$ '+long_name
            obj[signal] = (['rho', 't'], data, 
                           {'signal': signal, 'long_name': long_name, 
                            'units': units,},
                    )
        except:
            continue
    obj = obj.where(obj >= 0, 0) # no negative values

    if plot:
        fig, ax = plt.subplots(3, 1, sharex = True)
        obj.Te.plot.imshow(ax=ax[0], robust=True, vmin=0)
        obj.ne.plot.imshow(ax=ax[1], robust=True, vmin=0)
        obj.pe.plot.imshow(ax=ax[2], robust=True, vmin=0)
        for axs in ax[:-1]: 
            axs.set_xlabel('')
            axs.set_xlim([0,10])

    return obj if xArrayOutput else to_dict_with_metadata(obj)

def get_IDI(shot, plot = False, xArrayOutput: bool = True, **kwargs):
    """
    Same as IDA, but for ions.

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  plot: quick plot of the data
    :param  xArrayOutput: get the output in xarray format or dictionary
    """

    IDI = safe_sfread('IDI', shot, **kwargs)
    if IDI is None:
        return None
    t = np.array(IDI.gettimebase('Ti'), dtype='f4')
    rho_Ti = np.array(IDI.getareabase('Ti'), dtype='f4')
    rho_vt = np.array(IDI.getareabase('vt'), dtype='f4')

    obj = xr.Dataset(
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}),
                'rho_ti': ('rho_ti', rho_Ti[:,0], {'long_name': r'$\rho_{pol}$'}),
                'rho_vt': ('rho_vt', rho_vt[:,0], {'long_name': r'$\rho_{pol}$'}),
                },
        attrs={'shot': shot, 'diag': 'IDI'}
    )

    for signal in ['Ti', 'Ti_unc']:
        try:
            data = np.array(IDI(signal), dtype='f4') / 1e3
            units = 'keV'
            long_name = r'$T_i$'
            if 'unc' in signal: long_name = r'$\Delta$'+long_name
            obj[signal] = (['rho_ti', 't'], data.T, 
                           {'signal': signal, 'long_name': long_name, 
                            'units': units,},
                    )
        except:
            continue
    for signal in ['vt', 'vt_unc']:
        try:
            data = np.array(IDI(signal), dtype='f4')
            units = 'rad/s'
            long_name = r'$\omega$'
            if 'unc' in signal: long_name = r'$\sigma\omega$'
            obj[signal] = (['rho_vt', 't'], data.T, 
                           {'signal': signal, 'long_name': long_name, 
                            'units': units,},
                    )
        except:
            continue
    obj = obj.where(obj >= 0, 0) # no negative values

    if plot:
        fig, ax = plt.subplots(2, 1, sharex = True)
        obj.Ti.plot.imshow(ax=ax[0], robust=True, vmin=0)
        obj.vt.plot.imshow(ax=ax[1], robust=True, vmin=0)
        for axs in ax[:-1]: 
            axs.set_xlabel('')
            axs.set_xlim([0,10])
        
    return obj if xArrayOutput else to_dict_with_metadata(obj)

def get_VTA(shot, plot = False, xArrayOutput: bool = True, **kwargs):
    """
    Thompson diagnostic

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  plot: quick plot of the data
    :param  xArrayOutput: get the output in xarray format or dictionary
    """

    VTA = safe_sfread('VTA', shot, **kwargs)
    if VTA is None:
        return None
    t = np.array(VTA.gettimebase('Te_c'), dtype=np.float32)

    obj = xr.Dataset(
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}),
                'channel': ('channel', np.arange(1,17), {'long_name': 'Channel'})},
        attrs={'shot': shot, 'diag': 'VTA',},
    )
    # data = np.array(VTA('Te_c'), dtype=np.float32)/1e3
    # data_mean = data.mean(axis=1, dtype=np.float32)
    obj['Te'] = (['channel','t'], np.array(VTA('Te_c'), dtype=np.float32).T/1e3, 
                 {'signal': 'Te_c', 'long_name': r'$T_{e,c}$', 'units': 'keV'},)
    obj['Te_unc'] = (['channel','t'], np.array(VTA('SigTe_c'), dtype=np.float32).T/1e3, 
                 {'signal': 'SigTe_c', 'long_name': r'$T_{e,c}$', 'units': 'keV'},)

    obj['ne'] = (['channel','t'], np.array(VTA('Ne_c'), dtype=np.float32).T/1e19, 
                 {'signal': 'Ne_c', 'long_name': r'$n_{e,c}$', 
                  'units': r'$10^{19}$ /m^3'},)
    obj['ne_unc'] = (['channel','t'], np.array(VTA('SigNe_c'), dtype=np.float32).T/1e19, 
                 {'signal': 'SigTe_c', 'long_name': r'$n_{e,c}$', 
                  'units': r'$10^{19}$ /m^3'},)

    if plot:
        fig, ax = plt.subplots(2, 1, sharex=True)
        val = obj.mean('channel').rolling(t=10, center=True, min_periods=1).mean()
        ax[0].fill_between(val.t, val.Te+val.Te_unc, val.Te-val.Te_unc, alpha=0.3)
        val.Te.plot(ax=ax[0])
        ax[0].set_ylim([0,20])
        ax[1].fill_between(val.t, val.ne+val.ne_unc, val.ne-val.ne_unc, alpha=0.3)
        val.ne.plot(ax=ax[1])
        ax[1].set_ylim([0,12])
        for axs in ax[:-1]: 
            axs.set_xlabel('')
            axs.set_xlim([0,10])

    return obj if xArrayOutput else to_dict_with_metadata(obj)

def get_DCX(shot, diag = 'DCK', xArrayOutput: bool = True, **kwargs):
    """
    Gets de integrated density form the interferometers

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  diag: interferometer diagnostic (default DCK)
    :param  xArrayOutput: get the output in xarray format or dictionary
    """

    if diag not in ('DCN', 'DCK'):
        raise ValueError(f'Invalid diagnostic. Supported: DCN or DCK.')

    DC = safe_sfread(diag, shot, **kwargs)
    if DC is None:
        return None
    t = np.array(DC.gettimebase('H-0'),dtype='f4')

    names = ['H-0','H-1','H-2','H-4','H-5']
    obj = xr.Dataset(
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}),},
        attrs={'shot': shot, 'diag': diag,},
    )
    for n in names:
        data = np.array(DC(n), dtype='f4') / 1e19
        var_name = n.replace('-', '')
        obj[var_name] = (['t'], data, 
                         {'signal': n, 'long_name': r'$n_e$', 
                          'units': r'$10^{19}\ \mathrm{/m^2}$',})

    return obj if xArrayOutput else to_dict_with_metadata(obj)

def get_CXRS(shot, diag = 'CEZ', xArrayOutput: bool = True, **kwargs):
    """
    Imports one of the CXRS diagnostics

    Alex Reyner: areyner@us.es

    :param  shot: shotnumber to read.
    :param  diag: CXRS diagnostic to import
    :param  xArrayOutput: get the output in xarray format or dictionary
    """
    if diag not in ('IDI', 'CEZ', 'CUZ', 'COZ'):
        raise Exception('Diagnostic not supported!')
    
    cxrs = safe_sfread(diag, shot, **kwargs)
    if cxrs is None:
        return None
    
    t = np.array(cxrs.gettimebase('Ti_c'), dtype='f4')
    R = np.array(cxrs('R'), dtype='f4').squeeze()
    z = np.array(cxrs('z'), dtype='f4').squeeze()
    line = np.arange(len(R))+1
    rhopol_vals = get_rho(shot=shot, Rin=R, zin=z, time=t)

    obj = xr.Dataset(
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}),
                'line': ('line', line, {'long_name': 'LOS',}),
                },
        attrs={'shot': shot, 'diag': diag}
    )

    obj['Ti'] = (['line','t'], np.array(cxrs('Ti_c'), dtype='f4').T /1e3, 
                    {'signal': 'Ti_c', 'long_name': r'$T_i$', 
                    'units': 'keV',},
            )
    obj['Ti_unc'] = (['line','t'], np.array(cxrs('err_Ti_c'), dtype='f4').T /1e3,
                    {'signal': 'err_Ti_c', 'long_name': r'$\Delta T_i$', 
                    'units': 'keV',},
            )
    obj['vt'] = (['line','t'], np.array(cxrs('vr_c')/R, dtype='f4').T, 
                    {'signal': 'vr_c', 'long_name': r'$\omega$', 
                    'units': 'rad/s',},
            ) 
    obj['vt_unc'] = (['line','t'], np.array(cxrs('err_vr_c')/R, dtype='f4').T, 
                    {'signal': 'err_vr_c', 'long_name': r'$\sigma\omega$', 
                    'units': 'rad/s',},
            )
    obj['z'] = (['line'], z, {'long_name': 'z', 'units': 'm'})
    obj['R'] = (['line'], R, {'long_name': 'R', 'units': 'm'})
    obj['rho'] = (['line','t'], rhopol_vals.T, {'long_name': r'\rho_{pol}'})

    valid_channels = (obj.R > 1.0) & (obj.R != 0) & (obj.z != 0)
    obj = obj.sel(line=valid_channels).sortby('R')

    return obj if xArrayOutput else to_dict_with_metadata(obj)

def _fit_profile(rho, data, err, nrho, rho0, rho1, smooth_factor):
    """
    Fitss profile data with UnivariateSpline, returns line

    :param  rho: input rho values
    :param  data: values to fit
    :param  err: associated error values
    :param  nrho: number output rho values
    :param  rho0: initial output rho
    :param  rho1: final output rho
    :param  smooth_factor: to smooth the curve
    """
    # Computes weights
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_vals = 1.0 / err

    # Filter invalid numbers and zeros
    flags = (np.isnan(rho) | np.isnan(data) | np.isnan(err) | 
             (err == np.inf) | (err == 0.0) | (data == 0.0) | 
             np.isinf(rho) | np.isinf(data))
            
    rho_clean = rho[~flags]
    data_clean = data[~flags]
    err_clean = weight_vals[~flags]

    # Define output mesh
    if rho0 is None:
        rho0 = float(rho_clean.min())
    if rho1 is None:
        rho1 = float(rho_clean.max())
    rho_out = np.linspace(rho0, rho1, num=nrho)

    if len(rho_clean) >= 8:
        # Sort
        sorted_idx = np.argsort(rho_clean)
        rho_clean = rho_clean[sorted_idx]
        data_clean = data_clean[sorted_idx]
        err_clean = err_clean[sorted_idx]
        # Spline
        splineFun = UnivariateSpline(x=rho_clean, y=data_clean, w=err_clean, 
                                     s=smooth_factor, ext=0.0)
        data_fitted = splineFun(rho_out)
        # Remove data out of range
        eps = 0
        extrap_flags = (rho_out < (rho_clean.min()-eps)) | (rho_out > (rho_clean.max()+eps))
        data_fitted[extrap_flags] = np.nan
    else:
        data_fitted = np.full_like(rho_out, np.nan)

    return data_fitted, rho_out


# -----------------------------------------------------------------------------
# --- ECE data (update 19/10/2026)
# -----------------------------------------------------------------------------
def get_ECE(shot, xArrayOutput: bool = True):
    '''
    Get data from ECE

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  xArrayOutput: get the output in xarray format or dictionary
    '''

    CEC = safe_sfread('CEC', shot)
    if CEC is None:
        return None
    
    RMC = safe_sfread('RMC', shot)
    if RMC is None:
        return None
    
    try:
        flag = True
        shot2 = shot
        while flag:
            RMD = sf.SFREAD('RMD', shot2)
            if RMD.status:
                flag = False
            else:
                shot2 += -1
    except:
        raise Exception('Could not retrieve the ECE calibration')
    
    Temp = np.clip(np.array(CEC('Trad-A'), dtype='f4')/1e3, 0, None)
    t_T = np.array(CEC.gettimebase('Trad-A'), dtype='f4')

    pos_R = np.array(CEC('R-A'), dtype='f4')
    pos_Z = np.array(CEC('z-A'), dtype='f4')
    t_pos = np.array(CEC.gettimebase('R-A'), dtype='f4')

    n_channels = Temp.shape[1] if Temp.ndim > 1 else 60
    channels = np.arange(n_channels)

    obj = xr.Dataset(
        coords={
            't': ('t', t_T, {'long_name': 'Time', 'units': 's'}),
            't2': ('t2', t_pos, {'long_name': 'Time', 'units': 's'}),
            'channel': ('channel', channels, {'long_name': 'Channel'}),
        },
        attrs={'shot': shot, 'diag': 'CEC'},
    )

    obj['Trad'] = (['channel', 't'], Temp.T, 
                   {'signal': 'Trad', 'long_name': r'$T_e$', 'units': 'keV',},)
    obj['R'] = (['channel', 't2'], pos_R.T, 
                   {'signal': 'R', 'long_name': r'$R$', 'units': 'm',},)
    obj['z'] = (['channel', 't2'], pos_Z.T, 
                   {'signal': 'z', 'long_name': r'$z$', 'units': 'm',},)


    rhopol_vals = get_rho(shot=shot, Rin=obj['R'].mean('t2').values, 
                          zin=obj['z'].mean('t2').values, time=obj['t2'].mean().values,)

    obj['rho'] = (['channel'], rhopol_vals.flatten(), 
                  {'long_name': r'$\rho_{pol}$', 'units': 'a.u.',},)
    
    return obj if xArrayOutput else to_dict_with_metadata(obj)

def get_ECE_old(shotnumber: int, timeWindow: Optional[list] = None, 
            fast: bool = True,
            rhopLimits: Optional[float] = None, 
            safetyChecks: bool = True):
    """
    Retrieves from the database the ECE data and the calibrations to obtain
    the electron temperature perturbations.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: shot Number to get the ECE data.
    :param  timeWindow: time window to get the data.
    :param  fast: fast reading implies that the position of each of the
    radiometers is averaged along the time window making it faster.
    :param  rhopLimits: limits in rho poloidal to retrieve the ECE. If None,
    the limits are set to [-1.0, 1.0], where the sign is conventional to
    HFS (-) and LFS (+)
    :param safetyChecks: If True, ony channles were the signal in the slow and
    fast channels 'agree' (their average are close) wil be considered
    """

    # --- Opening the shotfiles.
    exp = 'AUGD'  # Only the AUGD data retrieve is supported with the dd.
    dt = 2*5.0e-3  # To get a bit more on the edges to make a better interp.
    method = 'linear'  # Interpolation method
    try:
        sf_cec = sf.SFREAD('CEC', shotnumber, experiment=exp,  edition=0) # Slow channel - ECE data.
        sf_rmc = sf.SFREAD('RMC', shotnumber, experiment=exp,  edition=0) # Fast channel - ECE data.
    except:
        raise Exception('Shotfile not existent for ECE/RMD/RMC')

    # eqh = sf.SFREAD('EQH', shotnumber, experiment=exp, edition=0)
    equ = sf.EQU(shotnumber, diag='EQH', exp='AUGD', ed=0)

    # Getting the calibration.
    try:
        flag = True
        shot2 = shotnumber
        while flag:
            sf_rmd = sf.SFREAD('RMD', shot2)
            if sf_rmd.status:
                flag = False
            else:
                shot2 += -1
    except:
        raise Exception('Could not retrieve the ECE calibration')

    # --- Time base retrieval.
    try:
        # Getting the fastest time basis.
        time = sf_rmc('TIME-AD0')
    except:
        raise Exception('Time base not available in shotfile!')

    if timeWindow is None:
        timeWindow = [time[0], time[-1]]
    else:
        timeWindow = np.atleast_1d(timeWindow)

    timeWindow[0] = np.maximum(time[0], timeWindow[0])
    timeWindow[1] = np.minimum(time[-1], timeWindow[-1])
    logger.info('Using time window %f, %f'%(timeWindow[0], timeWindow[1]))
    itminTB = np.argmin(np.abs(timeWindow[0] - time))
    itmaxTB = np.argmin(np.abs(timeWindow[1] - time))


    # --- Getting the calibration from RMD
    params = sf_rmd.getparset('eCAL-A1')
    params2 = sf_rmd.getparset('eCAL-A2')
    mult_cal = np.array(params['MULTIA00'])
    mult_cal = np.append(mult_cal, params2['MULTIA00'])

    shift_cal = np.array(params['SHIFTB00'])
    shift_cal = np.append(shift_cal, params2['SHIFTB00'])

    # Getting the availability.
    # This is kinda bugged in the shot file, so fuck it, I will set it to true
    # and in the comprobation if the averaged agrees between slow and fast,
    # we will see if there is or not a broken channel
    avail_cec = sf_cec.getparset('parms-A')['AVAILABL'].astype(bool)
    avail_rmd = sf_rmd.getparset('parms-A')['AVAILABL'].astype(bool)
    avail_rmc = sf_rmc.getparset('parms-A')['AVAILABL'].astype(bool)
    # avail_flag = np.array((avail_cec * avail_rmd * avail_rmc), dtype=bool)
    avail_flag = np.ones(avail_cec.size, bool)
    for i in range(len(avail_cec)):
        logger.debug('Avail data %i: CEC %r, RMD %r, RMC %r, General %r' %
                     (i+1, avail_cec[i], avail_rmd[i], avail_rmc[i], avail_flag[i]))
    del avail_cec
    del avail_rmd
    del avail_rmc

    # Getting the positions (R, Z) -> rho_pol
    rA_rmd = sf_cec('R-A')
    zA_rmd = sf_cec('z-A')

    timebase2 = sf_cec.gettimebase('z-A')
    itmin = np.argmin(np.abs(timebase2 - timeWindow[0]))
    itmax = np.argmin(np.abs(timebase2 - timeWindow[1]))
    if fast:
        if itmin != itmax:
            rA = np.mean(rA_rmd.data[itmin:itmax], axis=0)
            zA = np.mean(zA_rmd.data[itmin:itmax], axis=0)

        else:
            rA = np.mean(rA_rmd.data[itmin:(itmin + 1)], axis=0)
            zA = np.mean(zA_rmd.data[itmin:(itmin + 1)], axis=0)
        
        time_avg = np.mean(timeWindow)
        rhop = sf.rz2rho(equ, rA, zA, t_in=time_avg, coord_out='rho_pol',
                        extrapolate=True).flatten()
        rhot = sf.rz2rho(equ, rA, zA, t_in=time_avg, coord_out='rho_tor',
                        extrapolate=True).flatten()

    else:
        raise Exception('Not supportted')

    # --- Getting the electron temperature from the RMC shotfile.
    Trad_rmc1 = np.array(sf_rmc('Trad-A1'))
    Trad_rmc2 = np.array(sf_rmc('Trad-A2'))

    time32 = sf_rmc.gettimebase('Trad-A1')
    itmin = np.argmin(np.abs(time32 - timeWindow[0]))
    itmax = np.argmin(np.abs(time32 - timeWindow[1]))
    Trad_rmc = np.append(Trad_rmc1[itmin:itmax,:], Trad_rmc2[itmin:itmax, :], axis=1)

    # With the new time-basis, we compute the actual positions of the ECE
    # radiometers in time.
    if fast:
        Rece = rA
        Zece = zA
        RhoP_ece = rhop
        RhoT_ece = rhot
    else:
        pass  # We raised the exception above

    del Trad_rmc1  # Releasing some memory.
    del Trad_rmc2  # Releasing some memory.
    del rA_rmd
    del zA_rmd
    del rhop
    del rhot

    # --- Applying the calibration.
    Trad = np.zeros(Trad_rmc.shape)       # Electron temp [eV]
    Trad_norm = np.zeros(Trad_rmc.shape)  # Norm'd by the maximum.
    for ii in range(Trad_rmc.shape[1]):
        Trad[:, ii] = Trad_rmc[:, ii]*mult_cal[ii] + shift_cal[ii]
        meanTe = np.mean(Trad[:, ii])
        # logger.debug('mean temperature for this chord %f', meanTe)
        Trad_norm[:, ii] = Trad[:, ii]/meanTe

    del Trad_rmc  # Release the uncalibrated signal from memory.

    # --- Checking that the slow channel is consistent with the fast-channels.
    Trad_slow = sf_cec('Trad-A')
    time45 = sf_cec.gettimebase('Trad-A')
    itmin = np.argmin(np.abs(time45 - timeWindow[0]))
    itmax = np.argmin(np.abs(time45 - timeWindow[1]))
    logger.debug('TRDAT shape %i, %i '%Trad_slow.data.shape)
    meanECE = np.mean(np.array(Trad_slow.data)[itmin:itmax, :], axis=0)
    logger.debug('meanECE shape %i'%meanECE.shape)
    maxRMD = np.max(Trad, axis=0)
    minRMD = np.min(Trad, axis=0)
    RmeanRMD = np.mean(Rece.flatten())

    # We erase the channels that have and average in the slow channels that
    # are not within the RMD signals. Also, take away those not within the
    # vessel.
    avail_flag &= (RmeanRMD > 1.03)
    print(avail_flag.sum())
    if safetyChecks:
       avail_flag &= ((meanECE > minRMD) & (meanECE < maxRMD))
    del meanECE
    del maxRMD
    del minRMD

    # --- Applying the rhop limits.
    if rhopLimits is None:
        rhopLimits = [-1.0, 1.0]

    if fast:
        avail_flag &= ((RhoP_ece > rhopLimits[0]) &
                       (RhoP_ece < rhopLimits[1]))
        Rece = Rece[avail_flag]
        Zece = Zece[avail_flag]
        RhoP_ece = RhoP_ece[avail_flag]
        RhoT_ece = RhoT_ece[avail_flag]
        Trad = Trad[:, avail_flag]
        Trad_norm = Trad_norm[:, avail_flag]
    else:
        pass

    # --- Adding a conventional sign to the rhopol and rhotor.
    # Getting the axis center.
    sfo = sf.SFREAD('GQH', shotnumber)
    rmag = sfo('Rmag')
    # zmag = sfo('Zmag')
    timebaseR = sfo.gettimebase('Rmag')
    itmin = np.argmin(np.abs(timebaseR - timeWindow[0]))
    itmax = np.argmin(np.abs(timebaseR - timeWindow[1]))

    if fast:
        Raxis = np.mean(rmag[itmin:itmax])
        sign_array = Rece < Raxis
        RhoP_ece[sign_array] = - RhoP_ece[sign_array]
    else:
        pass

    # --- Saving output.
    output = {
        'time': np.asarray(time[itminTB:itmaxTB], dtype=np.float32),
        'r': np.asarray(Rece, dtype=np.float32),
        'z': np.asarray(Zece, dtype=np.float32),
        'rhop': np.asarray(RhoP_ece, dtype=np.float32),
        'rhot': np.asarray(RhoT_ece, dtype=np.float32),
        'Trad': np.asarray(Trad, dtype=np.float32),
        'Trad_norm': np.asarray(Trad_norm, dtype=np.float32),
        'Raxis': np.asarray(Raxis, dtype=np.float32),
        'fast_rhop': fast,
        'active': avail_flag,
        'channels': (np.arange(len(avail_flag))+1)[avail_flag],
        'shotnumber': shotnumber
    }

    return output

def correctShineThroughECE(ecedata: dict, diag: str = 'IDA', exp: str = 'AUGD',
                           edition: int = 0):
    """
    For a given data-set of the ECE data, a new entry will be provided with
    the temperature divided by the electron temperature gradient, trying to
    correct the shine-through effect.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  ecedata: dictionary with the data from the ECE.
    :param  diag: Diagnostic from which the eletron gradient will be retrieved.
    By default is set to PED in AUGD, because its separatrix value are more
    reliable. If the PED shotfile is not found, then the IDA shotfile will be
    opened.
    @parad exp: experiment from which the electron temperature will be
    extracted. By default, AUGD.
    :param  ed: Edition to retrieve from equilibrium. The latest is taken by
    default.
    """
    # --- Trying to open the PED shotfile.
    using_PED = False
    if diag == 'PED':
        raise Exception('Not implemented')
        # sf = dd.shotfile(diagnostic='PED', experiment=exp,
        #                   pulseNumber=ecedata['shotnumber'],
        #                   edition=edition)

        # name_te = 'TeFit'
        # name_rhop = 'rhoFit'
        # using_PED = True
    else:
        sf = sf.SFREAD('IDA',ecedata['shotnumber'])

        name_te = 'dTe_dr'
        name_rhop = 'rhop'
    timebase = sf.gettimebase(name_te)
    itmin = np.argmin(np.abs(timebase - ecedata['fft']['time'][0]))
    itmax = np.argmin(np.abs(timebase - ecedata['fft']['time'][-1]))
    rhop = sf(name_rhop)

    te_data = sf(name_te)

    if using_PED:

        # dte = np.abs(np.diff(te_data.data.squeeze()))
        # drhop = np.diff(rhop.data.squeeze())
        # dtedrho = dte/drhop
        # rhop_c = (rhop.data.squeeze()[1:] + rhop.data.squeeze()[:-1])/2.0

        # dte_eval = interp1d(rhop_c, dtedrho, kind='linear',
        #                     bounds_error=False, fill_value='extrapolate')\
        #                     (np.abs(ecedata['rhop']))

        # ecedata['fft']['dTe'] = dtedrho
        # ecedata['fft']['dTe_base'] = rhop_c

        # ecedata['fft']['spec_dte'] = ecedata['fft']['spec']
        # for ii in range(ecedata['fft']['spec_dte'].shape[0]):
        #     for jj in range(ecedata['fft']['spec_dte'].shape[1]):
        #         ecedata['fft']['spec_dte'][ii, jj, :] /= dte_eval
        pass

    else:
        time = timebase
        t0 = np.maximum(0,
                        np.abs(time.flatten()
                                - ecedata['fft']['time'][0]).argmin() - 1)
        t1 = np.minimum(len(time)-1,
                        np.abs(time.flatten()
                                - ecedata['fft']['time'][-1]).argmin() + 1)

        t0 = np.asarray(t0, dtype=int)
        t1 = np.asarray(t1, dtype=int)

        # Reducing the size, to make easier the interpolation.

        time = time[t0:t1]
        te_data = np.array(te_data.data)[..., t0:t1]
        rhop = rhop[:, t0]

        dte_eval_fun = interp2d(time, rhop, np.abs(te_data))

        dte_eval = np.abs(dte_eval_fun(ecedata['fft']['time'],
                                        np.abs((ecedata['rhop'])))).T

        ecedata['fft']['dTe_base'] = rhop
        ecedata['fft']['dTe'] = np.abs(np.mean(te_data, axis=1))
        ecedata['fft']['spec_dte'] = ecedata['fft']['spec']
        for ii in range(ecedata['fft']['spec'].shape[1]):
            ecedata['fft']['spec_dte'][:, ii, :] /= dte_eval
    return ecedata

def get_Zeff(shot, plot = False, xArrayOutput: bool = True):
    '''
    Get the Zeff effective profile from IDZ

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  plot: quick plot of the data
    :param  xArrayOutput: get the output in xarray format or dictionary
    '''

    IDZ = safe_sfread('IDZ', shot)
    if IDZ is None:
        return None
    
    t = np.array(IDZ.gettimebase('Zeff'), dtype='f4')

    obj = xr.Dataset(coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}),},
                     attrs={'shot': shot, 'diag': 'IDZ'})

    for signal in ['Zeff', 'Zeff_unc']:
        try:
            data = np.array(IDZ(signal), dtype='f4')
            units = 'q'
            long_name = r'$Z_eff$'
            if 'unc' in signal: long_name = r'$\Delta$'+long_name
            obj[signal] = (['t'], data[:,0], 
                           {'signal': signal, 'long_name': long_name, 
                            'units': units,},
                    )
        except:
            continue 

    if plot:
        fig, ax = plt.subplots()
        ax.fill_between(obj.t, obj.Zeff+obj.Zeff_unc, obj.Zeff-obj.Zeff_unc, 
                           alpha=0.3)
        obj.Zeff.plot(ax=ax)

    return obj if xArrayOutput else to_dict_with_metadata(obj)