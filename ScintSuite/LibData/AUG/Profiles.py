"""Routines to interact with the AUG database."""
import numpy as np
import xarray as xr
import aug_sfutils as sfutils
import ScintSuite.errors as errors
from typing import Optional
from ScintSuite._Paths import Path
from scipy.interpolate import interp1d, interp2d, UnivariateSpline

from ScintSuite.LibData.AUG.Equilibrium import get_rho, get_shot_basics
import logging
logger = logging.getLogger('ScintSuite.Data')
pa = Path()


# -----------------------------------------------------------------------------
# --- Electron density and temperature profiles.
# -----------------------------------------------------------------------------
def get_ne(shotnumber: int, time: float = None, exp: str = 'AUGD',
           diag: str = 'IDA', edition: int = 0, sf=None,
           xArrayOutput: bool = False, flag_avg: bool=False,
           t_avg: float=8.0):
    """
    Wrap the different diagnostics to read the electron density profile.

    It supports IDA and PED profiles.

    Pablo Oyola - pablo.oyola@ipp.mpg.de and J. Rueda: jrrueda@us.es

    :param  shot: Shot number
    :param  time: Time point to read the profile.
    :param  exp: Experiment name.
    :param  diag: diagnostic from which 'ne' will extracted.
    :param  edition: edition of the shotfile to be read.
    :param  xArrayOutput: flag to return the output as dictionary of xarray
    :param flag_avg: flag to average the profiles in time.
    :param t_avg: time window to average the profiles [ms].

    :return output: a dictionary containing the electron density evaluated
        in the input times and the corresponding rhopol base.


    Use example:
        >>> import Lib as ss
        >>> ne = ss.dat.get_ne(41091, 3.55, xArrayOutput=True)
    """
    if diag not in ('IDA', 'PED'):
        raise Exception('Diagnostic non supported!')

    if diag == 'PED':
        return get_ne_ped(shotnumber=shotnumber, time=time, exp=exp,
                          edition=edition, sf=sf, xArrayOutput=xArrayOutput)
    elif diag == 'IDA':
        return get_ne_ida(shotnumber=shotnumber, time=time, exp=exp,
                          edition=edition, sf=sf, xArrayOutput=xArrayOutput,
                          flag_avg=flag_avg, t_avg=t_avg)


def get_ne_ped(shotnumber: int, time: float = None, exp: str = 'AUGD',
               edition: int = 0, sf=None, xArrayOutput: bool = False):
    """
    Reads from the PED shotfile the electron density profile.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: shot number to get the data.
    :param  time: time window to retrieve the toroidal rotation data.
    :param  exp: experiment name where the shotfile is stored.
    :param  edition: edition of the shotfile.
    :param  sf: shotfile opened for the PED.
    """

    # --- Opening the shotfile.
    if sf is None:
        sf = sfutils.SFREAD(shotnumber, 'PED', edition=edition, exp=exp)

        if not sf.status:
            raise Exception('Cannot open PED shotfile for #%05d' % shotnumber)

    # --- Trying to read the toroidal rotation.
    try:
        ne = np.array(sf('neFit'))
        ne_unc = np.array(sf('dneFit'))
        rhop = np.array(sf('rhoFit'))
        timebase = np.array(sf('time'))
    except:
        raise Exception('Cannot read ne in #05d' % shotnumber)

    if time is not None:
        time = np.atleast_1d(time)
        if (timebase > time.max()) or (timebase < time.min()):
            raise Exception('Time window cannot be located in PED shotfile!')

    if not xArrayOutput:
        output = {
            'rhop': rhop,
            'data': ne,
            'uncertainty': ne_unc,
            'time': timebase,
                 }
    else:
        output = xr.Dataset()
        output['data'] = xr.DataArray(
            ne.T, dims=('rho', 't'),
            coords={'rho': rhop, 't': timebase})
        output['data'].attrs['long_name'] = '$n_e$'
        output['data'].attrs['units'] = '$10^{19} m^3$'
        output['uncertainty'] = xr.DataArray(ne_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta n_e$'
        output['uncertainty'].attrs['units'] = '$10^{19} m^3$'

        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'PED'
        output.attrs['shot'] = shotnumber
    return output


def get_ne_ida(shotnumber: int, time: float = None, exp: str = 'AUGD',
               edition: int = 0, sf=None, xArrayOutput: bool = False,
               flag_avg: bool=False, t_avg: float=8.0):
    """
    Wrap to get AUG electron density using the IDA profiles.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: Shot number
    :param  time: Time point to read the profile.
    :param  exp: Experiment name.
    :param  edition: edition of the shotfile to be read.
    :param  sf: shotfile opened with the IDA to be accessed.
    :param  xArrayOutput: flag to return the output as dictionary of xarray
    :param flag_avg: flag to average the profiles in time.
    :param t_avg: time window to average the profiles [ms].

    :return output: a dictionary containing the electron density evaluated
    in the input times and the corresponding rhopol base.
    """
    # --- Opening the shotfile.
    if sf is None:
        sf = sfutils.SFREAD(shotnumber, 'IDA', edition=edition, exp=exp)

        if not sf.status:
            raise Exception('Cannot open IDA shotfile for #%05d' % shotnumber)

    # --- Reading from the database
    try:
        ne = np.array(sf('ne')).T
        ne_unc = np.array(sf('ne_unc')).T
        rhop = np.array(sf.getareabase('ne'))
        timebase = np.array(sf('time'))
    except:
        raise Exception('Cannot read the density from the IDA #%05d'%shotnumber)

    # We will return the data in the same spatial basis as provided by IDA.
    if time is None:
        time = timebase
        tmp_ne = ne
        tmp_unc = ne_unc
    else:
        if flag_avg:
            logger.info('averaging profiles over %.3d ms' %t_avg)
            if len(time) != 1:
                logger.info('can only average arounf one time. First value will be used.')
            t_avg *= 1e-3
            t_idx0 = np.argmin(abs(timebase - (time[0] - t_avg/2)))
            t_idx1 = np.argmin(abs(timebase - (time[0] + t_avg/2)))
            tmp_ne = np.mean([ne[i,:] for i in np.arange(t_idx0, t_idx1)], axis=0)
            tmp_unc = np.mean([ne_unc[i,:] for i in np.arange(t_idx0, t_idx1)], axis=0)
        else:
            tmp_ne = interp1d(timebase, ne, kind='linear', axis=0,
                              bounds_error=False, fill_value=np.nan,
                              assume_sorted=True)(time).T
            tmp_unc = interp1d(timebase, ne_unc,
                               kind='linear', axis=0,
                               bounds_error=False,
                               fill_value=np.nan,
                               assume_sorted=True)(time).T


    if not xArrayOutput:
        output = {
            'rhop': rhop[:, 0],
            'time': time,
            'uncertainty': tmp_unc,
            'data': tmp_ne
        }
    else:
        tmp_ne = np.atleast_2d(tmp_ne)
        tmp_unc = np.atleast_2d(tmp_unc)
        time = np.atleast_1d(time)

        output = xr.Dataset()
        output['data'] = xr.DataArray(
            tmp_ne.T/1.0e19, dims=('rho', 't'),
            coords={'rho': rhop[:, 0], 't': time})
        output['data'].attrs['long_name'] = '$n_e$'
        output['data'].attrs['units'] = '$10^{19} m^3$'
        output['uncertainty'] = xr.DataArray(tmp_unc.T/1.0e19, dims=('rho',
                                                                     't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta n_e$'
        output['uncertainty'].attrs['units'] = '$10^{19} m^3$'

        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'IDA'
        output.attrs['shot'] = shotnumber
    return output


def get_Te(shotnumber: int, time: float = None, exp: str = 'AUGD',
           diag: str = 'IDA', edition: int = 0, sf=None,
           xArrayOutput: bool = False, flag_avg: bool=False, t_avg: float=8.0):

    """
    Wrapper to the different diagnostics to read the electron density profile.
    It supports IDA and PED profiles.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: Shot number
    :param  time: Time point to read the profile.
    :param  exp: Experiment name.
    :param  diag: diagnostic from which 'ne' will extracted.
    :param  edition: edition of the shotfile to be read.
    :param  sf: shotfile object opened with the shotfile to be accessed.
    :param  xArrayOutput: flag to return the output as a xarray
    :param  flag_avg: flag to average the profiles in time.
    :param  t_avg: time window to average the profiles [ms].

    :return output: a dictionary containing the electron density evaluated
    in the input times and the corresponding rhopol base.
    """

    if diag not in ('IDA', 'PED'):
        raise Exception('Diagnostic non supported!')

    if diag == 'PED':
        return get_Te_ped(shotnumber=shotnumber, time=time, exp=exp,
                          edition=edition, sf=sf, xArrayOutput=xArrayOutput)
    elif diag == 'IDA':
        return get_Te_ida(shotnumber=shotnumber, time=time, exp=exp,
                          edition=edition, sf=sf, xArrayOutput=xArrayOutput,
                          flag_avg=flag_avg, t_avg=t_avg)


def get_Te_ped(shotnumber: int, time: float = None, exp: str = 'AUGD',
               edition: int = 0, sf=None, xArrayOutput: bool = False):
    """
    Reads from the PED shotfile the electron density profile.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: shot number to get the data.
    :param  time: time window to retrieve the toroidal rotation data.
    :param  exp: experiment name where the shotfile is stored.
    :param  edition: edition of the shotfile.
    :param  sf: shotfile opened for the PED.
    """

    # --- Opening the shotfile.
    if sf is None:
        sf = sfutils.SFREAD(shotnumber, 'PED', edition=edition, exp=exp)

        if not sf.status:
            raise Exception('Cannot open PED shotfile for #%05d' % shotnumber)
    # --- Trying to read the timebasis.
    try:
        timebasis = sf(name='time')
    except:
        raise Exception('Cannot read the timebasis for vT in #%05d'%shotnumber)

    # --- Trying to read the toroidal rotation.
    try:
        te = np.array(sf('TeFit'))
        te_unc = np.array(sf('dTeFit'))
        rhop = np.array(sf('rhoFit'))
    except:
        sf.close()
        raise Exception('Cannot read Te in #%05d' % shotnumber)
    if not xArrayOutput:
        output = {
            'rhop': rhop,
            'data': te,
            'uncertainty': te_unc,
            'time': timebasis
                 }
    else:
        output = xr.Dataset()
        output['data'] = xr.DataArray(
            te.T, dims=('rho', 't'),
            coords={'rho': rhop[:, 0], 't': timebasis})
        output['data'].attrs['long_name'] = '$T_e$'
        output['data'].attrs['units'] = 'eV'
        output['uncertainty'] = xr.DataArray(te_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta T_e$'
        output['uncertainty'].attrs['units'] = '$eV$'

        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'PED'
        output.attrs['shot'] = shotnumber

    return output


def get_Te_ida(shotnumber: int, time: float = None, exp: str = 'AUGD',
               edition: int = 0, sf=None, xArrayOutput: bool = False,
               flag_avg: bool=False, t_avg: float=8.0):
    """
    Wrap to get AUG electron temperature from the IDA shotfile.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: Shot number
    :param  time: Time point to read the profile.
    :param  exp: Experiment name.
    :param  diag: diagnostic from which 'Te' will extracted.
    :param  edition: edition of the shotfile to be read.
    :param  sf: shotfile object opened with the shotfile to be accessed.
    :param  xArrayOutput: flag to return the output as a xarray
    :param  flag_avg: flag to average the profiles in time.
    :param  t_avg: time window to average the profiles [ms].

    :return output: a dictionary containing the electron temp. evaluated
    in the input times and the corresponding rhopol base.
    """

    # --- Opening the shotfile.
    if sf is None:
        sf = sfutils.SFREAD(shotnumber, 'IDA', edition=edition, exp=exp)

        if not sf.status:
            raise Exception('Cannot open IDA shotfile for #%05d' % shotnumber)

    # Reading from the database
    try:
        te = np.array(sf('Te')).T
        te_unc = np.array(sf('Te_unc')).T
        rhop = np.array(sf.getareabase('Te'))
        timebase = np.array(sf('time'))

    except:
        raise Exception('Cannot read the temperature from the IDA #%05d'%shotnumber)


    # We will return the data in the same spatial basis as provided by IDA.
    if time is None:
        time = timebase
        tmp_te = te
        tmp_unc = te_unc
    else:
        if flag_avg:
            t_avg *= 1e-3 # [ms] -> [s]
            t_idx0 = np.argmin(abs(timebase - (time[0] - t_avg/2)))
            t_idx1 = np.argmin(abs(timebase - (time[0] + t_avg/2)))
            tmp_te = np.mean([te[i,:] for i in np.arange(t_idx0, t_idx1)], axis=0)
            tmp_unc = np.mean([te_unc[i,:] for i in np.arange(t_idx0, t_idx1)], axis=0)
        else:
            tmp_te = interp1d(timebase, te, kind='linear', axis=0,
                              bounds_error=False, fill_value=np.nan,
                              assume_sorted=True)(time).T
            tmp_unc = interp1d(timebase, te_unc,
                               kind='linear', axis=0,
                               bounds_error=False, fill_value=np.nan,
                               assume_sorted=True)(time).T

    if not xArrayOutput:
        output = {
            'rhop': rhop[:, 0],
            'time': time,
            'data': tmp_te,
            'uncertainty': tmp_unc}
    else:
        output = xr.Dataset()
        tmp_te = np.atleast_2d(tmp_te)
        time = np.atleast_1d(time)
        output['data'] = xr.DataArray(
            tmp_te.T, dims=('rho', 't'),
            coords={'rho': rhop[:, 0], 't': time})
        output['data'].attrs['long_name'] = '$T_e$'
        output['data'].attrs['units'] = 'eV'
        tmp_unc = np.atleast_2d(tmp_unc)
        output['uncertainty'] = xr.DataArray(tmp_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta T_e$'
        output['uncertainty'].attrs['units'] = '$eV$'

        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'IDA'
        output.attrs['shot'] = shotnumber

    return output


# -----------------------------------------------------------------------------
# --- Ion temperature
# -----------------------------------------------------------------------------
def get_Ti(shot: int, time: float = None, diag: str = 'IDI', exp: str = 'AUGD',
           edition: int = 0, sf=None, xArrayOutput: bool = False,
           flag_avg: bool=False, t_avg: float=8.0):
    """"
    Wrapper to all the routines to read the ion temperature.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: Shot number
    :param  time: Time point to read the profile.
    :param  diag: diagnostic to read the ion temperature. By default, IDI.
    :param  exp: Experiment name.
    :param  diag: diagnostic from which 'Te' will extracted.
    :param  edition: edition of the shotfile to be read.
    :param  sf: shotfile object opened with the shotfile to be accessed.
    :param  xArrayOutput: flag to return the output as a xarray
    :param  flag_avg: flag to average the profiles in time.
    :param  t_avg: time window to average the profiles [ms].

    :return output: a dictionary containing the electron temp. evaluated
    in the input times and the corresponding rhopol base.
    """
    if diag not in ('IDI', 'CXRS'):
        raise Exception('Diagnostic non supported!')

    if diag == 'IDI':
        return get_Ti_idi(shotnumber=shot, time=time, exp=exp,
                          edition=edition, sf=sf, xArrayOutput=xArrayOutput,
                          flag_avg=flag_avg, t_avg=t_avg)
    elif diag == 'CXRS':
        return get_Ti_cxrs(shotnumber=shot, time=time, exp=exp,
                           edition=edition, xArrayOutput=xArrayOutput)
    else:
        raise Exception('Diagnostic non supported!')


def get_Ti_idi(shotnumber: int, time: float = None, exp: str = 'AUGD',
               edition: int = 0, sf=None, xArrayOutput: bool = False,
               flag_avg: bool=False, t_avg: float=8.0):
    """
    Wrap to get AUG ion temperature from the IDI shotfile.

    Jose Rueda: jrrueda@us.es

    Copy of get_Te_ida

    :param  shot: Shot number
    :param  time: Time point to read the profile.
    :param  exp: Experiment name.
    :param  diag: diagnostic from which 'Te' will extracted.
    :param  edition: edition of the shotfile to be read.
    :param  sf: shotfile object opened with the shotfile to be accessed.
    :param  xArrayOutput: flag to return the output as a xarray
    :param  flag_avg: flag to average the profiles in time.
    :param  t_avg: time window to average the profiles [ms].

    :return output: a dictionary containing the electron temp. evaluated
    in the input times and the corresponding rhopol base.
    """
    # --- Opening the shotfile.
    if sf is None:
        sf = sfutils.SFREAD(shotnumber, 'IDI', edition=edition, exp=exp)

        if not sf.status:
            raise Exception('Cannot open IDI shotfile for #%05d' % shotnumber)
    # Reading from the database
    try:
        ti = np.array(sf('Ti'))
        ti_unc = np.array(sf('Ti_unc'))
        rhop = np.array(sf.getareabase('Ti'))
        timebase = np.array(sf('time'))
    except:
        raise Exception('Cannot read the density from the IDA #%05d' % shotnumber)

    # We will return the data in the same spatial basis as provided by IDI.

    if time is None:
        time = timebase
        tmp_ti = ti
        tmp_unc = ti_unc
    else:
        if flag_avg:
            t_avg *= 1e-3
            t_idx0 = np.argmin(abs(timebase - (time[0] - t_avg/2)))
            t_idx1 = np.argmin(abs(timebase - (time[0] + t_avg/2)))
            tmp_ti = np.mean([ti[i,:] for i in np.arange(t_idx0, t_idx1)], axis=0)
            tmp_unc = np.mean([ti_unc[i,:] for i in np.arange(t_idx0, t_idx1)], axis=0)
        else:
            tmp_ti = interp1d(timebase, ti, kind='linear', axis=0,
                              bounds_error=False, fill_value=np.nan,
                              assume_sorted=True)(time).T
            tmp_unc = interp1d(timebase, ti_unc,
                               kind='linear', axis=0,
                               bounds_error=False, fill_value=np.nan,
                               assume_sorted=True)(time).T

    if not xArrayOutput:
        output = {
            'rhop': rhop[:, 0],
            'time': time,
            'data': tmp_ti,
            'uncertainty': tmp_unc
        }
    else:
        output = xr.Dataset()
        tmp_ti = np.atleast_2d(tmp_ti)
        tmp_unc = np.atleast_2d(tmp_unc)
        time = np.atleast_1d(time)
        output['data'] = xr.DataArray(
            tmp_ti.T, dims=('rho', 't'),
            coords={'rho': rhop[:, 0], 't': time})
        output['data'].attrs['long_name'] = '$T_i$'
        output['data'].attrs['units'] = 'eV'
        output['uncertainty'] = xr.DataArray(tmp_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta T_i$'
        output['uncertainty'].attrs['units'] = '$eV$'

        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'IDI'
        output.attrs['shot'] = shotnumber
    return output


def get_Ti_cxrs(shotnumber: int, time: float = None,
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
    sf = list()
    for ii in diags:
        sf_aux = sfutils.SFREAD(shotnumber, ii, exp=exp, edition=edition)
        if not sf_aux.status:
            print('Cannot open %s for shot #%05d\n' % (ii, shotnumber))
            continue

        nshotfiles += 1
        sf.append(sf_aux)

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
    for ii in np.arange(len(sf), dtype=int):
        Ti_data = np.array(sf[ii](signals[ii])).squeeze()
        zaux = np.array(sf[ii](name='z')).squeeze()
        Raux = np.array(sf[ii](name='R')).squeeze()
        err_aux = np.array(sf[ii](name=error_signals[ii])).squeeze()
        time_aux = np.array(sf[ii].gettimebase(signals[ii]))

        if Ti_data.size == 0:
            nshotfiles -= 1
            continue

        if len(sf) == 0:
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

            if len(sf) == 0:
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

            if len(sf) == 0:
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
# --- Toroidal rotation velocity
# -----------------------------------------------------------------------------
def get_tor_rotation(shotnumber: int, time: float = None, diag: str = 'IDI',
                     exp: str = 'AUGD', edition: int=0, **kwargs):
    """
    Retrieves from the database the toroidal velocity velocity (omega_tor).
    To get the linear velocity (i.e., vtor) multiply by the major radius.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: shotnumber to read.
    :param  time: time window to get the data. If None, all the available times
    are read.
    :param  diag: the diagnostic can only be 'IDI' or 'CXRS'. In the first, the
    profiles are obtained directly from the IDI reconstruction. For the option
    CXRS the diagnostics 'CEZ'/'CMZ' are used.
    :param  exp: experiment under which the shotfile is stored.
    :param  edition: edition of the shotfile to read
    :param  cxrs_options: extra parameters to send to the fitting procedure
    that reads all the rotation velocities.
    """

    if diag == 'IDI':
        return get_tor_rotation_idi(shotnumber, time, exp, edition)
    elif diag == 'CXRS':
        return get_tor_rotation_cxrs(shotnumber, time, exp, edition, **kwargs)
    elif diag == 'PED':
        return get_tor_rotation_ped(shotnumber, time, exp, edition)
    else:
        raise NameError('Diagnostic not available for the toroidal rotation')


def get_tor_rotation_idi(shotnumber: int, time: float = None,
                         exp: str = 'AUGD', edition: int = 0, sf=None,
                         xArrayOutput: bool = False):

    """
    Reads from the IDI shotfile the toroidal rotation velocity.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: shotnumber to read.
    :param  time: time window to get the data. If None, all the available times
    are read.
    :param  exp: experiment under which the shotfile is stored.
    :param  edition: edition of the shotfile to read
    """

    # --- Opening the shotfile.
    if sf is None:
        sf = sfutils.SFREAD(shotnumber, 'IDI', edition=edition, exp=exp)

        if not sf.status:
            raise errors.DatabaseError('Cannot open IDI shotfile for #%05d' %
                                       shotnumber)

    # --- Getting the data
    data = np.array(sf('vt'))
    unc = np.array(sf('vt_unc'))
    timebase = np.array(sf.gettimebase('vt'))

    # --- If a time window is provided, we cut out the data.
    if time is not None:
        t0, t1 = timebase.searchsorted(time)
        data = data[t0:t1, :]
        unc = unc[t0:t1, :]
        timebase = timebase[t0:t1]

    # --- Saving to a dictionary and output:
    if not xArrayOutput:
        output = {
            'data': data,
            'time': timebase,
            'rhop': np.array(sf.getareabase('vt')[t0, ...])
        }
    else:
        output = xr.Dataset()
        time = np.atleast_1d(timebase)
        output['data'] = xr.DataArray(
                data.T, dims=('rho', 't'),
                coords={'rho': np.array(sf.getareabase('vt')[..., 0]),
                        't': timebase})
        output['uncertainty'] = xr.DataArray(unc.T, dims=('rho', 't'))
        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output['data'].attrs['long_name'] = '$\\omega$'
        output['data'].attrs['units'] = 'rad/s'

        output['uncertainty'].attrs['long_name'] = '$\\sigma\\omega$'
        output['uncertainty'].attrs['units'] = 'rad/s'
        output.attrs['diag'] = 'IDI'
        output.attrs['shot'] = shotnumber

    return output


def get_tor_rotation_cxrs(shotnumber: int, time: float = None,
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
    sf = list()
    for ii in diags:
        sf_aux = sfutils.SFREAD(shotnumber, ii, exp=exp, edition=edition)
        if not sf_aux.status:
            print('Cannot open %s for shot #%05d\n' % (ii, shotnumber))
            continue

        nshotfiles += 1
        sf.append(sf_aux)

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
    for ii in range(len(sf)):
        vtor_data = np.array(sf[ii](signals[ii])).squeeze()
        zaux = np.array(sf[ii](name='z')).squeeze()
        Raux = np.array(sf[ii](name='R')).squeeze()
        err_aux = np.array(sf[ii](name=error_signals[ii])).squeeze()
        time_aux = np.array(sf[ii].gettimebase(signals[ii]))

        if vtor_data.size == 0:
            nshotfiles -= 1
            continue

        if len(sf) == 0:
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
            if len(sf) == 0:
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
            if len(sf) == 0:
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


def get_tor_rotation_ped(shotnumber: int, time: float = None,
                         exp: str = 'AUGD', edition: int = 0, sf=None):
    """
    Reads from the shotfile created with augped the toroidal rotation.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shotnumber: shot number to get the data.
    :param  time: time window to retrieve the toroidal rotation data.
    :param  exp: experiment name where the shotfile is stored.
    :param  edition: edition of the shotfile.
    """

    # --- Opening the shotfile.
    if sf is None:
        sf = sfutils.SFREAD(shotnumber, 'PED', edition=edition, exp=exp)

        if not sf.status:
            raise Exception('Cannot open PED shotfile for #%05d' % shotnumber)


    # --- Trying to read the timebasis.
    try:
        timebasis = np.array(sf('time'))
    except:
        raise Exception('Cannot read the timebasis for vT in #05d'% shotnumber)

    # --- Trying to read the toroidal rotation.
    try:
        vT_sig = sf('vTFit')
        vT = np.array(vT_sig)
        rhop = np.array(sf.getareabase('vTFit'))
    except:
        sf.close()
        raise Exception('Cannot read vT in #05d' % shotnumber)

    output = {
        'rhop': rhop,
        'data': vT,
        'time': timebasis
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
    equ = sfutils.EQU(shotnumber, diag=equ_diag['diag'],
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
# %% ECE data.
# -----------------------------------------------------------------------------
def get_ECE(shotnumber: int, timeWindow: Optional[list] = None, 
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
        sf_cec = sfutils.SFREAD('CEC', shotnumber, experiment=exp,  edition=0) # Slow channel - ECE data.
        sf_rmc = sfutils.SFREAD('RMC', shotnumber, experiment=exp,  edition=0) # Fast channel - ECE data.
    except:
        raise Exception('Shotfile not existent for ECE/RMD/RMC')

    # eqh = sfutils.SFREAD('EQH', shotnumber, experiment=exp, edition=0)
    equ = sfutils.EQU(shotnumber, diag='EQH', exp='AUGD', ed=0)

    # Getting the calibration.
    try:
        flag = True
        shot2 = shotnumber
        while flag:
            sf_rmd = sfutils.SFREAD('RMD', shot2)
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
        rhop = sfutils.rz2rho(equ, rA, zA, t_in=time_avg, coord_out='rho_pol',
                        extrapolate=True).flatten()
        rhot = sfutils.rz2rho(equ, rA, zA, t_in=time_avg, coord_out='rho_tor',
                        extrapolate=True).flatten()

    else:

        # # As the R and z are not calculated for all the points that ECE is
        # # measuring, we build here an interpolator.
        # rA_interp = interp1d(rA_rmd.time, rA_rmd.data, axis=0,
        #                       fill_value=np.nan, kind=method,
        #                       assume_sorted=True, bounds_error=False)
        # zA_interp = interp1d(zA_rmd.time, zA_rmd.data, axis=0,
        #                       fill_value=np.nan, kind=method,
        #                       assume_sorted=True, bounds_error=False)

        # # Getting the rho_pol equivalent to the positions.

        # rhop = np.zeros(rA_rmd.data.shape)
        # rhot = np.zeros(rA_rmd.data.shape)
        # for ii in tqdm(np.arange(rA_rmd.data.shape[0])):
        #     rhop[ii, :] = equ.rz2rho(rA_rmd.data[ii, :], zA_rmd.data[ii, :],
        #                               t_in=rA_rmd.time[ii],
        #                               coord_out='rho_pol',
        #                               extrapolate=True)
        #     rhot[ii, :] = equ.rz2rho(rA_rmd.data[ii, :], zA_rmd.data[ii, :],
        #                               t_in=rA_rmd.time[ii],
        #                               coord_out='rho_tor',
        #                               extrapolate=True)

        # # Creating the interpolators.
        # rhop_interp = interp1d(rA_rmd.time, rhop, axis=0,
        #                         fill_value=np.nan, kind=method,
        #                         assume_sorted=True, bounds_error=False)
        # rhot_interp = interp1d(rA_rmd.time, rhot, axis=0,
        #                         fill_value=np.nan, kind=method,
        #                         assume_sorted=True, bounds_error=False)
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
        # Rece = rA_interp(time)
        # Zece = zA_interp(time)
        # RhoP_ece = rhop_interp(time)
        # RhoT_ece = rhot_interp(time)
        # del rA_interp
        # del zA_interp
        # del rhop_interp
        # del rhot_interp
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
        # flags = ((RhoP_ece > rhopLimits[0]) & (RhoP_ece < rhopLimits[1]))
        # avail_flag &= np.all(flags, axis=0)  # If the channel is continuously
        # #                                    # outside, then discard it.
        # Rece = Rece[:, avail_flag]
        # Zece = Zece[:, avail_flag]
        # RhoP_ece = RhoP_ece[:, avail_flag]
        # RhoT_ece = RhoT_ece[:, avail_flag]
        # Trad = Trad[:, avail_flag]
        # Trad_norm = Trad_norm[:, avail_flag]

        # # Now, set to NaN the values that are outside the limits.
        # flags = not flags[:, avail_flag]
        # Rece[flags] = np.nan
        # Zece[flags] = np.nan
        # RhoP_ece[flags] = np.nan
        # RhoT_ece[flags] = np.nan
        # Trad[flags] = np.nan
        # Trad_norm[flags] = np.nan

    # --- Adding a conventional sign to the rhopol and rhotor.
    # Getting the axis center.
    sfo = sfutils.SFREAD('GQH', shotnumber)
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
        # eqh_time = eqh_time[t0:t1]
        # Raxis = interp1d(eqh_time, Raxis_eqh,
        #                   fill_value='extrapolate', kind=method,
        #                   assume_sorted=True, bounds_error=False)(time)

        # # If R_ECE < R(magnetic axis), then the rhopol is negative.
        # for ii in range(Trad.shape[1]):
        #     sign_array = Rece[:, ii] < Raxis
        #     RhoP_ece[sign_array, ii] = - RhoP_ece[sign_array, ii]
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
        sf = sfutils.SFREAD('IDA',ecedata['shotnumber'])

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


# -----------------------------------------------------------------------------
# %% ECE data.
# -----------------------------------------------------------------------------
def get_Zeff(shot: int):
    """
    Get the Zeff effective profile from IDZ
    """
    IDZ = sfutils.SFREAD('IDZ', shot)
    if not IDZ.status:
        raise errors.DatabaseError('Not shotfile')
    Zeff = IDZ ('Zeff')
    t = IDZ.gettimebase('Zeff')
    rho = IDZ.getareabase('Zeff')
    unc = IDZ('Zeff_unc')
    z = xr.Dataset()
    z['data'] = xr.DataArray(Zeff, dims=('t', 'rho'), 
                             coords={'t':t, 'rho':rho[:, 0]})
    z['uncertainty'] = xr.DataArray(unc, dims=('t', 'rho'), 
                             coords={'t':t, 'rho':rho[:, 0]})
    return z