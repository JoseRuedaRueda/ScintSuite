"""
Routines to interact with the TCV database implemented for ScintSuite

Wrapper routines to acces kinetic profile data from the CONF node via the tcvpy package
MDSplus needed as well as the tcvpy package to call these functions

Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch
"""
import sys
sys.path.append('/home/jansen/NoTivoli/ascot-tcv/')
import tcvpy.results.conf as conf
import numpy as np


import numpy as np
import xarray as xr

import ScintSuite.errors as errors
from typing import Optional
from ScintSuite._Paths import Path
from scipy.interpolate import interp1d, interp2d, UnivariateSpline


import logging
logger = logging.getLogger('ScintSuite.Data')
pa = Path()


# -----------------------------------------------------------------------------
# --- Electron density and temperature profiles.
# -----------------------------------------------------------------------------
def get_ne(shotnumber: int, time: float = None,
           diag: str = 'CONF',
           xArrayOutput: bool = False):
    """
    Wrap the different diagnostics to read the electron density profile.

    It supports CONF and PROFFIT and Thomson profiles.

    Anton Jansen van Vuuren - anton.jansenvanvuuren

    :param  shot: Shot number
    :param  time: Time point to read the profile.

    :param  diag: diagnostic from which 'ne' will extracted.

    :param  xArrayOutput: flag to return the output as dictionary of xarray

    :return output: a dictionary containing the electron density evaluated
        in the input times and the corresponding rhopol base.


    Use example:
        >>> import Lib as ss
        >>> ne = ss.dat.get_ne(41091, 3.55, xArrayOutput=True)
    """
    if diag not in ('CONF', 'PROFFIT', 'THOMSON'):
        raise Exception('Diagnostic non supported!')

    if diag == 'CONF':
        return get_ne_conf(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'PROFFIT':
        pass
        #TODO
        #return get_ne_proffit(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'THOMSON':
        pass
        #TODO
        #return get_ne_thomson(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)    

def get_ne_conf(shotnumber: int, time: float = None, 
           xArrayOutput: bool = False):
    """
    Read the electron density profile from the TCV MODS conf node.
    see https://spcwiki.epfl.ch/wiki/Chie_TCV_nodes#Full_CONF_mds_nodes_list
    :param  shot: Shot number
    :param  time: Time point to read the profile.

    :param  xArrayOutput: flag to return the output as dictionary of xarray

    :return output: a dictionary containing the electron density evaluated
        in the input times and the corresponding rhopol base.


    Use example:
        >>> import Lib as ss
        >>> ne = ss.dat.get_ne(41091, 3.55, xArrayOutput=True)
    """
    # --- Reading from the database
    try:
        ne_conf_data = conf.ne(shotnumber,  trial_indx=1)
        ne = ne_conf_data.data.T
        ne_unc = np.zeros(np.shape(ne)) * np.nan
        rho = ne_conf_data.rho.data
        timebase = ne_conf_data.t.data
    except:
        raise Exception('Cannot read the density from the IDA #%05d'%shotnumber)

    # We will return the data in the same spatial basis as provided by IDA.
    if time is None:
        time = timebase
        tmp_ne = ne
        tmp_unc = np.zeros(np.shape(ne)) * np.nan  #conf node doesn't inlcude uncertainty
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
            'rho': rho,
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
            coords={'rho': rho, 't': time})
        output['data'].attrs['long_name'] = '$n_e$'
        output['data'].attrs['units'] = '$10^{19} m^3$'
        output['uncertainty'] = xr.DataArray(tmp_unc.T/1.0e19, dims=('rho',
                                                                     't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta n_e$'
        output['uncertainty'].attrs['units'] = '$10^{19} m^3$'

        output['rho'].attrs['long_name'] = ne_conf_data.dim_labels[0] #'$\\rho_{TCV}$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'CONF'
        output.attrs['shot'] = shotnumber
    return output

def get_Te(shotnumber: int, time: float = None,
           diag: str = 'CONF',
           xArrayOutput: bool = False):
    """
    Wrap the different diagnostics to read the electron temperature profile.

    It supports CONF and PROFFIT and Thomson profiles.

    Anton Jansen van Vuuren - anton.jansenvanvuuren

    :param  shot: Shot number
    :param  time: Time point to read the profile.

    :param  diag: diagnostic from which 'Te' will extracted.

    :param  xArrayOutput: flag to return the output as dictionary of xarray

    :return output: a dictionary containing the electron temperature evaluated
        in the input times and the corresponding rhopol base.


    Use example:
        >>> import Lib as ss
        >>> ne = ss.dat.get_Te(41091, 3.55, xArrayOutput=True)
    """
    if diag not in ('CONF', 'PROFFIT', 'THOMSON'):
        raise Exception('Diagnostic non supported!')

    if diag == 'CONF':
        return get_Te_conf(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'PROFFIT':
        pass
        #TODO
        #return get_Te_proffit(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'THOMSON':
        pass
        #TODO
        #return get_Te_thomson(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)    

def get_Te_conf(shotnumber: int, time: float = None, 
           xArrayOutput: bool = False):
    """
    Read the electron temperature profile from the TCV MODS conf node.
    see https://spcwiki.epfl.ch/wiki/Chie_TCV_nodes#Full_CONF_mds_nodes_list
    :param  shot: Shot number
    :param  time: Time point to read the profile.

    :param  xArrayOutput: flag to return the output as dictionary of xarray

    :return output: a dictionary containing the electron density evaluated
        in the input times and the corresponding rhopol base.


    Use example:
        >>> import Lib as ss
        >>> ne = ss.dat.get_Te(41091, 3.55, xArrayOutput=True)
    """
    # --- Reading from the database
    try:
        te_conf_data = conf.te(shotnumber,  trial_indx=1)
        te = te_conf_data.data.T
        te_unc = np.zeros(np.shape(te)) * np.nan
        rho = te_conf_data.rho.data
        timebase = te_conf_data.t.data
    except:
        raise Exception('Cannot read the electron temperature from the conf nodes for shot: #%05d'%shotnumber)

    # We will return the data in the same spatial basis as provided by IDA.
    if time is None:
        time = timebase
        tmp_te = te
        tmp_unc = te_unc
        #tmp_dTe = gradTe
    else:
        tmp_te = interp1d(timebase, te, kind='linear', axis=0,
                          bounds_error=False, fill_value=np.nan,
                          assume_sorted=True)(time).T
        tmp_unc = interp1d(timebase, te_unc,
                           kind='linear', axis=0,
                           bounds_error=False, fill_value=np.nan,
                           assume_sorted=True)(time).T
        #tmp_dTe = interp1d(timebase, gradTe,
        #                   kind='linear', axis=0,
        #                   bounds_error=False, fill_value=np.nan,
        #                   assume_sorted=True)(time).T
    if not xArrayOutput:
        output = {
            'rho': rho,
            'time': time,
            'data': tmp_te,
            'uncertainty': tmp_unc}#,
            #'gradient': tmp_dTe}
    else:
        output = xr.Dataset()
        tmp_te = np.atleast_2d(tmp_te)
        time = np.atleast_1d(time)
        output['data'] = xr.DataArray(
            tmp_te.T, dims=('rho', 't'),
            coords={'rho': rho, 't': time})
        output['data'].attrs['long_name'] = '$T_e$'
        output['data'].attrs['units'] = 'eV'
        tmp_unc = np.atleast_2d(tmp_unc)
        output['uncertainty'] = xr.DataArray(tmp_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta T_e$'
        output['uncertainty'].attrs['units'] = '$eV$'
        #tmp_dTe = np.atleast_2d(tmp_dTe)
        #output['gradient'] = xr.DataArray(tmp_dTe.T, dims=('rho', 't'))
        # output['gradient'].attrs['long_name'] = '$\\Nabla T_e$'
        # output['gradient'].attrs['units'] = '$eV$'

        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'CONF'
        output.attrs['shot'] = shotnumber

    return output


# -----------------------------------------------------------------------------
# --- Ion temperature
# -----------------------------------------------------------------------------
def get_Ti(shotnumber: int, time: float = None,
           diag: str = 'CONF',
           xArrayOutput: bool = False):
    """
    Wrap the different diagnostics to read the ion temperature profile.

    It supports CONF and PROFFIT and Thomson profiles.

    Anton Jansen van Vuuren - anton.jansenvanvuuren

    :param  shot: Shot number
    :param  time: Time point to read the profile.

    :param  diag: diagnostic from which 'Ti' will extracted.

    :param  xArrayOutput: flag to return the output as dictionary of xarray

    :return output: a dictionary containing the ion temperature evaluated
        in the input times and the corresponding rhopol base.


    Use example:
        >>> import Lib as ss
        >>> ne = ss.dat.get_Ti(41091, 3.55, xArrayOutput=True)
    """
    if diag not in ('CONF', 'CXRS'):
        raise Exception('Diagnostic non supported!')

    if diag == 'CONF':
        return get_ne_conf(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'CXRS':
        pass
        #TODO
        #return get_ne_cxrs(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)

def get_Ti_conf(shot: int, time: float = None,  xArrayOutput: bool = False):
    """"
    Read the ion temperature profile from the TCV MODS conf node.

    see https://spcwiki.epfl.ch/wiki/Chie_TCV_nodes#Full_CONF_mds_nodes_list
    
    :param  shot: Shot number
    :param  time: Time point to read the profile.
    :param  edition: edition of the shotfile to be read.

    :return output: a dictionary containing the electron temp. evaluated
    in the input times and the corresponding rhopol base.
    """

    # --- Reading from the database
    try:
        ti_conf_data = conf.ti(shot,  trial_indx=1)
        ti = ti_conf_data.data.T
        ti_unc = np.zeros(np.shape(ti)) * np.nan
        rho = ti_conf_data.rho.data
        timebase = ti_conf_data.t.data
    except:
        raise Exception('Cannot read the ion temperature from the conf nodes for shot: #%05d'%shot)

    if time is None:
        time = timebase
        tmp_ti = ti
        tmp_unc = ti_unc
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
            'rhop': rho,
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
            coords={'rho': rho, 't': time})
        output['data'].attrs['long_name'] = '$T_i$'
        output['data'].attrs['units'] = 'eV'
        output['uncertainty'] = xr.DataArray(tmp_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta T_i$'
        output['uncertainty'].attrs['units'] = '$eV$'

        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'CONF'
        output.attrs['shot'] = shot
    return output

# -----------------------------------------------------------------------------
# --- Toroidal rotation velocity
# -----------------------------------------------------------------------------

def get_tor_rotation(shotnumber: int, time: float = None, **kwargs):
    """
    Retrieves from the database the toroidal velocity (vtor).
    To get the linear velocity (i.e., vtor) multiply by the major radius.



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

    #TODO

# -----------------------------------------------------------------------------
# Zeff
# -----------------------------------------------------------------------------
def get_Zeff(shot: int, **kwargs):
    """
    Wrap the different diagnostics to read Z effective.

    It supports CONF and PROFFIT (or it will eventually ;) ).

    Anton Jansen van Vuuren - anton.jansenvanvuuren

    :param  shot: Shot number
    :param  time: Time point to read the profile.
    :param  diag (**kwarg): diagnostic from which 'Ti' will extracted.

    :param  xArrayOutput: flag to return the output as dictionary of xarray
    :return output: a dictionary containing the Z effective


    Use example:
        >>> import Lib as ss
        >>> z_eff = ss.dat.get_Zeff(41091)
    """

    default_options = {
        'diag': 'CONF'
    }
    default_options.update(kwargs)

    if default_options['diag'] not in ('CONF', 'CXRS'):
        raise Exception('Diagnostic non supported!')

    if default_options['diag'] == 'CONF':
        return get_Zeff_conf(shot=shot)
    elif default_options['diag'] == 'CXRS':
        pass
        #TODO
        #return get_zeff_cxrs(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)

def get_Zeff_conf(shot: int):
    """
    Get the Zeff effective
    """

    # --- Reading from the database
    try:
        Zeff_conf_data = conf.ti(shot,  trial_indx=1)
        Zeff = Zeff_conf_data.data
        rho = Zeff_conf_data.rho.data
        Zeff_unc = Zeff * np.nan
        timebase = Zeff_conf_data.t.data
    except:
        raise Exception('Cannot read Z effective from the conf nodes for shot: #%05d'%shot)

    z = xr.Dataset()
    
    z['data'] = xr.DataArray(Zeff.T, dims=('t', 'rho'), 
                            coords={'t':timebase, 'rho':rho})
    z['uncertainty'] = xr.DataArray(Zeff_unc.T, dims=('t', 'rho'), 
                            coords={'t':timebase, 'rho':rho})
    return z