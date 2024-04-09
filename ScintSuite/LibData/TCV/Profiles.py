"""
Routines to interact with the TCV database implemented for ScintSuite

Wrapper routines to acces kinetic profile data from the CONF node via the tcvpy package
MDSplus needed as well as the tcvpy package to call these functions

Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch
"""
import sys

try:
    sys.path.append('/home/jansen/NoTivoli/ascot-tcv/')
    import tcvpy.results.conf as conf
    import tcvpy.tcv as tcv
except:
    pass

try:
    import eqtools
except:
    pass


import numpy as np
import xarray as xr

import ScintSuite.errors as errors
from ScintSuite._Paths import Path
from scipy.interpolate import interp1d, interp2d, UnivariateSpline


import logging
logger = logging.getLogger('ScintSuite.Data')
pa = Path()

import ScintSuite._Video._MATfiles as mat
# -----------------------------------------------------------------------------
# --- Electron density and temperature profiles.
# -----------------------------------------------------------------------------
def get_ne(shotnumber: int, time: float = None,
           diag: str = 'MAT',
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
    if diag not in ('CONF', 'PROFFIT', 'THOMSON', 'MAT'):
        raise Exception('Diagnostic non supported!')

    if diag == 'CONF':
        return get_ne_conf(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'PROFFIT':
        return get_ne_proffit(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'THOMSON':
        pass
        #TODO
        #return get_ne_thomson(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)    
    elif diag == 'MAT':
        return get_ne_MAT(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    

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

def get_ne_proffit(shotnumber: int, time: float = None, 
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
        >>> import ScintSuite.LibData as ssdat
        >>> ne = ssdat.get_ne(41091, 3.55, xArrayOutput=True)
    """
    # --- Reading from the database
    try:
        MDS_Connection = tcv.shot(shotnum = shotnumber)
        MDS_tdi_command = r'\tcv_shot::top.results.proffit.local_time:neft'
        ne_proffit_data = MDS_Connection.tdi(MDS_tdi_command).values 

        ne_proffit_rho  = MDS_Connection.tdi('dim_of(%s, 0)'%MDS_tdi_command).values 
        ne_proffit_time  = MDS_Connection.tdi('dim_of(%s, 1)'%MDS_tdi_command).values 

        ne = ne_proffit_data
        ne_unc = np.zeros(np.shape(ne)) * np.nan
        rho = ne_proffit_rho
        timebase = ne_proffit_time
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

        output['rho'].attrs['long_name'] = ne_proffit_data.dim_labels[0] #'$\\rho_{TCV}$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'proffit'
        output.attrs['shot'] = shotnumber
    return output

def get_ne_MAT(shotnumber: int, time: float = None, 
           xArrayOutput: bool = False):
    """
    Read the electron density profile from MAT file.
    
    :param  shot: Shot number
    :param  time: Time point to read the profile.

    :param  xArrayOutput: flag to return the output as dictionary of xarray

    :return output: a dictionary containing the electron density evaluated
        in the input times and the corresponding rhopol base.


    Use example:
        >>> import ScintSuite.LibData as ssdat
        >>> ne = ssdat.get_ne(41091, 3.55, xArrayOutput=True)
    """
    # --- Reading from the database
    try:
        file = '/NoTivoli/jansen/SF/NE/%i.mat'%shotnumber
        ne_mat = mat.read_file(file)['output'][0,0]
    except:
        raise Exception('Cannot read the density from the IDA #%05d'%shotnumber)

    ne = ne_mat[0]#['ne_profile']
    rho = ne_mat[2][:, 0]#['rho_pol_norm']
    timebase = ne_mat[5][0,:]#['time']
    ne_unc = ne*0

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
            tmp_ne/1.0e19, dims=('rho', 't'),
            coords={'rho': rho, 't': time})
        output['data'].attrs['long_name'] = '$n_e$'
        output['data'].attrs['units'] = '$10^{19} m^3$'
        output['uncertainty'] = xr.DataArray(tmp_unc/1.0e19, dims=('rho',
                                                                     't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta n_e$'
        output['uncertainty'].attrs['units'] = '$10^{19} m^3$'

        output['rho'].attrs['long_name'] = '$\\rho_{TCV}$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'MAT'
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
        return get_Te_proffit(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
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

def get_Te_proffit(shotnumber: int, time: float = None, 
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
        >>> import ScintSuite.LibData as ssdat
        >>> ne = ssdat.get_ne(41091, 3.55, xArrayOutput=True)
    """
    # --- Reading from the database
    try:
        MDS_Connection = tcv.shot(shotnum = shotnumber)
        MDS_tdi_command = r'\tcv_shot::top.results.proffit.avg_time:teft'
        Te_proffit_data = MDS_Connection.tdi(MDS_tdi_command).values 

        Te_proffit_rho  = MDS_Connection.tdi('dim_of(%s, 0)'%MDS_tdi_command).values 
        Te_proffit_time  = MDS_Connection.tdi('dim_of(%s, 1)'%MDS_tdi_command).values 

        Te = Te_proffit_data
        Te_unc = np.zeros(np.shape(Te)) * np.nan
        rho = Te_proffit_rho
        timebase = Te_proffit_time
    except:
        raise Exception('Cannot read the density from the IDA #%05d'%shotnumber)

    # We will return the data in the same spatial basis as provided by IDA.
    if time is None:
        time = timebase
        tmp_ne = Te
        tmp_unc = np.zeros(np.shape(Te)) * np.nan  #conf node doesn't inlcude uncertainty
    else:
        tmp_ne = interp1d(timebase, Te, kind='linear', axis=0,
                          bounds_error=False, fill_value=np.nan,
                          assume_sorted=True)(time).T
        tmp_unc = interp1d(timebase, Te_unc,
                           kind='linear', axis=0,
                           bounds_error=False,
                           fill_value=np.nan,
                           assume_sorted=True)(time).T

    if not xArrayOutput:
        output = {
            'rho': rho,
            'time': time,
            'uncertainty': tmp_unc,
            'data': tmp_Te
        }
    else:
        tmp_Te = np.atleast_2d(tmp_Te)
        tmp_unc = np.atleast_2d(tmp_unc)
        time = np.atleast_1d(time)

        output = xr.Dataset()
        output['data'] = xr.DataArray(
            tmp_Te.T, dims=('rho', 't'),
            coords={'rho': rho, 't': time})
        output['data'].attrs['long_name'] = '$T_e$'
        output['data'].attrs['units'] = '$eV$'
        output['uncertainty'] = xr.DataArray(tmp_unc.T, dims=('rho',
                                                                     't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta T_e$'
        output['uncertainty'].attrs['units'] = '$eV$'

        output['rho'].attrs['long_name'] = Te_proffit_data.dim_labels[0] #'$\\rho_{TCV}$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'proffit'
        output.attrs['shot'] = shotnumber
    return output

def get_Pe(shotnumber: int, time: float = None,
           diag: str = 'CONF',
           xArrayOutput: bool = False):
    """
    Wrap the different diagnostics to read the electron pressure profile.

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
        pass
        #TODO
        #return get_Pe_conf(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'PROFFIT':
        return get_Pe_proffit(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'THOMSON':
        pass
        #TODO
        #return get_ne_thomson(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)    

def get_Pe_proffit(shotnumber: int, time: float = None, 
           xArrayOutput: bool = False):
    """
    Read the electron pressure profile from the TCV MODS conf node.
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
        MDS_Connection = tcv.shot(shotnum = shotnumber)
        MDS_tdi_command = r'\tcv_shot::top.results.proffit.avg_time:peft'
        Pe_proffit_data = MDS_Connection.tdi(MDS_tdi_command).values 

        Pe_proffit_rho  = MDS_Connection.tdi('dim_of(%s, 0)'%MDS_tdi_command).values 
        Pe_proffit_time  = MDS_Connection.tdi('dim_of(%s, 1)'%MDS_tdi_command).values 

        Pe = Pe_proffit_data
        Pe_unc = np.zeros(np.shape(Pe)) * np.nan
        rho = Pe_proffit_rho
        timebase = Pe_proffit_time
    except:
        raise Exception('Cannot read the electron temperature from the conf nodes for shot: #%05d'%shotnumber)

    # We will return the data in the same spatial basis as provided by IDA.
    if time is None:
        time = timebase
        tmp_Pe = Pe
        tmp_unc = Pe_unc
        #tmp_dPe = gradPe
    else:
        tmp_te = interp1d(timebase, Pe, kind='linear', axis=0,
                          bounds_error=False, fill_value=np.nan,
                          assume_sorted=True)(time).T
        tmp_unc = interp1d(timebase, Pe_unc,
                           kind='linear', axis=0,
                           bounds_error=False, fill_value=np.nan,
                           assume_sorted=True)(time).T
        #tmp_dTe = interp1d(timebase, gradPe,
        #                   kind='linear', axis=0,
        #                   bounds_error=False, fill_value=np.nan,
        #                   assume_sorted=True)(time).T
    if not xArrayOutput:
        output = {
            'rho': rho,
            'time': time,
            'data': tmp_Pe,
            'uncertainty': tmp_unc}#,
            #'gradient': tmp_dPe}
    else:
        output = xr.Dataset()
        tmp_Pe = np.atleast_2d(tmp_Pe)
        time = np.atleast_1d(time)
        output['data'] = xr.DataArray(
            tmp_Pe.T, dims=('rho', 't'),
            coords={'rho': rho, 't': time})
        output['data'].attrs['long_name'] = '$P_e$'
        output['data'].attrs['units'] = 'J/m^3'
        tmp_unc = np.atleast_2d(tmp_unc)
        output['uncertainty'] = xr.DataArray(tmp_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta P_e$'
        output['uncertainty'].attrs['units'] = '$J/m^3$'
        #tmp_dPe = np.atleast_2d(tmp_dPe)
        #output['gradient'] = xr.DataArray(tmp_dPe.T, dims=('rho', 't'))
        # output['gradient'].attrs['long_name'] = '$\\Nabla P_e$'
        # output['gradient'].attrs['units'] = '$J/m^3$'

        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'proffit'
        output.attrs['shot'] = shotnumber

    return output

def get_R_Lne(shotnumber: int, time: float = None,
           diag: str = 'CONF',
           xArrayOutput: bool = False):
    """
    Wrap the different diagnostics to read the 
    normalized density gradient lenght R0*grad(ne)/ne, 
    where grad ne=dne/drho*grad(rho)

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
        pass
        #TODO
        #return get_R_Lne_conf(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'PROFFIT':
        return get_R_Lne_proffit(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'THOMSON':
        pass
        #TODO
        #return get_ne_thomson(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)    

def get_R_Lne_proffit(shotnumber: int, time: float = None, 
           xArrayOutput: bool = False):
    """
    Read the normalized density gradient lenght R0*grad(ne)/ne, 
    where grad ne=dne/drho*grad(rho)
    from proffit nodes
    see https://spcwiki.epfl.ch/wiki/Proffit_nodes
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
        MDS_Connection = tcv.shot(shotnum = shotnumber)
        MDS_tdi_command = r'\tcv_shot::top.results.proffit.avg_time:r_lne'
        R_Lne_proffit_data = MDS_Connection.tdi(MDS_tdi_command).values 

        R_Lne_proffit_rho  = MDS_Connection.tdi('dim_of(%s, 0)'%MDS_tdi_command).values 
        R_Lne_proffit_time  = MDS_Connection.tdi('dim_of(%s, 1)'%MDS_tdi_command).values 

        R_Lne = R_Lne_proffit_data
        R_Lne_unc = np.zeros(np.shape(R_Lne)) * np.nan
        rho = R_Lne_proffit_rho
        timebase = R_Lne_proffit_time
    except:
        raise Exception('Cannot read the electron temperature from the conf nodes for shot: #%05d'%shotnumber)

    # We will return the data in the same spatial basis as provided by IDA.
    if time is None:
        time = timebase
        tmp_Pe = R_Lne
        tmp_unc = R_Lne_unc
        #tmp_dPe = gradPe
    else:
        tmp_R_Lne = interp1d(timebase, R_Lne, kind='linear', axis=0,
                          bounds_error=False, fill_value=np.nan,
                          assume_sorted=True)(time).T
        tmp_unc = interp1d(timebase, R_Lne_unc,
                           kind='linear', axis=0,
                           bounds_error=False, fill_value=np.nan,
                           assume_sorted=True)(time).T

    if not xArrayOutput:
        output = {
            'rho': rho,
            'time': time,
            'data': tmp_R_Lne,
            'uncertainty': tmp_unc}
        
    else:
        output = xr.Dataset()
        tmp_R_Lne = np.atleast_2d(tmp_R_Lne)
        time = np.atleast_1d(time)
        output['data'] = xr.DataArray(
            tmp_R_Lne.T, dims=('rho', 't'),
            coords={'rho': rho, 't': time})
        output['data'].attrs['long_name'] = '$R_{Lne}$'
        output['data'].attrs['units'] = 'R_{Lne}'
        tmp_unc = np.atleast_2d(tmp_unc)
        output['uncertainty'] = xr.DataArray(tmp_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta R_{Lne}$'
        output['uncertainty'].attrs['units'] = '$R_{Lne}$'


        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'proffit'
        output.attrs['shot'] = shotnumber

    return output

def get_R_Lte(shotnumber: int, time: float = None,
           diag: str = 'CONF',
           xArrayOutput: bool = False):
    """
    Wrap the different diagnostics to read the 
    normalized density gradient lenght R0*grad(ne)/ne, 
    where grad ne=dne/drho*grad(rho)

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
        pass
        #TODO
        #return get_R_Lte_conf(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'PROFFIT':
        return get_R_Lte_proffit(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)
    elif diag == 'THOMSON':
        pass
        #TODO
        #return get_ne_thomson(shotnumber=shotnumber, time=time, xArrayOutput=xArrayOutput)    

def get_R_Lte_proffit(shotnumber: int, time: float = None, 
           xArrayOutput: bool = False):
    """
    Read the normalized temperature gradient lenght R0*grad(ne)/ne, 
    where grad te=dte/drho*grad(rho)
    from proffit nodes
    see https://spcwiki.epfl.ch/wiki/Proffit_nodes
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
        MDS_Connection = tcv.shot(shotnum = shotnumber)
        MDS_tdi_command = r'\tcv_shot::top.results.proffit.avg_time:r_lte'
        R_Lte_proffit_data = MDS_Connection.tdi(MDS_tdi_command).values 

        R_Lte_proffit_rho  = MDS_Connection.tdi('dim_of(%s, 0)'%MDS_tdi_command).values 
        R_Lte_proffit_time  = MDS_Connection.tdi('dim_of(%s, 1)'%MDS_tdi_command).values 

        R_Lte = R_Lte_proffit_data
        R_Lte_unc = np.zeros(np.shape(R_Lte)) * np.nan
        rho = R_Lte_proffit_rho
        timebase = R_Lte_proffit_time
    except:
        raise Exception('Cannot read the electron temperature from the conf nodes for shot: #%05d'%shotnumber)

    # We will return the data in the same spatial basis as provided by IDA.
    if time is None:
        time = timebase
        tmp_R_Lte = R_Lte
        tmp_unc = R_Lte_unc
        #tmp_dPe = gradPe
    else:
        tmp_R_Lte = interp1d(timebase, R_Lte, kind='linear', axis=0,
                          bounds_error=False, fill_value=np.nan,
                          assume_sorted=True)(time).T
        tmp_unc = interp1d(timebase, R_Lte_unc,
                           kind='linear', axis=0,
                           bounds_error=False, fill_value=np.nan,
                           assume_sorted=True)(time).T

    if not xArrayOutput:
        output = {
            'rho': rho,
            'time': time,
            'data': tmp_R_Lte,
            'uncertainty': tmp_unc}
        
    else:
        output = xr.Dataset()
        tmp_R_Lte = np.atleast_2d(tmp_R_Lte)
        time = np.atleast_1d(time)
        output['data'] = xr.DataArray(
            tmp_R_Lte.T, dims=('rho', 't'),
            coords={'rho': rho, 't': time})
        output['data'].attrs['long_name'] = '$R_{Lte}$'
        output['data'].attrs['units'] = 'R_{Lte}'
        tmp_unc = np.atleast_2d(tmp_unc)
        output['uncertainty'] = xr.DataArray(tmp_unc.T, dims=('rho', 't'))
        output['uncertainty'].attrs['long_name'] = '$\\Delta R_{Lte}$'
        output['uncertainty'].attrs['units'] = '$R_{Lte}$'


        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output.attrs['diag'] = 'proffit'
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
def get_tor_rotation(shotnumber: int, time: float = None, diag: str = 'CXRS'
                     ,**kwargs):
    """
    Retrieves from the database the toroidal velocity velocity (v_tor).
    Wrapper to call diagnistic spefic functions

    :param  shotnumber: shotnumber to read.
    :param  time: time window to get the data. If None, all the available times
    are read.

    """

    if diag == 'CXRS':
        return get_tor_rotation_cxrs_fit(shotnumber, time, **kwargs)

def get_tor_rotation_cxrs_fit(shotnumber: int, time: float = None, 
                              rad_per_s = True,
                              nans_to_zero = True,
                              **kwargs):
    """
    Retrieves from the fitted toroidal velocity (v_tor) from CXRS data in [km/s].
    See: https://spcwiki.epfl.ch/wiki/CXRS
    
    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch

    To get the angular velocity (i.e., omega_tor) devide by the major radius.
    Use keyword rad_per_s = True to convert from radial to angular velocity.

    :param  shotnumber: shotnumber to read.
    :param  time: time window to get the data. If None, all the available times
    are read.

    """
    default_options = {
        'xArrayOutput': True
    }
    default_options.update(kwargs)

    try:
        MDS_tdi_command = r'\RESULTS::CXRS.PROFFIT:VI_TOR'
        MDS_Connection = tcv.shot(shotnum = shotnumber)  
        vtor = MDS_Connection.tdi(MDS_tdi_command)
        vtor = vtor.values
        vtor_unc = vtor * np.nan
        rho = MDS_Connection.tdi('dim_of(%s, 0)'%MDS_tdi_command).values    # dimension is (rho [101], t [42])
        rho = rho[:, 0]
        timebase = MDS_Connection.tdi('dim_of(%s, 1)'%MDS_tdi_command).values
        MDS_Connection.close()

    except:
        
        print('\033[91m %s MDS data not available for %i \033[0m'%('toroidal rotation', shotnumber))
        return None

    if nans_to_zero:
        idx_nans = np.argwhere(np.isnan(vtor))
        vtor[idx_nans[:, 0], idx_nans[:, 1]] = 0

    rotation = vtor
    rotation_unc = vtor_unc

    rotation_units = 'km/s'
    rotation_label = '$vtor$'
    rotation_unc_label = '$\\sigma vtor$'

    if rad_per_s:
        #convert the v_tor rotation form km/s to rad/s
        eq = eqtools.TCVLIUQEMATTree(shotnumber)
        eq.getTimeBase()

        Rmid = eq.rho2rho('sqrtpsinorm', 'Rmid', rho, timebase).T

        v_omega = vtor/ (Rmid / 1000 )

        if nans_to_zero:
            idx_nans = np.argwhere(np.isnan(v_omega))
            v_omega[idx_nans[:, 0], idx_nans[:, 1]] = 0

        rotation = v_omega
        rotation_unc = v_omega * np.nan
        rotation_units = 'rad/s'
        rotation_label = '$\\omega$'
        rotation_unc_label = '$\\sigma\\omega$'

    # --- If a time window is provided, we cut out the data.
    if time is not None:
        t0, t1 = timebase.searchsorted(time)
        data = data[t0:t1, :]
        unc = unc[t0:t1, :]
        timebase = timebase[t0:t1]

    # --- Saving to a dictionary and output:
    if not default_options['xArrayOutput']:
        output = {
            'data': rotation,
            'uncertainty': rotation_unc,
            'time': timebase,
            'rhop': rho,
            'units': rotation_units,
            'label': rotation_label
        }
    else:
        output = xr.Dataset()
        time = np.atleast_1d(timebase)
        output['data'] = xr.DataArray(
                rotation, dims=('rho', 't'),
                coords={'rho': rho,
                        't': timebase})
        output['uncertainty'] = xr.DataArray(rotation_unc, dims=('rho', 't'))
        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'
        output['data'].attrs['long_name'] = rotation_label
        output['data'].attrs['units'] = rotation_units

        output['uncertainty'].attrs['long_name'] = rotation_unc_label
        output['uncertainty'].attrs['units'] = rotation_units
        output.attrs['diag'] = 'FITPROF_CXRS'
        output.attrs['shot'] = shotnumber

    return output

# -----------------------------------------------------------------------------
# Zeff
# -----------------------------------------------------------------------------
def get_Zeff(shot: int, **kwargs):
    """
    Wrap the different diagnostics to read Z effective.

    It supports CONF and PROFFIT (or it will eventually ;) ).
    CONF Zeff data is constant over rho profile
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
        Zeff_conf_data = conf.zeff(shot,  trial_indx=1)
        Zeff = Zeff_conf_data.data
        Zeff = np.expand_dims(Zeff, axis = 1)
        nrho = 41
        Zeff = np.repeat(Zeff , nrho, axis = 1)

        Zeff_unc = Zeff * np.nan
        timebase = Zeff_conf_data.t.data
    except:
        raise Exception('Cannot read Z effective from the conf nodes for shot: #%05d'%shot)

    z = xr.Dataset()
    rho = np.arange(nrho)/ (nrho-1)  # conf Zeff is not not resolved in
    z['data'] = xr.DataArray(Zeff, dims=('t', 'rho'), 
                            coords={'t':timebase, 'rho':rho})
    z['uncertainty'] = xr.DataArray(Zeff_unc, dims=('t', 'rho'), 
                            coords={'t':timebase, 'rho':rho})
    return z