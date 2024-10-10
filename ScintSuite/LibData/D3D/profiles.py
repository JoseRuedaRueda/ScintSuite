"""
Module to load the profiles from D3D database
"""
import ScintSuite as ss
import numpy as np
import xarray as xr
import os
import logging
from ScintSuite.LibData.D3D._gadata import gadata
logger = logging.getLogger('ScintSuite.D3D')

# ------------------------------------------------------------------------------
# %%
# Individual routines
# ------------------------------------------------------------------------------
def get_ne(shot, time: float = None, exp: str = 'D3D', diag: str = 'zipfit',
           **kwargs):
    """
    Get the electron density from the D3D database
    
    Jose Rueda Rueda: jruedaru@uci.edu
    
    get the electron density, we could get this directly from zfit, but this is 
    a wrapper to be compatible with the already existing AUG library
    
    No keyword arguments are used, it is just to be compatible with the AUG
    """
    if diag.lower() == 'zipfit':
        ne = ss.dat.gadata('profile_fits.zipfit.edensfit', shot, 
                           tree='electrons')
    else:
        raise ValueError(f'Unknown diagnostic {diag}')
    NE = xr.DataArray(ne.zdata, dims=['rho', 't'],
                      coords={'rho': ne.xdata, 't': ne.ydata/1000.0})
    NE.attrs['units'] = ne.zunits
    NE.attrs['long_name'] = 'Electron density'
    
    NE.t.attrs['units'] = 's'
    NE.t.attrs['long_name'] = 'Time'
    NE.rho.attrs['long_name'] = r'$\rho_p$'
    # Get the desired time
    if time is not None:
        NE = NE.interp(t=time)
    # Return a dataset (in the future uncertainities could be added)
    NE2 = xr.Dataset({'data': NE})
    NE2.attrs = NE.attrs
    NE2.attrs['shot'] = shot
    NE2.attrs['diag'] = diag
    return NE2


def get_Te(shot, time: float = None, exp: str = 'D3D', diag: str = 'zipfit',
           **kwargs):
    """
    Get the electron temperature from the D3D database
    
    Jose Rueda Rueda: jruedaru@uci.edu
    
    :param shot: Shot number
    :param time: Time where we want the profiles, if None, all available times are returned
    """
    if diag.lower() == 'zipfit':
        Te = ss.dat.gadata('profile_fits.zipfit.etempfit', shot, 
                           tree='electrons')
    else:
        raise ValueError(f'Unknown diagnostic {diag}')
    TE = xr.DataArray(Te.zdata, dims=['rho', 't'],
                      coords={'rho': Te.xdata, 't': Te.ydata/1000.0})
    TE.attrs['units'] = Te.zunits
    TE.attrs['long_name'] = 'Electron temperature'
    TE.t.attrs['units'] = 's'
    TE.t.attrs['long_name'] = 'Time'
    TE.rho.attrs['long_name'] = r'$\rho_p$'
    # Get the desired time
    if time is not None:
        TE = TE.interp(t=time)
    # Return a dataset (in the future uncertainities could be added)
    TE2 = xr.Dataset({'data': TE})
    TE2.attrs = TE.attrs
    TE2.attrs['shot'] = shot
    TE2.attrs['diag'] = diag
    return TE2


def get_Ti(shot, time: float = None, exp: str = 'D3D', diag: str = 'zipfit',
           **kwargs):
    """
    Get the ion temperature from the D3D database
     
    Jose Rueda Rueda:
     
    :param shot: Shot number
    :param time: Time where we want the profiles, if None, all available times are returned
    """
    if diag.lower() == 'zipfit':
        Ti = ss.dat.gadata('profile_fits.zipfit.itempfit', shot, 
                           tree='ions')
    else:
        raise ValueError(f'Unknown diagnostic {diag}')
    TI = xr.DataArray(Ti.zdata, dims=['rho', 't'],
                      coords={'rho': Ti.xdata, 't': Ti.ydata/1000.0})
    TI.attrs['units'] = Ti.zunits
    TI.attrs['long_name'] = 'Ion temperature'
    TI.t.attrs['units'] = 's'
    TI.t.attrs['long_name'] = 'Time'
    TI.rho.attrs['long_name'] = r'$\rho_p$'
    # Get the desired time
    if time is not None:
        TI = TI.interp(t=time)
    # Return a dataset (in the future uncertainities could be added)
    TI2 = xr.Dataset({'data': TI})
    TI2.attrs = TI.attrs
    TI2.attrs['shot'] = shot
    TI2.attrs['diag'] = diag
    return TI2

def get_tor_rotation(shot, time: float = None, exp: str = 'D3D',
                     diag: str = 'zipfit', **kwargs):
    """
    Get the toroidal rotation from the D3D database
    
    Jose Rueda Rueda:
    
    :param shot: Shot number
    :param time: Time where we want the profiles, if None, all available times are returned
    """
    if diag.lower() == 'zipfit':
        tor = ss.dat.gadata('profile_fits.zipfit.trotfit', shot, 
                            tree='ions')
        tor.zdata *= 1000.0  # Convert to Hz
        tor.zunits = 'Hz'
    else:
        raise ValueError(f'Unknown diagnostic {diag}')
    TROT = xr.DataArray(tor.zdata, dims=['rho', 't'],
                        coords={'rho': tor.xdata, 't': tor.ydata/1000.0})
    TROT.attrs['units'] = tor.zunits
    TROT.attrs['long_name'] = 'Toroidal rotation'
    TROT.t.attrs['units'] = 's'
    TROT.t.attrs['long_name'] = 'Time'
    TROT.rho.attrs['long_name'] = r'$\rho_p$'
    # Get the desired time
    if time is not None:
        TROT = TROT.interp(t=time)
    # Return a dataset (in the future uncertainities could be added)
    TROT2 = xr.Dataset({'data': TROT})
    TROT2.attrs = TROT.attrs
    TROT2.attrs['shot'] = shot
    TROT2.attrs['diag'] = diag
    return TROT2

def get_nimp(shot, time: float = None, exp: str = 'D3D', diag: str = 'zipfit',
                **kwargs):
    """
    Get the impurity density from the D3D database
    
    Jose Rueda Rueda:
    
    :param shot: Shot number
    :param time: Time where we want the profiles, if None, all available times are returned
    """
    if diag.lower() == 'zipfit':
        nimp = ss.dat.gadata('profile_fits.zipfit.zdensfit', shot, 
                            tree='ions')
    else:
        raise ValueError(f'Unknown diagnostic {diag}')
    NIMP = xr.DataArray(nimp.zdata, dims=['rho', 't'],
                        coords={'rho': nimp.xdata, 't': nimp.ydata/1000.0})
    NIMP.attrs['units'] = nimp.zunits
    NIMP.attrs['long_name'] = 'Impurity density'
    NIMP.t.attrs['units'] = 's'
    NIMP.t.attrs['long_name'] = 'Time'
    NIMP.rho.attrs['long_name'] = r'$\rho_p$'
    logger.warning('Assumed carbon as impurity')
    NIMP.attrs['impurity'] = 'C'
    NIMP.attrs['Z'] = 6
    # Get the desired time
    if time is not None:
        NIMP = NIMP.interp(t=time)
    # Return a dataset (in the future uncertainities could be added)
    NIMP2 = xr.Dataset({'data': NIMP})
    NIMP2.attrs = NIMP.attrs
    NIMP2.attrs['shot'] = shot
    NIMP2.attrs['diag'] = diag
    return NIMP2

def get_Zeff(shot, time: float = None, exp: str = 'D3D', diag: str = 'zipfit',
                **kwargs):
    """
    Get the effective charge from the D3D database
    
    Jose Rueda Rueda:
    
    :param shot: Shot number
    :param time: Time where we want the profiles, if None, all available times are returned
    """
    # Get the electron density
    ne = get_ne(shot, time, diag=diag)
    # Get the impurity density
    nimp = get_nimp(shot, time, diag=diag)
    # Defive the ion density
    Zimp = nimp.attrs['Z']
    ni = ne.data - Zimp*nimp.data.interp_like(ne.data)
    # Get the Zeffective
    Zeff = xr.Dataset()
    Zeff['data'] = (Zimp**2*nimp.data.interp_like(ne.data) + ni) / ne.data
    return Zeff
                  

def get_profiles(shot: int, time: float = None, source: str = 'zipfit'):
    """
    Get the profiles from the D3D database
    
    For the moment, only zipfit support added, but we can add more
    
    Jose Rueda Rueda: jruedaru@uci.edu
    
    :param shot: Shot number
    :param time: Time where we want the profiles, if None, all available times are returned
    """
    if source == 'zipfit':
        return get_profiles_zipfit(shot, time)


def get_profiles_zipfit(shot: int, time: float = None):
    """
    Get the profiles from the MDSplus ZfitRun
    
    Jose Rueda Rueda: jruedaru@uci.edu
    
    :param shot: Shot number
    :param time: Time where we want the profiles, if None, all available times are returned
    """
    # Get the electron density
    ne = get_ne(shot, time, diag='zipfit')
    # Get the electron temperature
    Te = get_Te(shot, time, diag='zipfit')
    # Get the ion temperature
    Ti = get_Ti(shot, time, diag='zipfit')
    # Get the toroidal rotation
    tor = get_tor_rotation(shot, time, diag='zipfit')
    # Get the impurity density
    nimp = get_nimp(shot, time, diag='zipfit')
    # Defive the ion density
    Zeff = xr.Dataset()
    ni = xr.Dataset()
    Zimp = nimp.attrs['Z']
    ni['data'] = ne.data - Zimp*nimp.data.interp_like(ne.data)
    # Get the Zeffective
    Zeff['data'] = (Zimp**2*nimp.data.interp_like(ne.data) + ni.data) / ne.data
    # Move to a data set
    profiles = {'ne': ne, 'Te': Te, 'Ti': Ti, 'trot': tor, 'nimp': nimp,
                           'ni': ni, 'Zeff': Zeff}
    return profiles
