"""
Library to handle ECE data
"""
import os
import logging
import numpy as np
import xarray as xr
import scipy.signal as signal
import ScintSuite.errors as errors
from ScintSuite.LibData.D3D._gadata import gadata


# --- Auxiliary objects
logger = logging.getLogger('ScintSuite.D3D.BES')

# --- Routines
def get_ece_fast(shot: int, ch: int, trange: list = [0.5, 1.0],
              normalize: bool = True):
    """
    Read the ECE fast data
    
    :param shot: The shot number
    :param ch: The channel number
    :param trange [s]: The time range to output
    :param normalize: Bool, if true, the fast signal will be normalized by the
        slow signal. (this is usefull to get (dT)/T)
    """
    if (shot < 115000) and (shot > 84000):
        pointnameFast = 'ecef'+str(ch)
    elif (shot > 115000) and (shot < 152600):
        pointnameFast = 'ecevf'+str(ch).zfill(2)
    elif (shot > 152700):
        pointnameFast = 'ecevs'+str(ch).zfill(2)
    else:
        raise errors.NotValidInput('Shot number not valid')
    
    f = gadata(pointnameFast, shot)
    fast = xr.DataArray(f.zdata, coords={'t': f.xdata/1000.0}, dims=['t'])
    # Detrent the fluctation data to get the slow variation
    df = signal.detrend(fast.values, type='linear', bp=np.arange(0, fast.size, 1024))
    df = xr.DataArray(df, coords={'t': fast.t}, dims=['t'])
    dslow = fast - df
    # Select the time range
    fast = fast.sel(t=slice(trange[0], trange[1]))
    df = df.sel(t=slice(trange[0], trange[1]))
    dslow = dslow.sel(t=slice(trange[0], trange[1]))
    if normalize:
        fast = df/dslow
        # Remove possible nans and infs
        flags = np.isnan(fast.values) | np.isinf(fast.values)
        fast [flags] = 0
    fast.attrs['shot'] = shot
    fast.attrs['channel'] = ch
    fast.attrs['units'] = f.zunits
    return fast


