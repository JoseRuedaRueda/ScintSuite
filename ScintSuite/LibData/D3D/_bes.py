"""
Library to handle BES data

Adapted from Xiaodi Du analysis of BES data
"""
import os
import logging
import numpy as np
import xarray as xr
import scipy.io as sio
import ScintSuite.errors as errors
from ScintSuite.LibData.D3D._gadata import gadata
from ScintSuite._Paths import Path
from ScintSuite._Machine import machine

# --- Auxiliary objects
paths = Path(machine)
logger = logging.getLogger('ScintSuite.D3D.BES')
homeSuite = os.getenv("ScintSuitePath")
if homeSuite is None:
    homeSuite = os.path.join(os.getenv("HOME"), 'ScintSuite')
# --- Routines
def get_bes_fast(shot: int, ch: int, trange: list = [0.5, 1.0],
         normalize: bool = True):
    """
    Read the BES fast data
    
    :param shot: The shot number
    :param ch: The channel number
    :param trange [s]: The time range to output
    :param normalize: Bool, if true, the fast signal will be normalized by the
        slow signal. (this is usefull to get (dn)/n)
    """
    pointnameFast = 'BESFU'+str(ch).zfill(2)
    pointnameSlow = 'BESSU'+str(ch).zfill(2)
    fast = gadata(pointnameFast, shot)
    fast = xr.DataArray(fast.zdata, coords={'t': fast.xdata/1000.0}, dims=['t'])
    fast = fast.sel(t=slice(trange[0], trange[1]))
    if normalize:
        try:
            slow = gadata(pointnameSlow, shot)
            # Remove the offset
            slow.zdata = slow.zdata - slow.zdata[0]
            slow = xr.DataArray(slow.zdata, coords={'t': slow.xdata/1000.0}, dims=['t'])
            slow = slow.sel(t=slice(trange[0], trange[1]))
        except Exception as e:
            logger.error(f'Error reading slow data: {e}')
            logger.error('No normalization possible')
            raise errors.DatabaseError('No slow data available')
        # Interpolate the slow data in the fast time base
        slow = slow.interp_like(fast, kwargs={"extrapolate": True})
        fast = fast/slow
        # Remove possible nans and infs
        fast = fast.fillna(0)
        fast = fast.where(fast != np.inf, 0)
        fast = fast.where(fast != -np.inf, 0)
    return fast

def transfer(ch: int):
    """
    Read the transfer function for a given channel
    
    This translate from fluctation in the BES signal to density fluctuation. 
    It includes gain of slow signal and response as the function of frequency

    
    :param ch: The channel number
    """
    fn = os.path.join(homeSuite, 'Data', 'Calibrations', 'BES', 'D3D', 
                      'bestransfer.sav')
    out = sio.readsav(fn)
    f = out['f_bes']
    coeff = np.sqrt(out['transfer_bes'][ch,:])
    return f, coeff