"""
Routines to work with the pellets
"""
import aug_sfutils as sf
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from ._common import safe_sfread, safe_equ, _clean_val, to_dict_with_metadata

xr.set_options(keep_attrs=True)
import logging
logger = logging.getLogger(__name__)


def get_pellets(shot, coars = 1000, plot = True, xArrayOutput: bool = True,):
    """
    Return the pellets time trace in terms of programmed value and pellets
    that reached the plasma.

    Based on the old routine written by P. Oyola

    Alex Reyner: areyner@us.es

    :param  shot: shot number
    :param  coars: reduce data points (default 1000)
    :param  plot: quick plot of the data
    :param  xArrayOutput: get the output in xarray format or dictionary
    """

    PID = safe_sfread('PID', shot)
    t = np.array(PID.gettimebase('5Co'), dtype='f4')
    real = np.array(PID('5Co'), dtype='f4')
    prog = np.array(PID('Pelarr'), dtype='f4')
    
    obj = xr.Dataset(
        data_vars={
            'real_value': (['t'], real,
                {'long_name': 'Measured Pellet', 'units': 'a.u.'},),
            'programmed': (['t'], prog, 
                {'long_name': 'Programmed Pellet', 'units': 'a.u.'},),
        },
        coords={'t': ('t', t, {'long_name': 'Time', 'units': 's'}),},
        attrs={'shot': shot, 'diag': 'PID'}
    )
    obj = obj.coarsen(t=coars, coord_func='mean', boundary='trim').mean()
    offset = obj.real_value.sel(t=slice(None,0)).mean().values
    obj.attrs['offset'] = offset
    obj['real_value'] -= offset

    if plot:
        fig, ax = plt.subplots()
        obj.programmed.plot(ax=ax, label = 'Programmed')
        obj.real_value.plot(ax=ax, label = 'Real')
        ax.set_ylabel('Value [-]')
        ax.legend()

    if xArrayOutput: return obj
    return to_dict_with_metadata(obj)