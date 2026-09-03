"""
Routines to work with the pellets
"""
import aug_sfutils as sf
import matplotlib.pyplot as plt
from .Misc import to_dict_with_metadata
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr



def get_pellets(shot, coars = 1000, plot = True, xArrayOutput: bool = True,):
    '''
    Return the pellets time trace in terms of programmed value and pellets
    that reached the plasma.

    :param shot: AUG shot number
    :param avg_window: Window size for block averaging to downsample high-frequency signal
    :param remove_offset: Correct the real value signal offset using pre-shot baseline
    :param xArrayOutput: Return xarray.Dataset if True, else converted dictionary
    '''

    PID = sf.SFREAD(shot, 'PID')
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