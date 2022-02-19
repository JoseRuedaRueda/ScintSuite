"""
Other routines.

This library will contain routines that because of its nature do not belong to
a particular category.
This library contains:

- FILD4 trajectories routines.
- ELM starting points.

"""

import dd                # Module to load shotfiles
import numpy as np
import os
import Lib.LibData.AUG.DiagParam as params
from Lib.LibPaths import Path
import matplotlib.pyplot as plt
pa = Path()

# -----------------------------------------------------------------------------
# --- GENERIC SIGNAL RETRIEVING.
# -----------------------------------------------------------------------------
def get_signal_generic(shot: int, diag: str, signame: str, exp: str='AUGD',
                       edition: int=0, tBegin: float=None, tEnd: float=None):
    """
    Function that generically retrieves a signal from the database in AUG.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shot: shotnumber of the shotfile to read.
    @param diag: diagnostic name.
    @param signame: signal name.
    @param exp: experiment where the shotfile is stored. Default to AUGD.
    @param edition: edition of the shotfile to open. If 0, the last closed
    edition is opened.
    @param tBegin: initial time point to read.
    @param tEnd: final time point to read.
    """

    # Reading the second diagnostic data.
    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shot,
                         edition=edition, experiment=exp)

        signal_obj = sf(name=signame, tBegin=tBegin, tEnd=tEnd)
        data = signal_obj.data
        time = signal_obj.time

    except:
        raise Exception('The signal data cannot be read for #%05d:%s:%s(%d)'\
                        (shot, diag, signame, edition))

    sf.close()

    return time, data


# -----------------------------------------------------------------------------
# --- SIGNAL OF FAST CHANNELS.
# -----------------------------------------------------------------------------
def get_fast_channel(diag: str, diag_number: int, channels, shot: int):
    """
    Get the signal for the fast channels (PMT, APD)

    Jose Rueda Rueda: jrrueda@us.es

    @param diag: diagnostic: 'FILD' or 'INPA'
    @param diag_number: 1-5
    @param channels: channel number we want, or arry with channels
    @param shot: shot file to be opened
    """
    # Check inputs:
    suported_diag = ['FILD']
    if diag not in suported_diag:
        raise Exception('No understood diagnostic')

    # Load diagnostic names:
    if diag == 'FILD':
        if (diag_number > 5) or (diag_number < 1):
            print('You requested: ', diag_number)
            raise Exception('Wrong fild number')
        info = params.FILD[diag_number - 1]
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

    # Open the shot file
    fast = dd.shotfile(diag_name, shot)
    dummy_name = signal_prefix + "{0:02}".format(ch[0])
    time = fast.getTimeBase(dummy_name.encode('UTF-8'))
    data = []
    for ic in range(nch):
        real_channel = ic + 1
        if real_channel in ch:
            name_channel = signal_prefix + "{0:02}".format(real_channel)
            channel_dat = fast.getObjectData(name_channel.encode('UTF-8'))
            data.append(channel_dat[:time.size])
        else:
            data.append(None)
    # get the time base (we will use last loaded channel)

    print('Number of requested channels: ', nch_to_load)
    return {'time': time, 'data': data, 'channels': ch}


# -----------------------------------------------------------------------------
# --- ELMs
# -----------------------------------------------------------------------------
def get_ELM_timebase(shot: int, time: float=None, edition: int=0,
                     exp: str='AUGD'):
    """
    Give the ELM onset and duration times

    Jose Rueda Rueda - jrrueda@us.es
    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shot: shot number
    @returns tELM: Dictionary with:
        -# t_onset: The time when each ELM starts
        -# dt: the duration of each ELM
        -# n: The number of ELMs
    """
    # --- Open the AUG shotfile
    ELM = dd.shotfile(diagnostic='ELM', pulseNumber=shot,
                      edition=edition, experiment=exp)
    tELM = {
        't_onset':  ELM('tELM'),
        'dt': ELM('dt_ELM').data,
        'energy': ELM(name='ELMENER').data,
        'f_ELM': ELM(name='f_ELM').data
    }

    if time is not None:
        # If one time is given, we find the nearest ELM.
        time = np.atleast_1d(time)
        if len(time) == 1:
            t0 = np.abs(tELM['t_onset'] - time[0]).argmin()
            tELM = { 't_onset': np.array((tELM['t_onset'][t0],)),
                     'dt': np.array((tELM['dt'][t0],)),
                     'energy': np.array((tELM['energy'][t0],)),
                     'f_ELM': np.array((tELM['f_ELM'][t0],))
                   }
        elif len(time) == 2:
            t0, t1 = np.searchsorted(tELM['t_onset'], time)
            t1 = min(len(tELM['t_onset']), t1+1)
            tELM = { 't_onset': tELM['t_onset'][t0:t1],
                     'dt': tELM['dt'][t0:t1],
                     'energy': tELM['energy'][t0:t1],
                     'f_ELM': tELM['f_ELM'][t0:t1]
                   }

        else:
            tidx = [np.abs(tELM['t_onset'] - time_val).argmin()\
                    for time_val in time]

            tELM = { 't_onset': tELM['t_onset'][tidx],
                     'dt': tELM['dt'][tidx],
                     'energy': tELM['energy'][tidx],
                     'f_ELM': tELM['f_ELM'][tidx]
                   }

    tELM['n'] = len(tELM['t_onset'])
    ELM.close()

    return tELM
