"""
Other routines.

This library will contain routines that because of its nature do not belong to
a particular category.
This library contains:

- FILD4 trajectories routines.
- ELM starting points.
- ICRH power

"""

import numpy as np
import xarray as xr
import aug_sfutils as sf
import ScintSuite.errors as errors
import ScintSuite.LibData.AUG.DiagParam as params

from ScintSuite._Paths import Path
import ScintSuite.errors as errors
import xarray as xr
from datetime import datetime

pa = Path()

# -----------------------------------------------------------------------------
# --- GENERIC SIGNAL RETRIEVING.
# -----------------------------------------------------------------------------
def get_signal_generic(shot: int, diag: str, signame: str, exp: str = 'augd',
                       edition: int = 0, tBegin: float = None,
                       tEnd: float = None):
    """
    Function that generically retrieves a signal from the database in AUG.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: shotnumber of the shotfile to read.
    :param  diag: diagnostic name.
    :param  signame: signal name.
    :param  exp: experiment where the shotfile is stored. Default to augd.
    :param  edition: edition of the shotfile to open. If 0, the last closed
    edition is opened.
    :param  tBegin: initial time point to read.
    :param  tEnd: final time point to read.
    """

    # Reading the second diagnostic data.
    sfo = sf.SFREAD(shot, diag, edition=edition, experiment=exp)

    if not sfo.status:
        raise errors.DatabaseError('The signal data cannot be read for #%05d:%s:%s(%d)'
                        % (shot, diag, signame, edition))

    data = sfo(name=signame)
    if data is None:
        raise errors.DatabaseError('Cannot find signal %s' % signame)
    time = sfo.gettimebase(signame)

    if tBegin is None:
        t0 = 0
    else:
        t0 = np.abs(time - tBegin).argmin()

    if tBegin is None:
        t1 = len(time)
    else:
        t1 = np.abs(time - tEnd).argmin()

    data = np.array(data[t0:t1, ...], dtype=float)
    time = np.array(time[t0:t1, ...], dtype=float)

    return time, data


# -----------------------------------------------------------------------------
# --- SIGNAL OF FAST CHANNELS.
# -----------------------------------------------------------------------------
def get_fast_channel(diag: str, diag_number: int, channels, shot: int,
                     ed: int = 0, exp: str = 'augd'):
    """
    Get the signal for the fast channels (PMT, APD)

    Jose Rueda Rueda: jrrueda@us.es

    :param  diag: diagnostic: 'FILD' or 'INPA'
    :param  diag_number: 1-5
    :param  channels: channel number we want, or arry with channels
    :param  shot: shot file to be opened
    """
    # Check inputs:
    suported_diag = ['FILD', 'INPA']
    if diag not in suported_diag:
        raise errors.NotValidInput('No understood diagnostic') 

    # Load diagnostic names:
    if diag.lower() == 'fild':
        if (diag_number > 5) or (diag_number < 1):
            print('You requested: ', diag_number)
            raise errors.NotValidInput('Wrong fild number')
        info = params.FILD[diag_number - 1]
        diag_name = info['diag']
        signal_prefix = info['channel']
        nch = info['nch']

    elif diag.lower() == 'inpa':
        if diag_number != 1:
            print('You requested: ', diag_number)
            raise errors.NotValidInput('Wrong INPA number')
        info = params.INPA[diag_number - 1]
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
        nch_to_load = channels
        if channels == 'all':
            nch_to_load = nch
            ch = np.arange(nch)+1 #all channels

    # Open the shot file
    fast = sf.SFREAD(diag_name, shot, ed=ed, exp=exp)
    dummy_name = signal_prefix + "{0:02}".format(ch[0])
    time = np.array(fast.gettimebase(dummy_name))
    data = []
    for ic in range(nch):
        real_channel = ic + 1
        if real_channel in ch:
            name_channel = signal_prefix + "{0:02}".format(real_channel)
            channel_dat = np.array(fast(name_channel))
            data.append(channel_dat[:time.size])
        else:
            pass
            # data.append(None)
    print('Number of requested channels: ', nch_to_load)
    return {'time': time, 'data': data, 'channels': ch}


# -----------------------------------------------------------------------------
# --- ELMs
# -----------------------------------------------------------------------------
def get_ELM_timebase(shot: int, time: float = None, edition: int = 0,
                     exp: str = 'augd'):
    """
    Give the ELM onset and duration times

    Jose Rueda Rueda - jrrueda@us.es
    Pablo Oyola - pablo.oyola@ipp.mpg.de

    :param  shot: shot number
    :returns tELM: Dictionary with:
        -# t_onset: The time when each ELM starts
        -# dt: the duration of each ELM
        -# n: The number of ELMs
    """
    # --- Open the AUG shotfile
    sfo = sf.SFREAD(shot, 'ELM', edition=edition, experiment=exp)

    if not sfo.status:
        raise Exception('Cannot access shotfile %s:#%05d:ELM' % (exp, shot))
    tELM = {
        't_onset':  sfo('tELM'),
        'dt': sfo('dt_ELM'),
        'energy': sfo('ELMENER'),
        'f_ELM': sfo('f_ELM')
    }

    if time is not None:
        # If one time is given, we find the nearest ELM.
        time = np.atleast_1d(time)
        if len(time) == 1:
            t0 = np.abs(tELM['t_onset'] - time[0]).argmin()
            tELM = {
                't_onset': np.array((tELM['t_onset'][t0],)),
                'dt': np.array((tELM['dt'][t0],)),
                'energy': np.array((tELM['energy'][t0],)),
                'f_ELM': np.array((tELM['f_ELM'][t0],))
            }
        elif len(time) == 2:
            t0, t1 = np.searchsorted(tELM['t_onset'], time)
            t1 = min(len(tELM['t_onset']), t1+1)
            tELM = {
                't_onset': tELM['t_onset'][t0:t1],
                'dt': tELM['dt'][t0:t1],
                'energy': tELM['energy'][t0:t1],
                'f_ELM': tELM['f_ELM'][t0:t1]
            }

        else:
            tidx = [np.abs(tELM['t_onset'] - time_val).argmin()
                    for time_val in time]

            tELM = {
                't_onset': tELM['t_onset'][tidx],
                'dt': tELM['dt'][tidx],
                'energy': tELM['energy'][tidx],
                'f_ELM': tELM['f_ELM'][tidx]
            }

    tELM['n'] = len(tELM['t_onset'])

    return tELM


# -----------------------------------------------------------------------------
# --- ICRH
# -----------------------------------------------------------------------------
def get_icrh(shot: int, coupled: bool = False):
    """
    Get the ICRH power time trace
    :param shot: shot number
    :type shot: int
    :param coupled: If true, the cupled power will be returned, defaults False
    :type coupled: bool, optional

    :return: xArray containing the power
    :rtype: TYPE

    """
    ICP = sf.SFREAD(shot, 'ICP')
    if coupled:
        name = 'PICRNc'
    else:
        name = 'PICRN'
    pICRH = ICP(name) / 1.0e6
    tICRH = ICP.gettimebase(name)
    output = xr.DataArray(pICRH, dims='t', coords={'t': tICRH})
    output.attrs['units'] = 'MW'
    output.attrs['long_name'] = '$P_{ICRH}$'
    output.attrs['shot'] = shot
    output.attrs['diag'] = 'ICP'
    output.attrs['signal'] = name
    return output

# -----------------------------------------------------------------------------
# --- OTHER USEFUL SIGNALS.
# -----------------------------------------------------------------------------
def get_neutron(shot: int, time: float = None, exp: str = 'augd', 
                xArrayOutput: bool = False):
    """"
    Reads the neutron rates for a given AUG shot in a given time interval.

    Pablo Oyola - poyola@us.es

    @param shot: shotnumber to get the neutron rate.
    @param time: time where to retrieve the neutron rate. 
        - If None, all the time points are return
        - If a single time point is provided, then it is interpolated to
         that point
        - If two time points are given, then the signal is returned in that 
        time range.
        - If a time basis is provided, then the windowed average is used.
    """

    if time is None:
        tBegin = None
        tEnd   = None
        time_mode = 0
    else:
        time = np.atleast_1d(time)
        if len(time) == 1:
            tBegin = time - 1.0e-2
            tEnd   = time + 1.0e-2
            time_mode = 1

        elif len(time) == 2:
            tBegin, tEnd = time.tolist()
            time_mode = 2
        else:
            tBegin, tEnd = time.min(), time.max()
            time_mode = 3
    
    # Reading the data from the shotfile.
    time_out, data = get_signal_generic(shot=shot, diag='ENR', 
                                        signame='NRATE_II', exp=exp,
                                        tBegin=tBegin, tEnd=tEnd)
    
    if time_mode == 1:
        data = np.interp(time, time_out, data)
        time_out = time
    elif time_mode == 2:
        bins = len(time)
        ranges = [tBegin, tEnd]
        data = np.histogram(time_out, bins=bins, range=ranges,
                            weights=data)
        norm = np.histogram(time_out, bins=bins, range=ranges)

        data /= norm
        time_out = np.linspace(tBegin, tEnd, bins)

    # We also compute the total neutron emitted during the whole time range
    if time_mode != 1:
        tot = np.trapz(data, time_out)
    else:
        tot = None

    # Preparing the output.
    if xArrayOutput:
        time = xr.DataArray(time_out, attrs={'units': 's'},
                            name='Time')

        attrs = { 'units': 'neutron/s',
                  'diag': 'ENR',
                  'exp': exp,
                  'total': tot
                }
        output = xr.DataArray(data, coords=(time,), 
                              dims=(time.name,), 
                              attrs=attrs, name='Neutron rate')
    else:
        output = { 'time': time,
                   'data': data,
                   'units': 'neutron/s',
                   'diag': 'ENR',
                   'exp': exp,
                   'base_units': 's',
                   'total': tot
                 }

    return output
    

def get_neutron_history(shot_0: int, shot_1: int):
    """
    Compute for all the shot in the range of shots given, the evolution of the
    neutron emitted.

    Pablo Oyola - poyola@us.es

    @param shot_0: starting shot.
    @param shot_1: ending shot to return.
    """

    if shot_0 == shot_1:
        shots = (shot_0,)
    else:
        shots = np.arange(shot_0, shot_1+1)
    
    shots = np.array(shots, dtype=int)
    nshots = len(shots)

    # Calling systematically the database to get the total 
    # neutron rate.
    neutron_evol = np.zeros_like(shots)
    abstime = np.zeros_like(shots, dtype=object)
    flags = np.ones_like(shots, dtype=bool)
    for ii, ishot in enumerate(shots):
        try:
            tmp = get_neutron(ishot, xArrayOutput=True)
            neutron_evol[ii] = tmp.total
        except errors.DatabaseError:
            neutron_evol[ii] = 0.0
            flags[ii] = False
        
        jou = sf.journal.getEntryForShot(int(ishot))
        if 'upddate' not in jou:
            abstime[ii] = np.nan
            flags[ii] = False
        else:
            abstime[ii] = datetime.strptime(jou['upddate'].decode(), 
                                            '%Y-%m-%d %H:%M:%S\n')

    # Removing useless points
    abstime = abstime[flags]
    neutron_evol = neutron_evol[flags]
    time = np.zeros_like(neutron_evol, dtype=float)

    # Preparing the output:
    for ii in range(len(neutron_evol)):
        time[ii] = (abstime[ii] - abstime[0]).total_seconds()

    sortidx = np.argsort(time)
    time = time[sortidx]
    neutron_evol = neutron_evol[sortidx]

    return time, neutron_evol
    
