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
# --- FILD 4 TRAJECTORIES
# -----------------------------------------------------------------------------
def load_FILD4_trajectory(shot, path=pa.FILD4_trayectories):
    """
    Load FILD4 trayectory

    Jose Rueda: jrrueda@us.es

    Note: This is a temporal function, in the future will be replaced by one to
    load trayectories for an arbitrary in-shot mobile FILD

    @param shot: Shot number to load
    @param path: Path to the main folder with FILD4 trajectories
    """
    # --- Load the power supply output data
    shot_str = str(shot)
    try:
        file = os.path.join(path, 'output_raw', shot_str[0:2],
                            'FILD_MDRS_' + shot_str + '.txt')
        print('Looking for file: ', file)
        data = np.loadtxt(file, skiprows=1)
        # Delete the last line of the data because is always zero
        dat = np.delete(data, -1, axis=0)
        # Delete points where PS output is zero. This **** instead of giving
        # as ouput the points where the trajectory was requested, it always
        # gives as ouput a given number of rows, and set to zero the non used
        # ones...
        fi = dat[:, 2] < 1
        fv = dat[:, 4] < 1
        flags = (fv * fi).astype(np.bool)
        PSouput = {
            'V_t_obj': dat[~flags, 0] / 1000.0,
            'V_obj': dat[~flags, 1],
            'I_t': dat[~flags, 2] * 1.0e-9,
            'I': dat[~flags, 3],
            'V_t': dat[~flags, 4] * 1.0e-9,
            'V': dat[~flags, 5]
        }
    except OSError:
        print('File with power supply outputs not found')
        PSouput = None
    # --- Load the reconstructed trajectory
    try:
        file = os.path.join(path, 'output_processed', shot_str[0:2],
                            shot_str + '.txt')
        print('Looking for file: ', file)
        data = np.loadtxt(file, skiprows=2, delimiter=',')
        insertion = {
            't': data[:, 0],
            'insertion': data[:, 1],
        }
    except OSError:
        print('File with trajectory not found')
        insertion = None

    return {'PSouput': PSouput, 'insertion': insertion}


def plot_FILD4_trayectory(shot, PS_output=False, ax=None, ax_PS=None,
                          line_params={}, line_params_PS={}, overlay=False,
                          unit='cm'):
    """
    Plot FILD4 trayectory

    Jose Rueda: jrrueda@us.es

    Note: this is in beta phase, improvement suggestions are wellcome

    @param shot: shot you want to plot
    @param PS_output: flag to plot the output of the power supply
    @param ax: axes where to plot the trajectory. If none, new axis will be
               created
    @param ax_PS: Array of two axes where we want to plot the PS data. ax_PS[0]
                  will be for the voltaje while ax_PS[1] for the intensity. If
                  None, new axis  will be created
    @param line_params: Line parameters for the trajectory plotting
    @param line_params_PS: Line parameters for the PS plots. Note: same dict
                           will be used for the Voltaje and intensity plots, be
                           carefull if you select the 'color'
    @param overlay: Flag to overlay the trayectory over the current plot. The
                    insertion will be plotted in arbitrary units on top of it.
                    ax input is mandatory for this
    """
    line_options = {
        'label': '#' + str(shot),
    }
    line_options.update(line_params)
    # ---
    factor = {
        'cm': 100.0,
        'm': 1.0,
        'inch': 100.0 / 2.54,
        'mm': 1000.0
    }
    # --- Load the position
    position = load_FILD4_trajectory(shot)

    # --- Plot the position
    if ax is None:
        fig, ax = plt.subplots()
    if overlay:  # Overlay the trayectory in an existing plot:
        print('Sorry, still not implemented')
    else:
        ax.plot(position['insertion']['t'],
                factor[unit] * position['insertion']['insertion'],
                **line_options)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Insertion [' + unit + ']')
        ax.set_xlim(0, 1.1 * position['insertion']['t'].max())
        ymax = 1.1 * factor[unit] * position['insertion']['insertion'].max()
        ax.set_ylim(0, ymax)
        ax.legend()

    # --- Plot the PS output
    if PS_output:
        if ax_PS is None:
            fig2, ax_PS = plt.subplots(2, 1, sharex=True)
        # Plot the voltage
        ax_PS[0].plot(position['PSouput']['V_t_obj'],
                      position['PSouput']['V_obj'],
                      label='Objective')
        ax_PS[0].plot(position['PSouput']['V_t'],
                      position['PSouput']['V'],
                      label='Real')
        ax_PS[0].set_ylabel('Voltage [V]')
        ax_PS[0].legend()
        ax_PS[1].plot(position['PSouput']['I_t'],
                      position['PSouput']['I'])
        ax_PS[1].set_ylabel('Intensity [A]')
    plt.show()


# -----------------------------------------------------------------------------
# --- ELMs
# -----------------------------------------------------------------------------
def get_ELM_timebase(shot: int, time:float=None, edition: int=0,
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
