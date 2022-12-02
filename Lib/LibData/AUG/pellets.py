"""Routines to work with the pellets"""
import aug_sfutils as sf
import matplotlib.pyplot as plt


def get_pellets_timeTraces(shot: int = None, plot: bool = False,
                           avg_window: int = 999, remove_offset: bool = True):
    """
    Return the pellets time trace in terms of programmed value and pellets
    that reached the plasma

    :param  shot
    :param  plot
    :param  avg_window: acqusition frequency of this diagnostic is really high.
        Choose the averaging window to reduce the size
    :param  remove_offset: correct the signal offset

    :return dictionary with the time traces
    """

    PID = sf.SFREAD(shot, 'PID')
    raw = {'real_value': PID.getobject('5Co'),
           'prog_value': PID.getobject('Pelarr'),
           'time': PID.getobject('Sio-Time')}
    pellets = {'real_value': [],
               'prog_value': [],
               'time': []}
    # Reduce the number of measured values
    i = 0
    n = 999
    flag = 0
    while flag == 0:
        if i + n > len(raw['time']):
            n = len(raw['time'])-i
            flag = 1
        pellets['real_value'].append(float(raw['real_value'][i:i+n].mean()))
        pellets['prog_value'].append(float(raw['prog_value'][i:i+n].mean()))
        pellets['time'].append(float(raw['time'][i:i+n].mean()))
        i += n

    # Remove the offset
    # offset_real_value = pellets['real_value'].pop()

    if plot:
        fig, ax = plt.subplots()
        ax.plot(pellets['time'], pellets['prog_value'], label='Programmed')
        ax.plot(pellets['time'], pellets['real_value'], label='Real')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Value [-]')

    return pellets
