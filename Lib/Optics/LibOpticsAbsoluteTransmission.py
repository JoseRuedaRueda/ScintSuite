"""
Routines to calculate the optic transmission of the system

Jose Rueda Rueda: jrrueda@us.es
"""

import numpy as np
import matplotlib.pyplot as plt
import LibIO as ssio


def read_spectral_response(file: str = None, plot: bool = False):
    """
    Read the camera spectral response

    Jose Rueda: jrrueda@us.es

    Data should follow the example of the PhantomV2512.txt format

    for the future: allow for unit changes (ie, receive the response in units
    different from the A/W)

    @param file: full path to the txt file
    @param plot: bool to plot or not
    @return out: dict containing:
        -# lambda: wavelength [nm]
        -# response: spectral response [A/W]
    """
    if file is None:
        file = ssio.ask_to_open(ext='*.txt')
        if file == '' or file == ():
            print('You canceled the reading')
            return
    else:
        file = ssio.check_open_file(file)

    # read the data
    [x, y] = np.loadtxt(file, skiprows=5, unpack=True)

    # plot
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=2)
        ax.set_xlabel('Wavelength [nm]', fontsize=14)
        ax.set_ylabel('Response [A/W]', fontsize=14)
        fig.show()

    out = {
        'lambda': x,
        'response': y
    }
    return out


def read_sphere_data(file: str = None, plot: bool = False):
    """
    Read integrating sphere data

    Jose Rueda Rueda: jrrueda@us.es

    @param file: full path to the file with the data
    @param plot: flag to decide if we must plot
    """
    # check if the file exist
    if file is None:
        file = ssio.ask_to_open(ext='*.txt')
        if file == '' or file == ():
            print('You canceled the reading')
            return
    else:
        file = ssio.check_open_file(file)
    # read the data
    [x, y] = np.loadtxt(file, skiprows=1, unpack=True)

    # plot
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=2)
        ax.set_xlabel('Wavelength [nm]', fontsize=14)
        ax.set_ylabel('spectral radiance [W/m**2/nm/sr]', fontsize=14)
        fig.show()

    out = {
        'lambda': x,
        'spectrum': y
    }
    return out
