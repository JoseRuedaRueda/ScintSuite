"""
Routines to calculate the optic transmission of the system

Jose Rueda Rueda: jrrueda@us.es
"""

import numpy as np
import matplotlib.pyplot as plt
import Lib.LibIO as ssio
try:
    from wand.image import Image
except ModuleNotFoundError:
    print('Wand image not found, you cannnot apply distortion to figures')


# -----------------------------------------------------------------------------
# --- Absolute calibration
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# --- Distortion analysis
# -----------------------------------------------------------------------------
def manual_find_grid_points(image):
    """
    Manually find the edges corners of the calibration grid

    Jose Rueda: jrrueda@us.es

    @param image: frame with the grid
    """
    print('Left mouse: add a point')
    print('Right mouse: remove a point')
    print('Middle mouse: stop input')
    points = plt.ginput(-1, timeout=0, show_cliks=True)
    return points


def distort_image(frame, params: dict = {}):
    """
    Apply distortion to the images
    """
    options = {
        'model': 'WandImage',
        'parameters': {
            'method': 'barrel',
            'arguments': (0.2, 0.1, 0.1, 0.6)
        },
    }
    options.update(params)
    if options['model'] == 'WandImage':
        maximum = frame.max()
        dummy = frame.astype(np.float) / maximum
        img = Image.from_array(dummy)
        img.virtual_pixel = 'transparent'
        img.distort(**options['parameters'])
        output = np.array(img)[:, :, 0].astype(np.float) * maximum / 255

    return output.astype(np.uint)
