"""
Routines to calculate the optic transmission of the system

Jose Rueda Rueda: jrrueda@us.es
"""

import numpy as np
import matplotlib.pyplot as plt
import Lib.LibIO as ssio


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
def edge_detection(image):
    """
    Detect a edges in an image thanks to an adaptive threshold

    José Rueda Rueda: jrrueda@us.es

    Adepted from:
    https://stackoverflow.com/questions/61589953/
    how-to-identify-the-complete-grid-in-the-image-using-python-opencv

    @param image: frame with the distorted grid. Should be RGB or gray
    """
    # If the image is rgb, translated to gray:
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get the threshold
    thresh = cv2.threshold(image, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh


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
