"""
Routines to calculate the optic transmission of the system

Jose Rueda Rueda: jrrueda@us.es
"""
import os
import math
import logging
import numpy as np
import Lib._IO as ssio
import Lib.errors as errors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import convolve
from Lib._Paths import Path
from Lib._Machine import machine
from Lib._Utilities import distmat
from Lib._Mapping._Common import XYtoPixel
from Lib._SideFunctions import gkern
from tqdm import tqdm
from scipy.sparse import lil_matrix
from scipy.ndimage import gaussian_filter
logger = logging.getLogger('ScintSuite.Optics')

paths = Path(machine)
try:
    from wand.image import Image
except ModuleNotFoundError:
    text = 'Wand image not found, you cannnot apply distortion to figures'
    logger.warning('10: %s' % text)
try:
    import lmfit
except ModuleNotFoundError:
    text = 'lmfit not found. You can not fit'
    logger.warning('10: %s' % text)


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

    :param  file: full path to the txt file
    :param  plot: bool to plot or not
    :return out: dict containing:
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

    :param  file: full path to the file with the data
    :param  plot: flag to decide if we must plot
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

    :param  image: frame with the grid
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


# -----------------------------------------------------------------------------
# --- Finite focusing
# -----------------------------------------------------------------------------
def createFocusMatrix(frame, coef_sigma=1.0,
                      center: tuple = (0, 0)):
    """
    Create the translation matrix for the Finite Focusing of the optics

    Jose Rueda Rueda: jrrueda@us.es

    :param  frame: original frame
    :param  coef_sigma: standard deviation of the focusing, in pixels. Can be
        an array if the focusing is a function of the distance to the optical
        axis.
    :param  center: coordinates (in pixels, of the optical axis)

    Warning, can be extremelly memory consuming, if you want to apply finite
    focus, use the function: defocus
    """
    # convert to numpy array, if the user gave us just a constant sigma
    try:
        len(coef_sigma)
    except TypeError:
        coef_sigma = np.array([coef_sigma])
    # Create the ouput matrix
    n1, n2 = frame.shape

    output = lil_matrix((n1*n2, n1*n2))

    # fill the matrix
    ic = center[0]
    jc = center[1]  # just to save notation

    for i in tqdm(range(n1)):
        for j in range(n2):
            single_matrix = np.zeros((n1, n2))
            single_matrix[i, j] = 1.0
            output[:, j + i*n2] = gaussian_filter(single_matrix, coef_sigma[0]
                                                  ).flatten()
            # d = math.sqrt((i - ic)**2 + (j - jc)**2)
            # sigma = np.polyval(coef_sigma, d)
            # index_distance = distmat(frame, (i, j))
            # output[i, j, ...] = np.e**(-index_distance**2 / 2 / sigma*2)\
            #     / 2 / math.pi / sigma**2
    return output.tocsr().copy()


def defocus(frame, coef_sigma=1.0,
            center: tuple = (0, 0)):
    """
    Defocus an image using assuming a gaussian focusing factor

    Jose Rueda Rueda: jrrueda@us.es

    :param  frame: original frame
    :param  coef_sigma: standard deviation of the focusing, in pixels. Can be
        an array if the focusing is a function of the distance to the optical
        axis. In this case, sigma = np.polyval(coef_sigma, d), where d is the
        distance to the optical axis
    :param  center: coordinates (in pixels, of the optical axis)
    """
    try:
        _ = len(coef_sigma)
        lista = True
    except TypeError:
        lista = False
    # If the signam is just one number, we will call the scipy convolution,
    # which is order of magnetude faster
    
    if not lista:
        kernel = gkern(int(6.0*coef_sigma)+1, sig=coef_sigma)
        output = convolve(frame, kernel, mode='same')
    else:
        # Create the ouput matrix
        n1, n2 = frame.shape
        output = np.zeros((n1, n2))
        # Crete the matrices to get the distances
        col, row = np.meshgrid(np.arange(n2), np.arange(n1))
        # fill the new matrix
        ic = center[0]
        jc = center[1]  # just to save notation
        axis_distance = np.sqrt((col - jc)**2 + (row - ic)**2)
        sigma = np.polyval(coef_sigma, axis_distance)

        for i in range(n1):
            for j in range(n2):
                index_distance = (col - j)**2 + (row - i)**2
                output += np.exp(-index_distance / 2 / sigma[i, j]**2) \
                    / 2 / np.pi / sigma[i, j]**2 * frame[i, j]
    return output


# -----------------------------------------------------------------------------
# --- Transmission factor
# -----------------------------------------------------------------------------
class FnumberTransmission():
    """
    F-number transmission

    Jose Rueda Rueda

    Simple object, just to read the F-number files and store the data, no
    more methods are foreseen in this class for now.
    """

    def __init__(self, file: str = None, diag: str = 'INPA',
                 machine: str = 'AUG', geomID: str = 'iAUG01',
                 fit_model: str = 'poly2'):
        """
        Read the file

        There is 2 ways of found the file to be read:
            -file: give the full name of the file
            -diag, machine, geomID: give the 'info' of the file to look and the
            code will look in the data folder

        :param  file: Full path to the file to open (optional)
        :param  diag: Diagnostic type, to look in the data folder (FILD or INPA)
        :param  machine: machine, to look in the data folder
        :param  geomID: diagnostic geometry ID to look in the data folder
        :param  fit_model: fit function to use in the fitting, default, 2nd
            order polynomial (poly2)

        For the fit, it is assumed that the tramission is always maximum in the
        axis
        """
        if file is None:
            file = os.path.join(paths.ScintSuite, 'Data', 'Calibrations',
                                diag, machine, geomID + '_F_number.txt')
            print(file)
        R, F = np.loadtxt(file, skiprows=12, unpack=True)

        Omega = np.pi / (2 * F)**2
        if fit_model == 'poly2':
            # Notice that b must be clamped, as it
            model = lmfit.models.ParabolicModel()
            params = model.guess(F, x=R)
            params['b'].value = 0
            params['b'].vary = False
            self._fit = model.fit(F, params, x=R)
        self._data = {'r': R, 'F': F, 'Omega': Omega}

    def f_number(self, r):
        """
        Evaluate the f_number coefficient as a given radius in the object plane

        :param : r (can be an array), the radial positions where to evaluate the
            f_number
        """

        return self._fit.eval(x=r)


# -----------------------------------------------------------------------------
# --- Distortion Grid
# -----------------------------------------------------------------------------
class DistortGrid(XYtoPixel):
    """
    Class to generate distortion grids, to calibrate the camera

    J. Rueda-Rueda : jrrueda@us.es
    """
    def __init__(self, x0: float, y0: float, d: float, nx: int = 10,
                 ny: int = 10):
        """
        Generate the grid lines

        :param  x0: x bottom left corner of the grid
        :param  y0: y bottom left corner of the grid
        :param  d: grid spacing
        :param  nx: number of grid points in the x direction
        :param  ny: number of grid points in the y direction

        @TODO include rotation
        """
        xCorner = x0 + np.arange(nx) * d
        yCorner = y0 + np.arange(ny) * d
        # Create the mesh
        XX, YY = np.meshgrid(xCorner, yCorner, indexing='ij')
        XYtoPixel.__init__(self)
        ## Coordinates of the vertex of the scintillator (X,Y,Z).
        self._coord_real['x1'] = XX
        self._coord_real['x2'] = YY

    def plot_real(self, ax=None, line_params: dict = {}):
        """
        Plot the grid in the real space

        :param  ax: axes where to plot, if none, new axis will be created
        :param  line_params: dictionary with the parameters for plt.plot()
        """
        # -- Initialise the plotting options
        line_options = {
            'color': 'k',
        }
        line_options.update(line_params)
        # -- Create the axis
        if ax is None:
            fig, ax = plt.subplots()
        # -- Plot the grid
        nx, ny = self._coord_real['x'].shape
        for ix in range(nx):
            ax.plot(self._coord_real['x'][ix, :],
                    self._coord_real['y'][ix, :], **line_options)
        for iy in range(ny):
            ax.plot(self._coord_real['x'][:, iy],
                    self._coord_real['y'][:, iy], **line_options)

    def plot_pix(self, ax=None, line_params: dict = {}):
        """
        Plot the grid in the real space

        :param  ax: axes where to plot, if none, new axis will be created
        :param  line_params: dictionary with the parameters for plt.plot()
        """
        # -- Initialise the plotting options
        line_options = {
            'color': 'w',
            'label': None,
            'alpha': 0.5
        }
        line_options.update(line_params)
        # -- Create the axis
        if ax is None:
            fig, ax = plt.subplots()
        # -- Plot the grid
        nx, ny = self._coord_pix['x'].shape
        for ix in range(nx):
            ax.plot(self._coord_pix['x'][ix, :],
                    self._coord_pix['y'][ix, :], **line_options)
        for iy in range(ny):
            ax.plot(self._coord_pix['x'][:, iy],
                    self._coord_pix['y'][:, iy], **line_options)
