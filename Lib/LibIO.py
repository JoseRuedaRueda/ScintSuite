"""
Input/output library

Contains a miscellany of routines related with the different diagnostics, for
example the routine to read the scintillator efficiency files, common for all
"""

import numpy as np
import time
import Lib.LibTimeTraces as sstt
import Lib.LibParameters as sspar
import Lib.LibMap as ssmapping
from scipy.io import netcdf
from Lib.version_suite import version
from Lib.LibPaths import Path
import os
import warnings
import tkinter as tk                       # To open UI windows
paths = Path()


# -----------------------------------------------------------------------------
# --- Checking and asking for files
# -----------------------------------------------------------------------------
def check_save_file(file):
    """
    Check if the file exist, if yes, open a window to select a new filename

    Jose Rueda: jrrueda@us.es

    @param file: the filename to test if exist

    @return out: fir file does not exist, return the same filename, if exist,
    open a window for the user to select the filename he wants
    """
    if not os.path.isfile(file):
        out = file
    else:
        warnings.warn('The file exist!!! you can choose the new name',
                      category=UserWarning)
        dir, name = os.path.split(file)
        dummy, ext = os.path.splitext(name)
        out = ask_to_save(dir, ext)
    return out


def check_open_file(file):
    """
    Check if the file exist, if no, open a window to select a new filename

    Jose Rueda: jrrueda@us.es

    @param file: the filename to test if exist

    @return out: fir file exists, return the same filename, if does not,
    open a window for the user to select the filename he wants
    """
    if os.path.isfile(file):
        out = file
    else:
        warnings.warn('The file does not exist!!! you can choose the new name',
                      category=UserWarning)
        dir, name = os.path.split(file)
        dummy, ext = os.path.splitext(name)
        out = ask_to_open(dir, ext)
    return out


def ask_to_save(dir=None, ext=None):
    """
    Open a dialogue to choose the file to be saved

    Jose Rueda: jrrueda@us.es

    @param dir: Initial directory to direct the GUI to open the file, if none,
    just the current directory will be opened
    @param ext: extension for filter the possible options, if none, no filter
    will be applied

    @return out: the filename selected by the user
    """
    root = tk.Tk()
    root.withdraw()   # To close the window after the selection
    out = tk.filedialog.asksaveasfilename(initialdir=dir, defaultextension=ext,
                                          filetypes=sspar.filetypes)
    return out


def ask_to_open(dir=None, ext=None):
    """
    Open a dialogue to choose the file to be opened

    Jose Rueda: jrrueda@us.es

    @param dir: Initial directory to direct the GUI to open the file, if none,
    just the current directory will be opened
    @param ext: extension for filter the possible options, if none, no filter
    will be applied

    @return out: the filename selected by the user
    """
    root = tk.Tk()
    root.withdraw()   # To close the window after the selection
    out = tk.filedialog.askopenfilename(initialdir=dir, defaultextension=ext,
                                        filetypes=sspar.filetypes)
    return out


# -----------------------------------------------------------------------------
# --- General reading
# -----------------------------------------------------------------------------
def read_variable_ncdf(file, varNames, human=True):
    """
    Read a variable from a  netCDF file

    Jose Rueda Rueda: jrrueda@us.es

    @param file: path to the .CDF file to be opened
    @param varNames: list with the variable names
    @param human: this is just a flag, if true, if the file is not found, it
    open a window for the user to select the proper file, if False, we assume
    that this functions used inside some kind of automatic way by some script
    so just give an exception
    @return data: values of the variable
    @return units: physical units
    @return long_name: name of the variable
    """
    # Try to locate the filename
    if human:
        file = check_open_file(file)
    # see if the inputs is a list/tupple or not
    try:
        varNames.append
        listNames = varNames.copy()
    except AttributeError:
        listNames = []
        listNames.append(varNames)
    out = []
    varfile = netcdf.netcdf_file(file, 'r', mmap=False).variables
    for ivar in range(len(listNames)):
        dummy = varfile[listNames[ivar]]
        out.append(dummy)
        del dummy
    return out


def print_netCDF_content(file):
    """
    Print the list of variables in a netcdf file

    Jose Rueda Rueda: jrrueda@us.es

    @param file: full path to the netCDF file
    """
    varfile = netcdf.netcdf_file(file, 'r', mmap=False).variables
    print('%20s' % ('Var name'),  '|  Description  | Dimensions')
    for key in sorted(varfile.keys()):
        print('%20s' % (key), varfile[key].long_name, varfile[key].dimensions)


# -----------------------------------------------------------------------------
# --- ROIs
# -----------------------------------------------------------------------------
def save_mask(mask, filename=None, nframe=None, shot=None, frame=None):
    """
    Save the mask used in timetraces and remap calculations

    Jose Rueda: jrrueda@us.es

    @param mask: Bynary mask to be saved
    @param filename: Name of the file
    @param nframe: the frame number used to define the roi (optional)
    @param shot: Shot number of the video used to define the roi (optional)
    @param frame: Frame used to define the roi
    """
    # --- Check if the file exist
    if filename is None:    # If no file is given, just take the 'standard'
        # name = 'mask.nc'
        # filename = os.path.join(paths.Results, name)
        filename = ask_to_save()
        if filename == '' or filename == ():
            print('You canceled the export')
            return
    else:  # Check if the file is there, to avoid overwriting
        filename = check_save_file(filename)

    nnx, nny = mask.shape
    print('Saving results in: ', filename)
    with netcdf.netcdf_file(filename, 'w') as f:
        f.history = 'Done with version ' + version

        f.createDimension('number', 1)
        f.createDimension('nx', nnx)
        f.createDimension('ny', nny)
        nx = f.createVariable('nx', 'i', ('number', ))
        nx[:] = nnx
        nx.units = ' '
        nx.long_name = 'Number of pixels in the first dimension'

        ny = f.createVariable('ny', 'i', ('number', ))
        ny[:] = nny
        ny.units = ' '
        ny.long_name = 'Number of pixels in the second dimension'

        if shot is not None:
            shott = f.createVariable('shot', 'i', ('number', ))
            shott[:] = int(shot)
            shott.units = ' '
            shott.long_name = 'shot number'

        if nframe is not None:
            nnframe = f.createVariable('nframe', 'i', ('number', ))
            nnframe[:] = nframe
            nnframe.units = ' '
            nnframe.long_name = 'Frame number used to define the mask'

        if frame is not None:
            frames = f.createVariable('frame', 'i', ('nx', 'ny'))
            frames[:] = frame.squeeze()
            frames.units = ' '
            frames.long_name = 'Counts'

        m = f.createVariable('mask', 'i', ('nx', 'ny'))
        m[:] = mask
        m.units = ' '
        m.long_name = 'Binary mask'


# -----------------------------------------------------------------------------
# --- TimeTraces
# -----------------------------------------------------------------------------
def read_timetrace(file=None):
    """
    Read a timetrace created with the suite

    Jose Rueda: jrrueda@us.es

    Note: If just a txt file is passed as input, only the trace will be loaded,
    as the mask and extra info are not saved in the txt. netcdf files are
    preferred

    @todo: implement netcdf part

    @param filename: full path to the file to load, if none, a window will
    pop-up to do this selection
    """
    if file is None:
        file = ' '
        file = check_open_file(file)
        if file == '':
            raise Exception('You must select a file!!!')
    TT = sstt.TimeTrace()
    TT.time_base, TT.sum_of_roi, TT.mean_of_roi, TT.std_of_roi =\
        np.loadtxt(file, skiprows=2, unpack=True, delimiter='   ,   ')
    return TT


# -----------------------------------------------------------------------------
# --- Calibration
# -----------------------------------------------------------------------------
def read_calibration(file=None):
    """
    Read a the used calibration from a remap netCDF file

    Jose Rueda: jrrueda@us.es

    @param filename: full path to the file to load, if none, a window will
    pop-up to do this selection
    """
    if file is None:
        file = ' '
        file = check_open_file(file)
        if file == '':
            raise Exception('You must select a file!!!')
    print('-.-. .- .-.. .. -... .-. .- - .. --- -.')
    cal = ssmapping.CalParams()
    list = ['xshift', 'yshift', 'xscale', 'yscale', 'deg']
    var = read_variable_ncdf(file, list, human=False)
    cal.xshift = var[0].data[:]
    cal.yshift = var[1].data[:]
    cal.xscale = var[2].data[:]
    cal.yscale = var[3].data[:]
    cal.deg = var[4].data[:]
    return cal


# -----------------------------------------------------------------------------
# --- Tomography
# -----------------------------------------------------------------------------
def save_FILD_W(W4D, grid_p, grid_s, W2D=None, filename: str = None,
                efficiency: bool = False):
    """
    Save the FILD_W to a .netcdf file

    Jose rueda: jrrueda@us.es

    @todo: include the units of W

    @param W4D: 4D Weight matrix to be saved
    @param grid_p: grid at the pinhole
    @param grid_s: grid at the scintillator
    @param W2D: optional, 2D contraction of W4D
    @param filename: Optional filename to use, if none, it will be saved at the
    results file with the name W_FILD_<date, time>.nc
    @param efficiency: bool to save at the file, to indicate if the efficiency
    was used in the calculation of W
    """
    print('.... . .-.. .-.. ---')
    if filename is None:
        a = time.localtime()
        name = 'W_FILD_' + str(a.tm_year) + '_' + str(a.tm_mon) + '_' +\
            str(a.tm_mon) + '_' + str(a.tm_hour) + '_' + str(a.tm_min) +\
            '.nc'
        filename = os.path.join(paths.Results, name)
    else:
        filename = check_save_file(filename)
    print('Saving results in: ', filename)
    with netcdf.netcdf_file(name, 'w') as f:
        f.history = 'Done with version ' + version

        # --- Save the pinhole grid
        f.createDimension('number', 1)
        nr_pin = f.createVariable('nr_pin', 'i', ('number', ))
        nr_pin[:] = grid_p['nr']
        nr_pin.units = ' '
        nr_pin.long_name = 'Number of points in r, pinhole'

        np_pin = f.createVariable('np_pin', 'i', ('number', ))
        np_pin[:] = grid_p['np']
        np_pin.units = ' '
        np_pin.long_name = 'Number of points in pitch, pinhole'

        f.createDimension('np_pin', grid_p['np'])
        p_pin = f.createVariable('p_pin', 'float64', ('np_pin', ))
        p_pin[:] = grid_p['p']
        p_pin.units = 'degrees'
        p_pin.long_name = 'Pitch values, pinhole'

        f.createDimension('nr_pin', grid_p['nr'])
        r_pin = f.createVariable('r_pin', 'float64', ('nr_pin', ))
        r_pin[:] = grid_p['r']
        r_pin.units = 'cm'
        r_pin.long_name = 'Gyroradius values, pinhole'

        # --- Save the scintillator grid
        nr_scint = f.createVariable('nr_scint', 'i', ('number', ))
        nr_scint[:] = grid_s['nr']
        nr_scint.units = ' '
        nr_scint.long_name = 'Number of points in r, scint'

        np_scint = f.createVariable('np_scint', 'i', ('number', ))
        np_scint[:] = grid_s['np']
        np_scint.units = ' '
        np_scint.long_name = 'Number of points in pitch, scint'

        f.createDimension('np_scint', grid_s['np'])
        p_scint = f.createVariable('p_scint', 'float64', ('np_scint', ))
        p_scint[:] = grid_s['p']
        p_scint.units = 'degrees'
        p_scint.long_name = 'Pitch values, scint'

        f.createDimension('nr_scint', grid_s['nr'])
        r_scint = f.createVariable('r_scint', 'float64', ('nr_scint', ))
        r_scint[:] = grid_s['r']
        r_scint.units = 'cm'
        r_scint.long_name = 'Gyroradius values, scint'

        # Save the 4D W
        W = f.createVariable('W4D', 'float64',
                             ('nr_scint', 'np_scint', 'nr_pin', 'np_pin'))
        W[:, :, :, :] = W4D
        W.units = 'a.u.'
        W.long_name = 'Intrument function'

        if W2D is not None:
            f.createDimension('n_scint', grid_s['nr'] * grid_s['np'])
            f.createDimension('n_pin', grid_p['nr'] * grid_p['np'])
            W2 = f.createVariable('W2D', 'float64', ('n_scint', 'n_pin'))
            W2[:, :] = W2D
            W2.units = 'a.u.'
            W2.long_name = 'Instrument function ordered in 2D'

        print('Make sure you are setting the efficiency flag properly!!!')
        if efficiency:
            eff = f.createVariable('efficiency', 'i', ('number',))
            eff[:] = int(efficiency)
            eff.units = ' '
            eff.long_name = '1 Means efficiency was activated to calculate W'
    print('-... -.-- . / -... -.-- .')
