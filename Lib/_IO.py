"""
Input/output library

Contains a miscellany of routines related with the different diagnostics, for
example the routine to read the scintillator efficiency files, common for all
"""
import time
import os
import pickle
import f90nml
import logging
import tarfile
import json
import numpy as np
import xarray as xr
import tkinter as tk                       # To open UI windows
import Lib._TimeTrace as sstt
import Lib._Parameters as sspar
import Lib.errors as errors
import Lib.version_suite as ver
from scipy.io import netcdf
from Lib._Mapping._Calibration import CalParams
from Lib.version_suite import version
from Lib._Paths import Path
from Lib._Video._FILDVideoObject import FILDVideo
from Lib._Video._INPAVideoObject import INPAVideo


# Initialise the objects
logger = logging.getLogger('ScintSuite.IO')
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
        logger.warning('13: The file exist!!! you can choose the new name')
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
        text = '4: The file does not exist!!! you can choose the new name'
        logger.warning(text)
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


def ask_to_open(dir: str = None, ext: str = None, filetype=None):
    """
    Open a dialogue to choose the file to be opened

    Jose Rueda: jrrueda@us.es

    @param dir: Initial directory to direct the GUI to open the file, if none,
    just the current directory will be opened
    @param ext: extension for filter the possible options, if none, no filter
    will be applied
    @filetype: filetype to search for in the folder. If None, only internal
    recognized filetypes will be opened. @see{Lib._Parameters.filetypes}

    @return out: the filename selected by the user
    """
    if filetype is None:
        filetype = sspar.filetypes

    root = tk.Tk()
    root.withdraw()   # To close the window after the selection
    out = tk.filedialog.askopenfilename(initialdir=dir, defaultextension=ext,
                                        filetypes=filetype)
    return out


def ask_to_open_dir(path: str = None):
    """
    Open a dialogue to choose the directory to be opened

    Jose Rueda: jrrueda@us.es

    @param dir: Initial directory to direct the GUI to open the directory,
    if none, just the current directory will be opened

    @return out: the filename selected by the user
    """
    root = tk.Tk()
    root.withdraw()   # To close the window after the selection
    out = tk.filedialog.askdirectory(initialdir=path)
    return out


# -----------------------------------------------------------------------------
# --- General reading
# -----------------------------------------------------------------------------
def read_variable_ncdf(file, varNames, human=True, verbose=True):
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
    if isinstance(varNames, (list, tuple)):
        listNames = varNames
    else:
        listNames = []
        listNames.append(varNames)
    out = []
    varfile = netcdf.netcdf_file(file, 'r', mmap=False).variables
    for ivar in range(len(listNames)):
        if verbose:
            print('Reading: ', listNames[ivar])
        try:
            dummy = varfile[listNames[ivar]]
            out.append(dummy)
            del dummy
        except KeyError:
            print('Var not found')
            out.append(None)
    return out


def print_netCDF_content(file, long_name=False):
    """
    Print the list of variables in a netcdf file

    Jose Rueda Rueda: jrrueda@us.es

    @param file: full path to the netCDF file
    """
    varfile = netcdf.netcdf_file(file, 'r', mmap=False).variables
    if long_name:
        print('%20s' % ('Var name'),  '|  Description  | Dimensions')
        for key in sorted(varfile.keys()):
            print('%20s' % (key), varfile[key].long_name,
                  varfile[key].dimensions)
    else:
        print('%20s' % ('Var name'),  '| Dimensions')
        for key in sorted(varfile.keys()):
            print('%20s' % (key), varfile[key].dimensions)


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


def load_mask(filename):
    """
    Load a binary mask to use in timetraces, remaps or VRT images

    Javier Hidalgo-Salaverri: jhsalaverri@us.es

    @param filename: Name of the netcdf file
    """
    frame = None
    mask = None
    nx = None
    ny = None
    shot = None
    file = netcdf.NetCDFFile(filename, 'r', mmap=False, verbose=False)
    if 'frame' in file.variables.keys():
        frame = file.variables['frame'][:]
    if 'mask' in file.variables.keys():
        mask = file.variables['mask'][:]
        mask = mask.astype(bool)
    if 'nx' in file.variables.keys():
        nx = file.variables['nx'][:]
    if 'ny' in file.variables.keys():
        ny = file.variables['ny'][:]
    if 'shot' in file.variables.keys():
        shot = file.variables['shot'][:]
    file.close()

    return {'frame': frame, 'mask': mask, 'nx': nx, 'ny': ny, 'shot': shot}


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
def read_calibration(file=None, verbose: bool = False):
    """
    Read the used calibration from a remap netCDF file

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
    cal = CalParams()
    list = ['xshift', 'yshift', 'xscale', 'yscale', 'deg', 'xcenter', 'ycenter',
            'c1', 'c2']
    var = read_variable_ncdf(file, list, human=False, verbose=False)
    cal.xshift = var[0].data[:]
    cal.yshift = var[1].data[:]
    cal.xscale = var[2].data[:]
    cal.yscale = var[3].data[:]
    cal.deg = var[4].data[:]
    if var[5] is not None:
        cal.xcenter = var[5].data[:]
        cal.ycenter = var[6].data[:]
        cal.c1 = var[7].data[:]
        cal.c2 = var[8].data[:]
    return cal


# -----------------------------------------------------------------------------
# --- Figures
# -----------------------------------------------------------------------------
def save_object_pickle(file, obj):
    """
    Just a wrapper to the pickle library to write files

    Jose Rueda: jrrueda@us.es
    @param file: full path to the file to write the object
    @param obj: object to be saved, can be a list if you want to save several
    ones
    """
    file = check_save_file(file)
    if file == '':
        print('You cancelled the export')
        return
    print('Saving object in: ', file)
    with open(file, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    return


def load_object_pickle(file):
    """
    Just a wrapper to the pickle library to load files

    Jose Rueda: jrrueda@us.es
    @param file: full path to the file to load the object

    @return: object saved in the file
    """
    file = check_open_file(file)
    print('Reading object from: ', file)
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj


# -----------------------------------------------------------------------------
# --- Camera properties
# -----------------------------------------------------------------------------
def read_camera_properties(file: str):
    """
    Read namelist with the camera properties

    Jose Rueda Rueda: jrrueda@us.es

    @param file: full path to the file to be loaded, or the name of the camera
    we want to load

    @return out: dictionary containing camera properties
    """
    if os.path.isfile(file):
        filename = file
    else:
        filename = os.path.join(paths.ScintSuite, 'Data',
                                'CameraGeneralParameters', file + '.txt')
    if not os.path.isfile(filename):
        print('Looking for: ', filename)
        raise Exception('File not found, revise camera name')

    nml = f90nml.read(filename)

    return nml['camera'].todict()


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
            str(a.tm_mday) + '_' + str(a.tm_hour) + '_' + str(a.tm_min) +\
            '.nc'
        filename = os.path.join(paths.Results, name)
    else:
        filename = check_save_file(filename)
    print('Saving results in: ', filename)
    with netcdf.netcdf_file(filename, 'w') as f:
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


# -----------------------------------------------------------------------------
# --- Remaped videos
# -----------------------------------------------------------------------------
def load_remap(filename, diag='FILD'):
    """
    Load a tar remap file into a video object
    """
    if not os.path.isdir(filename):
        if filename is None:
            filename = ' '
            filename = check_open_file(filename)
        if filename == '' or filename == ():
            raise Exception('You must select a file!!!')
    
        # decompress the file
        dummyFolder = os.path.join(paths.Results, 'tmp')
        os.mkdir(dummyFolder)
        # extract the file
        tar = tarfile.open(filename)
        tar.extractall(path=dummyFolder)
        tar.close()
    else:
        dummyFolder = filename
    if diag.lower() == 'fild':
        vid = FILDVideo(empty=True)  # Open the empty Object
    elif diag.lower() == 'inpa':
        vid = INPAVideo(empty=True)
    else:
        raise errors.NotValidInput('Not suported diagnostic')
    vid.remap_dat = xr.load_dataset(os.path.join(dummyFolder, 'remap.nc'))
    vid.Bangles = xr.load_dataset(os.path.join(dummyFolder, 'Bfield.nc'))
    vid.BField = xr.load_dataset(os.path.join(dummyFolder, 'BfieldAngles.nc'))
    vid.CameraCalibration = \
        read_calibration(os.path.join(dummyFolder, 'CameraCalibration.nc'))
    v = ver.readVersion(os.path.join(dummyFolder, 'version.txt'))
    noise_frame = os.path.join(dummyFolder, 'noiseFrame.nc')
    position = os.path.join(dummyFolder, 'position.json')
    orientation = os.path.join(dummyFolder, 'orientation.json')
    if os.path.isfile(noise_frame):
        vid.exp_dat = xr.Dataset()
        vid.exp_dat['frame_noise'] = xr.load_dataarray(noise_frame)
    with open(os.path.join(dummyFolder, 'metadata.txt'), 'r') as f:
        vid.shot = int(f.readline().split(':')[-1])
        vid.diag_ID = int(f.readline().split(':')[-1])
        vid.diag = diag.upper()
        vid.geometryID = f.readline().split(':')[-1].split('\n')[0].strip()
        vid.settings = {}
        vid.settings['RealBPP'] = int(f.readline().split(':')[-1])
    vid.position = json.load(open(position))
    vid.orientation = \
        {k:np.array(v) for k,v in json.load(open(orientation)).items()}
    logger.info('Remap generated with version %i.%i.%i'%(v[0], v[1], v[2]))
    return vid

def load_FILD_remap(filename: str = None, verbose=True,
                    encoding: str = 'utf-8'):
    """
    Load all the data in a remap file into a video object.

    Jose Rueda Rueda: jrrueda@us.es

    @param filename: netCDF file to read
    @param verbose: flag to print information in the console
    @param encoding: encode to decode the strings

    @return vid: FILDvideoObject with the remap loaded

    Notice: Only the modulus of the field is saved, not the complete field, so
    the dictionary vid.Bfield will not be initialised, call yoursef getBfield
    if you need it
    """
    if filename is None:
        filename = ' '
        filename = check_open_file(filename)
        if filename == '' or filename == ():
            raise Exception('You must select a file!!!')
    vid = FILDVideo(empty=True)  # Open the empty file

    with netcdf.netcdf_file(filename, 'r', mmap=False,) as f:
        var = f.variables.keys()  # list of all available variables
        history = f.history.decode(encoding)   # Version used for the remap
        # Get the version numbers
        if 'versionIDa' in var:  # @Todo: else reading it from history
            va = f.variables['versionIDa'][:]
            vb = f.variables['versionIDb'][:]
            vc = f.variables['versionIDc'][:]
        # Initialise the dictionaries for saving the data
        vid.remap_dat = {'options': {}}
        vid.exp_dat = {}
        vid.Bangles = {}
        vid.BField = {}
        vid.orientation = {}
        vid.position = {}
        vid.settings = {}
        # Read and save the 'standard' data
        vid.shot = f.variables['shot'][0]  # Shot number

        if 'avg_flag' in var:  # this is a don with version 0.8.0 or greater
            vid.remap_dat['options']['use_average'] = \
                bool(f.variables['use_average'][0])
        else:  # This is done with old suite, only exp_remap was possible
            vid.remap_dat['options']['use_average'] = False

        if 'geom_ID' in var:
            vid.geometryID = f.variables['geom_ID'][:]

        vid.remap_dat['tframes'] = f.variables['tframes'][:]

        vid.remap_dat['xaxis'] = f.variables['xaxis'][:]
        vid.remap_dat['xunits'] = f.variables['xaxis'].units.decode(encoding)
        vid.remap_dat['xlabel'] = \
            f.variables['xaxis'].long_name.decode(encoding)

        vid.remap_dat['yaxis'] = f.variables['yaxis'][:]
        vid.remap_dat['yunits'] = f.variables['yaxis'].units.decode(encoding)
        vid.remap_dat['ylabel'] = \
            f.variables['yaxis'].long_name.decode(encoding)

        vid.remap_dat['frames'] = f.variables['frames'][:]

        vid.remap_dat['sprofx'] = f.variables['sprofx'][:]
        vid.remap_dat['sprofxlabel'] = \
            f.variables['sprofx'].long_name.decode(encoding)

        vid.remap_dat['sprofy'] = f.variables['sprofy'][:]
        vid.remap_dat['sprofylabel'] = \
            f.variables['sprofy'].long_name.decode(encoding)

        vid.CameraCalibration = CalParams()
        vid.CameraCalibration.xshift = f.variables['xshift'][:]
        vid.CameraCalibration.yshift = f.variables['yshift'][:]
        vid.CameraCalibration.xscale = f.variables['xscale'][:]
        vid.CameraCalibration.yscale = f.variables['yscale'][:]
        vid.CameraCalibration.deg = f.variables['deg'][:]

        if 't1_noise' in var:
            vid.exp_dat['t1_noise'] = f.variables['t1_noise'][0]
            vid.exp_dat['t2_noise'] = f.variables['t2_noise'][0]
        if 'frame_noise' in var:
            vid.exp_dat['frame_noise'] = f.variables['frame_noise'][:]

        vid.exp_dat['n_pixels_gt_threshold'] = \
            f.variables['n_pixels_gt_threshold'][:]
        vid.exp_dat['threshold_for_counts'] = \
            f.variables['threshold_for_counts'][:]

        vid.Bangles['theta'] = f.variables['theta'][:]
        vid.Bangles['phi'] = f.variables['phi'][:]

        vid.BField['B'] = f.variables['bfield'][:]

        vid.remap_dat['theta'] = f.variables['theta'][:]
        vid.remap_dat['phi'] = f.variables['phi'][:]
        if 'theta_used' in var:
            vid.remap_dat['theta_used'] = f.variables['theta'][:]
            vid.remap_dat['phi_used'] = f.variables['phi'][:]

        vid.remap_dat['options']['rmax'] = f.variables['rmax'][0]
        vid.remap_dat['options']['rmin'] = f.variables['rmin'][0]
        vid.remap_dat['options']['dr'] = f.variables['dr'][0]

        vid.remap_dat['options']['pmax'] = f.variables['pmax'][0]
        vid.remap_dat['options']['pmin'] = f.variables['pmin'][0]
        vid.remap_dat['options']['dp'] = f.variables['dp'][0]

        vid.remap_dat['options']['pprofmax'] = f.variables['pprofmax'][0]
        vid.remap_dat['options']['pprofmin'] = f.variables['pprofmin'][0]
        vid.remap_dat['options']['rprofmin'] = f.variables['rprofmin'][0]
        vid.remap_dat['options']['rprofmax'] = f.variables['rprofmax'][0]

        vid.position['R'] = f.variables['rfild'][0]
        vid.position['z'] = f.variables['zfild'][0]
        if 'phifild' in var:
            vid.position['phi'] = f.variables['phifild'][0]
        else:  # if phi is not present, old remap file, tokamak. Phi irrelevant
            vid.position['phi'] = 0.0

        vid.orientation['alpha'] = f.variables['alpha'][0]
        vid.orientation['beta'] = f.variables['beta'][0]
        if 'gamma' in var:
            vid.orientation['gamma'] = f.variables['gamma'][0]
        else:  # if phi is not present, old remap file, tokamak. Phi irrelevant
            vid.orientation['gamma'] = 0.0

        if 'bits' in var:
            vid.settings['RealBPP'] = f.variables['RealBPP'][0]
        vid.diag = 'FILD'
        if 'diag_ID' in var:
            vid.diag_ID = f.variables['diag_ID'][0]
        else:
            print('Assuming FILD1')
            vid.diag_ID = 1

    if verbose:
        print(history)
        print('Shot: ', vid.shot)
    return vid
