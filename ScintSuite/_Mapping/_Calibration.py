"""Calibration and database objects."""
import logging
import numpy as np
import xarray as xr
import pandas as pd
import ScintSuite.errors as errors
from scipy.io import netcdf                # To export remap data

# Initialise the auxiliary objects
logger = logging.getLogger('ScintSuite.Calibration')


# -----------------------------------------------------------------------------
# ---- Aux functions
# -----------------------------------------------------------------------------
def readCameraCalibrationDatabase(filename: str, n_header: int = 0,
                                  verbose: bool = True):
    """
    Read camera calibration database.

    Jose Rueda: jrrueda@us.es

    :param  filename: Complete path to the file with the calibrations
    :param  n_header: Number of header lines (5 in the oficial format)
    :param  verbose: if true, print some information in the command line
        verbose is no longer used

    :return database: Pandas dataframe with the database
    """
    data = {'CalID': [], 'camera': [], 'shot1': [], 'shot2': [],
            'xshift': [], 'yshift': [], 'xscale': [], 'yscale': [],
            'deg': [], 'cal_type': [], 'diag_ID': [], 'c1': [],
            'xcenter': [], 'ycenter': []}

    # Read the file
    database = pd.read_csv(filename, skiprows=n_header, delim_whitespace=True)
    # There are several ways of calling the colums, as some users have a
    # different format. This is a way to make it compatible with all of them
    acepted = {
        'CalID': ['CalID', 'ID', 'cal_ID', 'calID',
                  '#CalID','#id', '#ID', '#Cal_ID', '#CAL_ID'],
        'camera': ['camera', 'cam', 'cam_1', 'cam_1_1', 'CAMERA'],
        'shot1': ['shot1', 'shot', 'shot_1', 'shot1_1', 'PULSE_1', 'SHOT1', 'PULSE1'],
        'shot2': ['shot2', 'shot_2', 'shot1_2', 'PULSE_2', 'SHOT2', 'PULSE2'],
        'xshift': ['xshift', 'x_shift', 'x_shift_1', 'Xsh', 'XS', 'XSHIFT'],
        'yshift': ['yshift', 'y_shift', 'y_shift_1', 'Ysh', 'YS', 'YSHIFT'], 
        'xscale': ['xscale', 'x_scale', 'x_scale_1', 'Xsc', 'XSCALE', 'X_SCALE'], 
        'yscale': ['yscale', 'y_scale', 'y_scale_1', 'Ysc', 'YSCALE', 'Y_SCALE'],
        'deg': ['DEG', 'Deg'],
        'cal_type': ['CAL', 'CALTYPE', 'CAL_TYPE'],
        'diag_ID': ['diag_ID', 'diagID', 'diag', 'diag_1', 'diag_1_1', 
                    'FILD_ID', 'fild_ID'], 
        'c1': ['c1', 'c1_1', 'c1_2'],
        'xcenter': ['xcenter', 'x_center', 'x_center_1', 'Xc'], 
        'ycenter': ['ycenter', 'y_center', 'y_center_1', 'Yc'],
        'nxpix': ['nxpix', 'nxpix_1', 'nxpix_1_1'],
        'nypix': ['nypix', 'nypix_1', 'nypix_1_1'],
        'type': ['type'],
        }
    for k in database.keys():
        for k2 in acepted.keys():
            if k in acepted[k2]:
                database[k2] = database.pop(k)
    return database


def readTimeDependentCalibration(filename):
    fields = {
        1: ['time', 'xshift', 'yshift', 'xscale', 'yscale',
            'deg', 'c1', 'xcenter', 'ycenter']
    }
    with open(filename) as f:
        logger.info('Reading calibration file: %s'%filename)
        # Read the header lines
        dummy = f.readline()  # Intro line
        dummy = f.readline()  # Date line
        kind, ver = f.readline().split('#')[0].split()
        ver=int(ver)
        camera = f.readline().split('#')[0].strip()
        geomID = f.readline().split('#')[0].strip()
        dummy = f.readline()  # info line
        data = {}
        for k in fields[ver]:
            data[k] = []
        # Database itself
        for line in f:
            dummy = line.split()
            for j, k in enumerate(fields[ver]):
                data[k].append(float(dummy[j]))
    # Transform to xarray
    calibration = xr.Dataset()
    for k in data.keys():
        calibration[k] = xr.DataArray(data[k], dims='t',
                                     coords={'t':data['time']})
    calibration.attrs['Camera'] = camera
    calibration.attrs['geomID'] = geomID
    return calibration



def get_calibration_method(data, shot: int = None, diag_ID: int = None, 
                           method: str = None):
    """
    Give the calibration parameters of a precise database entry

    Jose Rueda Rueda: jrrueda@us.es
    Hannah Lindl: hannah.lindl@ipp.mpg.de

    :param shot: shotnumber of which we want the calibration
    :param camera: name of the camera
    :param diag_ID: ID of the diagnostic
    :param type: calibration method
    """

    flags = np.zeros_like(data['CalID'])
    for i in range(len(flags)):
        if 'type' in data.keys():
            if (data['shot1'][i] <= shot) * \
                    (data['shot2'][i] >= shot) * \
                    (data['diag_ID'][i] == diag_ID) * \
                    (data['type'][i] == method):
                flags[i] = True
        else:
            if (data['shot1'][i] <= shot) * \
                    (data['shot2'][i] >= shot) * \
                    (data['diag_ID'][i] == diag_ID):
                flags[i] = True

    n_true = sum(flags)

    if n_true == 0:
        raise errors.NotFoundCameraCalibration(
            'No entry found in the database, revise it')

    elif n_true > 1:
        print('Several entries fulfill the condition')
        print('Possible entries:')
        print(data['ID'][flags])
        raise errors.FoundSeveralCameraCalibration()

    else:
        dummy = np.argmax(np.array(flags))
        cal = CalParams()
        for ikey in data.keys():
            cal.__dict__[ikey] = data[ikey][dummy]

    return cal


# ------------------------------------------------------------------------------
# ---- Calibration database object
# ------------------------------------------------------------------------------
class CalibrationDatabase:
    """Database of parameter to align the scintillator."""

    def __init__(self, filename: str, n_header: int = 3):
        """
        Read the calibration database, to align the strike maps.

        See database page for a full documentation of each field

        @author Jose Rueda Rueda: jrrueda@us.es

        :param  filename: Complete path to the file with the calibrations
        :param  n_header: Number of header lines
        :return database: Dictionary containing the database information
        """
        ## Name of file with the data
        self.file = filename
        ## Header of the file
        self.header = []
        ## Dictionary with the data from the calibration. See @ref database
        ## for a full description of the meaning of each field
        self.data = {'CalID': [], 'camera': [], 'shot1': [], 'shot2': [],
                     'xshift': [], 'yshift': [], 'xscale': [], 'yscale': [],
                     'deg': [], 'cal_type': [], 'diag_ID': [], 'c1': [],
                     'xcenter': [], 'ycenter': [], 'type': [],
                     'nxpix': [], 'nypix': []}

        # Read the file
        logger.info('Reading Camera database from: %s', filename)
        self.data = readCameraCalibrationDatabase(filename, n_header=n_header)

    def write_database_to_txt(self, file: str = None):
        """
        Write database into a txt.

        If no name is given, the name of the loaded file will be used but a
        'new' will be added. Example: if the file from where the info has
        been loaded is 'calibration.txt' the new file would be
        'calibration_new.txt'. This is just to be save and avoid overwriting
        the original database.

        :param  file: name of the file where to write the results
        """
        if file is None:
            file = self.file[:-4] + '_new.txt'
        with open(file, 'w') as f:
            # Write the header
            for i in range(len(self.header)):
                f.write(self.header[i])
            # Write the database information
            for i in range(len(self.data['CalID'])):
                line = str(self.data['CalID'][i]) + ' ' + \
                       self.data['camera'][i] + ' ' + \
                       str(self.data['shot1'][i]) + ' ' + \
                       str(self.data['shot2'][i]) + ' ' + \
                       str(self.data['xshift'][i]) + ' ' + \
                       str(self.data['yshift'][i]) + ' ' + \
                       str(self.data['xscale'][i]) + ' ' + \
                       str(self.data['yscale'][i]) + ' ' + \
                       str(self.data['deg'][i]) + ' ' + \
                       self.data['cal_type'][i] + ' ' + \
                       str(self.data['diag_ID'][i]) + ' '
                if 'c1' in self.data.keys():
                    line = line \
                        + str(self.data['c1'][i]) + ' '\
                        + str(self.data['xcenter'][i]) + ' '\
                        + str(self.data['ycenter'][i]) + ' ' + '\n'
                else:
                    line = line + '\n'
                f.write(line)
            logger.info('File %s writen', file)

    def get_calibration(self, shot, diag_ID, type=None):
        """
        Give the calibration parameter of a precise database entry.

        :param  shot: Shot number for which we want the calibration
        :param  camera: Camera used
        :param  cal_type: Type of calibration we want
        :param  diag_ID: ID of the diagnostic we want
        :param  typ: type of calibration. Poly or non-poly. Only used
                     when there is type-dependent calibrations.

        :return cal: CalParams() object
        """
        flags = np.zeros(len(self.data['CalID']), dtype=bool)
        for i in range(len(self.data['CalID'])):
            if (self.data['shot1'][i] <= shot) * \
                    (self.data['shot2'][i] >= shot) * \
                    (self.data['diag_ID'][i] == diag_ID):
                flags[i] = True
            if type is not None:
                if self.data['type'][i] != type:
                    flags[i] = flags[i] * False

        n_true = sum(flags)

        if n_true == 0:
            raise errors.NotFoundCameraCalibration(
                'No entry find in the database, revise database')
        elif n_true > 1:
            print('Several entries fulfill the condition')
            print('Possible entries:')
            print(self.data['diag_ID'][flags])
            raise errors.FoundSeveralCameraCalibration()
        else:
            dummy = np.argmax(np.array(flags))
            cal = CalParams()
            for ikey in self.data.keys():
                cal.__dict__[ikey] = self.data[ikey][dummy]

        return cal
    
    def get_nearest_calibration(self, shot: int):
        """
        Return the calibration data from the nearest shot in the database.

        Pablo Oyola - poyola@us.es

        :param shot: Shot number for which we want the calibration.
        """
        # Find the nearest calibration
        shot1 = self.data['shot1']
        shot2 = self.data['shot2']

        idx = np.argmin(np.abs(shot - shot1))
        if shot > shot1[idx]:
            idx = np.argmin(np.abs(shot - shot2))
        cal = CalParams()
        for ikey in self.data.keys():
            cal.__dict__[ikey] = self.data[ikey][idx]

        return cal


# ------------------------------------------------------------------------------
# --- Calibration parameters object
# ------------------------------------------------------------------------------
class CalParams:
    """
    Information to relate points in the camera sensor the scintillator.

    In a future, it will contain the correction of the optical distortion and
    all the methods necessary to correct it.

    :Example of Use:

    >>> # Initialise the calibration object
    >>> import Lib as ss
    >>> import numpy as np
    >>> cal = ss.mapping.CalParams()
    >>> # Fill the calibration
    >>> cal.xscale = cal.yscale = 27.0
    >>> cal.xshift = cal.yshift = 0.0
    >>> cal.deg = 25.0
    """

    def __init__(self):
        """Initialize the class"""
        # To transform the from real coordinates to pixel (see
        # transform_to_pixel())
        ## pixel/cm in the x direction
        self.xscale = 0.0
        ## pixel/cm in the y direction
        self.yscale = 0.0
        ## Offset to align 0,0 of tssiohe sensor with the scintillator
        self.xshift = 0.0
        ## Offset to align 0,0 of the sensor with the scintillator
        self.yshift = 0.0
        ## Rotation angle to transform from the sensor to the scintillator
        self.deg = 0.0
        ## Camera type
        self.camera = ''
        ## First order factor for the distortion
        self.c1 = 0.0
        ## Second order factor for the distortion
        self.c2 = 0.0
        ## X-pixel position of the optical axis
        self.xcenter = 0.0
        ## Y-pixel position of the optical axis
        self.ycenter = 0.0
        ## camera size in x
        self.nxpix = 0
        ## camera size in y
        self.nypix = 0
        ## used calibration method
        self.type = ''

    def print(self):
        """
        Print calibration.
        
        >>> # Assume you have in your workspace a calibration called call
        >>> cal.print()
        """
        for ikey in self.__dict__.keys():
            print('%s:'%ikey, self.__dict__[ikey])

    def __str__(self):
        """
        Print the calibration via the print routine.

        Pablo Oyola - poyola@us.es
        """
        text =  ''
        text =  'xscale: ' + str(self.xscale) + '\n'
        text += 'yscale: ' + str(self.yscale) + '\n'
        text += 'xshift: ' + str(self.xshift) + '\n'
        text += 'yshift: ' + str(self.yshift) + '\n'
        text += 'deg: ' + str(self.deg) + '\n'
        text += 'xcenter: ' + str(self.xcenter) + '\n'
        text += 'ycenter: ' + str(self.ycenter) + '\n'
        text += 'c1: ' + str(self.c1) + '\n'
        text += 'c2: ' + str(self.c2) + '\n'
        text += 'nxpix:' + str(self.nxpix) + '\n'
        text += 'nypix:' + str(self.nypix) + '\n'
        text += 'type:' + str(self.type)
        return text

    def save2netCDF(self, filename):
        """
        Save the calibration in a netCDF file

        :param filename: (str) name of the file for the calibration

        Jose Rueda: jrrueda@us.es

        :Example of use:

        >>> # Assume you have in your workspace a calibration called call
        >>> cal.save2netCDF('calibration.nc')
        """
        logger.info('Saving results in: %s', filename)
        with netcdf.netcdf_file(filename, 'w') as f:
            # Create the dimensions for the variables:
            f.createDimension('number', 1)  # For numbers
            # Save the calibration
            xscale = f.createVariable('xscale', 'float64', ('number', ))
            xscale[:] = self.xscale
            xscale.long_name = 'x scale of the used calibration'

            yscale = f.createVariable('yscale', 'float64', ('number', ))
            yscale[:] = self.yscale
            yscale.long_name = 'y scale of the used calibration'

            xshift = f.createVariable('xshift', 'float64', ('number', ))
            xshift[:] = self.xshift
            xshift.units = 'px'
            xshift.long_name = 'x shift of the used calibration'

            yshift = f.createVariable('yshift', 'float64', ('number', ))
            yshift[:] = self.yshift
            yshift.units = 'px'
            yshift.long_name = 'y shift of the used calibration'

            deg = f.createVariable('deg', 'float64', ('number', ))
            deg[:] = self.deg
            deg.units = 'degrees'
            deg.long_name = 'alpha angle the used calibration'

            xcenter = f.createVariable('xcenter', 'float64', ('number', ))
            xcenter[:] = self.xcenter
            xcenter.units = 'px'
            xcenter.long_name = 'x center of the used calibration'

            ycenter = f.createVariable('ycenter', 'float64', ('number', ))
            ycenter[:] = self.ycenter
            ycenter.units = 'px'
            ycenter.long_name = 'y center of the used calibration'

            c1 = f.createVariable('c1', 'float64', ('number', ))
            c1[:] = self.c1
            c1.long_name = 'c1 (distortion) of the used calibration'

            c2 = f.createVariable('c2', 'float64', ('number', ))
            c2[:] = self.c2
            c2.long_name = 'c2 (distortion) of the used calibration'

            nxpix = f.createVariable('nxpix', 'float64', ('number', ))
            nxpix[:] = self.nxpix
            nxpix.long_name = 'nxpix of the used calibration'

            nypix = f.createVariable('nypix', 'float64', ('number', ))
            nypix[:] = self.nypix
            nypix.long_name = 'nypix of the used calibration'

            camera = f.createVariable('camera', 'S1', ('number', ))
            camera[:] = self.camera
            camera.long_name = 'camera type of the used calibration'

            type = f.createVariable('type', 'S1', ('number', ))
            type[:] = self.type
            type.long_name = 'type of the used calibration'
            
