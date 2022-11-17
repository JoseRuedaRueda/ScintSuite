"""Calibration and database objects."""
import logging
import numpy as np
import xarray as xr
import pandas as pd
import Lib.errors as errors
from scipy.io import netcdf                # To export remap data

# Initialise the auxiliary objects
logger = logging.getLogger('ScintSuite.Calibration')


# -----------------------------------------------------------------------------
# --- Aux functions
# -----------------------------------------------------------------------------
def readCameraCalibrationDatabase(filename: str, n_header: int = 5,
                                  verbose: bool = True):
    """
    Read camera calibration database

    This function is different from the one implemented in the __init__ of the
    old object of the CalibrationDatabase. This one return the database as a
    pandas dataframe to be used in the new logbook, which are considered to be
    the proper way of interacting with the camera calibrations. The
    object CalibrationDatabase (see below) is deprecated for FILD and INPA

    Jose Rueda: jrrueda@us.es

    :param  filename: Complete path to the file with the calibrations
    :param  n_header: Number of header lines (5 in the oficial format)
    :param  verbose: if true, print some information in the command line

    :return database: Pandas dataframe with the database
    """
    data = {'CalID': [], 'camera': [], 'shot1': [], 'shot2': [],
            'xshift': [], 'yshift': [], 'xscale': [], 'yscale': [],
            'deg': [], 'cal_type': [], 'diag_ID': [], 'c1': [],
            'xcenter': [], 'ycenter': []}

    # Read the file
    if verbose:
        print('Reading Camera database from: ', filename)
    with open(filename) as f:
        for i in range(n_header):
            dummy = f.readline()
        # Database itself
        for line in f:
            dummy = line.split()
            data['CalID'].append(int(dummy[0]))
            data['camera'].append(dummy[1])
            data['shot1'].append(int(dummy[2]))
            data['shot2'].append(int(dummy[3]))
            data['xshift'].append(float(dummy[4]))
            data['yshift'].append(float(dummy[5]))
            data['xscale'].append(float(dummy[6]))
            data['yscale'].append(float(dummy[7]))
            data['deg'].append(float(dummy[8]))
            data['cal_type'].append(dummy[9])
            data['diag_ID'].append(int(dummy[10]))
            try:  # If there is distortion information, it would be here
                data['c1'].append(float(dummy[11]))
                data['xcenter'].append(float(dummy[12]))
                data['ycenter'].append(float(dummy[13]))
            except IndexError:
                continue
    # If the c1 and c2 fields are empty, delete them to avoid issues in the
    # pandas dataframe
    if (len(data['c1']) == 0):
        data.pop('c1')
        data.pop('xcenter')
        data.pop('ycenter')
    # Transform to pandas
    database = pd.DataFrame(data)
    return database

def readCcalibrationFile(filename):
    """
    Read the time/position dependent calibration files
    :param  filename: file to be read
    :return: 
    """
    pass

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

# ------------------------------------------------------------------------------
# --- Calibration database object
# ------------------------------------------------------------------------------
class CalibrationDatabase:
    """Database of parameter to align the scintillator."""

    def __init__(self, filename: str, n_header: int = 5):
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
                     'xcenter': [], 'ycenter': []}

        # Read the file
        logger.info('Reading Camera database from: %s', filename)
        with open(filename) as f:
            for i in range(n_header):
                dummy = f.readline()
            # Database itself
            for line in f:
                dummy = line.split()
                self.data['CalID'].append(int(dummy[0]))
                self.data['camera'].append(dummy[1])
                self.data['shot1'].append(int(dummy[2]))
                self.data['shot2'].append(int(dummy[3]))
                self.data['xshift'].append(float(dummy[4]))
                self.data['yshift'].append(float(dummy[5]))
                self.data['xscale'].append(float(dummy[6]))
                self.data['yscale'].append(float(dummy[7]))
                self.data['deg'].append(float(dummy[8]))
                self.data['cal_type'].append(dummy[9])
                self.data['diag_ID'].append(int(dummy[10]))
                try:  # If there is distortion information, it would be here
                    self.data['c1'].append(float(dummy[11]))
                    self.data['xcenter'].append(float(dummy[12]))
                    self.data['ycenter'].append(float(dummy[13]))
                except IndexError:
                    continue
        # If the c1 and c2 fields are empty, delete them to avoid issues in the
        # pandas self.dataframe
        if len(self.data['c1']) == 0:
            self.data.pop('c1')
            self.data.pop('xcenter')
            self.data.pop('ycenter')

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

    def get_calibration(self, shot, diag_ID):
        """
        Give the calibration parameter of a precise database entry.

        :param  shot: Shot number for which we want the calibration
        :param  camera: Camera used
        :param  cal_type: Type of calibration we want
        :param  diag_ID: ID of the diagnostic we want

        :return cal: CalParams() object
        """
        flags = np.zeros(len(self.data['CalID']))
        for i in range(len(self.data['CalID'])):
            if (self.data['shot1'][i] <= shot) * \
                    (self.data['shot2'][i] >= shot) * \
                    (self.data['diag_ID'][i] == diag_ID):
                flags[i] = True

        n_true = sum(flags)

        if n_true == 0:
            raise errors.NotFoundCameraCalibration(
                'No entry find in the database, revise database')
        elif n_true > 1:
            print('Several entries fulfill the condition')
            print('Possible entries:')
            print(self.data['ID'][flags])
            raise errors.FoundSeveralCameraCalibration()
        else:
            dummy = np.argmax(np.array(flags))
            cal = CalParams()
            for ikey in self.data.keys():
                cal.__dict__[ikey] = self.data[ikey][dummy]

        return cal


# ------------------------------------------------------------------------------
# --- Calibration parameters object
# ------------------------------------------------------------------------------
class CalParams:
    """
    Information to relate points in the camera sensor the scintillator.

    In a future, it will contain the correction of the optical distortion and
    all the methods necessary to correct it.
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

    def print(self):
        """
        Print calibration

        Jose Rueda: jrrueda@us.es
        """
        print('xscale: ', self.xscale)
        print('yscale: ', self.yscale)
        print('xshift: ', self.xshift)
        print('yshift: ', self.yshift)
        print('deg: ', self.deg)
        print('xcenter: ', self.xcenter)
        print('ycenter: ', self.ycenter)
        print('c1: ', self.c1)
        print('c2: ', self.c2)

    def save2netCDF(self, filename):
        """
        Save the calibration in a netCDF file

        Jose Rueda: jrrueda@us.es
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
