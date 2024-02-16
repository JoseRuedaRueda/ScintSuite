"""
Contain FILD object.

Jose Rueda: jrrueda@us.es and others
"""

import os
import f90nml
import logging
import numpy as np
import pandas as pd
from urllib.error import HTTPError
from ScintSuite._Paths import Path
from ScintSuite._Machine import machine
from ScintSuite._Mapping._Calibration import CalParams, readCameraCalibrationDatabase
import ScintSuite.LibData.MU.DiagParam as params
paths = Path(machine)
logger = logging.getLogger('ScintSuite.MU.FILD')

# --- Default files:
_cameraDatabase = os.path.join(paths.ScintSuite, 'Data', 'Calibrations',
                               'FILD', 'MU', 'calibration_database.txt')
_geometryDatabase = os.path.join(paths.ScintSuite, 'Data',
                                 'Calibrations', 'FILD', 'MU',
                                 'Geometry_logbook.txt')
_positionDatabase = paths.FILDPositionDatabase
_geometrdefault = os.path.join(paths.ScintSuite, 'Data',
                               'Calibrations', 'FILD', 'MU',
                               'GeometryDefaultParameters.txt')
_defaultFILDdata = f90nml.read(_geometrdefault)


# --- Auxiliar routines to find the path towards the camera files
def guessFILDfilename(shot: int, diag_ID: int = 1):
    """
    Guess the filename of a video

    Jose Rueda Rueda: jrrueda@us.es

    Note Juanfran criteria of organising files is assumed: .../<shot>/...

    :param  shot: shot number
    :param  diag_ID: FILD manipulator number

    :return file: the name of the file/folder
    """
    base_dir = params.FILD[diag_ID-1]['path'](shot)
    extension = params.FILD[diag_ID-1]['extension'](shot)
    prefix = params.FILD[diag_ID-1]['prefix'](shot)
    shot_str = str(shot)
    name = prefix + shot_str + extension
    file = os.path.join(base_dir, name)
    return file


# --- FILD object
class FILD_logbook:
    """
    Contains all geometrical parameters and path information of FILD

    Jose Rueda - jrrueda@us.es
    Lina Velarde - linvelgal@alum.us.es
    Juan Rivero - juan.rivero-rodriguez@ukaea.uk

    Introduced in version 0.8.3

    Public methods:
        - getCameraCalibration(): find the camera parameters
        - getGeomID(): Get the id of the geometry installed in a manipulator
        - getPosition(): get the position of the FILD head
        - getOrientation(): get the orientation of the FILD head
        - getGeomShots(): find all shots where a given collimator was installed
    """

    def __init__(self,
                 cameraFile: str = _cameraDatabase,
                 geometryFile: str = _geometryDatabase,
                 positionFile: str = _positionDatabase,
                 verbose: bool = True):
        """
        Initialise the object

        Read the three data bases and save them in atributes of the object

        :param  cameraFile: path to the ACSII file containing the data
        :param  geometryFile: path to the ACSII file containing the data
        :param  positionFile: path to the excel file containing the data (the
            url poiting to the internet logbook. It can be a path to a local
            excel)
        """
        if verbose:
            print('.-.. --- --. -... --- --- -.-')
        # Load the camera database
        self.CameraCalibrationDatabase = \
            readCameraCalibrationDatabase(cameraFile, verbose=verbose,
                                          n_header=0)
        # Load the position database
        # The position database is not distributed with the ScintSuite, so it
        # can happend that it is not available. For that reason, just in case
        # we do a try
        try:
            self.positionDatabase = \
                self._readPositionDatabase(positionFile, verbose=verbose)
            self.flagPositionDatabase = True
        except (FileNotFoundError, HTTPError):
            self.flagPositionDatabase = False
            print('Not found position database, we will use the defaults')
        # Load the geometry database
        self.geometryDatabase = \
            self._readGeometryDatabase(geometryFile, verbose=verbose)
        print('..-. .. -. .- .-.. .-.. -.--')

    def _readPositionDatabase(self, filename: str, verbose: bool = True):
        """
        Read the excel containing the position database

        :param  filename: path or url pointing to the logbook
        :param  verbose: flag to print some info
        """
        if verbose:
            print('Looking for the position database: ', filename)
        dummy = pd.read_excel(filename, engine='openpyxl', header=[0, 1])
        dummy['shot'] = dummy.Shot.Number.values.astype(int)
        return dummy

    def _readGeometryDatabase(self, filename: str, n_header: int = 3,
                              verbose: bool = True):
        """
        Read the Geometry database

        See the help PDF located at the readme file for a full description of
        each available parameter

        @author Jose Rueda Rueda: jrrueda@us.es

        :param  filename: Complete path to the file with the calibrations
        :param  n_header: Number of header lines (5 in the oficial format)

        :return database: Pandas dataframe with the database
        """
        data = {'CalID': [], 'shot1': [], 'shot2': [],
                'GeomID': [], 'diag_ID': []}

        # Read the file
        if verbose:
            print('Reading Geometry database from: ', filename)
        with open(filename) as f:
            for i in range(n_header):
                dummy = f.readline()
            # Database itself
            for line in f:
                dummy = line.split()
                data['CalID'].append(int(dummy[0]))
                data['shot1'].append(int(dummy[1]))
                data['shot2'].append(int(dummy[2]))
                data['GeomID'].append(dummy[3])
                data['diag_ID'].append(int(dummy[4]))
        # Transform to pandas
        database = pd.DataFrame(data)
        return database

    def getCameraGeneralParameters(self, shot: int, diag_ID: int = 1):
        """
        Read the camera general properties
        """
        calib = self.getCameraCalibration(shot, diag_ID)
        filename = os.path.join(paths.ScintSuite, 'Data',
                                'CameraGenralParameters', 
                                calib.camera.lower()+'.txt')
        return f90nml.read(filename)
    
    def getCameraCalibration(self, shot: int, diag_ID: int = 1):
        """
        Get the camera calibration parameters for a shot

        :param  shot: Shot number for which we want the calibration
        :param  cal_type: Type of calibration we want
        :param  diag_ID: ID of the diagnostic we want

        :return cal: CalParams() object

        @todo: overcome the need of camera inputs
        """
        flags = (self.CameraCalibrationDatabase['shot1'] <= shot) & \
            (self.CameraCalibrationDatabase['shot2'] >= shot) & \
            (self.CameraCalibrationDatabase['cal_type'] == 'PIX') & \
            (self.CameraCalibrationDatabase['diag_ID'] == diag_ID)

        n_true = sum(flags)

        if n_true == 0:
            raise Exception('No entry find in the database, revise database')
        elif n_true > 1:
            print('Several entries fulfill the condition')
            print('Possible entries:')
            print(self.data['CalID'][flags])
            raise Exception()
        else:
            cal = CalParams()
            row = self.CameraCalibrationDatabase[flags]
            cal.xscale = row.xscale.values[0]
            cal.yscale = row.yscale.values[0]
            cal.xshift = row.xshift.values[0]
            cal.yshift = row.yshift.values[0]
            cal.deg = row.deg.values[0]
            cal.camera = row.camera.values[0]
        return cal

    def getGeomID(self, shot: int, FILDid: int = 1):
        """
        Get the geometry id of the FILD manipulator for a given shot

        :param  shot: integer, shot number
        :param  FILDid: manipulator number
        """
        flags = (self.geometryDatabase['shot1'] <= shot) & \
            (self.geometryDatabase['shot2'] >= shot) & \
            (self.geometryDatabase['diag_ID'] == FILDid)
        n_true = sum(flags)
        if n_true == 0:
            raise Exception('No entry found in database')
        elif n_true > 1:
            raise Exception('More than onw entry found, revise')
        else:
            id = self.geometryDatabase[flags].GeomID.values[0]
        return id

    def getPosition(self, shot: int, FILDid: int = 1):
        """
        Get the position of the FILD detector.

        Jose Rueda - jrrueda@us.es

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id
        """
        # Get always the default as a reference:
        geomID = self.getGeomID(shot, FILDid)
        default = self._getPositionDefault(geomID)
        # First check that we have loaded the position logbook
        if not self.flagPositionDatabase:
            logger.warning('Position database not loaded, returning default values')
            return default
        # Get the shot index in the database
        if shot in self.positionDatabase['shot'].values:
            i, = np.where(self.positionDatabase['shot'].values == shot)[0]
            flag = True
        else:
            logger.warning('Shot not found in logbook, returning the default values')
            return default
        # --- Get the postion
        position = {        # Initialise the position
            'R': 0.0,
            'z': 0.0,
            'phi': 0.0,
        }
        dummy = self.positionDatabase['FILD'+str(FILDid)]
        if 'R [m]' in dummy.keys():  # Look for R
            position['R'] = dummy['R [m]'].values[i]
        else:  # Take the default approx value
            print('R not in the logbook, returning default')
            position['R'] = default['R']
        if 'Z [m]' in dummy.keys() and flag:  # Look for Z
            position['z'] = dummy['Z [m]'].values[i]
        else:  # Take the default approx value
            print('Z not in the logbook, returning default')
            position['z'] = default['z']
        if 'Phi [deg]' in dummy.keys() and flag:  # Look for phi
            position['phi'] = dummy['Phi [deg]'].values[i]
        else:  # Take the default approx value
            print('Phi not in the logbook, returning default')
            position['phi'] = default['phi']
        
        return position

    def getOrientation(self, shot, FILDid):
        """
        Get the orientation

        In MU the beta angle can change.

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id
        """
        geomID = self.getGeomID(shot, FILDid)
        default = self._getOrientationDefault(geomID)
        # First check that we have loaded the position logbook
        if not self.flagPositionDatabase:
            logger.warning('Logbook not loaded, returning default values')
            return default
        # Get the shot index in the database
        if shot in self.positionDatabase['shot'].values:
            i, = np.where(self.positionDatabase['shot'].values == shot)[0]
            flag = True
        else:
            logger.warning('Shot not found in logbook, returning the default values')
            return default
        # --- Get the angle
        dummy = self.positionDatabase['FILD'+str(FILDid)]
        if 'Gamma [deg]' in dummy.keys():  # Look for angle
            # Provisional negative sign.
            beta = - dummy['Gamma [deg]'].values[i]
            # Todo: change sign here and in notebook
            print('Provisional comments:')
            print(
                'Please make sure the beta angle in the notebook is contrary to convention')
            print(
                'Convention is: negative when anticlockwise, looked from outside the vessel')
        else:  # Take the default approx value
            print('Beta angle not in the logbook, returning default')
            return default
        default['beta'] = beta
        if default['beta'] == np.nan:
            print('Beta is Nan. Be careful!!')
        return default

    def getAdqFreq(self, shot: int, diag_ID: int = 1):
        """
        Get the adquisition frequency from the database
        Since XIMEA in use, this is deprecated, as the frames per second are stored in the video.

        Jose Rueda - jrrueda@us.es
        Lina Velarde - lvelarde@us.es

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id
        """
        # Get always the default as a reference:
        default = params.FILD[diag_ID-1]['adqfreq'](shot)
        # First check that we have loaded the position logbook
        if not self.flagPositionDatabase:
            logger.warning('Logbook not loaded, returning default values')
            return default
        # Get the shot index in the database
        if shot in self.positionDatabase['shot'].values:
            i, = np.where(self.positionDatabase['shot'].values == shot)[0]
            flag = True
        else:
            logger.warning('Shot not found in logbook, returning the default values')
            return default
        # --- Get the postion
        dummy = self.positionDatabase['FILD'+str(diag_ID)]
        if 'CCDqe freq [Hz]' in dummy.keys():  # Look for adqfreq
            adqfreq = dummy['CCDqe freq [Hz]'].values[i]
        else:  # Take the default approx value
            print('Adquisition frequency not in the logbook, returning default')
            adqfreq = default
        return adqfreq

    def gettTrig(self, shot: int, diag_ID: int = 1):
        """
        Get the triger time from the database

        Jose Rueda - jrrueda@us.es
        Lina Velarde - lvelarde@us.es

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id
        """
        # Get always the default as a reference:
        default = params.FILD[diag_ID-1]['t_trig'](shot)
        # First check that we have loaded the position logbook
        if not self.flagPositionDatabase:
            logger.warning('Logbook not loaded, returning default values')
            return default
        # Get the shot index in the database
        if shot in self.positionDatabase['shot'].values:
            i, = np.where(self.positionDatabase['shot'].values == shot)[0]
            flag = True
        else:
            logger.warning('Shot not found in logbook, returning the default values')
            return default
        # --- Get the postion
        dummy = self.positionDatabase['FILD'+str(diag_ID)]
        if 'CCDqe trigger time [s]' in dummy.keys():  # Look for tTrig
            tTrig = dummy['CCDqe trigger time [s]'].values[i]
        else:  # Take the default approx value
            logger.warning('Trigger time not in the logbook, returning default')
            tTrig = default
        return tTrig

    def getGeomShots(self, geomID, maxR: float = None):
        """
        Return all shots in the database position database with a geomID

        :param  geomID: ID of the geometry we are insterested in. E.g.: MU02.
        :param  maxR: if present, only shots for which R < maxR will be
            considered. Default values are, for each manipulator:
                1: 1.8 m
        """
        # Minimum insertion
        minin = {1: 1.8}
        # get the shot interval for this geometry
        flags_geometry = self.geometryDatabase['GeomID'] == geomID
        n_instalations = sum(flags_geometry)
        if n_instalations == 0:
            raise Exception('Not found geometry? revise input')

        instalations = self.geometryDatabase[flags_geometry]
        print('This geometry was installed %i times:' % n_instalations)
        for i in range(n_instalations):
            print('From shot %i to %i' % (instalations.shot1.values[i],
                                          instalations.shot2.values[i]))
        if instalations.diag_ID.values[0] == 4:
            raise Exception('This not work for FILD4, sorry')

        # Look in the postition in the database
        shots = np.empty(0, dtype=int)
        for i in range(n_instalations):
            shot1 = instalations.shot1.values[i]
            shot2 = instalations.shot2.values[i]
            diag_ID = instalations.diag_ID.values[i]
            FILD_name = 'FILD' + str(diag_ID)
            # find all shot in which FILD measured
            flags1 = (self.positionDatabase.shot >= shot1) &\
                (self.positionDatabase.shot <= shot2)
            # get the positions, to determine if the given FILD was inserted
            if maxR is None:
                maxR = minin[diag_ID]

            flags2 = self.positionDatabase[flags1][FILD_name]['R [m]'] < maxR
            shots = \
                np.append(shots,
                          self.positionDatabase[flags1].shot.values[flags2][:])
        return shots

    def _getPositionDefault(self, geomID: str):
        """Get the default postition of a FILD, given the geometry id"""
        dummy = _defaultFILDdata[geomID]
        return {'R': dummy['r'], 'z': dummy['z'], 'phi': dummy['phi']}

    def _getOrientationDefault(self, geomID: str):
        """Get the default postition of a FILD, given the geometry id"""
        dummy = _defaultFILDdata[geomID]
        output = {
            'alpha': dummy['alpha'],
            'beta': dummy['beta'],
            'gamma': dummy['gamma']
        }
        return output
