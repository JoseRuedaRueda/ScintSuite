"""
Contain FILD object

Jose Rueda Rueda: jruedaru@uci.edu
"""
import os
import f90nml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ScintSuite.errors as errors
from s3get_pure import getfiles, clean_up
from ScintSuite._Paths import Path
from ScintSuite._Machine import machine
from ScintSuite.decorators import deprecated
from ScintSuite._Mapping._Calibration import CalParams, readCameraCalibrationDatabase

logger = logging.getLogger('ScintSuite.Data')
paths = Path(machine)

__all__ = ['guessFILDfilename', 'FILD_logbook']
# -----------------------------------------------------------------------------
# %% Paths and files
# -----------------------------------------------------------------------------
homeSuite = os.getenv("ScintSuitePath")
if homeSuite is None:
    homeSuite = os.path.join(os.getenv("HOME"), 'ScintSuite')
# --- Default files:
_cameraDatabase = os.path.join(homeSuite, 'Data', 'Calibrations',
                               'FILD', 'D3D', 'calibration_database.txt')
_geometryDatabase = os.path.join(homeSuite, 'Data',
                                 'Calibrations', 'FILD', 'D3D',
                                 'Geometry_logbook.txt')
_positionDatabase = 'dummy'

_geometrdefault = os.path.join(homeSuite, 'Data',
                               'Calibrations', 'FILD', 'D3D',
                               'GeometryDefaultParameters.txt')
_defaultFILDdata = f90nml.read(_geometrdefault)

def guessFILDfilename(shot: int, diag_ID: int = 1):
    """
    Guess the filename of a video.
    
    This is actually misleading, as it is not guessing, but rather
    downloading the video from the system. This is a very bad practice,
    but is is the easiest way of maintaining compatibility with the
    rest of the suite (machines)

    Jose Rueda Rueda: 

    :param  shot: shot number
    :param  diag_ID: FILD manipulator number

    :return file: the name of the file/folder
    """
    h5_path = '{}_F{}.h5'
    Fn = ['','L','M']  
    folder = os.path.join(homeSuite, 'tmpFILDVideos')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fileName = os.path.join(homeSuite, 
                            'tmpFILDVideos',
                            h5_path.format(shot,Fn[diag_ID]))
    if not os.path.isfile(fileName):
        getfiles(shot,'fild',output_dir=folder,
                  specific_file=h5_path.format(shot,Fn[diag_ID]),
                  force_flag=False)
    return fileName


# -----------------------------------------------------------------------------
# %% Logbook
# -----------------------------------------------------------------------------
class FILD_logbook:
    """
    Contain all geometrical parameters and path information of FILD

    Jose Rueda - jruedaru@uci.edu
    
    :param  cameraFile: path to the ACSII file containing the data
    :param  geometryFile: path to the ACSII file containing the data
    :param  positionFile: path to the excel file containing the data (the
            url poiting to the internet logbook. It can be a path to a local
            excel)

    Public methods:
        - getCameraCalibration(): find the camera parameters
        - getGeomID(): Get the id of the geometry installed in a manipulator
        - getPosition(): get the position of the FILD head
        - getOrientation(): get the orientation of the FILD head
        - getGeomShots(): find all shots where a given collimator was installed
    
    Introduced in version 0.7.2
    Re-written in version 0.7.8
    """

    def __init__(self,
                 cameraFile: str = _cameraDatabase,
                 geometryFile: str = _geometryDatabase,
                 positionFile: str = _positionDatabase,
                 verbose: bool = True):
        """
        Initialise the object.

        Read the three data bases and save them in atributes of the object
        """
        if verbose:
            logger.info('.-.. --- --. -... --- --- -.-')
        # Load the camera database
        self.CameraCalibrationDatabase = \
            readCameraCalibrationDatabase(cameraFile, verbose=verbose,
                                          n_header=3)
        # Load the position database
        # The position database is not distributed with the ScintSuite, so it
        # can happend that it is not available. For that reason, just in case
        # we do a try
        try:
            self.positionDatabase = \
                self._readPositionDatabase(positionFile, verbose=verbose)
            self.flagPositionDatabase = True
        except FileNotFoundError:
            self.flagPositionDatabase = False
            logger.warning('Not found position database, we will use the defaults')
        # Load the geometry database
        self.geometryDatabase = \
            self._readGeometryDatabase(geometryFile, verbose=verbose)
        logger.info('..-. .. -. .- .-.. .-.. -.--')

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
        # Check if there is information on the overheating in the file
        if 'Overheating' not in dummy['FILD1'].keys():
            self.positionDatabaseVersion = 1
        else:
            self.positionDatabaseVersion = 2
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
                logger.debug(dummy)
                data['CalID'].append(int(dummy[0]))
                data['shot1'].append(int(dummy[1]))
                data['shot2'].append(int(dummy[2]))
                data['GeomID'].append(dummy[3])
                data['diag_ID'].append(int(dummy[4]))
        # Transform to pandas
        database = pd.DataFrame(data)
        return database

    def getCameraCalibration(self, shot: int,  FILDid: int = 1,
                             diag_ID: int = None,):
        """
        Get the camera calibration parameters for a shot

        :param  shot: Shot number for which we want the calibration
        :param  cal_type: Type of calibration we want
        :param  diag_ID: ID of the diagnostic we want
        :param  FILDid: alias for diag_ID, to be consistent with the rest of the
            functions in the logbook but also keep retrocompatibility. Notice
            that if diag_ID is not None, diag_ID value will prevail

        :return cal: CalParams() object
        """
        # --- Settings
        if diag_ID is None:
            diag_ID = FILDid
        # --- Select the possible entries
        flags = (self.CameraCalibrationDatabase['shot1'] <= shot) & \
            (self.CameraCalibrationDatabase['shot2'] >= shot) & \
            (self.CameraCalibrationDatabase['cal_type'] == 'PIX') & \
            (self.CameraCalibrationDatabase['diag_ID'] == diag_ID)
        # --- Be sure that there is only one entrance
        n_true = sum(flags)

        if n_true == 0:
            raise errors.NotFoundCameraCalibration(
                'No entry find in the database, revise database')
        elif n_true > 1:
            print('Several entries fulfill the condition')
            print('Possible entries:')
            print(self.data['CalID'][flags])
            raise errors.FoundSeveralCameraCalibration()
        else:
            cal = CalParams()
            row = self.CameraCalibrationDatabase[flags]
            cal.xscale = row.xscale.values[0]
            cal.yscale = row.yscale.values[0]
            cal.xshift = row.xshift.values[0]
            cal.yshift = row.yshift.values[0]
            cal.deg = row.deg.values[0]
            cal.camera = row.camera.values[0]
            if 'c1' in self.CameraCalibrationDatabase.keys():
                cal.c1 = row.c1.values[0]
                cal.c2 = row.c2.values[0]
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
            raise errors.NotFoundGeomID('No entry found in database')
        elif n_true > 1:
            raise errors.FoundSeveralGeomID(
                'More than onw entry found, revise')
        else:
            id = self.geometryDatabase[flags].GeomID.values[0]
        return id

    def getPosition(self, shot: int, FILDid: int = 1, insertion: float = None):
        """
        Get the position of the FILD detector.

        Jose Rueda - jrrueda@us.es

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id
        """
        # Get always the default as a reference:
        geomID = self.getGeomID(shot, FILDid)
        default = self._getPositionDefault(geomID)
        if insertion is None:
            return default
        # If we arrrived this point, we have data
        if FILDid == 1:    
            if shot < 155792:
                phi = np.deg2rad(21.8)
                # r,z coordinates of center of shield
                R = 2.24253 + (0.02765+(insertion-5.618)*0.0254) * np.cos(phi)
                # r coordinate of aperture
                z = -0.674 + 0.0254 * np.sin(phi) * (5.618-insertion)
                # z coordinate of aperture
            elif shot < 168751:
                R = 0.0236127 * insertion + 2.14397
                z = -0.0053842 * insertion - 0.643745
            else:
                R = 0.0234701 * insertion + 2.153225
                z = -0.0096081 * insertion - 0.6128247
        elif FILDid == 2:
            if shot < 155792:
                R = 2.347 + (b - 3.25) * 0.0254
                z = -0.134
            else:
                R = 2.26891 + b * 0.0253754
                Z = -0.134
        position = {
            'R': R,
            'z': z,
            'phi': default['phi'],
        }
        return position

    def getOrientation(self, shot, FILDid):
        """
        Get the orientation

        Note that in AUG the orientation of the diagnostic never changes, so
        this function just return always the default parameters

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id
        """
        geomID = self.getGeomID(shot, FILDid)
        return self._getOrientationDefault(geomID)

    def getGeomShots(self, geomID, maxR: float = None, verbose: bool = True):
        """
        Return all shots in the database position database with a geomID

        :param  geomID: ID of the geometry we are insterested in
        :param  maxR: if present, only shots for which R < maxR will be
            considered. Default values are, for each manipulator:
                1: 2.5 m
                2: 2.2361 m
                5: 1.795 m
        :param  verbose: flag to print in the console the number of shots found
            using that geometry
        """
        if not self.flagPositionDatabase:
            t = 'Not found position database, maybe the path to the logbook'\
                + 'was not given'
            raise errors.NotLoadedPositionDatabase(t)
        # Minimum insertion
        minin = {
            1: 2.5,
            2: 2.2361,
            5: 1.795,
        }
        # get the shot interval for this geometry
        flags_geometry = self.geometryDatabase['GeomID'] == geomID
        n_instalations = sum(flags_geometry)
        if n_instalations == 0:
            raise errors.NotFoundGeomID('Not found geometry? revise input')

        instalations = self.geometryDatabase[flags_geometry]
        if verbose:
            print('This geometry was installed %i times:' % n_instalations)
        for i in range(n_instalations):
            print('From shot %i to %i' % (instalations.shot1.values[i],
                                          instalations.shot2.values[i]))
        if instalations.diag_ID.values[0] == 4:
            raise errors.NotImplementedError('Not for FILD4, sorry')

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

    def getOverheating(self, shot: int, FILDid: int = 1):
        """
        Check if a FILD head was overheated during shots

        :param  shot: shot number (int) or array of shots
        :param  FILDid: Manipulator number

        :return: integer (or array) indicating the overheating
            > -1 : No information on the logbook
            > 0  : No overhating
            > 1  : Slight overheating
            > 2  : Strong overheating
        """
        # Prepare the shot list
        if isinstance(shot, int):
            shotl = np.array([shot])
        elif isinstance(shot, (tuple, list)):
            shotl = np.array(shot)
        elif isinstance(shot, np.ndarray):
            shotl = shot
        else:
            raise errors.NotValidInput('Check shot input')

        # Check the overheating
        overheating = -np.ones(shotl.size, dtype=int)
        dummy = self.positionDatabase['FILD%i' % FILDid]
        if self.positionDatabaseVersion >= 2:
            for ks, s in enumerate(shotl):
                if s in self.positionDatabase['shot'].values:
                    i, = np.where(self.positionDatabase['shot'].values == s)[0]
                    overheating[ks] = dummy['Overheating'].values[i]
        else:
            text = 'This logbook does not contain overheating information'
            logger.warning('')
        return overheating

    def getComment(self, shot: int):
        """
        Get the comment line

        :param  shot: shot number (int) or array of shots

        :return: string containing the comment written by the FILD operator
        """
        # Prepare the shot list
        if isinstance(shot, int):
            shotl = np.array([shot])
        elif isinstance(shot, (tuple, list)):
            shotl = np.array(shot)
        elif isinstance(shot, np.ndarray):
            shotl = shot
        else:
            raise errors.NotValidInput('Check shot input')

        # Check the overheating
        comment = []
        if self.positionDatabaseVersion >= 2:
            dummy = self.positionDatabase['Comments']['Comments']
            for ks, s in enumerate(shotl):
                if s in self.positionDatabase['shot'].values:
                    i, = np.where(self.positionDatabase['shot'].values == s)[0]
                    comment.append(dummy.values[i])
        else:
            text = 'Comments can not be read in this logbook version'
            logger.warning('22: %s' % text)
            comment = ['' for s in shotl]
        return comment
