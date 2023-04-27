"""
Contain routines to interact with the AUG - INPA database

Jose Rueda: jrrueda@us.es
"""

import os
import math
import f90nml
import logging
import numpy as np
import pandas as pd
import Lib.errors as errors
import Lib.LibData.AUG.DiagParam as params
from Lib._Paths import Path
from Lib._Machine import machine
from Lib._Mapping._Calibration import CalParams, readCameraCalibrationDatabase


# --- Auxiliary objects
paths = Path(machine)
logger = logging.getLogger('ScintSuite.INPAlogbook')


# --- Default files:
_cameraDatabase = os.path.join(paths.ScintSuite, 'Data', 'Calibrations',
                               'INPA', 'AUG', 'calibration_database.txt')
_geometryDatabase = os.path.join(paths.ScintSuite, 'Data',
                                 'Calibrations', 'INPA', 'AUG',
                                 'Geometry_logbook.txt')
_INPALogbook = paths.INPALogbook

_geometrdefault = os.path.join(paths.ScintSuite, 'Data',
                               'Calibrations', 'INPA', 'AUG',
                               'GeometryDefaultParameters.txt')
_defaultINPAdata = f90nml.read(_geometrdefault)


# --- Auxiliar routines to find the path towards the camera files
def guessINPAfilename(shot: int, diag_ID: int = 1):
    """
    Guess the filename of a video file/folder

    Jose Rueda Rueda: jrrueda@us.es

    :param  shot: shot number
    :param  diag_ID: INPA number

    :return f: the name of the file/folder
    """
    base_dir = params.INPA[diag_ID-1]['path'](shot)
    extension = params.INPA[diag_ID-1]['extension'](shot)
    shot_str = str(shot)

    if shot < 99999:  # PCO camera, stored in AFS
        name = shot_str + extension
        if shot < 41202:
            f = os.path.join(base_dir, shot_str[0:4], name)
        else:
            f = os.path.join(base_dir, shot_str[0:2], name)
    return f


# --- INPA object
class INPA_logbook:
    """
    Contain all geometrical parameters and camera calibration for INPA

    Jose Rueda - jrrueda@us.es

    Introduced in version 0.8.1

    Public methods:
        - getCameraCalibration(): find the camera parameters for a given shot
        - getGeomID(): Get the id of the geometry installed for a given shot
        - getPositionOrientation(): get the position and orientation of INPA
        - getGeomShots(): find all shots where a given geometry was installed
        - getComment(): Get the comment from the INPA logbook for a given shot

    Private methods:
        - _readExcelLogbook(): Read the INPA excel logbook
        - _readGeometryDatabase(): Read the geometry database
        - _getPositionOrientationDefault(): Return the default pos/orientation

    """

    def __init__(self,
                 cameraFile: str = _cameraDatabase,
                 geometryFile: str = _geometryDatabase,
                 logbookFile: str = _INPALogbook,
                 verbose: bool = True):
        """
        Initialise the object.

        Read the three data bases and save them in atributes of the object

        :param  cameraFile: path to the ACSII file containing the data
        :param  geometryFile: path to the ACSII file containing the data
        :param  logbookFile: path to the excel file containing the data (the
            url poiting to the internet logbook. It can be a path to a local
            excel)
        """
        if verbose:
            print('.-.. --- --. -... --- --- -.-')
        # Side attributes / space reservation
        self.logbookVersion = None
        # Load the camera database
        self.CameraCalibrationDatabase = \
            readCameraCalibrationDatabase(cameraFile, verbose=verbose)
        # Load the position database
        # The position database is not distributed with the ScintSuite, so it
        # can happend that it is not available. For that reason, just in case
        # we do a try
        try:
            self.excelLogbook = \
                self._readExcelLogbook(logbookFile, verbose=verbose)
        except FileNotFoundError:
            logger.warning('29: No excel logbook found')
        # Load the geometry database
        self.geometryDatabase = \
            self._readGeometryDatabase(geometryFile, verbose=verbose)

        print('..-. .. -. .- .-.. .-.. -.--')

    # -------------------------------------------------------------------------
    # --- Reading routines (private)
    # -------------------------------------------------------------------------
    def _readExcelLogbook(self, filename: str, verbose: bool = True):
        """
        Read the excel containing the position database

        :param  filename: path or url pointing to the logbook
        :param  verbose: flag to print some info

        :return excelLogbook: Pandas object with the excell logbook

        Note:
            - Will set the internal variable logbookVersion
        """
        if verbose:
            print('Looking for the position database: ', filename)
        excelLogbook = pd.read_excel(filename, engine='odf', header=[0, 1])
        # dummy['shot'] = dummy.Shot.Shot.values.astype(int)
        self.logbookVersion = 0  # Up to now, this is no usefull as there is
        #                          only one format, but this is a place holder
        #                          for the future, just in case
        return excelLogbook

    def _readGeometryDatabase(self, filename: str, n_header: int = 3,
                              verbose: bool = True):
        """
        Read the Geometry database

        See the help PDF located at the readme file for a full description of
        each available parameter

        Jose Rueda Rueda: jrrueda@us.es

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

    # -------------------------------------------------------------------------
    # --- Get routines
    # -------------------------------------------------------------------------
    def getCameraCalibration(self, shot: int,
                             diag_ID: int = 1, cal_type: str = 'PIX'):
        """
        Get the camera calibration parameters for a shot

        Jose Rueda Rueda: jrrueda@us.es

        :param  shot: Shot number for which we want the calibration
        :param  cal_type: Type of calibration we want
        :param  diag_ID: ID of the diagnostic we want
        :param  FILDid: alias for diag_ID, to be consistent with the rest of the
            functions in the logbook but also keep retrocompatibility. Notice
            that if diag_ID is not None, diag_ID value will prevail

        :return cal: CalParams() object
        """
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
            print(self.CameraCalibrationDatabase['CalID'][flags])
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
                cal.xcenter = row.xcenter.values[0]
                cal.ycenter = row.ycenter.values[0]
        return cal

    def getGeomID(self, shot: int, diag_ID: int = 1):
        """
        Get the geometry id of the INPA detector for a given shot

        Jose Rueda Rueda: jrrueda@us.es

        :param  shot: integer, shot number
        :param  FILDid: manipulator number

        :return GeomID: str, geometry ID installed for that shot
        """
        flags = (self.geometryDatabase['shot1'] <= shot) & \
            (self.geometryDatabase['shot2'] >= shot) & \
            (self.geometryDatabase['diag_ID'] == diag_ID)
        n_true = sum(flags)
        if n_true == 0:
            raise errors.NotFoundGeomID('No entry found in database')
        elif n_true > 1:
            raise errors.FoundSeveralGeomID(
                'More than onw entry found, revise')
        else:
            GeomID = self.geometryDatabase[flags].GeomID.values[0]
        return GeomID

    def getPositionOrientation(self, shot: int, diag_ID: int = 1):
        """
        Get the position of the INPA detector.

        Jose Rueda - jrrueda@us.es

        :param  shot: shot number to look in the database
        :param  diag_ID: INPA id number

        :return position: Dict with scintillator and pinhole position
        :return orientation: Dict with scintillator reference vectors

        Note:
            - The INPA does not move, so just take the default values from geom
        """
        # Get always the default as a reference:
        geomID = self.getGeomID(shot, diag_ID)
        default = self._getPositionOrientationDefault(geomID)
        position = {
            'R_pinhole': default['R_pinhole'],
            'z_pinhole': default['z_pinhole'],
            'phi_pinhole': default['phi_pinhole'],
            # Reference point on the scintillator (to get B)
            'R_scintillator': default['R_scintillator'],
            'z_scintillator': default['z_pinhole'],
            'phi_scintillator': default['phi_pinhole'],
        }
        # Transform to cylindrical vector in the choosen scintillator point
        phi = default['phi_scintillator'] * math.pi / 180.0
        ur = np.array([math.cos(phi), math.sin(phi), 0])
        uphi = np.array([-math.sin(phi), math.cos(phi), 0])
        uz = np.array([0, 0, 1])
        orientation = {
              's1': np.array(default['s1']),
              's2': np.array(default['s2']),
              's3': np.array(default['s3']),
        }
        orientation['s1rzt'] = np.array([(default['s1'] * ur).sum(),
                                         (default['s1'] * uz).sum(),
                                         (default['s1'] * uphi).sum()])
        orientation['s2rzt'] = np.array([(default['s2'] * ur).sum(),
                                         (default['s2'] * uz).sum(),
                                         (default['s2'] * uphi).sum()])
        orientation['s3rzt'] = np.array([(default['s3'] * ur).sum(),
                                         (default['s3'] * uz).sum(),
                                         (default['s3'] * uphi).sum()])
        return position, orientation

    def getComment(self, shot: int):
        """
        Get the comment line

        :param  shot: shot number (int) or array of shots

        :return comment: string (or list) with comments written by the operator
        """
        # Prepare the shot list
        if isinstance(shot, int):
            shotl = np.array([shot])
        elif isinstance(shot, (tuple, list)):
            shotl = np.array(shot)
        elif isinstance(shot, np.ndarray):
            shotl = shot
        else:
            print('Input type: ', type(shot))
            raise errors.NotValidInput('Check shot input')

        # Check the overheating
        comment = []
        if self.logbookVersion >= 0:
            dummy = self.excelLogbook['Comment']
            for ks, s in enumerate(shotl):
                if s in self.excelLogbook['shot'].values:
                    i, = np.where(self.excelLogbook['shot'].values == s)[0]
                    comment.append(dummy.values[i])
        else:
            text = 'Comments can not be read in this logbook version'
            logger.warning('22: %s' % text)
            comment = ['' for s in shotl]
        return comment

    def _getPositionOrientationDefault(self, geomID: str):
        """Get the default postition of an INPA, given the geometry id."""
        dummy = _defaultINPAdata[geomID]
        out = {
          # Pinhole position
          'R_pinhole': dummy['r_pinhole'],
          'z_pinhole': dummy['z_pinhole'],
          'phi_pinhole': dummy['phi_pinhole'],
          # Reference point on the scintillator (to get B)
          'R_scintillator': dummy['r_scintillator'],
          'z_scintillator': dummy['z_scintillator'],
          'phi_scintillator': dummy['phi_scintillator'],
          # Reference system in the pinhole
          's1': np.array(dummy['s1']),
          's2': np.array(dummy['s2']),
          's3': np.array(dummy['s3']),
        }
        return out
