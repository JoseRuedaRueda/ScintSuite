"""
Contain INPA object.

Jose Rueda: jrrueda@us.es
"""

import os
import f90nml
import math
import numpy as np
import pandas as pd
from Lib._Machine import machine
from Lib._Paths import Path
from Lib._Mapping._Calibration import CalParams, readCameraCalibrationDatabase
import Lib.LibData.AUG.DiagParam as params
import Lib.errors as errors
paths = Path(machine)


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
    Guess the filename of a video

    Jose Rueda Rueda: jrrueda@us.es

    @param shot: shot number
    @param diag_ID: INPA number

    @return f: the name of the file/folder
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
        - getCameraCalibration(): find the camera parameters
        - getGeomID(): Get the id of the geometry installed in a manipulator
        - getPositionOrientation(): get the position and orientation of INPA
        - getGeomShots(): find all shots where a given collimator was installed
    """

    def __init__(self,
                 cameraFile: str = _cameraDatabase,
                 geometryFile: str = _geometryDatabase,
                 logbookFile: str = _INPALogbook,
                 verbose: bool = True):
        """
        Initialise the object

        Read the three data bases and save them in atributes of the object

        @param cameraFile: path to the ACSII file containing the data
        @param geometryFile: path to the ACSII file containing the data
        @param logbookFile: path to the excel file containing the data (the
            url poiting to the internet logbook. It can be a path to a local
            excel)
        """
        if verbose:
            print('.-.. --- --. -... --- --- -.-')
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
            print('No excel logbook found')
        # Load the geometry database
        self.geometryDatabase = \
            self._readGeometryDatabase(geometryFile, verbose=verbose)
        print('..-. .. -. .- .-.. .-.. -.--')

    def _readExcelLogbook(self, filename: str, verbose: bool = True):
        """
        Read the excel containing the position database

        @param filename: path or url pointing to the logbook
        @param verbose: flag to print some info
        """
        if verbose:
            print('Looking for the position database: ', filename)
        dummy = pd.read_excel(filename, engine='openpyxl', header=[0, 1])
        dummy['shot'] = dummy.Shot.Shot.values.astype(int)

        return dummy

    def _readGeometryDatabase(self, filename: str, n_header: int = 3,
                              verbose: bool = True):
        """
        Read the Geometry database

        See the help PDF located at the readme file for a full description of
        each available parameter

        @author Jose Rueda Rueda: jrrueda@us.es

        @param filename: Complete path to the file with the calibrations
        @param n_header: Number of header lines (5 in the oficial format)

        @return database: Pandas dataframe with the database
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

    def getCameraCalibration(self, shot: int,
                             diag_ID: int = 1, cal_type: str = 'PIX'):
        """
        Get the camera calibration parameters for a shot

        @param shot: Shot number for which we want the calibration
        @param cal_type: Type of calibration we want
        @param diag_ID: ID of the diagnostic we want
        @param FILDid: alias for diag_ID, to be consistent with the rest of the
            functions in the logbook but also keep retrocompatibility. Notice
            that if diag_ID is not None, diag_ID value will prevail

        @return cal: CalParams() object
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
                cal.xcenter = row.xcenter.values[0]
                cal.ycenter = row.ycenter.values[0]
        return cal

    def getGeomID(self, shot: int, diag_ID: int = 1):
        """
        Get the geometry id of the INPA detector for a given shot

        @param shot: integer, shot number
        @param FILDid: manipulator number
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
            id = self.geometryDatabase[flags].GeomID.values[0]
        return id

    def getPositionOrientation(self, shot: int, diag_ID: int = 1):
        """
        Get the position of the INPA detector.

        Jose Rueda - jrrueda@us.es

        @param shot: shot number to look in the database
        @param FILDid: manipulator id

        Note: The INPA does not move, as FILD, so just take the default from
        the installation
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

    def getGeomShots(self, geomID, maxR: float = None, verbose: bool = True):
        """
        Return all shots in the database position database with a geomID

        @param geomID: ID of the geometry we are insterested in
        @param maxR: if present, only shots for which R < maxR will be
            considered. Default values are, for each manipulator:
                1: 2.5 m
                2: 2.2361 m
                5: 1.795 m
        @param verbose: flag to print in the console the number of shots found
            using that geometry
        """
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

    def _getPositionOrientationDefault(self, geomID: str):
        """Get the default postition of an INPA, given the geometry id"""
        dummy = _defaultINPAdata[geomID]
        out = {
          # Pinhole position
          'R_pinhole': dummy['r_pinhole'],
          'z_pinhole': dummy['z_pinhole'],
          'phi_pinhole': dummy['phi_pinhole'],
          # Reference point on the scintillator (to get B)
          'R_scintillator': dummy['r_pinhole'],
          'z_scintillator': dummy['z_pinhole'],
          'phi_scintillator': dummy['phi_pinhole'],
          # Reference system in the pinhole
          's1': np.array(dummy['s1']),
          's2': np.array(dummy['s2']),
          's3': np.array(dummy['s3']),
        }
        return out
