"""
Contain FILD object.

Jose Rueda: jrrueda@us.es and others
Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch
Jesus Poley-Sanjuan - jesus.poley@epfl.ch
"""

import os
import f90nml
import numpy as np
import pandas as pd
from ScintSuite._Machine import machine
from ScintSuite._Paths import Path
from ScintSuite._Mapping._Calibration import CalParams, readCameraCalibrationDatabase
import ScintSuite.LibData.TCV.DiagParam as params
paths = Path(machine)

##
import requests
from bs4 import BeautifulSoup
##
##
import requests
from bs4 import BeautifulSoup
##

# --- Default files:
_cameraDatabase = os.path.join(paths.ScintSuite, 'Data', 'Calibrations',
                               'FILD', 'TCV', 'calibration_database.txt')
_geometryDatabase = os.path.join(paths.ScintSuite, 'Data',
                                 'Calibrations', 'FILD', 'TCV',
                                 'Geometry_logbook.txt')
#_positionDatabase = paths.FILDPositionDatabase
#_geometrdefault = os.path.join(paths.ScintSuite, 'Data',
#                              'Calibrations', 'FILD', 'TCV',
#                               'GeometryDefaultParameters.txt')
#_defaultFILDdata = f90nml.read(_geometrdefault)


# --- Auxiliar routines to find the path towards the camera files
def guessFILDfilename(shot: int, diag_ID: int = 1):
   """
   Guess the filename of a video

   Jose Rueda Rueda: jrrueda@us.es

   :param  shot: shot number
   :param  diag_ID: FILD manipulator number

   :return file: the name of the file/folder
   """
   base_dir = params.FILD[diag_ID-1]['path']
   extension = params.FILD[diag_ID-1]['extension'](shot)
   shot_str = str(shot)
   name = shot_str + extension
   file = os.path.join(base_dir, name)
   return file


# --- FILD object
class FILD_logbook:
    """
    Contains all geometrical parameters and path information of FILD

    Jose Rueda - jrrueda@us.es
    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch
    Jesus Poley-Sanjuan - jesus.poley@epfl.ch

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
                 positionFile: str = 'https://spcwiki.epfl.ch/wiki/FILD/Logbook',
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
        try:    
           self.CameraCalibrationDatabase = \
           readCameraCalibrationDatabase(cameraFile, verbose=verbose,
                                        n_header=0)
        except:
           pass
        # Load the geometry database
        try:
           self.geometryDatabase = \
           self._readGeometryDatabase(geometryFile, verbose=verbose)
        except:
           print('could not read the geometry database')
           pass

        self.wikilink = positionFile
        
        return 
 
    def _readPositionDatabase(self, shot: int, verbose: bool = True):
        """
        Read the excel containing the position database

        :param  filename: path or url pointing to the logbook
        :param  verbose: flag to print some info
      
        #dummy = pd.read_excel(filename, engine='openpyxl', header=[0, 1])
        #dummy['shot'] = dummy.Shot.Number.values.astype(int)
        
        """
        
        return None

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

    def get_collimator_data(self, shot = int,
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
        
         # First load the logbook from the SPC-wiki
        response = requests.get(self.wikilink,verify=False)
        
        wiki_data = response.text
        soup = BeautifulSoup(wiki_data, 'html.parser')
        # Load from the soup the tables of the logbook wikipage
        wiki_table = soup.find_all('table',class_='wikitable') 
        # Choose the logbook table 
        if shot <= 77469:
            Logbook_table = wiki_table[3] #year 2022
            year = 2022
        else:
            Logbook_table = wiki_table[4] #year 2023
            year = 2023
                          
        if Logbook_table:
            rows = Logbook_table.find_all('tr')
            wiki_data = [] 
            for row in rows:
                columns = row.find_all(['td', 'th']) 
                row_data = [col.text.strip() for col in columns]
                wiki_data.append(row_data)
                
            # Identify the row where the shot is stored
            irow = 1
            for i in range(1,np.shape(wiki_data,)[0]):     
                if shot == int(wiki_data[i][0]):
                    irow = i

        geometry =  'dummy'
        if shot == int(wiki_data[irow][0]):            
            collimator_width = float(wiki_data[irow][2][10:13])
            slit = int(wiki_data[irow][2][25])
            geometry = 'col' + str(collimator_width) + 'mm_slit' + str(slit) + \
                        '_' + str(year)
        
        return geometry

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

    def getPosition(self, shot: int, FILDid: int = 1, verbose=True):
        """
        Get the position of the FILD detector. If a None shot is given return 0

        Jesus Poley-Sanjuan - jesus.poley@epfl.ch

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id
        """
        if shot == None:
            return 0


        # First load the logbook from the SPC-wiki
        response = requests.get(self.wikilink,verify=False)
        
        wiki_data = response.text
        soup = BeautifulSoup(wiki_data, 'html.parser')
        # Load from the soup the tables of the logbook wikipage
        wiki_table = soup.find_all('table',class_='wikitable') 
        # Choose the logbook table 
        if shot <= 77469:
            Logbook_table = wiki_table[3] #year 2022
            year = 2022
        else:
            Logbook_table = wiki_table[4] #year 2023
            year = 2023
                          
        if Logbook_table:
            rows = Logbook_table.find_all('tr')
            wiki_data = [] 
            for row in rows:
                columns = row.find_all(['td', 'th']) 
                row_data = [col.text.strip() for col in columns]
                wiki_data.append(row_data)
                
            # Identify the row where the shot is stored
            for i in range(1,np.shape(wiki_data,)[0]):     
                if shot == int(wiki_data[i][0]):
                    row = i
                   
        position = 0
        if verbose:
            print('Looking for the position database: ')
            position = int(wiki_data[i][3])
  
        return position

    def getOrientation(self, shot, FILDid):
        """
        Get the orientation

        Only the old probehead could be rotated

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id

        To be implemented:

        """


        '''
        geomID = self.getGeomID(shot, FILDid)
        default = self._getOrientationDefault(geomID)
        # First check that we have loaded the position logbook
        if not self.flagPositionDatabase:
            print('Logbook not loaded, returning default values')
            return default
        # Get the shot index in the database
        if shot in self.positionDatabase['shot'].values:
            i, = np.where(self.positionDatabase['shot'].values == shot)[0]
            flag = True
        else:
            print('Shot not found in logbook, returning the default values')
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

        '''
        return 0#default

    def getAdqFreq(self, shot: int, diag_ID: int = 1):
        """
        Get the adquisition frequency from the database

        Jose Rueda - jrrueda@us.es
        Lina Velarde - lvelarde@us.es

        :param  shot: shot number to look in the database
        :param  FILDid: manipulator id
        """
        # Get always the default as a reference:
        default = params.FILD[diag_ID-1]['adqfreq']
        # First check that we have loaded the position logbook
        if not self.flagPositionDatabase:
            print('Logbook not loaded, returning default values')
            return default
        # Get the shot index in the database
        if shot in self.positionDatabase['shot'].values:
            i, = np.where(self.positionDatabase['shot'].values == shot)[0]
            flag = True
        else:
            print('Shot not found in logbook, returning the default values')
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
        default = params.FILD[diag_ID-1]['t_trig']
        # First check that we have loaded the position logbook
        if not self.flagPositionDatabase:
            print('Logbook not loaded, returning default values')
            return default
        # Get the shot index in the database
        if shot in self.positionDatabase['shot'].values:
            i, = np.where(self.positionDatabase['shot'].values == shot)[0]
            flag = True
        else:
            print('Shot not found in logbook, returning the default values')
            return default
        # --- Get the postion
        dummy = self.positionDatabase['FILD'+str(diag_ID)]
        if 'CCDqe trigger time [s]' in dummy.keys():  # Look for adqfreq
            adqfreq = dummy['CCDqe trigger time [s]'].values[i]
        else:  # Take the default approx value
            print('Trigger time not in the logbook, returning default')
            adqfreq = default
        return adqfreq

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
