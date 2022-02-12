"""
Contain FILD object.

Jose Rueda: jrrueda@us.es
"""

import os
import f90nml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from Lib.LibMachine import machine
from Lib.LibPaths import Path
from Lib.LibMap.Calibration import CalParams
import Lib.LibData.AUG.DiagParam as params
import Lib.errors as errors
paths = Path(machine)


# --- Default files:
_cameraDatabase = os.path.join(paths.ScintSuite, 'Data', 'Calibrations',
                               'FILD', 'AUG', 'calibration_database.txt')
_geometryDatabase = os.path.join(paths.ScintSuite, 'Data',
                                 'Calibrations', 'FILD', 'AUG',
                                 'Geometry_logbook.txt')
_positionDatabase = paths.FILDPositionDatabase

_geometrdefault = os.path.join(paths.ScintSuite, 'Data',
                               'Calibrations', 'FILD', 'AUG',
                               'GeometryDefaultParameters.txt')
_defaultFILDdata = f90nml.read(_geometrdefault)


# --- Auxiliar routines to find the path towards the camera files
def guessFILDfilename(shot: int, diag_ID: int = 1):
    """
    Guess the filename of a video

    Jose Rueda Rueda: jrrueda@us.es

    Note AUG criteria of organising files is assumed: .../38/38760/...

    @param shot: shot number
    @param diag_ID: FILD manipulator number

    @return file: the name of the file/folder
    """
    base_dir = params.FILD[diag_ID-1]['path']
    extension = params.FILD[diag_ID-1]['extension']
    shot_str = str(shot)
    name = shot_str + extension
    file = os.path.join(base_dir, shot_str[0:2], name)
    return file


# --- Auxiliar routines to load and plot FILD4 trajectory:
def load_FILD4_trajectory(shot, path=paths.FILD4_trayectories):
    """
    Load FILD4 trayectory

    Jose Rueda: jrrueda@us.es

    Note: This is a temporal function, in the future will be replaced by one to
    load trayectories from shotfiles

    @param shot: Shot number to load
    @param path: Path to the main folder with FILD4 trajectories
    """
    # --- Load the power supply output data
    shot_str = str(shot)
    try:
        file = os.path.join(path, 'output_raw', shot_str[0:2],
                            'FILD_MDRS_' + shot_str + '.txt')
        print('Looking for file: ', file)
        data = np.loadtxt(file, skiprows=1)
        # Delete the last line of the data because is always zero
        dat = np.delete(data, -1, axis=0)
        # Delete points where PS output is zero. This **** instead of giving
        # as ouput the points where the trajectory was requested, it always
        # gives as ouput a given number of rows, and set to zero the non used
        # ones...
        fi = dat[:, 2] < 1
        fv = dat[:, 4] < 1
        flags = (fv * fi).astype(bool)
        PSouput = {
            'V_t_obj': dat[~flags, 0] / 1000.0,
            'V_obj': dat[~flags, 1],
            'I_t': dat[~flags, 2] * 1.0e-9,
            'I': dat[~flags, 3],
            'V_t': dat[~flags, 4] * 1.0e-9,
            'V': dat[~flags, 5]
        }
    except OSError:
        print('File with power supply outputs not found')
        PSouput = None
    # --- Load the reconstructed trajectory
    R4_lim = [2.082410, 2.015961]    # R min and max of FILD4
    Z4_lim = [-0.437220, -0.437906]  # z min and max of FILD4
    ins_lim = [0, 0.066526]          # Maximum insertion
    try:
        file = os.path.join(path, 'output_processed', shot_str[0:2],
                            shot_str + '.txt')
        print('Looking for file: ', file)
        data = np.loadtxt(file, skiprows=2, delimiter=',')
        insertion = {
            't': data[:, 0],
            'insertion': data[:, 1],
        }
        if data[:, 1].max() > ins_lim[1]:
            warnings.warn('FILD4 insertion larger than the maximum!!!')
        position = {
            't': data[:, 0],
            'R': R4_lim[0] + (data[:, 1]-ins_lim[0])/(ins_lim[0]-ins_lim[1])
            * (R4_lim[0]-R4_lim[1]),
            'z': Z4_lim[0]+(data[:, 1]-ins_lim[0])/(ins_lim[0]-ins_lim[1])
            * (Z4_lim[0]-Z4_lim[1])
        }
    except OSError:
        print('File with trajectory not found')
        position = None
        insertion = None
        PSouput = None

    return {'PSouput': PSouput, 'insertion': insertion, 'position': position}


def plot_FILD4_trajectory(shot, PS_output=False, ax=None, ax_PS=None,
                          line_params={}, line_params_PS={}, overlay=False,
                          unit='cm'):
    """
    Plot FILD4 trayectory

    Jose Rueda: jrrueda@us.es

    Note: this is in beta phase, improvement suggestions are wellcome

    @param shot: shot you want to plot
    @param PS_output: flag to plot the output of the power supply
    @param ax: axes where to plot the trajectory. If none, new axis will be
               created
    @param ax_PS: Array of two axes where we want to plot the PS data. ax_PS[0]
                  will be for the voltaje while ax_PS[1] for the intensity. If
                  None, new axis  will be created
    @param line_params: Line parameters for the trajectory plotting
    @param line_params_PS: Line parameters for the PS plots. Note: same dict
                           will be used for the Voltaje and intensity plots, be
                           carefull if you select the 'color'
    @param overlay: Flag to overlay the trayectory over the current plot. The
                    insertion will be plotted in arbitrary units on top of it.
                    ax input is mandatory for this
    """
    line_options = {
        'label': '#' + str(shot),
    }
    line_options.update(line_params)
    # ---
    factor = {
        'cm': 100.0,
        'm': 1.0,
        'inch': 100.0 / 2.54,
        'mm': 1000.0
    }
    # --- Load the position
    position = load_FILD4_trajectory(shot)

    # --- Plot the position
    if ax is None:
        fig, ax = plt.subplots()
    if overlay:  # Overlay the trayectory in an existing plot:
        print('Sorry, still not implemented')
    else:
        ax.plot(position['insertion']['t'],
                factor[unit] * position['insertion']['insertion'],
                **line_options)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Insertion [' + unit + ']')
        ax.set_xlim(0, 1.1 * position['insertion']['t'].max())
        ymax = 1.1 * factor[unit] * position['insertion']['insertion'].max()
        ax.set_ylim(0, ymax)
        ax.legend()

    # --- Plot the PS output
    if PS_output:
        if ax_PS is None:
            fig2, ax_PS = plt.subplots(2, 1, sharex=True)
        # Plot the voltage
        ax_PS[0].plot(position['PSouput']['V_t_obj'],
                      position['PSouput']['V_obj'],
                      label='Objective')
        ax_PS[0].plot(position['PSouput']['V_t'],
                      position['PSouput']['V'],
                      label='Real')
        ax_PS[0].set_ylabel('Voltage [V]')
        ax_PS[0].legend()
        ax_PS[1].plot(position['PSouput']['I_t'],
                      position['PSouput']['I'])
        ax_PS[1].set_ylabel('Intensity [A]')
    plt.show()


# --- FILD object
class FILD_logbook:
    """
    Contain all geometrical parameters and path information of FILD

    Jose Rueda - jrrueda@us.es

    Introduced in version 0.7.2
    Re-written in version 0.7.8

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

        @param cameraFile: path to the ACSII file containing the data
        @param geometryFile: path to the ACSII file containing the data
        @param positionFile: path to the excel file containing the data (the
            url poiting to the internet logbook. It can be a path to a local
            excel)
        """
        if verbose:
            print('.-.. --- --. -... --- --- -.-')
        # Load the camera database
        self.CameraCalibrationDatabase = \
            self._readCameraCalibrationDatabase(cameraFile, verbose=verbose)
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
            print('Not found position database, we will use the defaults')
        # Load the geometry database
        self.geometryDatabase = \
            self._readGeometryDatabase(geometryFile, verbose=verbose)
        print('..-. .. -. .- .-.. .-.. -.--')

    def _readCameraCalibrationDatabase(self, filename: str, n_header: int = 5,
                                       verbose: bool = True):
        """
        Read the calibration database, to align the strike maps.

        See the help PDF located at the readme file for a full description of
        each available parameter

        @author Jose Rueda Rueda: jrrueda@us.es

        @param filename: Complete path to the file with the calibrations
        @param n_header: Number of header lines (5 in the oficial format)

        @return database: Pandas dataframe with the database
        """
        data = {'CalID': [], 'camera': [], 'shot1': [], 'shot2': [],
                'xshift': [], 'yshift': [], 'xscale': [], 'yscale': [],
                'deg': [], 'cal_type': [], 'diag_ID': []}

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
        # Transform to pandas
        database = pd.DataFrame(data)
        return database

    def _readPositionDatabase(self, filename: str, verbose: bool = True):
        """
        Read the excel containing the position database

        @param filename: path or url pointing to the logbook
        @param verbose: flag to print some info
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

    def getCameraCalibration(self, shot: int,  FILDid: int = 1,
                             diag_ID: int = None,):
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
        return cal

    def getGeomID(self, shot: int, FILDid: int = 1):
        """
        Get the geometry id of the FILD manipulator for a given shot

        @param shot: integer, shot number
        @param FILDid: manipulator number
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

    def getPosition(self, shot: int, FILDid: int = 1):
        """
        Get the position of the FILD detector.

        Jose Rueda - jrrueda@us.es

        @param shot: shot number to look in the database
        @param FILDid: manipulator id
        """
        # Get always the default as a reference:
        geomID = self.getGeomID(shot, FILDid)
        default = self._getPositionDefault(geomID)
        # First check that we have loaded the position logbook
        if not self.flagPositionDatabase:
            print('Position database not loaded, returning default values')
            return default
        # Get the shot index in the database
        if shot in self.positionDatabase['shot'].values:
            i, = np.where(self.positionDatabase['shot'].values == shot)[0]
            flag = True
        else:
            print('Shot not found in logbook, returning the default values')
            return default
        # --- Get the postion
        position = {        # Initialise the position
            'R': 0.0,
            'z': 0.0,
            'phi': 0.0,
        }
        dummy = self.positionDatabase['FILD'+str(FILDid)]
        if FILDid != 4:
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
        else:  # We have FILD4, the movable FILD
            # Ideally, we will have an optic calibration database which is
            # position dependent, therefore we should keep all the FILD
            # trayectory.
            # However, this calibration database was not given by the previous
            # operator of the 'in-shot' movable FILD, neither any details of
            # the optical design which could allow us to create it. So until we
            # dismount and examine the diagnostic piece by piece, this
            # trayectory is irrelevant, we will just keep the average position,
            # which is indeed 'okeish', as the resolution of this FILD in AUG
            # is poor, so a small missalignement will not be noticed.
            # To calculate this average position, I will take the average of
            # the positions at least 5 mm further from the minimum limit
            dummy2 = load_FILD4_trajectory(shot)
            if dummy2['position'] is not None:
                min = dummy2['position']['R'].min()
                flags = dummy2['position']['R'] > (min + 0.005)
                position['R'] = dummy2['position']['R'][flags].mean()
                position['z'] = dummy2['position']['z'][flags].mean()
            else:    # Shot not found in the database
                position['R'] = default['R']
                position['z'] = default['z']
            # FILD4 phi is always the same:
            print('Phi not in the logbook, returning default')
            position['phi'] = default['phi']
        return position

    def getOrientation(self, shot, FILDid):
        """
        Get the orientation

        Note that in AUG the orientation of the diagnostic never changes, so
        this function just return always the default parameters

        @param shot: shot number to look in the database
        @param FILDid: manipulator id
        """
        geomID = self.getGeomID(shot, FILDid)
        return self._getOrientationDefault(geomID)

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
