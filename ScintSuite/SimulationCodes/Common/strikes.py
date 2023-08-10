"""
Strike object for SINPA and FILDSIM codes.

Maintaned by Jose Rueda: jrrueda@us.es

Contains the Strike object, which stores the information of the strike points
calculated by the code and plot the different information on it
"""
import os
import math
import f90nml
import logging
import numpy as np
import xarray as xr
import ScintSuite.errors as errors
import ScintSuite._Plotting as ssplt
import matplotlib.pyplot as plt
from ScintSuite.version_suite import exportVersion
from copy import deepcopy
from typing import Union, List, Tuple, Dict, Any, Optional
from ScintSuite._Paths import Path
from ScintSuite._Machine import machine
from mpl_toolkits.mplot3d import Axes3D
from ScintSuite._SideFunctions import createGrid
from ScintSuite._Mapping._Common import transform_to_pixel, remap
from ScintSuite.SimulationCodes.Common.strikeHeader import orderStrikes as order
from ScintSuite._Plotting import axisEqual3D, clean3Daxis


# -----------------------------------------------------------------------------
# --- Prepare auxiliary objects
# ----------------------------------------------------------------------------
logger = logging.getLogger('ScintSuite.SimCod')
paths = Path(machine)


# ----------------------------------------------------------------------------
# --- Reading routines
# ----------------------------------------------------------------------------
def readSINPAstrikes(filename: str, verbose: bool = False):
    """
    Read the strike points from a SINPA simulation

    Jose Rueda: jrrueda@us.es

    :param  filename: filename of the file
    :param  verbose: flag to print information on the file

    Note: in order to load the proper header (with information on which
    variables are stored in the file), the code will guess which kind of file
    it is by the name of the file. Therefore, official name of the SINPA files
    should not be changed, if not, this routine does no longer work
    """
    # --- Identify which can of file we are dealing with:
    if filename.endswith('spmap'):
        plate = 'scintillator'
    elif filename.endswith('spcmap'):
        plate = 'collimator'
    elif filename.endswith('spsignal'):
        plate = 'signalscintillator'
    elif filename.endswith('spcsignal'):
        plate = 'collimator'
    elif filename.endswith('wmmap'):
        plate = 'wrong'
    else:
        raise Exception('File not understood. Has you changed the ext???')

    # --- Open the file and read
    with open(filename, 'rb') as fid:
        header = {
            'versionID1': np.fromfile(fid, 'int32', 1)[0],
            'versionID2': np.fromfile(fid, 'int32', 1)[0],
        }
        if header['versionID1'] <= 4:
            # Keys of what we have in the file:
            header['runID'] = np.fromfile(fid, 'S50', 1)[:]
            header['ngyr'] = np.fromfile(fid, 'int32', 1)[0]
            header['gyroradius'] = np.fromfile(fid, 'float64', header['ngyr'])
            header['nXI'] = np.fromfile(fid, 'int32', 1)[0]
            header['XI'] = np.fromfile(fid, 'float64', header['nXI'])
            header['FILDSIMmode'] = \
                np.fromfile(fid, 'int32', 1)[0].astype(bool)
            header['ncolumns'] = np.fromfile(fid, 'int32', 1)[0]
            if header['versionID1'] >= 4:
                header['kindOfFile'] = np.fromfile(fid, 'int32', 1)[0]
            header['counters'] = \
                np.zeros((header['nXI'], header['ngyr']), int)
            data = np.empty((header['nXI'], header['ngyr']),
                            dtype=np.ndarray)
            header['scint_limits'] = {
                'xmin': 300.,
                'xmax': -300.,
                'ymin': 300.,
                'ymax': -300.
            }
            # get the information. Notice that from one SINPA version to the
            # following, it could happen that the strikes files was unchanged,
            # So there is no a different strike header, therefore, we need to
            # try backwards until we find the proper one
            found_header = False
            id_version = header['versionID1']
            if header['FILDSIMmode']:
                key_to_look = 'sinpa_FILD'
            else:
                key_to_look = 'sinpa_INPA'
            while not found_header:
                if header['versionID1'] < 4:
                    try:
                        header['info'] = deepcopy(
                            order[key_to_look][id_version][plate.lower()])
                        found_header = True
                    except KeyError:
                        id_version -= 1
                    # if the id_version is already -1, just stop, something
                    # went wrong
                    if id_version < 0:
                        raise Exception('Not undestood SINPA version')
                else:
                    try:
                        header['info'] = deepcopy(
                            order[key_to_look][id_version][plate.lower()][header['kindOfFile']])
                        found_header = True
                    except KeyError:
                        id_version -= 1
                    # if the id_version is already -1, just stop, something
                    # went wrong
                    if id_version < 0:
                        raise Exception('Not undestood SINPA version')
            # Load the data from each gyroradius and xi values. If the loaded
            # plate is scintillator, get the edge of the markers distribution,
            # for the latter histogram calculation
            scints = ['scintillator',  'signalscintillator']
            if plate.lower() in scints:
                ycolum = header['info']['x1']['i']
                zcolum = header['info']['x2']['i']
            for ig in range(header['ngyr']):
                for ia in range(header['nXI']):
                    header['counters'][ia, ig] = \
                        np.fromfile(fid, 'int32', 1)[0]
                    if header['counters'][ia, ig] > 0:
                        data[ia, ig] = np.reshape(
                            np.fromfile(fid, 'float64',
                                        header['ncolumns']
                                        * header['counters'][ia, ig]),
                            (header['counters'][ia, ig],
                             header['ncolumns']), order='F')
                        if plate.lower() in scints:
                            header['scint_limits']['xmin'] = \
                                min(header['scint_limits']['xmin'],
                                    data[ia, ig][:, ycolum].min())
                            header['scint_limits']['xmax'] = \
                                max(header['scint_limits']['xmax'],
                                    data[ia, ig][:, ycolum].max())
                            header['scint_limits']['ymin'] = \
                                min(header['scint_limits']['ymin'],
                                    data[ia, ig][:, zcolum].min())
                            header['scint_limits']['ymax'] = \
                                max(header['scint_limits']['ymax'],
                                    data[ia, ig][:, zcolum].max())
            # Read the time
            if plate.lower() == 'signalscintillator':
                header['time'] = float(np.fromfile(fid, 'float32', 1)[0])
                header['shot'] = int(np.fromfile(fid, 'int32', 1)[0])
        # Read the extra information included in version 2
        if header['versionID1'] >= 2:
            header['FoilElossModel'] = np.fromfile(fid, 'int32', 1)[0]
            if header['FoilElossModel'] == 1:
                header['FoilElossParameters'] = np.fromfile(fid, 'float64', 2)
            elif header['FoilElossModel'] == 2:
                header['FoilElossParameters'] = np.fromfile(fid, 'float64', 3) 
            elif header['FoilElossModel'] == 5:
                header['FoilElossParameters'] = np.fromfile(fid, 'float64', 3)
            header['FoilYieldModel'] = np.fromfile(fid, 'int32', 1)[0]
            if header['FoilYieldModel'] == 1:
                header['FoilYieldParameters'] = np.fromfile(fid, 'float64', 1)
            elif header['FoilYieldModel'] == 2:
                header['FoilYieldParameters'] = np.fromfile(fid, 'float64', 4)
            header['ScintillatorYieldModel'] = np.fromfile(fid, 'int32', 1)[0]
            if header['ScintillatorYieldModel'] == 1:
                header['ScintillatorYieldParameters'] = \
                    np.fromfile(fid, 'float64', 1)
            elif header['ScintillatorYieldModel'] == 2:
                header['ScintillatorYieldParameters'] = \
                    np.fromfile(fid, 'float64', 2)
        # Calculate the radial position if it is a INPA simulation
        if not header['FILDSIMmode']:
            iix = header['info']['x0']['i']
            iiy = header['info']['y0']['i']
            for ig in range(header['ngyr']):
                for ia in range(header['nXI']):
                    if header['counters'][ia, ig] > 0:
                        R = np.atleast_2d(np.sqrt(data[ia, ig][:, iix]**2
                                                  + data[ia, ig][:, iiy]**2)).T
                        data[ia, ig] = np.append(data[ia, ig], R, axis=1)
            # Update the headers.
            Old_number_colums = len(header['info'])
            extra_column = {
                'R0': {
                    'i': Old_number_colums,  # Column index in the file
                    'units': 'm',  # Units
                    'longName': 'Radial position of the CX event',
                    'shortName': '$R$',
                },
            }
            # Update the header
            header['info'].update(extra_column)
        # See if we have some rl or some XI for which any markers have arrived
        counts_rl = np.sum(header['counters'], axis=0, dtype=int)
        flags_0 = counts_rl == 0
        # Now remove the unwanted gyroradius
        dummy = np.arange(header['ngyr'])
        index_to_remove = dummy[flags_0]
        header['counters'] = np.delete(header['counters'], index_to_remove,
                                       axis=1)
        data = np.delete(data, index_to_remove, axis=1)
        header['gyroradius'] = np.delete(header['gyroradius'], index_to_remove)
        # Now remove the unwanted XI
        counts_XI = np.sum(header['counters'], axis=1, dtype=int)
        flags_0 = counts_XI == 0
        dummy = np.arange(header['nXI'])
        index_to_remove = dummy[flags_0]
        header['counters'] = np.delete(header['counters'], index_to_remove,
                                       axis=0)
        data = np.delete(data, index_to_remove, axis=0)
        header['XI'] = np.delete(header['XI'], index_to_remove)
        # Update the counters
        header['nXI'], header['ngyr'] = header['counters'].shape
        # Small retrocompatibility part
        # Just for old FILDSIM user which may have their routines based on the
        # Strike map points object, make a copy of the XI values as in the old
        # notation (they are just 10 numbers, so it will not be the end of the
        # world)
        if header['FILDSIMmode']:
            header['npitch'] = header['nXI']
            header['pitch'] = header['XI']
        # ---- Get the geometry id
        # Try to get the grometry id
        try:
            resultDir, name = os.path.split(filename)
            runID = name.split('.')[0]
            mainDir, dummy = os.path.split(resultDir)
            namelistFile = os.path.join(mainDir, 'inputs', runID + '.cfg')
            nml = f90nml.read(namelistFile)
            dummy, geomDir = os.path.split(nml['config']['geomfolder'])
            header['geomID'] = geomDir
        except FileNotFoundError:
            header['geomID'] = None
            logger.warning('Not found SINPA namelist')
        if verbose:
            print('File %s'%filename)
            print('Total number of strike points: ',
                  np.sum(header['counters']))
            print('SINPA version: ', header['versionID1'], '.',
                  header['versionID2'])
            print('Average number of strike points per centroid: ',
                  int(header['counters'].mean()))
        return header, data


def readFILDSIMstrikes(filename: str, verbose: bool = False):
    """
    Load the strike points from a FILDSIM simulation.

    Jose Rueda: ruejo@ipp.mpg.de

    :param  runID: runID of the FILDSIM simulation
    :param  plate: plate to collide with (Collimator or Scintillator)
    :param  file: if a filename is provided, data will be loaded from this
    file, ignoring the SINPA folder structure (and runID)
    :param  verbose. flag to print some info in the command line
    """
    if verbose:
        print('Reading strike points: ', filename)
    dummy = np.loadtxt(filename, skiprows=3)
    header = {
        'FILDSIMmode': True,
        'XI': np.unique(dummy[:, 1]),
        'gyroradius': np.unique(dummy[:, 0])
    }
    header['nXI'] = header['XI'].size
    header['ngyr'] = header['gyroradius'].size
    # --- Order the strike points in gyroradius and pitch angle
    data = np.empty((header['nXI'], header['ngyr']), dtype=np.ndarray)
    header['counters'] = np.zeros((header['nXI'], header['ngyr']),
                                  dtype=int)
    header['scint_limits'] = {  # for later histogram making
        'xmin': 300.,
        'xmax': -300.,
        'ymin': 300.,
        'ymax': -300.
    }
    nmarkers, ncolum = dummy.shape
    for ir in range(header['ngyr']):
        for ip in range(header['nXI']):
            data[ip, ir] = dummy[
                (dummy[:, 0] == header['gyroradius'][ir])
                * (dummy[:, 1] == header['XI'][ip]), 2:]
            header['counters'][ip, ir], ncolums = data[ip, ir].shape
            # Update the scintillator limit for the histogram
            if header['counters'][ip, ir] > 0:
                header['scint_limits']['xmin'] = \
                    min(header['scint_limits']['xmin'],
                        data[ip, ir][:, 2].min())
                header['scint_limits']['xmax'] = \
                    max(header['scint_limits']['xmax'],
                        data[ip, ir][:, 2].max())
                header['scint_limits']['ymin'] = \
                    min(header['scint_limits']['ymin'],
                        data[ip, ir][:, 3].min())
                header['scint_limits']['ymax'] = \
                    max(header['scint_limits']['ymax'],
                        data[ip, ir][:, 3].max())
    # Check with version of FILDSIM was used
    if ncolum == 9:
        print('Old FILDSIM format, initial position NOT included')
        versionID = 0
    elif ncolum == 12:
        print('New FILDSIM format, initial position included')
        versionID = 1
    else:
        print('Detected number of columns: ', ncolum)
        raise Exception('Error loading file, not recognised columns')
    # Write some help
    header['info'] = order['fildsim_FILD'][versionID]
    # Check number of markers
    total_counter = np.sum(header['counters'])
    if nmarkers != total_counter:
        print('Total number of strike points: ', nmarkers)
        print('Total number of counters: ', total_counter)
        raise Exception('Total number of markers not matching!!!')
    if verbose:
        print('Total number of strike points: ', total_counter)
        print('Average number of strike points per centroid: ',
              int(header['counters'].mean()))
    # Small retrocompatibility part
    # Just for old FILDSIM user which may have their routines based on the
    # Strike map points object, make a copy of the XI values as in the old
    # notation (they are just 10 numbers, so it will not be the end of the
    # world)
    header['npitch'] = header['nXI']
    header['pitch'] = header['XI']
    return header, data


# -----------------------------------------------------------------------------
# --- Main Object
# -----------------------------------------------------------------------------
class Strikes:
    """
    StrikePoint class.

    Jose Rueda: jrrueda@us.es

    Stores the information of the strike points calculated by the code and plot
    the different information on it
    """

    def __init__(self, 
                 runID: Optional[str] = None, 
                 type: Optional [str] = 'MapScintillator',
                 file: Optional[str] = None, 
                 verbose: Optional[bool] = True, 
                 code: Optional[str] = 'SINPA'):
        """
        Initialise the object reading data from a SINPA file.

        Jose Rueda: jrrueda@us.es

        :param  runID: runID of the simulation
        :param  type: file to load (mapcollimator, mapscintillator, mapwrong
            signalcollimator or signalscintillator).Not used if code=='FILDSIM'
        :param  file: if a filename is provided, data will be loaded from this
            file, ignoring the code folder structure (and runID)
        :param  verbose. flag to print some info in the command line
        :param  code: name of the code where the data is coming from
        """
        # --- Get the name of the file
        if file is None:
            if code.lower() == 'sinpa':
                # Guess the name of the file
                if (type.lower() == 'mapscintillator'
                        or type.lower() == 'scintillatormap'):
                    name = runID + '.spmap'
                elif (type.lower() == 'mapcollimator'
                        or type.lower() == 'collimatormap'):
                    name = runID + '.spcmap'
                elif (type.lower() == 'signalscintillator'
                        or type.lower() == 'scintillatorsignal'):
                    name = runID + '.spsignal'
                elif (type.lower() == 'signalcollimator'
                        or type.lower() == 'collimatorsignal'):
                    name = runID + '.spcsignal'
                elif type.lower() == 'mapwrong' or type.lower() == 'wrongmap':
                    name = runID + '.wmmap'
                elif (type.lower() == 'selfshadowmap'
                        or type.lower() == 'mapselfshadow'):
                    name = runID + '.spcself'
                else:
                    raise Exception('Type not understood, revise inputs')
                file = os.path.join(paths.SINPA, 'runs', runID, 'results',
                                    name)
            elif code.lower() == 'fildsim':
                name = '_strike_points.dat'
                file = os.path.join(paths.FILDSIM, 'results',
                                    runID + name)
        self.file = file
        # --- read the file
        logger.info('Reading file: %s', file)
        if code.lower() == 'sinpa':
            self.header, self.data = readSINPAstrikes(file, verbose)
        elif code.lower() == 'fildsim':
            self.header, self.data = readFILDSIMstrikes(file, verbose)
        else:
            raise Exception('Code not understood')
        # Save the size
        self._shape = self.header['counters'].shape
        # --- Initialise the rest of the object
        # ## Histogram of Scintillator strikes
        # self.ScintHistogram = None
        ## Code used
        self.code = code
        ## Rest of the histograms
        self.histograms = {}
        ## Magnetic field at the detector
        self.B = None

    # -------------------------------------------------------------------------
    # --- Histogram calculation
    # -------------------------------------------------------------------------
    def calculate_2d_histogram(self, varx: str = 'xcx', vary: str = 'yxc',
                               binsx: Optional[Union[int, np.ndarray]] = None,
                               binsy: Optional[Union[int, np.ndarray]] = None) -> None:
        """
        Calculate any 2D histogram of strike points variables

        Jose Rueda Rueda: jrrueda@us.es

        :param  varx: variable selected for the x axis
        :param  vary: variable selected for the y axis
        :param  binsx: bining for the x variable, if a number, this number of
            bins will be created between the xmin and xmax. If an array, it
            will be interpreted as bin edges. By default, 25 bins are
            considered
        :param  binsy: similar to binsx but for the y variable

        The function creates on the histogram atribute of the object 3
        dictionaries named as <varx + '_' + vary> for the counts
        <varx + '_' + vary + '_w'> for the weight into the scintillator
        and <varx + '_' + vary + '_w0'> for the weight at the detector entrance
        [These last two only present if 'weight' and 'weoght0' are inside the
        data]. Each dict will contain 0: Total histogram, i: kind separated
        histograms (for FIDASIM markers only). On each one you will have:
            'xcen': cell centers on the x axis,
            'ycen': cell centers on the y axis,
            'xedges': bin edges on the x axis,
            'yedges': bin edges on the y axis,
            'H': Histogram matrix, [nx, ny], normalised to bin area
        """
        # --- Check if the variables we need actually exist
        if (varx not in self.header['info'].keys()) or \
           (vary not in self.header['info'].keys()):
            print('Variables available: ', list(self.header['info'].keys()))
            raise Exception('Variables not found')
        # --- Check if the histogram is already there
        if (varx + '_' + vary) in self.histograms.keys():
            logger.warning('11: Histogram present, overwritting')
        # --- Find the needed colums:
        if not varx.endswith('cam'):
            jx = self.header['info'][varx]['i']
            jy = self.header['info'][vary]['i']
        else:
            # This is to avoid issues with the remap of the camera frame, as
            # latter we will adopt the IDL criteria for camera frames and all
            #  is a bit messy. Sorry
            text = 'varx and vary exchanged'
            logger.warning('a3: %s' % text)
            jx = self.header['info'][vary]['i']
            jy = self.header['info'][varx]['i']

        try:   # FILD strike points has no weight
            jw = self.header['info']['weight']['i']
        except KeyError:
            jw = None
        try:   # For 2.0 SINPA files with 2 weights
            jw0 = self.header['info']['weight0']['i']
        except KeyError:
            jw0 = None
        try:   # We can have optics in the camera, which include optical models
            jwc = self.header['info']['wcam']['i']
        except KeyError:
            jwc = None
        try:   # For 2.0 SINPA files with 2 weights
            jk = self.header['info']['kind']['i']
        except KeyError:
            jk = None
        # --- Define the grid for the histogram
        if (binsx is None) or isinstance(binsx, int):
            xmin = np.inf
            xmax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        xmin = min(self.data[ia, ig][:, jx].min(), xmin)
                        xmax = max(self.data[ia, ig][:, jx].max(), xmax)
            if binsx is None:
                edgesx = np.linspace(xmin, xmax, 25)
            else:
                edgesx = np.linspace(xmin, xmax, binsx+1)
        else:
            edgesx = binsx
        if (binsy is None) or isinstance(binsx, int):
            ymin = np.inf
            ymax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        ymin = min(self.data[ia, ig][:, jy].min(), ymin)
                        ymax = max(self.data[ia, ig][:, jy].max(), ymax)
            if binsy is None:
                edgesy = np.linspace(ymin, ymax, 25)
            else:
                edgesy = np.linspace(ymin, ymax, binsy+1)
        else:
            edgesy = binsy
        # --- Preallocate the data
        histName = varx + '_' + vary
        self.histograms[histName] = xr.Dataset()
        # kind of markers:
        supportedKinds = [0, 5, 6, 7, 8]
        if self.header['FILDSIMmode']:
            supportedKinds = [0,]
        nkinds = len(supportedKinds)
        # Prepare the matrices
        # Basic (counts)
        data = np.zeros((edgesx.size - 1, edgesy.size - 1, nkinds))
        # For the weight at thedetecor entrance
        if jw0 is not None:
            data0 = np.zeros((edgesx.size - 1, edgesy.size - 1, nkinds))
        # For the weight at the scintillator
        if jw is not None:
            dataS = np.zeros((edgesx.size - 1, edgesy.size - 1, nkinds))
        # For the weight of the camera
        if jwc is not None:
            dataC = np.zeros((edgesx.size - 1, edgesy.size - 1, nkinds))
        for ik, k in enumerate(supportedKinds):
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 1:
                        # Skip if there are not markers of that kind
                        if k != 0:
                            f = self.data[ig, ia][:, jk].astype(int) == k
                            if f.sum() == 0:
                                continue
                        else:
                            f = np.ones(self.data[ig, ia][:, 0].size, bool)
                        # Count histogram
                        H, xedges, yedges = \
                            np.histogram2d(self.data[ia, ig][f, jx],
                                           self.data[ia, ig][f, jy],
                                           bins=(edgesx, edgesy))
                        data[:, :, ik] += H
                        # Weight histogram
                        if jw is not None:
                            H, xedges, yedges = \
                                np.histogram2d(self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               bins=(edgesx, edgesy),
                                               weights=self.data[ia, ig][f, jw])
                            dataS[:, :, ik] += H
                        # Entrance weight histogram
                        if jw0 is not None:
                            H, xedges, yedges = \
                                np.histogram2d(self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               bins=(edgesx, edgesy),
                                               weights=self.data[ia, ig][f, jw0])
                            data0[:, :, ik] += H
                        if jwc is not None:
                            H, xedges, yedges = \
                                np.histogram2d(self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               bins=(edgesx, edgesy),
                                               weights=self.data[ia, ig][f, jwc])
                            dataC[:, :, ik] += H
        xcen = 0.5 * (xedges[1:] + xedges[:-1])
        ycen = 0.5 * (yedges[1:] + yedges[:-1])
        deltax = xcen[1] - xcen[0]
        deltay = ycen[1] - ycen[0]
        data /= deltax * deltay
        self.histograms[histName]['markers'] = xr.DataArray(
            data, dims=('x', 'y', 'kind'),
            coords={'x': xcen, 'y': ycen, 'kind': supportedKinds}
        )

        #  Set the attributes for the particular histogram
        self.histograms[histName]['markers'].attrs['Description'] = \
            'Number of markers histogram'
        self.histograms[histName]['markers'].attrs['units'] = \
            '#/(' + self.header['info'][varx]['units'] + '$\\cdot$' +\
            self.header['info'][vary]['units'] + ')'
        self.histograms[histName]['markers'].attrs['long_name'] = 'Markers'
        if jw is not None:
            dataS /= deltax * deltay
            self.histograms[histName]['w'] = xr.DataArray(
                dataS, dims=('x', 'y', 'kind'),
                coords={'x': xcen, 'y': ycen, 'kind': supportedKinds}
            )
            self.histograms[histName]['w'].attrs['Description'] = \
                'Weight at the scintillator'
            self.histograms[histName]['w'].attrs['units'] = \
                self.header['info']['weight']['units'] +\
                '/(' + self.header['info'][varx]['units'] + '$\\cdot$' +\
                self.header['info'][vary]['units'] + ')'
            self.histograms[histName]['w'].attrs['long_name'] = '$W_{Scint}$'
        if jw0 is not None:
            data0 /= deltax * deltay
            self.histograms[histName]['w0'] = xr.DataArray(
                data0, dims=('x', 'y', 'kind'),
                coords={'x': xcen, 'y': ycen, 'kind': supportedKinds}
            )
            self.histograms[histName]['w0'].attrs['Description'] = \
                'Weight at the pinhole'
            self.histograms[histName]['w0'].attrs['units'] = \
                self.header['info']['weight0']['units'] +\
                '/(' + self.header['info'][varx]['units'] + '$\\cdot$' +\
                self.header['info'][vary]['units'] + ')'
            self.histograms[histName]['w0'].attrs['long_name'] = '$W_{Pin}$'
        if jwc is not None:
            dataC /= deltax * deltay
            self.histograms[histName]['wcam'] = xr.DataArray(
                dataC, dims=('x', 'y', 'kind'),
                coords={'x': xcen, 'y': ycen, 'kind': supportedKinds}
            )
            self.histograms[histName]['wcam'].attrs['Description'] = \
                'Weight at the camera'
            self.histograms[histName]['wcam'].attrs['units'] = '[a.u.]'
            self.histograms[histName]['wcam'].attrs['long_name'] = '$W_{cam}$'
        # Set the variables attributes
        self.histograms[histName]['x'].attrs['long_name'] = \
            self.header['info'][varx]['shortName']
        self.histograms[histName]['y'].attrs['long_name'] = \
            self.header['info'][vary]['shortName']
        self.histograms[histName]['x'].attrs['units'] = \
            self.header['info'][varx]['units']
        self.histograms[histName]['y'].attrs['units'] = \
            self.header['info'][vary]['units']
        self.histograms[histName]['kind'].attrs['long_name'] = 'Marker kind'
        # Set the attributes of the data set
        self.histograms[histName].attrs['xedges'] = xedges
        self.histograms[histName].attrs['yedges'] = yedges
        self.histograms[histName].attrs['area'] = deltax * deltay

    def calculate_1d_histogram(self, var: str = 'xcx',
                               bins: Optional[Union[int, np.ndarray]] = None) -> None:
        """
        Calculate any 1D histogram of strike points variables

        Jose Rueda Rueda: jrrueda@us.es

        :param  var: variable selected for the x axis
        :param  bins: bining for the x variable, if a number, this number of
            bins will be creaded between the xmin and xmax. If an array, it
            will be interpreted as bin edges. By default, 25 bins are
            considered

        The function creates on the histogram atribute of the object 3
        dictionaries named as <var> for the counts
        <var + '_w'> for the weight into the scintillator
        and <var + '_w0'> for the weight at the detector entrance
        [These last two only present if 'weight' and 'weoght0' are inside the
        data]. Each dict will contain 0: Total histogram, i: kind separated
        histograms (for FIDASIM markers only). On each one you will have:
            'xcen': cell centers on the x axis,
            'xedges': bin edges on the x axis,
            'H': Histogram array, [nx], normalised to bin area
        """
        # --- Check if the variables we need actually exist
        if (var not in self.header['info'].keys()):
            print('Variables available: ', list(self.header['info'].keys()))
            raise Exception('Variables not found')
        # --- Check if the histogram is already there
        if var in self.histograms.keys():
            logger.warning('11: Histogram present, overwritting')
        # --- Find the needed colums:
        dat = self(var)
        w = self('weight')
        w0 = self('weight0')
        k = self('kind')
        # --- Define the grid for the histogram
        if (bins is None) or isinstance(bins, int):
            xmin = dat.min()
            xmax = dat.max()
            if bins is None:
                edgesx = np.linspace(xmin, xmax, 25)
            else:
                edgesx = np.linspace(xmin, xmax, bins+1)
        else:
            edgesx = bins
        # --- Preallocate the data
        varw = var + '_w'
        varw0 = var + 'w_0'
        # Basic (counts)
        self.histograms[var] = \
            {0: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        # For the weight at thedetecor entrance
        if w0 is not None:
            self.histograms[varw0] = \
                {0: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        # For the weight at the scintillator
        if w is not None:
            self.histograms[varw] = \
                {0: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        # We could calculate the complete histogram (case k = 0) as the sum of
        # the individual histogram, this will be more efficient but we would
        # have the issue that the FILD strike points does not have this kind
        # separation. So we would need to duplicate, for FILD straight
        # calculation, and for INPA signal the sum. Computationally speaking is
        # fast, so to simplify, I would perform the full calculation
        H, xedges = \
            np.histogram(dat, bins=edgesx)
        xcen = 0.5 * (xedges[1:] + xedges[:-1])
        deltax = xcen[1] - xcen[0]
        self.histograms[var][0] = {
            'xcen': xcen,
            'xedges': xedges,
            'H': H.astype(float)/deltax
        }
        if w is not None:
            H, xedges = \
                np.histogram(dat, bins=edgesx, weights=w)
            self.histograms[varw][0] = {
                'xcen': xcen,
                'xedges': xedges,
                'H': H.astype(float)/deltax
            }
        if w0 is not None:
            H, xedges = \
                np.histogram(dat, bins=edgesx, weights=w0)
            self.histograms[varw0][0] = {
                'xcen': xcen,
                'xedges': xedges,
                'H': H.astype(float)/deltax
            }
        # Now repeat the same for the different kinds
        if k is not None:
            for kind in [5, 6, 7, 8]:
                flags = k == kind
                # Count histogram
                H, xedges = np.histogram(dat[flags], bins=edgesx)
                xcen = 0.5 * (xedges[1:] + xedges[:-1])
                deltax = xcen[1] - xcen[0]
                self.histograms[var][kind] = {
                    'xcen': xcen,
                    'xedges': xedges,
                    'H': H.astype(float)/deltax
                }
                if w is not None:
                    H, xedges = \
                        np.histogram(dat[flags], bins=edgesx, weights=w[flags])
                    self.histograms[varw][kind] = {
                        'xcen': xcen,
                        'xedges': xedges,
                        'H': H.astype(float)/deltax
                    }
                if w0 is not None:
                    H, xedges = \
                        np.histogram(dat[flags], bins=edgesx,
                                     weights=w0[flags])
                    self.histograms[varw0][kind] = {
                        'xcen': xcen,
                        'xedges': xedges,
                        'H': H.astype(float)/deltax
                    }

    def calculate_3d_histogram(self, varx: str = 'xcx', vary: str = 'yxc',
                               varz: str = 'zxc',
                               binsx: Optional[Union[int, np.ndarray]] = None,
                               binsy: Optional[Union[int, np.ndarray]] = None,
                               binsz: Optional[Union[int, np.ndarray]] = None) -> None:
        """
        Calculate any 3D histogram of strike points variables

        Jose Rueda Rueda: jrrueda@us.es

        :param  varx: variable selected for the x axis
        :param  vary: variable selected for the y axis
        :param  binsx: bining for the x variable, if a number, this number of
            bins will be created between the xmin and xmax. If an array, it
            will be interpreted as bin edges. By default, 25 bins are
            considered
        :param  binsy: similar to binsx but for the y variable

        The function creates on the histogram atribute of the object 3
        dictionaries named as <varx + '_' + vary> for the counts
        <varx + '_' + vary + '_w'> for the weight into the scintillator
        and <varx + '_' + vary + '_w0'> for the weight at the detector entrance
        [These last two only present if 'weight' and 'weoght0' are inside the
        data]. Each dict will contain 0: Total histogram, i: kind separated
        histograms (for FIDASIM markers only). On each one you will have:
            'xcen': cell centers on the x axis,
            'ycen': cell centers on the y axis,
            'xedges': bin edges on the x axis,
            'yedges': bin edges on the y axis,
            'H': Histogram matrix, [nx, ny], normalised to bin area
        """
        # --- Check if the variables we need actually exist
        if (varx not in self.header['info'].keys()) or \
           (vary not in self.header['info'].keys()) or \
           (varz not in self.header['info'].keys()):
            print('Variables available: ', list(self.header['info'].keys()))
            raise Exception('Variables not found')
        # --- Check if the histogram is already there
        if (varx + '_' + vary + '_' + varz) in self.histograms.keys():
            logger.warning('11: Histogram present, overwritting')
        # --- Find the needed colums:
        if not varx.endswith('cam'):
            jx = self.header['info'][varx]['i']
            jy = self.header['info'][vary]['i']
            jz = self.header['info'][varz]['i']
        else:
            if varz.endswith('cam'):
                raise Exception('Sorry not implemented, permute variables')
            # This is to avoid issues with the remap of the camera frame, as
            # latter we will adopt the IDL criteria for camera frames and all
            #  is a bit messy. Sorry
            text = 'varx and vary exchanged'
            logger.warning('a3: %s' % text)
            jx = self.header['info'][vary]['i']
            jy = self.header['info'][varx]['i']
            jz = self.header['info'][varz]['i']

        try:   # FILD strike points has no weight
            jw = self.header['info']['weight']['i']
        except KeyError:
            jw = None
        try:   # For 2.0 SINPA files with 2 weights
            jw0 = self.header['info']['weight0']['i']
        except KeyError:
            jw0 = None
        try:   # We can have optics in the camera, which include optical models
            jwc = self.header['info']['wcam']['i']
        except KeyError:
            jwc = None
        try:   # For 2.0 SINPA files with 2 weights
            jk = self.header['info']['kind']['i']
        except KeyError:
            jk = None
        # --- Define the grid for the histogram
        if (binsx is None) or isinstance(binsx, int):
            xmin = np.inf
            xmax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        xmin = min(self.data[ia, ig][:, jx].min(), xmin)
                        xmax = max(self.data[ia, ig][:, jx].max(), xmax)
            if binsx is None:
                edgesx = np.linspace(xmin, xmax, 25)
            else:
                edgesx = np.linspace(xmin, xmax, binsx+1)
        else:
            edgesx = binsx
        if (binsy is None) or isinstance(binsy, int):
            ymin = np.inf
            ymax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        ymin = min(self.data[ia, ig][:, jy].min(), ymin)
                        ymax = max(self.data[ia, ig][:, jy].max(), ymax)
            if binsy is None:
                edgesy = np.linspace(ymin, ymax, 25)
            else:
                edgesy = np.linspace(ymin, ymax, binsy+1)
        else:
            edgesy = binsy
        if (binsz is None) or isinstance(binsz, int):
            zmin = np.inf
            zmax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        zmin = min(self.data[ia, ig][:, jz].min(), zmin)
                        zmax = max(self.data[ia, ig][:, jz].max(), zmax)
            if binsz is None:
                edgesz = np.linspace(zmin, zmax, 25)
            else:
                edgesz = np.linspace(zmin, zmax, binsz+1)
        else:
            edgesz = binsz
        # --- Preallocate the data
        histName = varx + '_' + vary + '_' + varz
        self.histograms[histName] = xr.Dataset()
        # kind of markers:
        supportedKinds = [0, 5, 6, 7, 8]
        if self.header['FILDSIMmode']:
            supportedKinds = [0,]
        nkinds = len(supportedKinds)
        # Prepare the matrices
        # Basic (counts)
        data = np.zeros((edgesx.size - 1, edgesy.size - 1, 
                         edgesz.size - 1, nkinds))

        # For the weight at the detecor entrance
        if jw0 is not None:
            data0 = np.zeros((edgesx.size - 1, edgesy.size - 1, 
                              edgesz.size - 1, nkinds))
        # For the weight at the scintillator
        if jw is not None:
            dataS = np.zeros((edgesx.size - 1, edgesy.size - 1, 
                              edgesz.size - 1, nkinds))
        # For the weight of the camera
        if jwc is not None:
            dataC = np.zeros((edgesx.size - 1, edgesy.size - 1, 
                              edgesz.size - 1, nkinds))
        for ik, k in enumerate(supportedKinds):
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 1:
                        # Skip if there are not markers of that kind
                        if k != 0:
                            f = self.data[ig, ia][:, jk].astype(int) == k
                            if f.sum() == 0:
                                continue
                        else:
                            f = np.ones(self.data[ig, ia][:, 0].size, bool)
                        # Count histogram
                        H, (xedges, yedges, zedges) = \
                            np.histogramdd((self.data[ia, ig][f, jx],
                                            self.data[ia, ig][f, jy],
                                            self.data[ia, ig][f, jz]),
                                           bins=(edgesx, edgesy, edgesz))
                        data[:, :, :, ik] += H
                        # Weight histogram
                        if jw is not None:
                            H, (xedges, yedges, zedges) = \
                                np.histogramdd((self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               self.data[ia, ig][f, jz]),
                                               bins=(edgesx, edgesy, edgesz),
                                               weights=self.data[ia, ig][f, jw])
                            dataS[:, :, :, ik] += H
                        # Entrance weight histogram
                        if jw0 is not None:
                            H, (xedges, yedges, zedges) = \
                                np.histogramdd((self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               self.data[ia, ig][f, jz]),
                                               bins=(edgesx, edgesy, edgesz),
                                               weights=self.data[ia, ig][f, jw0])
                            data0[:, :, :, ik] += H
                        if jwc is not None:
                            H, (xedges, yedges, zedges) = \
                                np.histogramdd((self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               self.data[ia, ig][f, jz]),
                                               bins=(edgesx, edgesy, edgesz),
                                               weights=self.data[ia, ig][f, jwc])
                            dataC[:, :, :, ik] += H
        xcen = 0.5 * (xedges[1:] + xedges[:-1])
        ycen = 0.5 * (yedges[1:] + yedges[:-1])
        zcen = 0.5 * (zedges[1:] + zedges[:-1])
        deltax = xcen[1] - xcen[0]
        deltay = ycen[1] - ycen[0]
        deltaz = zcen[1] - zcen[0]
        data /= deltax * deltay * deltaz
        self.histograms[histName]['markers'] = xr.DataArray(
            data, dims=('x', 'y', 'z', 'kind'),
            coords={'x': xcen, 'y': ycen, 'z': zcen, 
                    'kind': supportedKinds}
        )

        #  Set the attributes for the particular histogram
        self.histograms[histName]['markers'].attrs['Description'] = \
            'Number of markers histogram'
        self.histograms[histName]['markers'].attrs['units'] = \
            '#/(' + self.header['info'][varx]['units'] + '$\\cdot$' +\
            self.header['info'][vary]['units'] + \
            self.header['info'][varz]['units'] + ')'
        self.histograms[histName]['markers'].attrs['long_name'] = 'Markers'
        if jw is not None:
            dataS /= deltax * deltay
            self.histograms[histName]['w'] = xr.DataArray(
                dataS, dims=('x', 'y', 'z', 'kind'),
                coords={'x': xcen, 'y': ycen, 'z': zcen, 
                        'kind': supportedKinds}
            )
            self.histograms[histName]['w'].attrs['Description'] = \
                'Weight at the scintillator'
            self.histograms[histName]['w'].attrs['units'] = \
                self.header['info']['weight']['units'] +\
                '/(' + self.header['info'][varx]['units'] + '$\\cdot$' +\
                self.header['info'][vary]['units'] + \
                self.header['info'][varz]['units'] + ')'
            self.histograms[histName]['w'].attrs['long_name'] = '$W_{Scint}$'
        if jw0 is not None:
            data0 /= deltax * deltay
            self.histograms[histName]['w0'] = xr.DataArray(
                data0, dims=('x', 'y', 'z', 'kind'),
                coords={'x': xcen, 'y': ycen, 'z': zcen, 
                        'kind': supportedKinds}
            )
            self.histograms[histName]['w0'].attrs['Description'] = \
                'Weight at the pinhole'
            self.histograms[histName]['w0'].attrs['units'] = \
                self.header['info']['weight0']['units'] +\
                '/(' + self.header['info'][varx]['units'] + '$\\cdot$' +\
                self.header['info'][vary]['units'] + \
                self.header['info'][varz]['units'] + ')'
            self.histograms[histName]['w0'].attrs['long_name'] = '$W_{Pin}$'
        if jwc is not None:
            dataC /= deltax * deltay
            self.histograms[histName]['wcam'] = xr.DataArray(
                dataC, dims=('x', 'y', 'z', 'kind'),
                coords={'x': xcen, 'y': ycen, 'z': zcen, 
                        'kind': supportedKinds}
            )
            self.histograms[histName]['wcam'].attrs['Description'] = \
                'Weight at the camera'
            self.histograms[histName]['wcam'].attrs['units'] = '[a.u.]'
            self.histograms[histName]['wcam'].attrs['long_name'] = '$W_{cam}$'
        # Set the variables attributes
        self.histograms[histName]['x'].attrs['long_name'] = \
            self.header['info'][varx]['shortName']
        self.histograms[histName]['y'].attrs['long_name'] = \
            self.header['info'][vary]['shortName']
        self.histograms[histName]['x'].attrs['units'] = \
            self.header['info'][varx]['units']
        self.histograms[histName]['y'].attrs['units'] = \
            self.header['info'][vary]['units']        
        self.histograms[histName]['z'].attrs['long_name'] = \
            self.header['info'][varz]['shortName']
        self.histograms[histName]['z'].attrs['units'] = \
            self.header['info'][varz]['units']
        self.histograms[histName]['kind'].attrs['long_name'] = 'Marker kind'
        # Set the attributes of the data set
        self.histograms[histName].attrs['xedges'] = xedges
        self.histograms[histName].attrs['yedges'] = yedges
        self.histograms[histName].attrs['zedges'] = zedges
        self.histograms[histName].attrs['area'] = deltax * deltay * deltaz
    
    def calculate_4d_histogram(self, varx1: str = 'xcx', varx2: str = 'yxc',
                               varx3: str = 'zxc', varx4: str = 'e0',
                               binsx1: Optional[Union[int, np.ndarray]] = None,
                               binsx2: Optional[Union[int, np.ndarray]] = None,
                               binsx3: Optional[Union[int, np.ndarray]] = None,
                               binsx4: Optional[Union[int, np.ndarray]] = None,
                               limitation: Optional[float] = None) -> None:
        """
        Calculate any 4D histogram of strike points variables.

        Jose Rueda Rueda: jrrueda@us.es

        :param  varx1: variable selected for the first axis
        :param  varx2: variable selected for the second axis
        :param  varx3: variable selected for the third axis
        :param  varx4: variable selected for the fourth axis
        :param  binsx1: bining for the x1 variable, if a number, this number of
            bins will be created between the x1min and x1max. If an array, it
            will be interpreted as bin edges. By default, 25 bins are
            considered
        :param  binsx2: similar to binsx but for the x2 variable
        :param  binsx3: similar to binsx but for the x3 variable
        :param  binsx4: similar to binsx but for the x4 variable
        """
        # --- Check if the variables we need actually exist
        if (varx1 not in self.header['info'].keys()) or \
           (varx2 not in self.header['info'].keys()) or \
           (varx3 not in self.header['info'].keys()) or \
           (varx4 not in self.header['info'].keys()):
            print('Variables available: ', list(self.header['info'].keys()))
            raise Exception('Variables not found')
        # --- Check if the histogram is already there
        histName = varx1 + '_' + varx2 + '_' + varx3 + '_' + varx4
        if histName in self.histograms.keys():
            logger.warning('11: Histogram present, overwritting')
        # --- Find the needed colums:
        if not varx1.endswith('cam'):
            jx = self.header['info'][varx1]['i']
            jy = self.header['info'][varx2]['i']
            jz = self.header['info'][varx3]['i']
            jt = self.header['info'][varx4]['i']
        else:
            if varx3.endswith('cam') or varx4.endswith('cam'):
                raise Exception('Sorry not implemented, permute variables')
            # This is to avoid issues with the remap of the camera frame, as
            # latter we will adopt the IDL criteria for camera frames and all
            #  is a bit messy. Sorry
            text = 'varx and vary exchanged'
            logger.warning('a3: %s' % text)
            jx = self.header['info'][varx2]['i']
            jy = self.header['info'][varx1]['i']
            jz = self.header['info'][varx3]['i']
            jt = self.header['info'][varx4]['i']

        try:   # FILD strike points has no weight
            jw = self.header['info']['weight']['i']
        except KeyError:
            jw = None
        try:   # For 2.0 SINPA files with 2 weights
            jw0 = self.header['info']['weight0']['i']
        except KeyError:
            jw0 = None
        try:   # We can have optics in the camera, which include optical models
            jwc = self.header['info']['wcam']['i']
        except KeyError:
            jwc = None
        try:   # For 2.0 SINPA files with 2 weights
            jk = self.header['info']['kind']['i']
        except KeyError:
            jk = None
        # --- Define the grid for the histogram
        if (binsx1 is None) or isinstance(binsx1, int):
            xmin = np.inf
            xmax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        xmin = min(self.data[ia, ig][:, jx].min(), xmin)
                        xmax = max(self.data[ia, ig][:, jx].max(), xmax)
            if binsx1 is None:
                edgesx = np.linspace(xmin, xmax, 25)
            else:
                edgesx = np.linspace(xmin, xmax, binsx1+1)
        else:
            edgesx = binsx1
        if (binsx2 is None) or isinstance(binsx2, int):
            ymin = np.inf
            ymax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        ymin = min(self.data[ia, ig][:, jy].min(), ymin)
                        ymax = max(self.data[ia, ig][:, jy].max(), ymax)
            if binsx2 is None:
                edgesy = np.linspace(ymin, ymax, 25)
            else:
                edgesy = np.linspace(ymin, ymax, binsx2+1)
        else:
            edgesy = binsx2
        if (binsx3 is None) or isinstance(binsx3, int):
            zmin = np.inf
            zmax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        zmin = min(self.data[ia, ig][:, jz].min(), zmin)
                        zmax = max(self.data[ia, ig][:, jz].max(), zmax)
            if binsx3 is None:
                edgesz = np.linspace(zmin, zmax, 25)
            else:
                edgesz = np.linspace(zmin, zmax, binsx3+1)
        else:
            edgesz = binsx3        
        if (binsx4 is None) or isinstance(binsx4, int):
            tmin = np.inf
            tmax = -np.inf
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 0:
                        tmin = min(self.data[ia, ig][:, jt].min(), tmin)
                        tmax = max(self.data[ia, ig][:, jt].max(), tmax)
            if binsx4 is None:
                edgest = np.linspace(tmin, tmax, 25)
            else:
                edgest = np.linspace(tmin, tmax, binsx4+1)
        else:
            edgest = binsx4
        # --- Preallocate the data
        self.histograms[histName] = xr.Dataset()
        # kind of markers:
        supportedKinds = [0, 5, 6, 7, 8]
        if self.header['FILDSIMmode']:
            supportedKinds = [0,]
        nkinds = len(supportedKinds)
        # Prepare the matrices
        # Basic (counts)
        data = np.zeros((edgesx.size - 1, edgesy.size - 1, 
                         edgesz.size - 1, edgest.size - 1, nkinds))

        # For the weight at the detecor entrance
        if jw0 is not None:
            data0 = np.zeros((edgesx.size - 1, edgesy.size - 1, 
                              edgesz.size - 1, edgest.size - 1, nkinds))
        # For the weight at the scintillator
        if jw is not None:
            dataS = np.zeros((edgesx.size - 1, edgesy.size - 1, 
                              edgesz.size - 1, edgest.size - 1, nkinds))
        # For the weight of the camera
        if jwc is not None:
            dataC = np.zeros((edgesx.size - 1, edgesy.size - 1, 
                              edgesz.size - 1, edgest.size - 1, nkinds))
        for ik, k in enumerate(supportedKinds):
            for ig in range(self.header['ngyr']):
                for ia in range(self.header['nXI']):
                    if self.header['counters'][ia, ig] > 1:
                        # Skip if there are not markers of that kind
                        if k != 0:
                            f = self.data[ig, ia][:, jk].astype(int) == k
                            if f.sum() == 0:
                                continue
                        else:
                            f = np.ones(self.data[ig, ia][:, 0].size, bool)
                        # Count histogram
                        H, (xedges, yedges, zedges, tedges) = \
                            np.histogramdd((self.data[ia, ig][f, jx],
                                            self.data[ia, ig][f, jy],
                                            self.data[ia, ig][f, jz],
                                            self.data[ia, ig][f, jt]),
                                           bins=(edgesx, edgesy, edgesz, edgest))
                        data[:, :, :, :, ik] += H
                        # Weight histogram
                        if jw is not None:
                            H, (xedges, yedges, zedges, tedges) = \
                                np.histogramdd((self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               self.data[ia, ig][f, jz],
                                               self.data[ia, ig][f, jt]),
                                               bins=(edgesx, edgesy, edgesz, edgest),
                                               weights=self.data[ia, ig][f, jw])
                            dataS[:, :, :, :, ik] += H
                        # Entrance weight histogram
                        if jw0 is not None:
                            H, (xedges, yedges, zedges, tedges) = \
                                np.histogramdd((self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               self.data[ia, ig][f, jz],
                                               self.data[ia, ig][f, jt]),
                                               bins=(edgesx, edgesy, edgesz, edgest),
                                               weights=self.data[ia, ig][f, jw0])
                            data0[:, :, :, :, ik] += H
                        if jwc is not None:
                            H, (xedges, yedges, zedges, tedges) = \
                                np.histogramdd((self.data[ia, ig][f, jx],
                                               self.data[ia, ig][f, jy],
                                               self.data[ia, ig][f, jz],
                                               self.data[ia, ig][f, jt]),
                                               bins=(edgesx, edgesy, edgesz, edgest),
                                               weights=self.data[ia, ig][f, jwc])
                            dataC[:, :, :, :, ik] += H
        xcen = 0.5 * (xedges[1:] + xedges[:-1])
        ycen = 0.5 * (yedges[1:] + yedges[:-1])
        zcen = 0.5 * (zedges[1:] + zedges[:-1])
        tcen = 0.5 * (tedges[1:] + tedges[:-1])
        deltax = xcen[1] - xcen[0]
        deltay = ycen[1] - ycen[0]
        deltaz = zcen[1] - zcen[0]
        deltat = tcen[1] - tcen[0]
        data /= deltax * deltay * deltaz * deltat
        self.histograms[histName]['markers'] = xr.DataArray(
            data, dims=('x1', 'x2', 'x3', 'x4', 'kind'),
            coords={'x1': xcen, 'x2': ycen, 'x3': zcen, 'x4': tcen, 
                    'kind': supportedKinds}
        )

        #  Set the attributes for the particular histogram
        self.histograms[histName]['markers'].attrs['Description'] = \
            'Number of markers histogram'
        self.histograms[histName]['markers'].attrs['units'] = \
            '#/(' + self.header['info'][varx1]['units'] + '$\\cdot$' +\
            self.header['info'][varx2]['units'] + \
            self.header['info'][varx3]['units'] + \
            self.header['info'][varx4]['units'] + ')'
        self.histograms[histName]['markers'].attrs['long_name'] = 'Markers'
        if jw is not None:
            dataS /= deltax * deltay * deltaz * deltat
            self.histograms[histName]['w'] = xr.DataArray(
                dataS, dims=('x1', 'x2', 'x3', 'x4', 'kind'),
                coords={'x1': xcen, 'x2': ycen, 'x3': zcen, 'x4': tcen, 
                        'kind': supportedKinds}
            )
            self.histograms[histName]['w'].attrs['Description'] = \
                'Weight at the scintillator'
            self.histograms[histName]['w'].attrs['units'] = \
                self.header['info']['weight']['units'] +\
                '/(' + self.header['info'][varx1]['units'] + '$\\cdot$' +\
                self.header['info'][varx2]['units'] + \
                self.header['info'][varx3]['units'] + \
                self.header['info'][varx4]['units'] + ')'
            self.histograms[histName]['w'].attrs['long_name'] = '$W_{Scint}$'
        if jw0 is not None:
            data0 /= deltax * deltay * deltaz * deltat
            self.histograms[histName]['w0'] = xr.DataArray(
                data0, dims=('x1', 'x2', 'x3', 'x4', 'kind'),
                coords={'x1': xcen, 'x2': ycen, 'x3': zcen, 'x4': tcen, 
                        'kind': supportedKinds}
            )
            self.histograms[histName]['w0'].attrs['Description'] = \
                'Weight at the pinhole'
            self.histograms[histName]['w0'].attrs['units'] = \
                self.header['info']['weight0']['units'] +\
                '/(' + self.header['info'][varx1]['units'] + '$\\cdot$' +\
                self.header['info'][varx2]['units'] + \
                self.header['info'][varx3]['units'] + \
                self.header['info'][varx4]['units'] + ')'
            self.histograms[histName]['w0'].attrs['long_name'] = '$W_{Pin}$'
        if jwc is not None:
            dataC /= deltax * deltay * deltaz * deltat
            self.histograms[histName]['wcam'] = xr.DataArray(
                dataC, dims=('x1', 'x2', 'x3', 'x4', 'kind'),
                coords={'x1': xcen, 'x2': ycen, 'x3': zcen, 'x4': tcen, 
                        'kind': supportedKinds}
            )
            self.histograms[histName]['wcam'].attrs['Description'] = \
                'Weight at the camera'
            self.histograms[histName]['wcam'].attrs['units'] = '[a.u.]'
            self.histograms[histName]['wcam'].attrs['long_name'] = '$W_{cam}$'
        # Set the variables attributes
        self.histograms[histName]['x1'].attrs['long_name'] = \
            self.header['info'][varx1]['shortName']
        self.histograms[histName]['x2'].attrs['long_name'] = \
            self.header['info'][varx2]['shortName']
        self.histograms[histName]['x1'].attrs['units'] = \
            self.header['info'][varx2]['units']
        self.histograms[histName]['x2'].attrs['units'] = \
            self.header['info'][varx3]['units']        
        self.histograms[histName]['x3'].attrs['long_name'] = \
            self.header['info'][varx3]['shortName']
        self.histograms[histName]['x3'].attrs['units'] = \
            self.header['info'][varx3]['units']
        self.histograms[histName]['kind'].attrs['long_name'] = 'Marker kind'
        # Set the attributes of the data set
        self.histograms[histName].attrs['x1edges'] = xedges
        self.histograms[histName].attrs['x2edges'] = yedges
        self.histograms[histName].attrs['x3edges'] = zedges
        self.histograms[histName].attrs['x4edges'] = tedges
        self.histograms[histName].attrs['area'] = deltax * deltay * deltaz * \
            deltat
    # -------------------------------------------------------------------------
    # --- Data handling block
    # -------------------------------------------------------------------------
    def get(self, var, gyroradius_index=None, XI_index=None)->np.ndarray:
        """
        Return an array with the values of 'var' for all strike points.

        Jose Rueda - jrrueda@us.es

        :param  var: variable to be returned
        :param  gyroradius_index: index (or indeces if given as an np.array) of
            gyroradii to plot
        :param  XI_index: index (or indeces if given as an np.array) of
            XIs (pitch or R) to plot

        :return variable: array of values, all of them will be concatenated
            in a single 1D array, indepentendly of gyroradius_index or XI_index
        """
        try:
            column_to_plot = self.header['info'][var]['i']
        except KeyError:
            print('Available variables: ')
            print(self.header['info'].keys())
            raise errors.NotFoundVariable()

        # --- get the values the markers:
        nXI, ngyr = self.header['counters'].shape

        # See which gyroradius / pitch we need
        if gyroradius_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyroradius_index, (list, np.ndarray)):
                index_gyr = gyroradius_index
            else:  # it should be just a number
                index_gyr = np.array([gyroradius_index])
        if XI_index is None:  # if None, use all gyroradii
            index_XI = range(nXI)
        else:
            # Test if it is a list or array
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:  # it should be just a number
                index_XI = np.array([XI_index])
        var = []
        for ig in index_gyr:
            for ia in index_XI:
                if self.header['counters'][ia, ig] > 0:
                    var.append(self.data[ia, ig][:, column_to_plot])
        return np.array(var).flatten()

    # -------------------------------------------------------------------------
    # --- Plotting functions
    # -------------------------------------------------------------------------
    def plot3D(self, per: float = 0.1, ax: Optional[plt.Axes] = None, 
               mar_params: dict = {},
               gyroradius_index=None, XI_index=None,
               where: str = 'Head'):
        """
        Plot the strike points in a 3D axis as scatter points.

        Jose Rueda: jrrueda@us.es

        :param  per: ratio of markers to be plotted (1=all of them)
        :param  ax: axes where to plot
        :param  mar_params: Dictionary with the parameters for the markers
        :param  gyroradius_index: index (or indeces if given as an np.array) of
            gyroradii to plot
        :param  XI_index: index (or indeces if given as an np.array) of
            XIs (pitch or R) to plot
        :param  where: string indicating where to plot: 'head', 'NBI',
        'ScintillatorLocalSystem'. First two are in absolute
        coordinates, last one in the scintillator coordinates (see SINPA
        documentation) [Head will plot the strikes in the collimator or
        scintillator]. For oldFILDSIM, use just head
        """
        # --- Default markers
        mar_options = {
            'marker': '.',
            'color': 'k'
        }
        mar_options.update(mar_params)
        # --- Create the axes
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            created = True
        else:
            created = False
        # --- Chose the variable we want to plot
        if where.lower() == 'head':
            column_to_plot = self.header['info']['x']['i']
        elif where.lower() == 'nbi':
            column_to_plot = self.header['info']['x0']['i']
        elif where.lower() == 'scintillatorlocalsystem':
            column_to_plot = self.header['info']['x3']['i']
        else:
            raise Exception('Not understood what do you want to plot')

        # --- Plot the markers:
        nXI, ngyr = self.header['counters'].shape
        minx = +100.0  # Dummy variables to set a decent axis limit
        miny = +100.0
        minz = +100.0
        maxx = -300.0
        maxy = -300.0
        maxz = -300.0
        # See which gyroradius / pitch (R) we need
        if gyroradius_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyroradius_index, (list, np.ndarray)):
                index_gyr = gyroradius_index
            else:  # it should be just a number
                index_gyr = np.array([gyroradius_index])
        if XI_index is None:  # if None, use all gyroradii
            index_XI = range(nXI)
        else:
            # Test if it is a list or array
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:  # it should be just a number
                index_XI = np.array([XI_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_XI:
                if self.header['counters'][ia, ig] > 0:
                    flags = np.random.rand(
                        self.header['counters'][ia, ig]) < per
                    if flags.sum() > 0:
                        x = self.data[ia, ig][flags, column_to_plot]
                        minx = min(minx, x.min())
                        maxx = max(maxx, x.max())
                        y = self.data[ia, ig][flags, column_to_plot + 1]
                        miny = min(miny, y.min())
                        maxy = max(maxy, y.max())
                        z = self.data[ia, ig][flags, column_to_plot + 2]
                        minz = min(minz, z.min())
                        maxz = max(maxz, z.max())
                        ax.scatter(x, y, z, **mar_options)
        # Set axis limits and beuty paramters
        if created:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_zlim(minz, maxz)
            # Set the aspect ratio to equal
            axisEqual3D(ax)
            # Get rid of the colored panes
            clean3Daxis(ax)
        return ax

    def plot1D(self, var='beta', gyroradius_index=None, XI_index=None, ax=None,
               ax_params: dict = {}, nbins: int = 20, includeW: bool = False,
               normalise: bool = False, var_for_threshold: str = None,
               levels: tuple = None):
        """
        Plot (and calculate) the histogram of the selected variable

        Jose Rueda: jrrueda@us.es

        :param  var: variable to plot
        :param  gyroradius_index: index (or indeces if given as an np.array) of
            gyroradii to plot
        :param  XI_index: index (or indeces if given as an np.array) of
            XIs (pitch or R) to plot
        :param  ax: axes where to plot
        :param  ax_params: parameters for the axis beauty
        :param  nbins: number of bins for the 1D histogram
        :param  includeW: include weight for the histogram
        """
        # --- Get the index:
        try:
            column_to_plot = self.header['info'][var]['i']
        except KeyError:
            print('Available variables: ')
            print(self.header['info'].keys())
            raise Exception()
        if includeW:
            column_of_W = self.header['info']['weight']['i']
        if var_for_threshold is not None:
            column_of_var = self.header['info'][var_for_threshold]['i']
        # --- Default plotting options
        ax_options = {
            'grid': 'both',
            'xlabel': self.header['info'][var]['shortName']
            + self.header['info'][var]['units'],
            'ylabel': '',
        }
        ax_options.update(ax_params)
        # --- Create the axes
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Plot the markers:
        nXI, ngyr = self.header['counters'].shape

        # See which gyroradius / pitch we need
        if gyroradius_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyroradius_index, (list, np.ndarray)):
                index_gyr = gyroradius_index
            else:  # it should be just a number
                index_gyr = np.array([gyroradius_index])
        if XI_index is None:  # if None, use all gyroradii
            index_XI = range(nXI)
        else:
            # Test if it is a list or array
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:  # it should be just a number
                index_XI = np.array([XI_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_XI:
                if self.header['counters'][ia, ig] > 0:
                    if var_for_threshold is not None:
                        var2 = self.data[ia, ig][:, column_of_var]
                        flags = (var2 > levels[0]) * (var2 < levels[1])
                        dat = self.data[ia, ig][flags, column_to_plot]
                        if includeW:
                            w = self.data[ia, ig][flags, column_of_W]
                        else:
                            w = np.ones(flags.sum())
                    else:
                        dat = self.data[ia, ig][:, column_to_plot]
                        if includeW:
                            w = self.data[ia, ig][:, column_of_W]
                        else:
                            w = np.ones(self.header['counters'][ia, ig])
                    H, xe = np.histogram(dat, weights=w, bins=nbins)
                    # Normalise H
                    H /= xe[1] - xe[0]
                    xc = 0.5 * (xe[:-1] + xe[1:])
                    if normalise:
                        H /= np.abs(H).max()
                    ax.plot(xc, H)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)
        return ax

    def plot_histogram(self, hist_name, ax=None, ax_params: dict = {},
                       cmap=None, kind=0, aspect='auto',
                       normalise: bool = True):
        """
        Plot the histogram of the scintillator strikes

        Jose Rueda: jrrueda@us.es
        :param  ax: axes where to plot
        :param  ax_params: parameters for the axis beauty
        :param  cmap: color map to be used, if none -> Gamma_II()
        :param  nbins: number of bins for the 1D histogram
        :param  kind: kind of markers to consider (for FILDSIM just 0, default)
        """
        # --- Check inputs
        if hist_name not in self.histograms:
            print('Available histograms: ', self.histograms.keys())
            raise Exception('You need to calculate first the histogram')
        # --- Identify if we deal with 1 or 2d data
        if 'yedges' in self.histograms[hist_name][0]:
            twoD = True
        else:
            twoD = False
        # --- Initialise potting options
        if twoD:
            # --- Get the axel levels:
            names = hist_name.split('_')
            xvar = names[0]
            yvar = names[1]
            ax_options = {
                'xlabel': '%s [%s]' % (self.header['info'][xvar]['shortName'],
                                       self.header['info'][xvar]['units']),
                'ylabel': '%s [%s]' % (self.header['info'][yvar]['shortName'],
                                       self.header['info'][yvar]['units']),
            }
        else:
            xvar = hist_name.split('_')[0]
            ax_options = {
                'xlabel': '%s [%s]' % (self.header['info'][xvar]['shortName'],
                                       self.header['info'][xvar]['units']),
            }
        ax_options.update(ax_params)
        if cmap is None:
            cmap = ssplt.Gamma_II()
        # --- Open the figure (if needed)
        if ax is None:
            fig, ax = plt.subplots()

            created = True
        else:
            created = False
        # --- Plot the matrix
        if twoD:
            # Camera histograms are without transpose to agree with the
            # creiteria of plotting the camera frames, which is from the
            # anciente IDL times
            if 'cam' in hist_name:
                dummy = self.histograms[hist_name][kind]['H']
            else:
                dummy = self.histograms[hist_name][kind]['H'].T

            if normalise:
                dummy = dummy.copy()/dummy.max()
            # @Todo: change this black for a cmap dependent background
            ax.set_facecolor((0.0, 0.0, 0.0))  # Set black bck
            if 'cam' in hist_name:
                extent = [self.histograms[hist_name][kind]['ycen'][0],
                          self.histograms[hist_name][kind]['ycen'][-1],
                          self.histograms[hist_name][kind]['xcen'][0],
                          self.histograms[hist_name][kind]['xcen'][-1]]
            else:
                extent = [self.histograms[hist_name][kind]['xcen'][0],
                          self.histograms[hist_name][kind]['xcen'][-1],
                          self.histograms[hist_name][kind]['ycen'][0],
                          self.histograms[hist_name][kind]['ycen'][-1]]

            ax.imshow(dummy,
                      extent=extent,
                      origin='lower', cmap=cmap, aspect=aspect)
        else:
            if normalise:
                factor = self.histograms[hist_name][kind]['H'].max()
            else:
                factor = 1.0
            ax.plot(self.histograms[hist_name][kind]['xcen'],
                    self.histograms[hist_name][kind]['H']/factor)
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)
        return ax

    def scatter(self, varx='y', vary='z', gyroradius_index=None,
                XI_index=None, ax=None, ax_params: dict = {},
                mar_params: dict = {}, per: float = 0.5,
                includeW: bool = False, xscale=1.0, yscale=1.0):
        """
        Scatter plot of two variables of the strike points

        Jose Rueda: jrrueda@us.es

        :param  varx: variable to plot in the x axis
        :param  vary: variable to plot in the y axis
        :param  per: ratio of markers to be plotted (1=all of them)
        :param  ax: axes where to plot
        :param  mar_params: Dictionary with the parameters for the markers
        :param  gyroradius_index: index (or indexes if given as an array) of
            gyroradii to plot
        :param  XI_index: index (or indexes if given as an array) of
            XIs (so pitch or R) to plot
        :param  ax_params: parameters for the axis beauty routine. Only applied
            if the axis was created inside the routine
        :param  xscale: Scale to multiply the variable plotted in the xaxis
        :param  yscale: Scale to multiply the variable plotted in the yaxis

        Note: The units will not be updates after the scaling, so you will need
        to change manually the labels via the ax_params()
        """
        # --- Get the index:
        xcolumn_to_plot = self.header['info'][varx]['i']
        ycolumn_to_plot = self.header['info'][vary]['i']
        if includeW:
            column_of_W = self.header['info']['weight']['i']
        # --- Default plotting options
        ax_options = {
            'grid': 'both',
            'xlabel': self.header['info'][varx]['shortName']
            + ' [' + self.header['info'][varx]['units'] + ']',
            'ylabel': self.header['info'][vary]['shortName']
            + ' [' + self.header['info'][vary]['units'] + ']'
        }
        ax_options.update(ax_params)
        mar_options = {
            'marker': '.',
            'color': 'k'
        }
        mar_options.update(mar_params)
        # --- Create the axes
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        # --- Plot the markers:
        nXI, ngyr = self.header['counters'].shape

        # See which gyroradius / pitch we need
        if gyroradius_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyroradius_index, (list, np.ndarray)):
                index_gyr = gyroradius_index
            else:  # it should be just a number
                index_gyr = np.array([gyroradius_index])
        if XI_index is None:  # if None, use all gyroradii
            index_XI = range(nXI)
        else:
            # Test if it is a list or array
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:  # it should be just a number
                index_XI = np.array([XI_index])
        # Proceed to plot
        for ig in index_gyr:
            for ia in index_XI:
                if self.header['counters'][ia, ig] > 0:
                    if isinstance(per, np.ndarray):
                        flags = per
                    else:
                        flags = np.random.rand(
                                self.header['counters'][ia, ig]) < per
                    x = self.data[ia, ig][flags, xcolumn_to_plot]
                    y = self.data[ia, ig][flags, ycolumn_to_plot]
                    if includeW:
                        w = self.data[ia, ig][flags, column_of_W]
                        ax.scatter(x * xscale, y * yscale, w, **mar_options)
                    else:
                        ax.scatter(x * xscale, y * yscale, **mar_options)
        # axis beauty:
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)
        plt.draw()
        return ax

    # -------------------------------------------------------------------------
    # --- Optics modeling
    # -------------------------------------------------------------------------
    def calculate_pixel_coordinates(self, calibration,):
        """
        Get the position of the markers in the camera sensor.

        Jose Rueda: jrrueda@us.es

        :param s calibration: object with the calibration parameters of the
            mapping class

        include in the data of the object the columns corresponding to the xcam
        and ycam, the position in the camera sensor of the strike points

        warning: Only fully tested for SINPA strike points
        """
        if self.header['FILDSIMmode']:
            logger.warning('20: Only fully tested for SINPA strike points')
        # See if there is already camera positions in the data
        if 'xcam' in self.header['info'].keys():
            text = 'The camera values are there, we will overwrite them'
            logger.warning('11: %s' % text)
            overwrite = True
            iixcam = self.header['info']['xcam']['i']
            iiycam = self.header['info']['ycam']['i']
        else:
            overwrite = False
        iix = self.header['info']['x1']['i']
        iiy = self.header['info']['x2']['i']
        for ig in range(self.header['ngyr']):
            for ia in range(self.header['nXI']):
                if self.header['counters'][ia, ig] > 0:
                    xp, yp = transform_to_pixel(self.data[ia, ig][:, iix],
                                                self.data[ia, ig][:, iiy],
                                                calibration)
                    if overwrite:
                        self.data[ia, ig][:, iixcam] = xp.copy()
                        self.data[ia, ig][:, iiycam] = yp.copy()
                    else:
                        n_strikes = self.header['counters'][ia, ig]
                        cam_data = np.zeros((n_strikes, 2))
                        cam_data[:, 0] = xp.copy()
                        cam_data[:, 1] = yp.copy()
                        self.data[ia, ig] = \
                            np.append(self.data[ia, ig], cam_data, axis=1)
        if not overwrite:
            Old_number_colums = len(self.header['info'])
            extra_column = {
                'xcam': {
                    'i': Old_number_colums,  # Column index in the file
                    'units': ' [px]',  # Units
                    'longName': 'X camera position',
                    'shortName': '$x_{cam}$',
                },
                'ycam': {
                    'i': Old_number_colums + 1,  # Column index in the file
                    'units': ' [px]',  # Units
                    'longName': 'Y camera position',
                    'shortName': '$y_{cam}$',
                },
            }
            self.header['info'].update(extra_column)
        # Now save the optical calibration for latter
        self.CameraCalibration = calibration

    def applyGeometricTramission(self, F_object, cal):
        """
        Modify markers weight taking into acocount geometric tramission

        Jose Rueda: jrrueda@us.es

        :param  F_object: FnumberTransmission() class of the LibOptics
        """
        # Get the index of the involved columns and if we need to overwrite
        if 'wcam' in self.header['info'].keys():
            print('The camera weights are there, we will overwrite them')
            overwrite = True
            iiwcam = self.header['info']['wcam']['i']
        else:
            overwrite = False
        iix = self.header['info']['x1']['i']
        iiy = self.header['info']['x2']['i']
        logger.warning('Scaling W0, not W scintillator')
        iiw = self.header['info']['weight0']['i']
        # --- Get the center coordinates in the scintillator space
        alpha = cal.deg * np.pi / 180
        xc = math.cos(alpha) * (cal.xcenter - cal.xshift) / cal.xscale \
            + math.sin(alpha) * (cal.ycenter - cal.yshift) / cal.yscale
        yc = - math.sin(alpha) * (cal.xcenter - cal.xshift) / cal.xscale \
            + math.cos(alpha) * (cal.ycenter - cal.yshift) / cal.yscale
        # --- Get the distance to the optical axis on the scintillator
        for ig in range(self.header['ngyr']):
            for ia in range(self.header['nXI']):
                if self.header['counters'][ia, ig] > 0:
                    rs = np.sqrt((self.data[ia, ig][:, iix] - xc)**2
                                 + (self.data[ia, ig][:, iiy] - yc)**2)
                    F = F_object.f_number(rs)
                    T = 1.0 /2.0/ (2*F)**2
                    if overwrite:
                        self.data[ia, ig][:, iiwcam] = \
                            T * self.data[ia, ig][:, iiw]
                    else:
                        shape = self.data[ia, ig].shape
                        dummy = np.zeros((shape[0], shape[1] + 1))
                        dummy[:, :-1] = self.data[ia, ig].copy()
                        dummy[:, -1] = T * self.data[ia, ig][:, iiw]
                        self.data[ia, ig] = dummy
        # --- Update the header
        if not overwrite:
            Old_number_colums = len(self.header['info'])
            extra_column = {
                'wcam': {
                    'i': Old_number_colums,  # Column index in the file
                    'units': ' [px]',  # Units
                    'longName': 'W at camera (only geom transmission)',
                    'shortName': '$W_{cam}^{geom}$',
                },
            }
            self.header['info'].update(extra_column)

    # -------------------------------------------------------------------------
    # --- Export block
    # -------------------------------------------------------------------------
    def points_to_txt(self, per: float = 0.1,
                      gyroradius_index=None, XI_index=None,
                      where: str = 'Head',
                      units: str = 'mm',
                      file_name_save: str = 'Strikes.txt'):
        """
        Store strike points to txt file to easily load in CAD software.

        Anton van Vuen: avanvuuren@us.es

        :param  per: ratio of markers to be plotted (1=all of them)
        :param  gyroradius_index: index (or indeces if given as an np.array) of
            gyroradii to plot
        :param  XI_index: index (or indeces if given as an np.array) of
            XIs (pitch or R) to plot
        :param  where: string indicating where to plot: 'head', 'NBI',
        'ScintillatorLocalSystem'. First two are in absolute
        coordinates, last one in the scintillator coordinates (see SINPA
        documentation) [Head will plot the strikes in the collimator or
        scintillator]. For oldFILDSIM, use just head
        :param  units: Units in which to save the strike positions.
        :param  filename: name of the text file to store strikepoints in

        :return file_name_save: name of the text file to store strikepoints in
        """
        # --- Chose the variable we want to plot
        if where.lower() == 'head':
            column_to_plot = self.header['info']['x']['i']
        elif where.lower() == 'nbi':
            column_to_plot = self.header['info']['x0']['i']
        elif where.lower() == 'scintillatorlocalsystem':
            column_to_plot = self.header['info']['x3']['i']
        else:
            raise Exception('Not understood what do you want to plot')

        nXI, ngyr = self.header['counters'].shape
        # See which gyroradius / pitch (R) we need
        if gyroradius_index is None:  # if None, use all gyroradii
            index_gyr = range(ngyr)
        else:
            # Test if it is a list or array
            if isinstance(gyroradius_index, (list, np.ndarray)):
                index_gyr = gyroradius_index
            else:  # it should be just a number
                index_gyr = np.array([gyroradius_index])
        if XI_index is None:  # if None, use all gyroradii
            index_XI = range(nXI)
        else:
            # Test if it is a list or array
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:  # it should be just a number
                index_XI = np.array([XI_index])

        # --- Check the scale
        if units not in ['m', 'cm', 'mm']:
            raise Exception('Not understood units?')
        possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
        factor = possible_factors[units]

        with open(file_name_save, 'weight') as f:
            for ig in index_gyr:
                for ia in index_XI:
                    if self.header['counters'][ia, ig] > 0:
                        flags = np.random.rand(
                            self.header['counters'][ia, ig]) < per
                        if flags.sum() > 0:
                            x = self.data[ia, ig][flags, column_to_plot]
                            y = self.data[ia, ig][flags, column_to_plot + 1]
                            z = self.data[ia, ig][flags, column_to_plot + 2]

                            for xs, ys, zs in zip(x, y, z):
                                f.write('%f %f %f \n'
                                        % (xs * factor,
                                           ys * factor,
                                           zs * factor))
        return file_name_save

    
    def exportHistograms(self, folder: str = 'Remaps', 
                          overwrite: bool = False) -> str:
        """
        Export the histograms to a folder.

        Jose Rueda: jrrueda@us.es

        :param  folder: folder where to store the histograms
        :param  overwrite: if True, overwrite the files

        :return folder: folder where the histograms have been stored
        """
        # --- Check the inputs
        if not os.path.isdir(folder):
            os.mkdir(folder)
        # --- Export the histograms
        for key, histogram in self.histograms.items():
            filename = os.path.join(folder, key + '.nc')
            if os.path.isfile(filename) and not overwrite:
                logger.warning('File %s already exists! Not saving' % filename)
                continue
            histogram.to_netCDF(filename, format='NETCDF4')
        # --- Export the version of the suite
        filename = os.path.join(folder, 'version.txt')
        if os.path.isfile(filename) and not overwrite:
            logger.warning('Version File already exsits! Not saving')
        else:
            exportVersion(filename)
        return folder

    # -------------------------------------------------------------------------
    # --- remap
    # -------------------------------------------------------------------------
    def remap(self, smap, options, variables_to_remap: tuple = ('R0', 'e0'),
              transformationMatrixExtraOptions: dict = {}) -> None:
        """
        Remap the camera histogram as it was a camera frame.

        Jose Rueda: jrrueda@us.es

        :param  strikemap: strike map to be used
        :param  options: disctionary containing the remaping options, like the
            one used for the video
        """
        # --- Check the inputs
        if 'xcam_ycam' not in self.histograms.keys():
            raise Exception('You need to calculate the camera histogram!')
        if options['remap_method'] == 'centers':
            options['MC_number'] = 0  # Turn off the transformation matrix calc
        # chek if there are weights
        if 'xcam_ycam_w' in self.histograms.keys():
            habia_peso = True
        else:
            habia_peso = False
        if 'xcam_ycam_w0' in self.histograms.keys():
            habia_peso_0 = True
        else:
            habia_peso_0 = False
        if 'xcam_ycam_wcam' in self.histograms.keys():
            habia_peso_cam = True
        else:
            habia_peso_cam = False
        # --- Prepare the grid
        frame_shape = self.histograms['xcam_ycam'].markers.shape
        nx, ny, xedges, yedges = createGrid(
            options['xmin'], options['xmax'], options['dx'],
            options['ymin'], options['ymax'], options['dy'],
            )
        xcenter = 0.5 * (xedges[:-1] + xedges[1:])
        ycenter = 0.5 * (yedges[:-1] + yedges[1:])
        # Interpolate the strike map
        # -- 1: Check if the variables_to_remap are already the ones we want
        if not (variables_to_remap[0] == smap._remap_var_names[0]
                and variables_to_remap[1] == smap._remap_var_names[1]):
            smap.setRemapVariables(variables_to_remap, verbose=False)
            changed_remap_variables = True
        else:
            changed_remap_variables = False
        # -- 2: Check if the pixel coordiantes are there
        if smap._coord_pix['x'] is None or changed_remap_variables:
            smap.calculate_pixel_coordinates(self.CameraCalibration)
        # -- 3: Perform the grid interpolation
        grid_options = {
            'xmin': options['xmin'],
            'xmax': options['xmax'],
            'dx': options['dx'],
            'ymin': options['ymin'],
            'ymax': options['ymax'],
            'dy': options['dy'],
        }
        if changed_remap_variables or smap._grid_interp is None:
            # In this case, this is un-avoidable
            smap.interp_grid(frame_shape, method=options['method'],
                             MC_number=options['MC_number'],
                             grid_params=grid_options,
                             **transformationMatrixExtraOptions)
        else:
            calc_is_needed = 0  # By default assume not
            # If we are not going to use MC, no need of recalculate
            if options['MC_number'] != 0:
                # See if the map has already a calculated transformation matrix
                name = variables_to_remap[0] + '_' + variables_to_remap[1]
                if name not in smap._grid_interp['transformation_matrix'].keys():
                    calc_is_needed = True
                else:  # There is a transformation matrix, let's see the axis
                    tol = 1e-3
                    diff = \
                        {key: grid_options[key]
                         - smap._grid_interp['transformation_matrix']
                         [name + '_grid'].get(key, 0)
                         for key in grid_options}
                    flags = [abs(diff[key]) > tol for key in diff]
                    flags = np.array(flags)
                    if flags.sum() > 0:
                        calc_is_needed = True
                if calc_is_needed:
                    smap.interp_grid(frame_shape, method=options['method'],
                                     MC_number=options['MC_number'],
                                     grid_params=grid_options)

        name = variables_to_remap[0] + '_' + variables_to_remap[1] + '_remap'
        self.histograms[name] = xr.Dataset()
        for k in self.histograms['xcam_ycam'].keys():
            nkinds = self.histograms['xcam_ycam'].kind.size
            data = np.zeros((xedges.size-1, yedges.size-1, nkinds))
            for j in range(nkinds):
                data[:, :, j] = remap(smap, self.histograms['xcam_ycam'][k].isel(kind=j).values,
                              x_edges=xedges, y_edges=yedges, mask=None,
                              method=options['remap_method'])
            self.histograms[name][k] = xr.DataArray(data, dims=('x', 'y', 'kind'),
                                                 coords={'x': xcenter,
                                                         'y': ycenter,
                                                         'kind': self.histograms['xcam_ycam'].kind})
        self.histograms[name].attrs = {
            'xedges': xedges,
            'yedges': yedges,
            }
        # Now repeat for the finite focus
        if 'xcam_ycam_finiteFocus' in self.histograms.keys():
            name = variables_to_remap[0] + '_' + variables_to_remap[1] +\
                '_remap_finiteFocus'
            self.histograms[name] = xr.Dataset()
            for k in self.histograms['xcam_ycam_finiteFocus'].keys():
                nkinds = self.histograms['xcam_ycam_finiteFocus'].kind.size
                data = np.zeros((xedges.size-1, yedges.size-1, nkinds))
                for j in range(nkinds):
                    data[:, :, j] = remap(smap, self.histograms['xcam_ycam_finiteFocus'][k].isel(kind=j).values,
                                  x_edges=xedges, y_edges=yedges, mask=None,
                                  method=options['remap_method'])
                self.histograms[name][k] = xr.DataArray(data, dims=('x', 'y', 'kind'),
                                                     coords={'x': xcenter,
                                                             'y': ycenter,
                                                             'kind': self.histograms['xcam_ycam_finiteFocus'].kind})
            self.histograms[name].attrs = {
                'xedges': xedges,
                'yedges': yedges,
                }

    @property
    def shape(self):
        return self._shape

    def __call__(self, var: str) -> np.ndarray:
        """Call for the object"""
        try:
            out = self.get(var)
        except KeyError:
            out = None
        return out
