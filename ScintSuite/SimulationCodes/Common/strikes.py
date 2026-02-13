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
import unyt
import numpy as np
import xarray as xr
import ScintSuite.errors as errors
import ScintSuite._Plotting as ssplt
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
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
from ScintSuite._Utilities import flatten

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
    :param  verbose: Not used anymore (kept for retrocompatibility)

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
    logger.debug('Identified plate: %s', plate)
    # --- Open the file and read
    with open(filename, 'rb') as fid:
        header = {
            'versionID1': np.fromfile(fid, 'int32', 1)[0],
            'versionID2': np.fromfile(fid, 'int32', 1)[0],
        }
        logger.info('File %s'%filename)
        logger.info('SINPA version: %i.%i'%(header['versionID1'],
                                            header['versionID2']))
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
            # print some debug information
            logger.debug('RunID: %s'%header['runID'])
            logger.debug('ngyr: %i'%header['ngyr'])
            printlog = [str(g) for g in header['gyroradius']]
            logger.debug('gyroradius: %s'% ', '.join(printlog))
            logger.debug('nXI: %s'%header['nXI'])
            printlog = [str(g) for g in header['XI']]
            logger.debug('XI: %s'% ', '.join(printlog))
            logger.debug('FILDSIMmose: %s'%header['FILDSIMmode'])
            logger.debug('Number of stored columns: %i'%header['ncolumns'])
            if header['versionID1'] >= 4 and plate.lower()!='collimator':
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
            logger.debug('Kind of File: %i', header['kindOfFile'])
            if header['FILDSIMmode']:
                key_to_look = 'sinpa_FILD'
            else:
                key_to_look = 'sinpa_INPA'
            while not found_header:
                logger.debug('Looking version %i'%id_version)
                if header['versionID1'] < 4:
                    try:
                        logger.debug('Looking version %i'%id_version)
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
                    
                    if plate.lower() != 'collimator':  
                        try:
                            header['info'] = deepcopy(
                                order[key_to_look][id_version][plate.lower()][header['kindOfFile']])
                            found_header = True
                        except KeyError:
                            id_version -= 1
                    else:
                        try:
                            logger.debug('Looking version %i'%id_version)
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
            logger.info('Reading strike points:')
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
        if header['versionID1'] >= 2 and plate.lower() != 'collimator':
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
        if not header['FILDSIMmode'] and plate.lower() != 'collimator':
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
        logger.info('Total number of strike points: %i'%
                    np.sum(header['counters']))
        logger.info('Average number of strike points per centroid: %i'%
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
    :param  verbose. flag to print some info in the command line, ignored, kept
    for retrocompatibility
    """
    logger.info('Reading strike points: ', filename)
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
    logger.info('Total number of strike points: ', total_counter)
    logger.info('Average number of strike points per centroid: ',
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
        :param  verbose. flag to print some info in the command line, useles, kept for retrocompatibility
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
        # Assembly the dataframe
        self.df = None
        self.df = self.to_dataframe()

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
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError('self.df is not available or not a DataFrame')
        if (varx not in self.df.columns) or (vary not in self.df.columns):
            print('Variables available: ', list(self.df.columns))
            raise Exception('Variables not found')
        # --- Check if the histogram is already there
        if (varx + '_' + vary) in self.histograms.keys():
            logger.warning('11: Histogram present, overwritting')
        # --- Resolve x/y column names (camera frame remap)
        if not varx.endswith('cam'):
            xcol = varx
            ycol = vary
        else:
            text = 'varx and vary exchanged'
            logger.warning('a3: %s' % text)
            xcol = vary
            ycol = varx

        # Optional weight/kind columns in self.df
        has_w = 'weight' in self.df.columns
        has_w0 = 'weight0' in self.df.columns
        has_wcam = 'wcam' in self.df.columns
        has_kind = 'kind' in self.df.columns

        # --- Define the grid for the histogram from self.df
        if (binsx is None) or isinstance(binsx, int):
            xmin = self.df[xcol].min()
            xmax = self.df[xcol].max()
            if np.isnan(xmin) or np.isnan(xmax):
                xmin, xmax = 0.0, 1.0
            if binsx is None:
                edgesx = np.linspace(xmin, xmax, 25)
            else:
                edgesx = np.linspace(xmin, xmax, binsx + 1)
        else:
            edgesx = np.asarray(binsx)
        if (binsy is None) or isinstance(binsy, int):
            ymin = self.df[ycol].min()
            ymax = self.df[ycol].max()
            if np.isnan(ymin) or np.isnan(ymax):
                ymin, ymax = 0.0, 1.0
            if binsy is None:
                edgesy = np.linspace(ymin, ymax, 25)
            else:
                edgesy = np.linspace(ymin, ymax, binsy + 1)
        else:
            edgesy = np.asarray(binsy)
        # --- Preallocate the data
        histName = varx + '_' + vary
        self.histograms[histName] = xr.Dataset()
        supportedKinds = [0, 5, 6, 7, 8]
        if self.header['FILDSIMmode'] or not has_kind:
            supportedKinds = [0,]
        nkinds = len(supportedKinds)
        data = np.zeros((edgesx.size - 1, edgesy.size - 1, nkinds))
        if has_w0:
            data0 = np.zeros((edgesx.size - 1, edgesy.size - 1, nkinds))
        if has_w:
            dataS = np.zeros((edgesx.size - 1, edgesy.size - 1, nkinds))
        if has_wcam:
            dataC = np.zeros((edgesx.size - 1, edgesy.size - 1, nkinds))
        # Only (gyroradius, XI) groups with more than one marker (match original)
        if 'gyroradius' in self.df.columns and 'XI' in self.df.columns:
            group_sizes = self.df.groupby(['gyroradius', 'XI']).size()
            valid_pairs = group_sizes[group_sizes > 1].index
            valid_df = pd.DataFrame(list(valid_pairs), columns=['gyroradius', 'XI'])
            df_hist = self.df.merge(valid_df, on=['gyroradius', 'XI'], how='inner')
        else:
            df_hist = self.df
        for ik, k in enumerate(supportedKinds):
            logger.debug('Histograming kind %i' % k)
            if k != 0 and has_kind:
                mask = df_hist['kind'].astype(int) == k
                if not mask.any():
                    continue
                sub = df_hist.loc[mask]
            else:
                sub = df_hist
            if len(sub) == 0:
                continue
            logger.debug('Histogram rows %i' % len(sub))
            H, xedges, yedges = np.histogram2d(
                sub[xcol].values, sub[ycol].values, bins=(edgesx, edgesy)
            )
            data[:, :, ik] += H
            if has_w:
                H, xedges, yedges = np.histogram2d(
                    sub[xcol].values, sub[ycol].values,
                    bins=(edgesx, edgesy), weights=sub['weight'].values
                )
                dataS[:, :, ik] += H
            if has_w0:
                H, xedges, yedges = np.histogram2d(
                    sub[xcol].values, sub[ycol].values,
                    bins=(edgesx, edgesy), weights=sub['weight0'].values
                )
                data0[:, :, ik] += H
            if has_wcam:
                H, xedges, yedges = np.histogram2d(
                    sub[xcol].values, sub[ycol].values,
                    bins=(edgesx, edgesy), weights=sub['wcam'].values
                )
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
        if has_w:
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
        if has_w0:
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
        if has_wcam:
            dataC /= deltax * deltay
            self.histograms[histName]['wcam'] = xr.DataArray(
                dataC, dims=('x', 'y', 'kind'),
                coords={'x': xcen, 'y': ycen, 'kind': supportedKinds}
            )
            self.histograms[histName]['wcam'].attrs['Description'] = \
                'Weight at the camera'
            self.histograms[histName]['wcam'].attrs['units'] = '[a.u.]'
            self.histograms[histName]['wcam'].attrs['long_name'] = '$W_{cam}$'
        # Set the variables attributes (use header when available for units/labels)
        if varx in self.header.get('info', {}):
            self.histograms[histName]['x'].attrs['long_name'] = \
                self.header['info'][varx]['shortName']
            self.histograms[histName]['x'].attrs['units'] = \
                self.header['info'][varx]['units']
        if vary in self.header.get('info', {}):
            self.histograms[histName]['y'].attrs['long_name'] = \
                self.header['info'][vary]['shortName']
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
        # --- Check if the variables we need actually exist (use self.df)
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError('self.df is not available or not a DataFrame')
        if var not in self.df.columns:
            print('Variables available: ', list(self.df.columns))
            raise Exception('Variables not found')
        # --- Check if the histogram is already there
        if var in self.histograms.keys():
            logger.warning('11: Histogram present, overwritting')
        dat = self.df[var].values
        w = self.df['weight'].values if 'weight' in self.df.columns else None
        w0 = self.df['weight0'].values if 'weight0' in self.df.columns else None
        k = self.df['kind'].values if 'kind' in self.df.columns else None
        # --- Define the grid for the histogram
        if (bins is None) or isinstance(bins, int):
            xmin = np.nanmin(dat)
            xmax = np.nanmax(dat)
            if np.isnan(xmin) or np.isnan(xmax):
                xmin, xmax = 0.0, 1.0
            if bins is None:
                edgesx = np.linspace(xmin, xmax, 25)
            else:
                edgesx = np.linspace(xmin, xmax, bins + 1)
        else:
            edgesx = np.asarray(bins)
        # --- Preallocate the data
        varw = var + '_w'
        varw0 = var + '_w0'
        # Basic (counts)
        self.histograms[var] = \
            {0: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        if w0 is not None:
            self.histograms[varw0] = {0: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        if w is not None:
            self.histograms[varw] = {0: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        H, xedges = np.histogram(dat, bins=edgesx)
        xcen = 0.5 * (xedges[1:] + xedges[:-1])
        deltax = xcen[1] - xcen[0] if len(xcen) > 1 else 1.0
        self.histograms[var][0] = {
            'xcen': xcen,
            'xedges': xedges,
            'H': H.astype(float) / deltax
        }
        if w is not None:
            H, xedges = np.histogram(dat, bins=edgesx, weights=w)
            self.histograms[varw][0] = {
                'xcen': xcen,
                'xedges': xedges,
                'H': H.astype(float) / deltax
            }
        if w0 is not None:
            H, xedges = np.histogram(dat, bins=edgesx, weights=w0)
            self.histograms[varw0][0] = {
                'xcen': xcen,
                'xedges': xedges,
                'H': H.astype(float) / deltax
            }
        # Now repeat the same for the different kinds
        if k is not None:
            for kind in [5, 6, 7, 8]:
                flags = k.astype(int) == kind
                if not np.any(flags):
                    continue
                H, xedges = np.histogram(dat[flags], bins=edgesx)
                xcen = 0.5 * (xedges[1:] + xedges[:-1])
                deltax = xcen[1] - xcen[0] if len(xcen) > 1 else 1.0
                self.histograms[var][kind] = {
                    'xcen': xcen,
                    'xedges': xedges,
                    'H': H.astype(float) / deltax
                }
                if w is not None:
                    H, xedges = np.histogram(dat[flags], bins=edgesx,
                                             weights=w[flags])
                    self.histograms[varw][kind] = {
                        'xcen': xcen,
                        'xedges': xedges,
                        'H': H.astype(float) / deltax
                    }
                if w0 is not None:
                    H, xedges = np.histogram(dat[flags], bins=edgesx,
                                             weights=w0[flags])
                    self.histograms[varw0][kind] = {
                        'xcen': xcen,
                        'xedges': xedges,
                        'H': H.astype(float) / deltax
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
        # --- Check if the variables we need actually exist (use self.df)
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError('self.df is not available or not a DataFrame')
        if (varx not in self.df.columns) or (vary not in self.df.columns) or \
           (varz not in self.df.columns):
            print('Variables available: ', list(self.df.columns))
            raise Exception('Variables not found')
        # --- Check if the histogram is already there
        if (varx + '_' + vary + '_' + varz) in self.histograms.keys():
            logger.warning('11: Histogram present, overwritting')
        # --- Resolve column names (camera frame remap)
        if not varx.endswith('cam'):
            xcol, ycol, zcol = varx, vary, varz
        else:
            if varz.endswith('cam'):
                raise Exception('Sorry not implemented, permute variables')
            logger.warning('a3: varx and vary exchanged')
            xcol, ycol, zcol = vary, varx, varz

        has_w = 'weight' in self.df.columns
        has_w0 = 'weight0' in self.df.columns
        has_wcam = 'wcam' in self.df.columns
        has_kind = 'kind' in self.df.columns

        # --- Define the grid for the histogram from self.df
        if (binsx is None) or isinstance(binsx, int):
            xmin, xmax = self.df[xcol].min(), self.df[xcol].max()
            if np.isnan(xmin) or np.isnan(xmax):
                xmin, xmax = 0.0, 1.0
            edgesx = np.linspace(xmin, xmax, 25 if binsx is None else binsx + 1)
        else:
            edgesx = np.asarray(binsx)
        if (binsy is None) or isinstance(binsy, int):
            ymin, ymax = self.df[ycol].min(), self.df[ycol].max()
            if np.isnan(ymin) or np.isnan(ymax):
                ymin, ymax = 0.0, 1.0
            edgesy = np.linspace(ymin, ymax, 25 if binsy is None else binsy + 1)
        else:
            edgesy = np.asarray(binsy)
        if (binsz is None) or isinstance(binsz, int):
            zmin, zmax = self.df[zcol].min(), self.df[zcol].max()
            if np.isnan(zmin) or np.isnan(zmax):
                zmin, zmax = 0.0, 1.0
            edgesz = np.linspace(zmin, zmax, 25 if binsz is None else binsz + 1)
        else:
            edgesz = np.asarray(binsz)
        # --- Preallocate the data
        histName = varx + '_' + vary + '_' + varz
        self.histograms[histName] = xr.Dataset()
        supportedKinds = [0, 5, 6, 7, 8]
        if self.header['FILDSIMmode'] or not has_kind:
            supportedKinds = [0,]
        nkinds = len(supportedKinds)
        data = np.zeros((edgesx.size - 1, edgesy.size - 1, edgesz.size - 1, nkinds))
        if has_w0:
            data0 = np.zeros((edgesx.size - 1, edgesy.size - 1, edgesz.size - 1, nkinds))
        if has_w:
            dataS = np.zeros((edgesx.size - 1, edgesy.size - 1, edgesz.size - 1, nkinds))
        if has_wcam:
            dataC = np.zeros((edgesx.size - 1, edgesy.size - 1, edgesz.size - 1, nkinds))
        if 'gyroradius' in self.df.columns and 'XI' in self.df.columns:
            group_sizes = self.df.groupby(['gyroradius', 'XI']).size()
            valid_pairs = group_sizes[group_sizes > 1].index
            valid_df = pd.DataFrame(list(valid_pairs), columns=['gyroradius', 'XI'])
            df_hist = self.df.merge(valid_df, on=['gyroradius', 'XI'], how='inner')
        else:
            df_hist = self.df
        for ik, k in enumerate(supportedKinds):
            if k != 0 and has_kind:
                mask = df_hist['kind'].astype(int) == k
                if not mask.any():
                    continue
                sub = df_hist.loc[mask]
            else:
                sub = df_hist
            if len(sub) == 0:
                continue
            H, (xedges, yedges, zedges) = np.histogramdd(
                (sub[xcol].values, sub[ycol].values, sub[zcol].values),
                bins=(edgesx, edgesy, edgesz)
            )
            data[:, :, :, ik] += H
            if has_w:
                H, _ = np.histogramdd(
                    (sub[xcol].values, sub[ycol].values, sub[zcol].values),
                    bins=(edgesx, edgesy, edgesz), weights=sub['weight'].values
                )
                dataS[:, :, :, ik] += H
            if has_w0:
                H, _ = np.histogramdd(
                    (sub[xcol].values, sub[ycol].values, sub[zcol].values),
                    bins=(edgesx, edgesy, edgesz), weights=sub['weight0'].values
                )
                data0[:, :, :, ik] += H
            if has_wcam:
                H, _ = np.histogramdd(
                    (sub[xcol].values, sub[ycol].values, sub[zcol].values),
                    bins=(edgesx, edgesy, edgesz), weights=sub['wcam'].values
                )
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
        if has_w:
            dataS /= deltax * deltay * deltaz
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
        if has_w0:
            data0 /= deltax * deltay * deltaz
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
        if has_wcam:
            dataC /= deltax * deltay * deltaz
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
        if varx in self.header.get('info', {}):
            self.histograms[histName]['x'].attrs['long_name'] = \
                self.header['info'][varx]['shortName']
            self.histograms[histName]['x'].attrs['units'] = \
                self.header['info'][varx]['units']
        if vary in self.header.get('info', {}):
            self.histograms[histName]['y'].attrs['long_name'] = \
                self.header['info'][vary]['shortName']
            self.histograms[histName]['y'].attrs['units'] = \
                self.header['info'][vary]['units']
        if varz in self.header.get('info', {}):
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
        # --- Check if the variables we need actually exist (use self.df)
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError('self.df is not available or not a DataFrame')
        for v in (varx1, varx2, varx3, varx4):
            if v not in self.df.columns:
                print('Variables available: ', list(self.df.columns))
                raise Exception('Variables not found')
        histName = varx1 + '_' + varx2 + '_' + varx3 + '_' + varx4
        if histName in self.histograms.keys():
            logger.warning('11: Histogram present, overwritting')
        # --- Resolve column names (camera frame remap)
        if not varx1.endswith('cam'):
            col1, col2, col3, col4 = varx1, varx2, varx3, varx4
        else:
            if varx3.endswith('cam') or varx4.endswith('cam'):
                raise Exception('Sorry not implemented, permute variables')
            logger.warning('a3: varx and vary exchanged')
            col1, col2, col3, col4 = varx2, varx1, varx3, varx4

        has_w = 'weight' in self.df.columns
        has_w0 = 'weight0' in self.df.columns
        has_wcam = 'wcam' in self.df.columns
        has_kind = 'kind' in self.df.columns

        # --- Define the grid for the histogram from self.df
        def _edges(col, bins, default_n=25):
            if (bins is None) or isinstance(bins, int):
                lo, hi = self.df[col].min(), self.df[col].max()
                if np.isnan(lo) or np.isnan(hi):
                    lo, hi = 0.0, 1.0
                return np.linspace(lo, hi, default_n if bins is None else bins + 1)
            return np.asarray(bins)
        edgesx = _edges(col1, binsx1)
        edgesy = _edges(col2, binsx2)
        edgesz = _edges(col3, binsx3)
        edgest = _edges(col4, binsx4)
        # --- Preallocate the data
        self.histograms[histName] = xr.Dataset()
        supportedKinds = [0, 5, 6, 7, 8]
        if self.header['FILDSIMmode'] or not has_kind:
            supportedKinds = [0,]
        nkinds = len(supportedKinds)
        data = np.zeros((edgesx.size - 1, edgesy.size - 1,
                         edgesz.size - 1, edgest.size - 1, nkinds))
        if has_w0:
            data0 = np.zeros((edgesx.size - 1, edgesy.size - 1,
                              edgesz.size - 1, edgest.size - 1, nkinds))
        if has_w:
            dataS = np.zeros((edgesx.size - 1, edgesy.size - 1,
                              edgesz.size - 1, edgest.size - 1, nkinds))
        if has_wcam:
            dataC = np.zeros((edgesx.size - 1, edgesy.size - 1,
                              edgesz.size - 1, edgest.size - 1, nkinds))
        if 'gyroradius' in self.df.columns and 'XI' in self.df.columns:
            group_sizes = self.df.groupby(['gyroradius', 'XI']).size()
            valid_pairs = group_sizes[group_sizes > 1].index
            valid_df = pd.DataFrame(list(valid_pairs), columns=['gyroradius', 'XI'])
            df_hist = self.df.merge(valid_df, on=['gyroradius', 'XI'], how='inner')
        else:
            df_hist = self.df
        for ik, k in enumerate(supportedKinds):
            if k != 0 and has_kind:
                mask = df_hist['kind'].astype(int) == k
                if not mask.any():
                    continue
                sub = df_hist.loc[mask]
            else:
                sub = df_hist
            if len(sub) == 0:
                continue
            H, (xedges, yedges, zedges, tedges) = np.histogramdd(
                (sub[col1].values, sub[col2].values, sub[col3].values, sub[col4].values),
                bins=(edgesx, edgesy, edgesz, edgest)
            )
            data[:, :, :, :, ik] += H
            if has_w:
                H, _ = np.histogramdd(
                    (sub[col1].values, sub[col2].values, sub[col3].values, sub[col4].values),
                    bins=(edgesx, edgesy, edgesz, edgest),
                    weights=sub['weight'].values
                )
                dataS[:, :, :, :, ik] += H
            if has_w0:
                H, _ = np.histogramdd(
                    (sub[col1].values, sub[col2].values, sub[col3].values, sub[col4].values),
                    bins=(edgesx, edgesy, edgesz, edgest),
                    weights=sub['weight0'].values
                )
                data0[:, :, :, :, ik] += H
            if has_wcam:
                H, _ = np.histogramdd(
                    (sub[col1].values, sub[col2].values, sub[col3].values, sub[col4].values),
                    bins=(edgesx, edgesy, edgesz, edgest),
                    weights=sub['wcam'].values
                )
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
        if has_w:
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
        if has_w0:
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
        if has_wcam:
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
        info = self.header.get('info', {})
        if varx1 in info:
            self.histograms[histName]['x1'].attrs['long_name'] = info[varx1]['shortName']
            self.histograms[histName]['x1'].attrs['units'] = info[varx1]['units']
        if varx2 in info:
            self.histograms[histName]['x2'].attrs['long_name'] = info[varx2]['shortName']
            self.histograms[histName]['x2'].attrs['units'] = info[varx2]['units']
        if varx3 in info:
            self.histograms[histName]['x3'].attrs['long_name'] = info[varx3]['shortName']
            self.histograms[histName]['x3'].attrs['units'] = info[varx3]['units']
        if varx4 in info:
            self.histograms[histName]['x4'].attrs['long_name'] = info[varx4]['shortName']
            self.histograms[histName]['x4'].attrs['units'] = info[varx4]['units']
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
        return np.array(flatten(var))

    def get_from_df(
        self,
        var: str,
        gyroradius_index: Optional[Union[int, List[int], np.ndarray]] = None,
        XI_index: Optional[Union[int, List[int], np.ndarray]] = None,
        gyroradius: Optional[Union[float, List[float], np.ndarray]] = None,
        XI: Optional[Union[float, List[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Return an array with the values of 'var' for all strike points using
        the data stored in self.df (pandas.DataFrame).

        Selection can be made either by indices or by direct values:
        - Use gyroradius_index and XI_index to select by header indices.
        - Use gyroradius and XI to select by actual parameter values.

        :param  var: variable name (column in self.df) to return
        :param  gyroradius_index: index or indices of gyroradii (see
            self.header['gyroradius']). None = all.
        :param  XI_index: index or indices of XI (see self.header['XI']).
            None = all.
        :param  gyroradius: gyroradius value(s) to filter by. If given,
            overrides gyroradius_index.
        :param  XI: XI value(s) to filter by. If given, overrides XI_index.

        :return: 1D array of values for the selected (gyroradius, XI) subset.
        """
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError('self.df is not available or not a DataFrame')
        if var not in self.df.columns:
            print('Available columns: ', list(self.df.columns))
            raise errors.NotFoundVariable(f'Variable "{var}" not in DataFrame')
        # Resolve selected gyroradius values
        if gyroradius is not None:
            sel_gyr = np.atleast_1d(gyroradius)
        elif gyroradius_index is not None:
            idx_g = np.atleast_1d(gyroradius_index)
            sel_gyr = np.asarray(self.header['gyroradius'])[idx_g]
        else:
            sel_gyr = np.asarray(self.header['gyroradius'])
        # Resolve selected XI values
        if XI is not None:
            sel_xi = np.atleast_1d(XI)
        elif XI_index is not None:
            idx_x = np.atleast_1d(XI_index)
            sel_xi = np.asarray(self.header['XI'])[idx_x]
        else:
            sel_xi = np.asarray(self.header['XI'])
        mask = (
            self.df['gyroradius'].isin(sel_gyr) & self.df['XI'].isin(sel_xi)
        )
        return self.df.loc[mask, var].values

    def to_dataframe(self, gyroradius_index=None, XI_index=None,
                     include_units: bool = True) -> pd.DataFrame:
        """
        Return all strike points as a single pandas DataFrame.

        Columns are taken from `self.header['info']` mapping. Two extra
        columns are added: `gyroradius` and `XI` with the corresponding
        parameter values for each marker.

        If `include_units` is True, a mapping of column->unit strings will
        be attached to `DataFrame.attrs['units']`.
        """
        # Build index selection similar to `get`
        nXI, ngyr = self.header['counters'].shape
        if gyroradius_index is None:
            index_gyr = range(ngyr)
        else:
            if isinstance(gyroradius_index, (list, np.ndarray)):
                index_gyr = gyroradius_index
            else:
                index_gyr = np.array([gyroradius_index])
        if XI_index is None:
            index_XI = range(nXI)
        else:
            if isinstance(XI_index, (list, np.ndarray)):
                index_XI = XI_index
            else:
                index_XI = np.array([XI_index])
            
        if self.df is not None:
            try:
                key_g = tuple(int(x) for x in index_gyr)
            except Exception:
                key_g = ('all',)
            try:
                key_x = tuple(int(x) for x in index_XI)
            except Exception:
                key_x = ('all',)
            cache_key = (key_g, key_x, bool(include_units))
            # Return cached copy if available
            cached = self.df.get(cache_key)
            if cached is not None:
                return cached.copy()

        info = self.header.get('info', {})
        if not info:
            raise errors.NotFoundVariable('No header info available')

        # Determine max column index and name mapping
        max_i = max([v['i'] for v in info.values()]) if len(info) > 0 else -1
        col_names = [f'col_{i}' for i in range(max_i + 1)]
        units_map: Dict[str, str] = {}
        for name, meta in info.items():
            idx = meta['i']
            if idx <= max_i:
                col_names[idx] = name
            units_map[name] = meta.get('units', '')

        frames: List['pd.DataFrame'] = []
        for ig in index_gyr:
            for ia in index_XI:
                if self.header['counters'][ia, ig] > 0:
                    arr = self.data[ia, ig]
                    ncols = arr.shape[1]
                    cols = col_names[:ncols]
                    df = pd.DataFrame(arr, columns=cols)
                    # add metadata columns
                    try:
                        df['gyroradius'] = self.header['gyroradius'][ig]
                    except Exception:
                        df['gyroradius'] = np.nan
                    try:
                        df['XI'] = self.header['XI'][ia]
                    except Exception:
                        df['XI'] = np.nan
                    frames.append(df)

        if len(frames) == 0:
            # empty dataframe with known columns
            df_empty = pd.DataFrame(columns=col_names + ['gyroradius', 'XI'])
            if include_units:
                df_empty.attrs['units'] = units_map
            # cache empty result as well
            try:
                self.df[cache_key] = df_empty.copy()
            except Exception:
                pass
            return df_empty

        result = pd.concat(frames, ignore_index=True)
        if include_units:
            result.attrs['units'] = units_map
        # Cache assembled dataframe for this selection
        try:
            self.df[cache_key] = result.copy()
        except Exception:
            pass
        return result

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
               levels: tuple = None, line_params: dict = {}):
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
                    ax.plot(xc, H, **line_params)
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
        :param  cmap: color map to be used, if none -> default()
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
            cmap = ssplt.default_cmap()
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
        # Invalidate assembled DataFrame cache
        try:
            self._df_cache.clear()
        except Exception:
            self._df_cache = {}

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

        # Invalidate assembled DataFrame cache (columns changed)
        try:
            self._df_cache.clear()
        except Exception:
            self._df_cache = {}

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
            histogram.to_netcdf(filename, format='NETCDF4')
        # --- Export the version of the suite
        filename = os.path.join(folder, 'version.txt')
        if os.path.isfile(filename) and not overwrite:
            logger.warning('Version File already exsits! Not saving')
        else:
            exportVersion(filename)
        return folder

    def exportVariables(self, file: str, vars: list,
                        overwrite: Optional[bool] = False):
        """
        Export a set of variable to an netCDF object
        """
        dummy = xr.Dataset()
        for var in vars:
            dummy[var] = self(var)
        if not os.path.isfile(file) or overwrite:
            logger.info('Saving into file: %s'%file)
            dummy.to_netcdf(file)
        else:
            logger.warning('File exist, doing nothing')
        return
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

    def __call__(self, var: str) -> unyt.array.unyt_array:
        """Call for the object"""
        try:
            out = self.get_from_df(var)
        except errors.NotFoundVariable:
            out = self.get(var)
        except KeyError:
            out = None
        return out
