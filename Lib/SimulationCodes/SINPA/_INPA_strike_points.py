"""
Class containing INPA strike points.

Jose Rueda Rueda: jrrueda@us.es
"""
import lmfit
import logging
import numpy as np
import xarray as xr
import Lib.LibData as ssdat
import Lib.errors as errors
# import Lib._StrikeMap as ssmap
from Lib.SimulationCodes.Common.strikes import Strikes
from Lib.SimulationCodes.FILDSIM.execution import get_gyroradius

from Lib._Optics import FnumberTransmission, defocus

# -----------------------------------------------------------------------------
# %% Auxiliary objects
# -----------------------------------------------------------------------------
logger = logging.getLogger('ScintSuite.INPAStrikes')


# -----------------------------------------------------------------------------
# %% Strikes Objects
# -----------------------------------------------------------------------------
class INPAStrikes(Strikes):
    """
    Strike object tunned for INPA.

    Public methods not present in the parent class:

        calculatePitch: calculate the pitch of the markers
    """

    # -------------------------------------------------------------------------
    # ---- Gyroradius and pitch angle
    # -------------------------------------------------------------------------
    def calculatePitch(self, Boptions: dict = {}, IPBtSign: float = -1.0):
        """
        Calculate the pitch of the markers.

        Jose Rueda: jrrueda@us.es

        Note: This function only works for INPA markers
        Note2: This is mainly though to be used for FIDASIM markers, no mapping
        for these guys, there is only one set of data, so is nice. If you use
        it for the mapping markers, there would be ngyr x nR set of data and
        the magnetic field shotfile will be open those number of times. Not the
        end of the world, but just to have it in mind. A small work arround
        could be added if needed, althoug that would imply something like:
        if machine == 'AUG'... which I would like to avoid

        :param  Boptions: extra parameters for the calculation of the magnetic
            field
        """
        # Only works for INPA markers, return error if we try with FILD
        if self.header['FILDSIMmode']:
            raise errors.NotValidInput('Hey! This is only for INPA')
        # See if we need to overwrite
        if 'pitch0' in self.header['info'].keys():
            print('The pitch values are there, we will overwrite them')
            overwrite = True
            ipitch0 = self.header['info']['pitch0']['i']
        else:
            overwrite = False
        # Get the index for the needed variables
        ir = self.header['info']['R0']['i']          # get the index
        iz = self.header['info']['z0']['i']
        ix = self.header['info']['x0']['i']
        iy = self.header['info']['y0']['i']
        ivz = self.header['info']['vz0']['i']
        ivx = self.header['info']['vx0']['i']
        ivy = self.header['info']['vy0']['i']
        for ig in range(self.header['ngyr']):
            for ia in range(self.header['nXI']):
                if self.header['counters'][ia, ig] > 0:
                    phi = np.arctan2(self.data[ia, ig][:, iy],
                                     self.data[ia, ig][:, ix])
                    br, bz, bt, bp =\
                        ssdat.get_mag_field(self.header['shot'],
                                            self.data[ia, ig][:, ir],
                                            self.data[ia, ig][:, iz],
                                            time=self.header['time'],
                                            **Boptions)
                    bx = br*np.cos(phi) - bt*np.sin(phi)
                    by = - br*np.cos(phi) + bt*np.cos(phi)
                    b = np.sqrt(bx**2 + by**2 + bz**2).squeeze()
                    v = np.sqrt(self.data[ia, ig][:, ivx]**2
                                + self.data[ia, ig][:, ivy]**2
                                + self.data[ia, ig][:, ivz]**2)
                    pitch = (self.data[ia, ig][:, ivx] * bx
                             + self.data[ia, ig][:, ivy] * by
                             + self.data[ia, ig][:, ivz] * bz).squeeze()
                    pitch /= v*b*IPBtSign
                    if overwrite:
                        self.data[ia, ig][:, ipitch0] = pitch.copy()
                    else:
                        self.data[ia, ig] = \
                            np.append(self.data[ia, ig],
                                      np.atleast_2d(pitch.copy()).T, axis=1)

        if not overwrite:
            Old_number_colums = len(self.header['info'])
            extra_column = {
                'pitch0': {
                    'i': Old_number_colums,  # Column index in the file
                    'units': ' []',  # Units
                    'longName': 'Pitch',
                    'shortName': '$\\lambda_{0}$',
                },
            }
            self.header['info'].update(extra_column)

    def calculateGyroradius(self, B, A=2.014, Z=1.0):
        # Only works for INPA markers, return error if we try with FILD
        if self.header['FILDSIMmode']:
            raise errors.NotValidInput('Hey! This is only for INPA')
        # See if we need to overwrite
        if 'gyroradius0' in self.header['info'].keys():
            print('The pitch values are there, we will overwrite them')
            overwrite = True
            ir0 = self.header['info']['gyroradius0']['i']
            irs = self.header['info']['gyroradiuss']['i']
        else:
            overwrite = False
        # Get the index for the needed variables
        ie0 = self.header['info']['e0']['i']
        ies = self.header['info']['es']['i']
        for ig in range(self.header['ngyr']):
            for ia in range(self.header['nXI']):
                if self.header['counters'][ia, ig] > 0:
                    pin = get_gyroradius(self.data[ia, ig][:, ie0]*1000.0, B,
                                         A, Z)
                    scint = get_gyroradius(self.data[ia, ig][:, ies]*1000.0, B,
                                           A, Z)
                    if overwrite:
                        self.data[ia, ig][:, ir0] = pin.copy()
                        self.data[ia, ig][:, irs] = scint.copy()
                    else:
                        self.data[ia, ig] = \
                            np.append(self.data[ia, ig],
                                      np.atleast_2d(pin.copy()).T, axis=1)
                        self.data[ia, ig] = \
                            np.append(self.data[ia, ig],
                                      np.atleast_2d(scint.copy()).T, axis=1)

        if not overwrite:
            Old_number_colums = len(self.header['info'])
            extra_column = {
                'gyroradius0': {
                    'i': Old_number_colums,  # Column index in the file
                    'units': ' []',  # Units
                    'longName': 'Gyroradius (pin)',
                    'shortName': '$cm$',
                },
                'gyroradiuss': {
                    'i': Old_number_colums + 1,  # Column index in the file
                    'units': ' []',  # Units
                    'longName': 'Gyroradius (scint)',
                    'shortName': '$cm$',
                },
            }
            self.header['info'].update(extra_column)

    # -------------------------------------------------------------------------
    # ---- Synthetic Signals
    # -------------------------------------------------------------------------
    def _getBHead(self, R, z):
        """
        Get the magnetic field.

        :param R: (float) Radial position to calculate the magnetic field, m
        :param z: (float) z position to calculate the magnetic field, m

        :Example:
        >>> # Load some INPA strikes in the strikes variable
        >>> strikes._getBHead(self, R=1.91121, z=0.95836)

        """
        # ---- Load the magnetic field
        br, bz, bt, bp = \
            ssdat.get_mag_field(self.header['shot'],
                                R,
                                z,
                                time=self.header['time'])
        Bmod = np.sqrt(bt**2 + bp**2)[0]
        self.B = {
            'Br': np.array(br).squeeze(),
            'Bz': np.array(bz).squeeze(),
            'Bt': np.array(bt).squeeze(),
            'B': Bmod,
            }

    def calculateSynthetiSignal(self,
                                smap,
                                R: float = 1.91121,
                                z: float = 0.95836,
                                nw: int = 480,
                                nh: int = 640,
                                opticalCalibration=None,
                                remap_options: dict = {},
                                remap_variables: tuple = ('R0', 'e0'),
                                sigmaOptics: float = 0.0,
                                includeOpticalTransmission: bool = True,
                                energyFit: str = None,
                                machine: str = 'AUG',
                                noiseLevel: float = 0.0
                                ):
        """
        Calculate the INPA synthetic signal starting from the strikes.

        :param R: (float) Radial position to calculate the magnetic field
        :param z: (float) z position to calculate the magnetic field
        :param noiseLevel: will be changed
        """
        # ---- Initialise the remap options
        remap_parameters = {
            'ymin': 10.0,      # Minimum energy [in keV]
            'ymax': 100.0,     # Maximum energy [in keV]
            'dy': 2.0,         # Interval of the gyroradius [in cm]
            'xmin': 1.5,       # Minimum radious [in m]
            'xmax': 2.12,      # Maximum radious [in m]
            'dx': 0.01,        # radius
            # methods for the interpolation
            'method': 2,       # 2 Spline, 1 Linear (smap interpolation)
            'decimals': 0,     # Precision for the strike map
            'remap_method': 'MC',  # Remap algorithm
            'MC_number': 200,
            }
        remap_parameters.update(remap_options)
        # ---- Prepare the magnetic field
        if self.B is None:
            self._getBHead(R, z)
        # ---- Prepare the strike map
        # smap = ssmap.Ismap(file=smapFile)
        smap.calculate_pixel_coordinates(opticalCalibration)
        smap.calculate_energy(self.B['B'])
        # ---- Transform to pixels
        if 'xcam' not in self.header['info'].keys():
            self.calculate_pixel_coordinates(opticalCalibration)
        else:
            text = 'Pixel position present in the object' +\
                ', assuming they are right'
            logger.warning(text)
        # ---- Include optical transmission
        if includeOpticalTransmission:
            logger.info('Including F number.')
            F = FnumberTransmission(diag='INPA', machine=machine,
                                    geomID=self.header['geomID'])
            self.applyGeometricTramission(F, opticalCalibration)
        else:
            logger.info('NOT including F number.')

        # ---- Perform the remap in the camera space
        if 'xcam_ycam' not in self.histograms.keys():
            self.calculate_2d_histogram('xcam', 'ycam',
                                        binsx=np.arange(nw+1),
                                        binsy=np.arange(nh+1))
            # ---- Add noise
            if noiseLevel > 0.0:
                logger.info('Adding random noise')
                noise = noiseLevel * np.random.rand(nw * nh).reshape(nw, nh)
                for k in self.histograms['xcam_ycam'].keys():
                    for jk, kind in enumerate(self.histograms[
                            'xcam_ycam'].kind.values):
                        self.histograms['xcam_ycam'][k].values[
                                :, :, jk] += noise
            # Apply finite focus
            if sigmaOptics > 0.05:
                self.histograms['xcam_ycam_finiteFocus'] = \
                    self.histograms['xcam_ycam'].copy()
                for k in self.histograms['xcam_ycam_finiteFocus'].keys():
                    for jk, kind in enumerate(self.histograms[
                            'xcam_ycam_finiteFocus'].kind.values):
                        total_counts = \
                           self.histograms['xcam_ycam_finiteFocus'][k].sel(
                                kind=kind).values.sum()
                        if total_counts < 0:
                            continue
                        defocus_matrix = defocus(
                            self.histograms['xcam_ycam_finiteFocus'][k].sel(
                                kind=kind).values,
                            coef_sigma=sigmaOptics)
                        self.histograms['xcam_ycam_finiteFocus'][k].values[
                            :, :, jk] = defocus_matrix.copy()
        else:
            logger.warning('Using xcam_ycam histogram present in the object')

        # ---- Remap the camera strike points
        self.remap(smap, remap_parameters, remap_variables)

        # ---- Apply energy fit:
        # Load the energy fit
        if energyFit is not None and 'e0' in remap_variables:
            fit = lmfit.model.load_modelresult(energyFit)
            remap_name = remap_variables[0] + '_' + remap_variables[1] + '_remap'
            yaxis = self.histograms[remap_name]['y']
            factor = fit.eval(x=yaxis.values)
            scale = xr.DataArray(factor, dims='y',
                                 coords={'y': yaxis})
            self.histograms[remap_name + '_Efit'] = \
                self.histograms[remap_name] * scale
            if sigmaOptics > 0.02:
                self.histograms[remap_name + '_finiteFocus' + '_Efit'] = \
                    self.histograms[remap_name + '_finiteFocus'] * scale
