"""
Strike map class

Jose Rueda: jrrueda@us.es
"""
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interp
import Lib.LibPlotting as ssplt
import Lib.SimulationCodes.FILDSIM as ssFILDSIM
# import Lib.SimulationCodes.SINPA as ssSINPA
from Lib.SimulationCodes.Common.strikes import Strikes
from Lib.LibMap.Common import XYtoPixel, _fit_to_model_
from Lib.LibMachine import machine
import Lib.LibPaths as p
from Lib.decorators import deprecated
import Lib.errors as errors
from tqdm import tqdm   # For waitbars
pa = p.Path(machine)
del p


class StrikeMap(XYtoPixel):
    """Class with the information of the strike map."""

    def __init__(self, flag=0, file: str = None, machine='AUG', theta=None,
                 phi=None, decimals=1):
        """
        Initialise the class.

        Thera are 2 ways of selecting the smap: give the full path to the file,
        or give the theta and phi angles and the machine, so the strike map
        will be selected from the remap database. This is still not implemented
        for the INPA

        @param flag: 0  means FILD, 1 means INPA, 2 means iHIBP (you can also
        write directly 'FILD', 'INPA', 'iHIBP')
        @param file: Full path to file with the strike map
        @param machine: machine, to look in the datbase
        @param theta: theta angle  (see FILDSIM doc) (zita for SINPA)
        @param phi: phi angle (see FILDSIM doc) (ipsilon for SINPA)
        @param decimals: decimals to look in the database

        Notes
        - Machine, theta and phi options introduced in version 0.4.14.
        - INPA compatibility included in version 0.6.0

        Warning for INPA: if alpha grows and R0 decrease there will be a
        decorrelation issue, maybe, contact Jose Rueda if this is your case
        """
        ## Init the parent class
        XYtoPixel.__init__(self)
        ## Associated diagnostic
        if flag == 0 or str(flag).lower() == 'fild':
            self.diag = 'FILD'
        elif flag == 1 or str(flag).lower() == 'inpa':
            self.diag = 'INPA'
        elif flag == 2 or str(flag).lower() == 'ihibp':
            self.diag = 'iHIBP'
            raise errors.NotImplementedError(
                'Diagnostic not implemented: Talk with Pablo')
        else:
            print('Flag: ', flag)
            raise errors.NotValidInput('Diagnostic not understood')
        # --- Initialise the part which are commond for the 3 diagnostics:
        ## X-position, in pixels, of the strike map (common)
        self.xpixel = None
        ## Y-Position, in pixels, of the strike map (common)
        self.ypixel = None
        ## file
        if file is not None:
            self.file = file
        ## Resolution of FILD (INPA) for each strike point
        self.resolution = None
        ## Interpolators (gyr, pitch)-> sigma_r, sigma_p, etc, (or gyr, aplha)
        self.interpolators = None
        ## x coordinates of map points
        self.x = None
        ## y coordinates of map points
        self.y = None
        ## z coordinates of map points
        self.z = None
        ## Translate from pixels in the camera to velocity space
        self.grid_interp = None
        ## Strike points used to calculate the map
        self.strike_points = None
        ## from strike position to phase space
        self.map_interpolators = None
        if self.diag == 'FILD':
            # Read the file
            if file is None:
                smap_folder = pa.FILDStrikeMapsRemap
                dumm = ssFILDSIM.guess_strike_map_name(phi, theta,
                                                       machine=machine,
                                                       decimals=decimals)
                file = os.path.join(smap_folder, dumm)
                self.file = file
            if not os.path.isfile(file):
                raise Exception('Strike map no found in the database')
            # Look for the code identifier, the second row of SINPA files start
            # with a 1:
            try:
                version = np.loadtxt(file, skiprows=1, max_rows=1)
                self.code = 'SINPA'
                version = version.astype(int)
                if version.size == 2:  # New strike map from SINPA 0.3
                    self.versionID1 = int(version[0])
                    self.versionID2 = int(version[1])
                else:   # Old SINPA strike map
                    self.versionID1 = 0
                    self.versionID2 = int(version)
            except ValueError:
                # If this second line did not started with a number, the strike
                # map was generated with FILDSIM. (yes, it could be that it was
                # generated with SINPA but the user modified manually the file,
                # we can't be preared for that)
                self.code = 'FILDSIM'
                self.versionID1 = 0
                self.versionID2 = 0
            if self.versionID1 < 2:
                dummy = np.loadtxt(file, skiprows=3)
                # See which rows has collimator factor larger than zero (ie see
                # for which combination of gyroradius and pitch some markers
                # arrived)
                ind = dummy[:, 7] > 0
                # Initialise the class
                ## Gyroradius of map points
                self.gyroradius = dummy[ind, 0]
                ## Simulated gyroradius (unique points of self.gyroradius)
                self.unique_gyroradius = np.unique(self.gyroradius)
                ## Energy of map points
                self.energy = None
                ## Pitch of map points
                self.pitch = dummy[ind, 1]
                ## Simulated pitches (unique points of self.pitch)
                self.unique_pitch = np.unique(self.pitch)
                # x coordinates of map points (common)
                self.x = dummy[ind, 2]
                # y coordinates of map points (common)
                self.y = dummy[ind, 3]
                # z coordinates of map points (common)
                self.z = dummy[ind, 4]
                ## Average initial gyrophase of map markers
                self.avg_ini_gyrophase = dummy[ind, 5]
                ## Number of markers striking in this area
                self.n_strike_points = dummy[ind, 6].astype(np.int)
                ## Collimator factor as defined in FILDSIM
                self.collimator_factor = dummy[ind, 7]
                ## Average incident angle of the FILDSIM markers
                self.avg_incident_angle = dummy[ind, 8]
                ## Colimator facror as a matrix
                # This simplify a lot W calculation and forward modelling:
                self.ngyr = len(self.unique_gyroradius)
                self.npitch = len(self.unique_pitch)
                self.collimator_factor_matrix = \
                    np.zeros((self.ngyr, self.npitch))
                for ir in range(self.ngyr):
                    for ip in range(self.npitch):
                        # By definition, flags can only have one True
                        flags = \
                            (self.gyroradius == self.unique_gyroradius[ir]) \
                            * (self.pitch == self.unique_pitch[ip])
                        if np.sum(flags) > 0:
                            self.collimator_factor_matrix[ir, ip] = \
                                self.collimator_factor[flags]
                # After SINPA arrived, the 'pitch' is named as XI, as basically
                # INPA and FILD are identical, just different names in this
                # variable, so a common name was choosen. All FILD user have
                # their routines constructed with the name of pitch, so I'll
                # not destroy it, but I'll add the new naming for future users:
                self.XI = self.pitch
                self.unique_XI = self.unique_pitch
                self.nXI = self.npitch
            else:
                raise Exception('Not recognised StrikeMap version')
        elif self.diag == 'INPA':
            try:
                version = np.loadtxt(file, skiprows=1, max_rows=1)
                self.code = 'SINPA'
                version = version.astype(int)
                if version.size == 2:  # New strike map from SINPA 0.3
                    self.versionID1 = int(version[0])
                    self.versionID2 = int(version[1])
                else:   # Old SINPA strike map
                    self.versionID1 = 0
                    self.versionID2 = int(version)
            except ValueError:
                raise Exception('Corrupted Strike Map.')
            dummy = np.loadtxt(file, skiprows=3)
            # See which rows has collimator factor larger than zero (ie see for
            # which combination of rl and alpha some markers arrived)
            ind = dummy[:, 7] > 0
            # Initialise the class
            ## Gyroradius of map points
            self.gyroradius = dummy[ind, 0]
            ## Simulated gyroradius (unique points of self.gyroradius)
            self.unique_gyroradius = np.unique(self.gyroradius)
            ## Energy of map points
            self.energy = None
            ## Alpha of map points
            self.alpha = dummy[ind, 1]
            ## Simulated pitches (unique points of self.pitch)
            self.unique_alpha = np.unique(self.alpha)
            # x coordinates of map points (common)
            self.x = dummy[ind, 2]
            # y coordinates of map points (common)
            self.y = dummy[ind, 3]
            # z coordinates of map points (common)
            self.z = dummy[ind, 4]
            ## x coordinates of closest point to NBI
            self.x0 = dummy[ind, 9]
            ## y coordinates of closest point to NBI
            self.y0 = dummy[ind, 10]
            ## z coordinates of closest point to NBI
            self.z0 = dummy[ind, 11]
            ## R position of the closes point to NBI
            self.R0 = np.sqrt(self.x0**2 + self.y0**2)
            ## distance to the NBI central line
            self.d0 = dummy[ind, 12]
            ## Collimator factor as defined in FILDSIM
            self.collimator_factor = dummy[ind, 7]
            ## Number of markers striking in this area
            self.n_strike_points = dummy[ind, 6]
            ## Number of markers striking in this area
            self.avg_beta_ini = dummy[ind, 5]
            ## Number of markers striking in this area
            self.avg_incident_angle = dummy[ind, 8]
            ## Colimator factor and R0 as a matrix
            # This simplify a lot W calculation and forward modelling:
            self.ngyr = len(self.unique_gyroradius)
            self.nalpha = len(self.unique_alpha)
            self.collimator_factor_matrix = np.zeros((self.ngyr, self.nalpha))
            R0 = np.zeros((self.ngyr, self.nalpha))
            R0[:] = np.NaN
            for ir in range(self.ngyr):
                for ip in range(self.nalpha):
                    # By definition, flags can only have one True
                    flags = (self.gyroradius == self.unique_gyroradius[ir]) \
                        * (self.alpha == self.unique_alpha[ip])
                    if flags.sum() > 1:
                        raise Exception('Revise StrikeMap')
                    if np.sum(flags) > 0:
                        self.collimator_factor_matrix[ir, ip] = \
                            self.collimator_factor[flags]
                        R0[ir, ip] = self.R0[flags]
            # average on gyroradius (as for limited precision there can be some
            # diferences)
            self.unique_R0 = np.nanmean(R0, axis=0)
            # In the case of INPA, the 'XI' variable, the joker, will be set as
            # the R0:
            self.XI = self.R0
            self.unique_XI = self.unique_R0
            self.nXI = self.nalpha

    def plot_real(self, ax=None,
                  marker_params: dict = {}, line_params: dict = {},
                  labels: bool = False,
                  rotation_for_gyr_label: float = 90.0,
                  rotation_for_pitch_label: float = 30.0,
                  factor: float = 1.0):
        """
        Plot the strike map (x,y = dimensions in the scintillator).

        Jose Rueda: jrrueda@us.es

        @param ax: Axes where to plot
        @param markers_params: parameters for plt.plot() to plot the markers
        @param line_params: parameters for plt.plot() to plot the markers
        @param labels: flag to add the labes (gyroradius, pitch) on the plot
        @param rotation_for_gyr_label: Rotation angle for the gyroradius label
        @param rotation_for_pitch_label: Rotation angle for the pitch label
        @param factor: scaling factor to plot the data. Dimensions will be
            multiplied by this factor. Notice that this is just to compare
            strike maps from different codes for the situations in which a code
            operate in cm and the oder in m.

        Note:
            - The rotation_for_pitch_label should be called in a different way
            (for example ..._xi_label using SINPA notation), but we will keep
            it like this to do not disturb FILD users which as this variable
            alrea
        """
        # Default plot parameters:
        marker_options = {
            'markersize': 6,
            'fillstyle': 'none',
            'color': 'k',
            'marker': 'o',
            'linestyle': 'none'
        }
        marker_options.update(marker_params)
        line_options = {
            'color': 'k',
            'marker': ''
        }
        line_options.update(line_params)

        if ax is None:
            fig, ax = plt.subplots()
        # Draw the line of constant Gyroradius (energy). 'Horizontal'
        calculated_delta = False
        j = 1
        for i in range(self.ngyr):
            flags = self.gyroradius == self.unique_gyroradius[i]
            ax.plot(self.y[flags] * factor, self.z[flags] * factor,
                    **line_options)
            # Add the gyroradius label, but just each 2 entries so the plot
            # does not get messy
            if i == j:
                try:
                    delta = abs(self.y[flags][1] - self.y[flags][0]) * factor
                    calculated_delta = True
                except IndexError:
                    j += 2
            if (i % 2 == 0) and labels and calculated_delta:  # add gyro radius labels
                # Delta variable just to adust nicelly the distance (as old
                # fildsim is in cm and new in m)
                ax.text((self.y[flags]).min() * factor - 0.5 * delta,
                        (self.z[flags]).min() * factor,
                        f'{float(self.unique_gyroradius[i]):g}',
                        horizontalalignment='right',
                        verticalalignment='center')
        if labels:
            ax.annotate('Gyroradius [cm]',
                        xy=(min(self.y) * factor - delta,
                            ((max(self.z) - min(self.z))/2 + min(self.z))
                            * factor),
                        rotation=rotation_for_gyr_label,
                        horizontalalignment='center',
                        verticalalignment='center')
        if self.diag == 'FILD':
            # Draw the lines of constant pitch. 'Vertical' lines
            for i in range(self.npitch):
                flags = self.pitch == self.unique_pitch[i]
                ax.plot(self.y[flags] * factor, self.z[flags] * factor,
                        **line_options)
                if i == 1:
                    delta = abs(self.z[flags][-1] - self.z[flags][-2]) * factor
                if labels:
                    ax.text((self.y[flags])[-1] * factor,
                            (self.z[flags])[-1] * factor - delta,
                            f'{float(self.unique_pitch[i]):g}',
                            horizontalalignment='center',
                            verticalalignment='top')
            if labels:
                ax.annotate('Pitch [$\\degree$]',
                            xy=(((max(self.y) - min(self.y))/2 + min(self.y))
                                * factor,
                                min(self.z) * factor - 1.5 * delta),
                            rotation=rotation_for_pitch_label,
                            horizontalalignment='center',
                            verticalalignment='center')
        elif self.diag == 'INPA':
            # Draw the lines of constant alpha. 'Vertical' lines
            for i in range(self.nalpha):
                flags = self.alpha == self.unique_alpha[i]
                if i == 0:
                    delta = abs(self.y[flags][1] - self.y[flags][0]) * factor

                ax.plot(self.y[flags] * factor, self.z[flags] * factor,
                        **line_options)

                ax.text((self.y[flags])[-1] * factor,
                        (self.z[flags])[-1] * factor - delta,
                        f'{float(self.unique_alpha[i]):g}',
                        horizontalalignment='center',
                        verticalalignment='top')

            ax.annotate('Alpha [rad]',
                        xy=(((max(self.y) - min(self.y))/2 + min(self.y))
                            * factor,
                            min(self.z) * factor - 1.5*delta),
                        rotation=rotation_for_pitch_label,
                        horizontalalignment='center',
                        verticalalignment='center')
        else:
            raise errors.NotValidInput('Diagnostic not implemented')

        # Plot some markers in the grid position
        ax.plot(self.y * factor, self.z * factor, **marker_options)

    def plot_pix(self, ax=None, marker_params: dict = {},
                 line_params: dict = {}, labels: bool = False,
                 rotation_for_gyr_label: float = 100.0,
                 rotation_for_xi_label: float = 30.0,):
        """
        Plot the strike map (x,y = pixels on the camera).

        Jose Rueda: jrrueda@us.es

        @param ax: Axes where to plot
        @param marker_params: parameters for the centroid plotting
        @param line_params: parameters for the lines plotting
        """
        # Default plot parameters:
        marker_options = {
            'markersize': 6,
            'fillstyle': 'none',
            'color': 'w',
            'marker': 'o',
            'linestyle': 'none'
        }
        marker_options.update(marker_params)
        line_options = {
            'color': 'w',
            'marker': ''
        }
        line_options.update(line_params)

        if ax is None:
            fig, ax = plt.subplots()

        # Draw the lines of constant gyroradius, energy, or rho (depending on
        # the particular diagnostic) [These are the 'horizontal' lines]
        if self.diag == 'FILD':
            # Lines of constant gyroradius
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_options)
            # Lines of constant pitch
            uniq = np.unique(self.pitch)
            n = len(uniq)
            for i in range(n):
                flags = self.pitch == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_options)
        elif self.diag == 'INPA':
            # Lines of constant gyroradius
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            j = 1
            calculated_delta = False
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_options)
                # Add the gyroradius label, but just each 2 entries so the plot
                # does not get messy
                if i == j:
                    try:
                        delta = abs(self.ypixel[flags]
                                    [1] - self.ypixel[flags][0])
                        calculated_delta = True
                    except IndexError:
                        j += 2
                if (i % 2 == 0) and labels and calculated_delta:  # add labels
                    # Delta variable just to adust nicelly the distance (as old
                    # fildsim is in cm and new in m)
                    ax.text((self.xpixel[flags][0]) - 0.5 * delta,
                            (self.ypixel[flags][0]),
                            '%.2f' % (float(self.unique_gyroradius[i])),
                            horizontalalignment='right',
                            verticalalignment='center',
                            color=line_options['color'])
                if i == round(n/2) and labels:
                    ax.text((self.xpixel[flags][0]) - 8.0 * delta,
                            (self.ypixel[flags][0]),
                            'Gyroradius [cm]',
                            horizontalalignment='center',
                            verticalalignment='center',
                            rotation=rotation_for_gyr_label,
                            color=line_options['color'])
            # Lines of constant pitch
            uniq = self.unique_XI
            n = len(uniq)
            j = 1
            calculated_delta = False
            for i in range(n):
                flags = abs(self.XI - uniq[i]) < 0.02
                ax.plot(self.xpixel[flags], self.ypixel[flags], **line_options)
                # Add the gyroradius label, but just each 2 entries so the plot
                # does not get messy
                if i == j:
                    try:
                        delta = abs(self.ypixel[flags]
                                    [1] - self.ypixel[flags][0])
                        calculated_delta = True
                    except IndexError:
                        j += 2
                if (i % 2 == 0) and labels and calculated_delta:  # add labels
                    # Delta variable just to adust nicelly the distance (as old
                    # fildsim is in cm and new in m)
                    ax.text((self.xpixel[flags][0]) - 0.5 * delta,
                            (self.ypixel[flags][0]),
                            '%.2f' % (float(self.unique_XI[i])),
                            horizontalalignment='right',
                            verticalalignment='center',
                            color=line_options['color'])
                if i == round(n/2) and labels:
                    ax.text((self.xpixel[flags][0]),
                            (self.ypixel[flags][0])+3*delta,
                            'R [m]',
                            horizontalalignment='center',
                            verticalalignment='center',
                            rotation=rotation_for_xi_label,
                            color=line_options['color'])
        else:
            raise errors.NotImplementedError('Not implemented diagnostic')

        # Plot some markers in the grid position
        ## @todo include labels energy/pitch in the plot
        ax.plot(self.xpixel, self.ypixel, **marker_options)

    def interp_grid(self, frame_shape, method=2, plot=False, verbose=False,
                    grid_params: dict = {}, MC_number: int = 100):
        """
        Interpolate grid values on the frames.

        @param frame_shape: Size of the frame used for the calibration (in px)
        @param method: method to calculate the interpolation:
            - 1: griddata linear (you can also write 'linear')
            - 2: griddata cubic  (you can also write 'cubic')
        @param plot: flag to perform a quick plot to see the interpolation
        @param verbose: flag to print some info along the way
        @param grid_params: grid options for the transformationn matrix grid
        @param MC_number: Number of MC markers for the transformation matrix,
        if this number < 0, the transformation matrix will not be calculated

        Info on the dictionary self.grid_interp:
            'gyroradius': gyroradius values of each pixel
            'pitch': pitch values of each pixel
            'collimator_factor': Collimator values of each pixel
            'interpolators': {
                'gyroradius': interpolator pixel -> rl
                'pitch': interpolator pixel-> pitch [only for FILD]
                'alpha': interpolator pixel-> alpha [only for INPA]
                'collimator_factor': interpolator pixel to fcol [only for FILD]
            },
            'transformation_matrix': 4D tensor for the remap
        """
        # --- 0: Check inputs
        if self.xpixel is None:
            raise Exception('Transform to pixel the strike map before')
        # Default grid options
        grid_options = {
            'ymin': 1.2,
            'ymax': 10.5,
            'dy': 0.1,
            'xmin': 20.0,
            'xmax': 90.0,
            'dx': 1.0
        }
        grid_options.update(grid_params)

        # --- 1: Create grid for the interpolation
        # Note, it seems transposed, but the reason is that the calibration
        # paramters were adjusted with the frame transposed (to agree with old
        # IDL implementation) therefore we have to transpose a bit almost
        # everything. Sorry for the headache
        grid_x, grid_y = np.mgrid[0:frame_shape[1], 0:frame_shape[0]]
        # --- 2: Interpolate the grid
        # Prepare the grid for te griddata method:
        dummy = np.column_stack((self.xpixel, self.ypixel))
        # Prepare the options and interpolators for later
        if method == 1 or str(method).lower() == 'linear':
            met = 'linear'
            interpolator = scipy_interp.LinearNDInterpolator
        elif method == 2 or str(method).lower == 'cubic':
            met = 'cubic'
            interpolator = scipy_interp.CloughTocher2DInterpolator
        else:
            raise errors.NotValidInput('Not recognized interpolation method')
        if verbose:
            print('Using %s interpolation of the grid' % met)
        if self.diag == 'FILD':
            # Initialise the structure
            self.grid_interp = {
                'gyroradius': None,
                'xi': None,
                'pitch': None,
                'collimator_factor': None,
                'interpolators': {
                    'gyroradius': None,
                    'pitch': None,
                    'xi': None,
                    'collimator_factor': None
                },
                'transformation_matrix': None
            }
            # Get gyroradius values of each pixel
            dummy2 = scipy_interp.griddata(dummy, self.gyroradius,
                                           (grid_x, grid_y), method=met,
                                           fill_value=1000)
            self.grid_interp['gyroradius'] = dummy2.copy().T
            # Get pitch values of each pixel
            dummy2 = scipy_interp.griddata(dummy, self.pitch, (grid_x, grid_y),
                                           method=met, fill_value=1000)
            self.grid_interp['xi'] = dummy2.copy().T
            self.grid_interp['pitch'] = self.grid_interp['xi']
            # Get collimator factor
            dummy2 = scipy_interp.griddata(dummy, self.collimator_factor,
                                           (grid_x, grid_y), method=met,
                                           fill_value=1000)
            self.grid_interp['collimator_factor'] = dummy2.copy().T
            # Calculate the interpolator
            grid = list(zip(self.xpixel, self.ypixel))
            self.grid_interp['interpolators']['gyroradius'] = \
                interpolator(grid, self.gyroradius, fill_value=1000)
            self.grid_interp['interpolators']['pitch'] = \
                interpolator(grid, self.pitch, fill_value=1000)
            self.grid_interp['interpolators']['xi'] = \
                self.grid_interp['interpolators']['pitch']
            self.grid_interp['interpolators']['collimator_factor'] = \
                interpolator(grid, self.collimator_factor, fill_value=1000)
            # --- Prepare the transformation matrix
            # Calculate the transformation matrix
            if MC_number > 0:
                # Initialise the random number generator
                rand = np.random.default_rng()
                generator = rand.uniform
                # Prepare the edges for the r, pitch histogram
                n_gyr = int((grid_options['ymax'] - grid_options['ymin'])
                            / grid_options['dy']) + 1
                n_pitch = int((grid_options['xmax'] - grid_options['xmin'])
                              / grid_options['dx']) + 1
                pitch_edges = grid_options['xmin'] - grid_options['dx']/2 \
                    + np.arange(n_pitch+1) * grid_options['dx']
                gyr_edges = grid_options['ymin'] - grid_options['dy']/2 \
                    + np.arange(n_gyr+1) * grid_options['dy']
                # Initialise the transformation matrix
                transform = np.zeros((n_pitch, n_gyr,
                                      frame_shape[0], frame_shape[1]))
                print('Calculating transformation matrix')
                for i in tqdm(range(frame_shape[0])):
                    for j in range(frame_shape[1]):
                        # Generate markers coordinates in the chip, note the
                        # first dimmension of the frame is y-pixel
                        # (IDL heritage)
                        x_markers = j + generator(size=MC_number)
                        y_markers = i + generator(size=MC_number)
                        # Calculate the r-pitch coordinates
                        r_markers = \
                            self.grid_interp['interpolators']['gyroradius'](
                                x_markers, y_markers)
                        p_markers = self.grid_interp['interpolators']['pitch'](
                            x_markers, y_markers)
                        # make the histogram in the r-pitch space
                        H, xedges, yedges = \
                            np.histogram2d(p_markers, r_markers,
                                           bins=[pitch_edges, gyr_edges])
                        transform[:, :, i, j] = H.copy()
                # Normalise the transformation matrix
                transform /= MC_number
                transform /= (grid_options['dx'] * grid_options['dy'])
                self.grid_interp['transformation_matrix'] = transform
        if self.diag == 'INPA':
            # Initialise the structure
            self.grid_interp = {
                'gyroradius': None,
                'xi': None,
                'collimator_factor': None,
                'interpolators': {
                    'gyroradius': None,
                    'xi': None,
                    'collimator_factor': None
                },
                'transformation_matrix': None
            }
            # Get gyroradius values of each pixel
            dummy2 = scipy_interp.griddata(dummy, self.gyroradius,
                                           (grid_x, grid_y), method=met,
                                           fill_value=1000)
            self.grid_interp['gyroradius'] = dummy2.copy().T
            # Get pitch values of each pixel
            dummy2 = scipy_interp.griddata(dummy, self.XI, (grid_x, grid_y),
                                           method=met, fill_value=1000)
            self.grid_interp['xi'] = dummy2.copy().T
            # Get collimator factor
            dummy2 = scipy_interp.griddata(dummy, self.collimator_factor,
                                           (grid_x, grid_y), method=met,
                                           fill_value=1000)
            self.grid_interp['collimator_factor'] = dummy2.copy().T
            # Calculate the interpolator
            grid = list(zip(self.xpixel, self.ypixel))
            self.grid_interp['interpolators']['gyroradius'] = \
                interpolator(grid, self.gyroradius, fill_value=1000)
            self.grid_interp['interpolators']['xi'] = \
                interpolator(grid, self.XI, fill_value=1000)
            self.grid_interp['interpolators']['collimator_factor'] = \
                interpolator(grid, self.collimator_factor, fill_value=1000)
            # --- Prepare the transformation matrix
            # Calculate the transformation matrix
            if MC_number > 0:
                # Initialise the random number generator
                rand = np.random.default_rng()
                generator = rand.uniform
                # Prepare the edges for the r, pitch histogram
                n_gyr = int((grid_options['ymax'] - grid_options['ymin'])
                            / grid_options['dy']) + 1
                n_pitch = int((grid_options['xmax'] - grid_options['xmin'])
                              / grid_options['dx']) + 1
                pitch_edges = grid_options['xmin'] - grid_options['dx']/2 \
                    + np.arange(n_pitch+1) * grid_options['dx']
                gyr_edges = grid_options['ymin'] - grid_options['dy']/2 \
                    + np.arange(n_gyr+1) * grid_options['dy']
                # Initialise the transformation matrix
                transform = np.zeros((n_pitch, n_gyr,
                                      frame_shape[0], frame_shape[1]))
                print('Calculating transformation matrix')
                for i in tqdm(range(frame_shape[0])):
                    for j in range(frame_shape[1]):
                        # Generate markers coordinates in the chip, note the
                        # first dimmension of the frame is y-pixel
                        # (IDL heritage)
                        x_markers = j + generator(size=MC_number)
                        y_markers = i + generator(size=MC_number)
                        # Calculate the r-pitch coordinates
                        r_markers = \
                            self.grid_interp['interpolators']['gyroradius'](
                                x_markers, y_markers)
                        p_markers = self.grid_interp['interpolators']['xi'](
                            x_markers, y_markers)
                        # make the histogram in the r-pitch space
                        H, xedges, yedges = \
                            np.histogram2d(p_markers, r_markers,
                                           bins=[pitch_edges, gyr_edges])
                        transform[:, :, i, j] = H.copy()
                # Normalise the transformation matrix
                transform /= MC_number
                transform /= (grid_options['dx'] * grid_options['dy'])
                self.grid_interp['transformation_matrix'] = transform

        # --- Plot
        if plot:
            if self.diag == 'INPA':
                fig, axes = plt.subplots(2, 2)
                # Plot the scintillator grid
                self.plot_pix(axes[0, 0], line_params={'color': 'k'})
                # Plot the interpolated gyroradius
                c1 = axes[0, 1].imshow(self.grid_interp['gyroradius'],
                                       cmap=ssplt.Gamma_II(),
                                       vmin=0, vmax=10, origin='lower')
                fig.colorbar(c1, ax=axes[0, 1], shrink=0.9)
                # Plot the interpolated pitch
                c2 = axes[1, 0].imshow(self.grid_interp['xi'],
                                       cmap=ssplt.Gamma_II(),
                                       vmin=0, vmax=90, origin='lower')
                fig.colorbar(c2, ax=axes[1, 0], shrink=0.9)
                # Plot the interpolated collimator factor
                c3 = axes[1, 1].imshow(self.grid_interp['collimator_factor'],
                                       cmap=ssplt.Gamma_II(),
                                       vmin=0, vmax=50, origin='lower')
                fig.colorbar(c3, ax=axes[1, 1], shrink=0.9)

    def get_energy(self, B0: float, Z: int = 1, A: float = 2.0):
        """
        Get the energy associated with each gyroradius.

        Jose Rueda: jrrueda@us.es

        @param B0: Magnetic field [in T]
        @param Z: the charge [in e units]
        @param A: the mass in amu
        """
        if self.diag == 'FILD':
            self.energy = ssFILDSIM.get_energy(self.gyroradius, B0, A=A, Z=Z)
        return

    def load_strike_points(self, file=None, verbose: bool = True):
        """
        Load the strike points used to calculate the map.

        Jose Rueda: ruejo@ipp.mpg.de

        @param file: File to be loaded. It should contain the strike points in
        FILDSIM format (if we are loading FILD). If none, name will be deduced
        from the self.file variable, so the strike points are supposed to be in
        the same folder than the strike map
        @param verbose: Flag to plot some information about the strike points
        @param newFILDSIM: Flag to decide if we are using the new FILDSIM or
        the old one
        """
        # Get the object we need to fill and the file to be load
        if self.diag == 'FILD':
            if file is None:
                if self.code.lower() == 'sinpa':
                    (filename, extension) = self.file.rsplit('.', 1)
                    file = filename + '.spmap'
                else:
                    file = self.file[:-14] + 'strike_points.dat'
        elif self.diag == 'INPA':
            if file is None:
                (filename, extension) = self.file.rsplit('.', 1)
                file = filename + '.spmap'
        # Load the strike points
        self.fileStrikePoints = file
        self.strike_points =\
            Strikes(file=file, verbose=verbose, code=self.code)
        # If the code was SINPA, perform the remap, as it is not done in
        # fortran:
        if self.code.lower() == 'sinpa':
            self.remap_strike_points()

    @deprecated('Please use smap.strike_points.scatter() instead')
    def plot_strike_points(self, ax=None, plt_param={}):
        """
        Scatter plot of the strik points. DEPRECATED!.

        Note, no weighting is done, just a scatter plot, this is not a
        sofisticated ready to print figure maker but just a quick plot to see
        what is going on.

        Note 2: Since version 0.6.0, there smap.strike_points is no longer a
        dictionary, but a complete object with its advanced plot routines.
        This mathod is left here for retrocompatibility. If you are planning a
        new implementation, use smap.strike_points.plot2D directly (or the hist
        plot also present there)

        @param ax: axes where to plot, if not given, new figure will pop up
        @param plt_param: options for the matplotlib scatter function
        """
        # Open the figure if needed:
        if ax is None:
            fig, ax = plt.subplots()

        self.strike_points.scatter(ax=ax)

    def calculate_resolutions(self, diag_params: dict = {},
                              min_statistics: int = 100,
                              adaptative: bool = True,
                              calculate_uncertainties: bool = False,
                              confidence_level: float = 0.9544997):
        """
        Calculate the resolution associated with each point of the map.

        Jose Rueda Rueda: jrrueda@us.es

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch. It contains:
            FOR FILD:
                -dpitch: pitch space used by default in the fit. 1.0 default
                -dgyr: giroradius space used by default in the fit. 0.1 default
                -p_method: Function to use in the pitch fit, default Gauss
                -g_method: Function to use in the gyroradius fit,default sGauss
        @param min_statistics: Minimum number of points for a given r,p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_statistics: Minimum number of counts to perform the fit
        @param adaptative: If true, the bin width will be adapted such that the
        number of bins in a sigma of the distribution is 4. If this is the
        case, dpitch, dgyr, will no longer have an impact
        @param confidence_level: confidence level for the uncertainty
            determination
        @param uncertainties: flag to calcualte the uncertainties of the fit
        """
        if self.strike_points is None:
            print('Trying to load the strike points')
            try:
                self.load_strike_points()
            except:
                print('Loading of the strike points failled.')
                raise Exception('You should load the strike points first!!')
        if self.diag == 'FILD':
            # --- Prepare options:
            diag_options = {
                'dpitch': 1.0,
                'dgyr': 0.1,
                'p_method': 'Gauss',
                'g_method': 'sGauss'
            }
            diag_options.update(diag_params)
            dpitch = diag_options['dpitch']
            dgyr = diag_options['dgyr']
            p_method = diag_options['p_method']
            g_method = diag_options['g_method']
            try:
                npitch = self.strike_points.header['npitch']
            except KeyError:
                npitch = self.strike_points.header['nXI']
            nr = self.strike_points.header['ngyr']
            # --- See which columns we need to consider
            if 'remap_rl' not in self.strike_points.header['info'].keys():
                raise Exception('Non remap data in the strike points options')
            else:
                iir = self.strike_points.header['info']['remap_rl']['i']
                iip = self.strike_points.header['info']['remap_pitch']['i']
            # --- Pre-allocate variables
            npoints = np.zeros((nr, npitch))  # Numer of strike points
            parameters_pitch = {
                'amplitude': np.ones((nr, npitch)),
                'center': np.zeros((nr, npitch)),
                'sigma': np.zeros((nr, npitch)),
                'gamma': np.zeros((nr, npitch))
            }   # Parameters of the instrument function
            parameters_gyr = {
                'amplitude': np.zeros((nr, npitch)),
                'center': np.zeros((nr, npitch)),
                'sigma': np.zeros((nr, npitch)),
                'gamma': np.zeros((nr, npitch))
            }   # Parameters of the instrument function
            unc_parameters_pitch = {
                'amplitude': np.zeros((nr, npitch)),
                'center': np.zeros((nr, npitch)),
                'sigma': np.zeros((nr, npitch)),
                'gamma': np.zeros((nr, npitch))
            }   # Parameters of the instrument function
            unc_parameters_gyr = {
                'amplitude': np.zeros((nr, npitch)),
                'center': np.zeros((nr, npitch)),
                'sigma': np.zeros((nr, npitch)),
                'gamma': np.zeros((nr, npitch))
            }   # Parameters of the instrument function
            # To store the fits
            fitg = np.empty((nr, npitch), dtype=np.ndarray)
            fitp = np.empty((nr, npitch), dtype=np.ndarray)
            normalization_g = np.zeros((nr, npitch))
            normalization_p = np.zeros((nr, npitch))

            print('Calculating resolutions ...')
            for ir in tqdm(range(nr)):
                for ip in range(npitch):
                    # --- Select the data
                    data = self.strike_points.data[ip, ir]

                    # --- See if there is enough points:
                    if self.strike_points.header['counters'][ip, ir] < min_statistics:
                        # Set paramters to NaN if there is no enough statistis
                        parameters_gyr['amplitude'][ir, ip] = np.nan
                        parameters_gyr['center'][ir, ip] = np.nan
                        parameters_gyr['sigma'][ir, ip] = np.nan
                        parameters_gyr['gamma'][ir, ip] = np.nan

                        parameters_pitch['amplitude'][ir, ip] = np.nan
                        parameters_pitch['center'][ir, ip] = np.nan
                        parameters_pitch['sigma'][ir, ip] = np.nan
                        parameters_pitch['gamma'][ir, ip] = np.nan

                        # Set the fits to None
                        fitp[ir, ip] = None
                        fitg[ir, ip] = None
                    else:  # If we have enough points, make the fit
                        # Prepare the bin edges according to the desired width
                        edges_pitch = \
                            np.arange(start=data[:, iip].min() - dpitch,
                                      stop=data[:, iip].max() + dpitch,
                                      step=dpitch)
                        edges_gyr = \
                            np.arange(start=data[:, iir].min() - dgyr,
                                      stop=data[:, iir].max() + dgyr,
                                      step=dgyr)
                        # --- Reduce (if needed) the bin width, we will set the
                        # bin width as 1/4 of the std, to ensure a good fitting
                        if adaptative:
                            n_bins_in_sigma = 4
                            sigma_r = np.std(data[:, iir])
                            new_dgyr = sigma_r / n_bins_in_sigma
                            edges_gyr = \
                                np.arange(start=data[:, iir].min() - new_dgyr,
                                          stop=data[:, iir].max() + new_dgyr,
                                          step=new_dgyr)
                            sigma_p = np.std(data[:, iip])
                            new_dpitch = sigma_p / n_bins_in_sigma
                            edges_pitch = \
                                np.arange(start=data[:, iip].min() - dpitch,
                                          stop=data[:, iip].max() + dpitch,
                                          step=new_dpitch)
                        # --- Proceed to fit
                        par_p, fitp[ir, ip], normalization_p[ir, ip], unc_p = \
                            _fit_to_model_(
                                data[:, iip], bins=edges_pitch, model=p_method,
                                confidence_level=confidence_level,
                                uncertainties=calculate_uncertainties)
                        par_g, fitg[ir, ip], normalization_g[ir, ip], unc_g = \
                            _fit_to_model_(
                                data[:, iir], bins=edges_gyr, model=g_method,
                                confidence_level=confidence_level,
                                uncertainties=calculate_uncertainties)
                        # --- Save the data in the matrices:
                        # pitch parameters:
                        parameters_pitch['amplitude'][ir, ip] = \
                            par_p['amplitude']
                        parameters_pitch['center'][ir, ip] = par_p['center']
                        parameters_pitch['sigma'][ir, ip] = par_p['sigma']
                        if p_method == 'Gauss':
                            parameters_pitch['gamma'][ir, ip] = np.nan
                        elif p_method == 'sGauss':
                            parameters_pitch['gamma'][ir, ip] = par_p['gamma']
                        # pitch uncertainty parameters:
                        unc_parameters_pitch['amplitude'][ir, ip] = \
                            unc_p['amplitude']
                        unc_parameters_pitch['center'][ir, ip] = \
                            unc_p['center']
                        unc_parameters_pitch['sigma'][ir, ip] = unc_p['sigma']
                        if p_method == 'Gauss':
                            unc_parameters_pitch['gamma'][ir, ip] = np.nan
                        elif p_method == 'sGauss':
                            unc_parameters_pitch['gamma'][ir, ip] = \
                                unc_p['gamma']
                        # gyroradius parameters:
                        parameters_gyr['amplitude'][ir, ip] = \
                            par_g['amplitude']
                        parameters_gyr['center'][ir, ip] = par_g['center']
                        parameters_gyr['sigma'][ir, ip] = par_g['sigma']
                        if g_method == 'Gauss':
                            parameters_gyr['gamma'][ir, ip] = np.nan
                        elif g_method == 'sGauss':
                            parameters_gyr['gamma'][ir, ip] = par_g['gamma']
                        # gyroradius uncertainty parameters:
                        unc_parameters_gyr['amplitude'][ir, ip] = \
                            unc_g['amplitude']
                        unc_parameters_gyr['center'][ir, ip] = unc_g['center']
                        unc_parameters_gyr['sigma'][ir, ip] = unc_g['sigma']
                        if g_method == 'Gauss':
                            unc_parameters_gyr['gamma'][ir, ip] = np.nan
                        elif g_method == 'sGauss':
                            unc_parameters_gyr['gamma'][ir, ip] = \
                                unc_g['gamma']

            self.resolution = {
                'Gyroradius': parameters_gyr,
                'Pitch': parameters_pitch,
                'nmarkers': npoints,
                'fits': {
                    'gyroradius': fitg,
                    'pitch': fitp,
                    'normalization_gyroradius': normalization_g,
                    'normalization_pitch': normalization_p
                },
                'gyroradius_model': g_method,
                'pitch_model': p_method,
                'Gyroradius_uncertainty': unc_parameters_gyr,
                'Pitch_uncertainty': unc_parameters_pitch
                }
            # --- Prepare the interpolators:
            self.calculate_interpolators()
        elif self.diag == 'INPA':
            # --- Prepare options:
            diag_options = {
                'dxi': 0.05,
                'dgyr': 0.1,
                'xi_method': 'Gauss',
                'g_method': 'Gauss'
            }
            diag_options.update(diag_params)
            dxi = diag_options['dxi']
            dgyr = diag_options['dgyr']
            xi_method = diag_options['xi_method']
            g_method = diag_options['g_method']
            nxi = self.strike_points.header['nXI']
            nr = self.strike_points.header['ngyr']
            # --- See which columns we need to consider
            if 'remap_rl' not in self.strike_points.header['info'].keys():
                raise Exception('Non remap data in the strike points options')
            else:
                iir = self.strike_points.header['info']['remap_rl']['i']
                iip = self.strike_points.header['info']['remap_XI']['i']
            # --- Pre-allocate variables
            npoints = np.zeros((nr, nxi))  # Numer of strike points
            parameters_xi = {
                'amplitude': np.ones((nr, nxi)),
                'center': np.zeros((nr, nxi)),
                'sigma': np.zeros((nr, nxi)),
                'gamma': np.zeros((nr, nxi))
            }   # Parameters of the instrument function
            parameters_gyr = {
                'amplitude': np.zeros((nr, nxi)),
                'center': np.zeros((nr, nxi)),
                'sigma': np.zeros((nr, nxi)),
                'gamma': np.zeros((nr, nxi))
            }   # Parameters of the instrument function
            unc_parameters_xi = {
                'amplitude': np.zeros((nr, nxi)),
                'center': np.zeros((nr, nxi)),
                'sigma': np.zeros((nr, nxi)),
                'gamma': np.zeros((nr, nxi))
            }   # Parameters of the instrument function
            unc_parameters_gyr = {
                'amplitude': np.zeros((nr, nxi)),
                'center': np.zeros((nr, nxi)),
                'sigma': np.zeros((nr, nxi)),
                'gamma': np.zeros((nr, nxi))
            }   # Parameters of the instrument function
            # To store the fits
            fitg = np.empty((nr, nxi), dtype=np.ndarray)
            fitxi = np.empty((nr, nxi), dtype=np.ndarray)
            normalization_g = np.zeros((nr, nxi))
            normalization_xi = np.zeros((nr, nxi))

            print('Calculating resolutions ...')
            for ir in tqdm(range(nr)):
                for ip in range(nxi):
                    # --- Select the data
                    data = self.strike_points.data[ip, ir]

                    # --- See if there is enough points:
                    if self.strike_points.header['counters'][ip, ir] < min_statistics:
                        # Set paramters to NaN if there is no enough statistis
                        parameters_gyr['amplitude'][ir, ip] = np.nan
                        parameters_gyr['center'][ir, ip] = np.nan
                        parameters_gyr['sigma'][ir, ip] = np.nan
                        parameters_gyr['gamma'][ir, ip] = np.nan

                        parameters_xi['amplitude'][ir, ip] = np.nan
                        parameters_xi['center'][ir, ip] = np.nan
                        parameters_xi['sigma'][ir, ip] = np.nan
                        parameters_xi['gamma'][ir, ip] = np.nan

                        # Set the fits to None
                        fitxi[ir, ip] = None
                        fitg[ir, ip] = None
                    else:  # If we have enough points, make the fit
                        # Prepare the bin edges according to the desired width
                        edges_xi = \
                            np.arange(start=data[:, iip].min() - dxi,
                                      stop=data[:, iip].max() + dxi,
                                      step=dxi)
                        edges_gyr = \
                            np.arange(start=data[:, iir].min() - dgyr,
                                      stop=data[:, iir].max() + dgyr,
                                      step=dgyr)
                        # --- Reduce (if needed) the bin width, we will set the
                        # bin width as 1/4 of the std, to ensure a good fitting
                        if adaptative:
                            n_bins_in_sigma = 4
                            sigma_r = np.std(data[:, iir])
                            new_dgyr = sigma_r / n_bins_in_sigma
                            edges_gyr = \
                                np.arange(start=data[:, iir].min() - new_dgyr,
                                          stop=data[:, iir].max() + new_dgyr,
                                          step=new_dgyr)
                            sigma_p = np.std(data[:, iip])
                            new_dpitch = sigma_p / n_bins_in_sigma
                            edges_xi = \
                                np.arange(start=data[:, iip].min() - dxi,
                                          stop=data[:, iip].max() + dxi,
                                          step=new_dpitch)
                        # --- Proceed to fit
                        # print(ir, ip)
                        # plt.hist(data[:, iip], bins=edges_pitch)
                        par_xi, fitxi[ir, ip], normalization_xi[ir, ip], unc_xi = \
                            _fit_to_model_(
                                data[:, iip], bins=edges_xi, model=xi_method,
                                confidence_level=confidence_level,
                                uncertainties=calculate_uncertainties)
                        par_g, fitg[ir, ip], normalization_g[ir, ip], unc_g = \
                            _fit_to_model_(
                                data[:, iir], bins=edges_gyr, model=g_method,
                                confidence_level=confidence_level,
                                uncertainties=calculate_uncertainties)
                        # --- Save the data in the matrices:
                        # pitch parameters:
                        parameters_xi['amplitude'][ir, ip] = \
                            par_xi['amplitude']
                        parameters_xi['center'][ir, ip] = par_xi['center']
                        parameters_xi['sigma'][ir, ip] = par_xi['sigma']
                        if xi_method == 'Gauss':
                            parameters_xi['gamma'][ir, ip] = np.nan
                        elif xi_method == 'sGauss':
                            parameters_xi['gamma'][ir, ip] = par_xi['gamma']
                        # pitch uncertainty parameters:
                        unc_parameters_xi['amplitude'][ir, ip] = \
                            unc_xi['amplitude']
                        unc_parameters_xi['center'][ir, ip] = \
                            unc_xi['center']
                        unc_parameters_xi['sigma'][ir, ip] = unc_xi['sigma']
                        if xi_method == 'Gauss':
                            unc_parameters_xi['gamma'][ir, ip] = np.nan
                        elif xi_method == 'sGauss':
                            unc_parameters_xi['gamma'][ir, ip] = \
                                unc_xi['gamma']
                        # gyroradius parameters:
                        parameters_gyr['amplitude'][ir, ip] = \
                            par_g['amplitude']
                        parameters_gyr['center'][ir, ip] = par_g['center']
                        parameters_gyr['sigma'][ir, ip] = par_g['sigma']
                        if g_method == 'Gauss':
                            parameters_gyr['gamma'][ir, ip] = np.nan
                        elif g_method == 'sGauss':
                            parameters_gyr['gamma'][ir, ip] = par_g['gamma']
                        # gyroradius uncertainty parameters:
                        unc_parameters_gyr['amplitude'][ir, ip] = \
                            unc_g['amplitude']
                        unc_parameters_gyr['center'][ir, ip] = unc_g['center']
                        unc_parameters_gyr['sigma'][ir, ip] = unc_g['sigma']
                        if g_method == 'Gauss':
                            unc_parameters_gyr['gamma'][ir, ip] = np.nan
                        elif g_method == 'sGauss':
                            unc_parameters_gyr['gamma'][ir, ip] = \
                                unc_g['gamma']

            self.resolution = {
                'Gyroradius': parameters_gyr,
                'Xi': parameters_xi,
                'nmarkers': npoints,
                'fits': {
                    'gyroradius': fitg,
                    'xi': fitxi,
                    'normalization_gyroradius': normalization_g,
                    'normalization_pitch': normalization_xi
                },
                'gyroradius_model': g_method,
                'xi_model': xi_method,
                'Gyroradius_uncertainty': unc_parameters_gyr,
                'Xi_uncertainty': unc_parameters_xi
                }
            # --- Prepare the interpolators:
            self.calculate_interpolators()
        else:
            raise errors.NotImplementedError(
                'Diagnostic still not implemented')

    def calculate_interpolators(self):
        """
        Calc interpolators from phase space to instrument function params

        Jose Rueda: jrrueda@us.es
        """
        if self.diag == 'FILD':
            # --- Prepare the interpolators:
            # Prepare grid
            try:
                xx, yy = np.meshgrid(self.strike_points.header['gyroradius'],
                                     self.strike_points.header['pitch'])
            except KeyError:
                xx, yy = np.meshgrid(self.strike_points.header['gyroradius'],
                                     self.strike_points.header['XI'])
            xxx = xx.flatten()
            yyy = yy.flatten()
            self.interpolators = {'pitch': {}, 'gyroradius': {}}
            for i in self.resolution['Gyroradius'].keys():
                dummy = self.resolution['Gyroradius'][i].T
                dummy = dummy.flatten()
                flags = np.isnan(dummy)
                x1 = xxx[~flags]
                y1 = yyy[~flags]
                z1 = dummy[~flags]
                if np.sum(~flags) > 4:
                    self.interpolators['gyroradius'][i] = \
                        scipy_interp.LinearNDInterpolator(
                            np.vstack((x1, y1)).T,
                            z1)
            for i in self.resolution['Pitch'].keys():
                dummy = self.resolution['Pitch'][i].T
                dummy = dummy.flatten()
                flags = np.isnan(dummy)
                x1 = xxx[~flags]
                y1 = yyy[~flags]
                z1 = dummy[~flags]
                if np.sum(~flags) > 4:
                    self.interpolators['pitch'][i] = \
                        scipy_interp.LinearNDInterpolator(
                            np.vstack((x1, y1)).T,
                            z1)
            # Collimator factor
            dummy = self.collimator_factor_matrix.T
            dummy = dummy.flatten()
            flags = np.isnan(dummy)
            x1 = xxx[~flags]
            y1 = yyy[~flags]
            z1 = dummy[~flags]
            if np.sum(~flags) > 4:
                self.interpolators['collimator_factor'] = \
                    scipy_interp.LinearNDInterpolator(np.vstack((x1, y1)).T,
                                                      z1)
            # positions:
            YMATRIX = np.zeros((self.npitch, self.ngyr))
            ZMATRIX = np.zeros((self.npitch, self.ngyr))
            for ir in range(self.ngyr):
                for ip in range(self.npitch):
                    flags = (self.gyroradius == self.unique_gyroradius[ir]) \
                        * (self.pitch == self.unique_pitch[ip])
                    if np.sum(flags) > 0:
                        # By definition, flags can only have one True
                        # yes, x is smap.y... FILDSIM notation
                        YMATRIX[ip, ir] = self.y[flags]
                        ZMATRIX[ip, ir] = self.z[flags]
            self.interpolators['x'] = \
                scipy_interp.LinearNDInterpolator(np.vstack((xxx.flatten(),
                                                             yyy.flatten())).T,
                                                  YMATRIX.flatten())
            self.interpolators['y'] = \
                scipy_interp.LinearNDInterpolator(np.vstack((xxx.flatten(),
                                                             yyy.flatten())).T,
                                                  ZMATRIX.flatten())
        return

    def calculate_mapping_interpolators(self,
                                        kernel: str = 'thin_plate_spline',
                                        degree=2):
        """
        Calculate interpolators scintillator position -> phase space.

        Jose Rueda: jrrueda@us.es

        @param kernel: kernel for the interpolator
        @param degree: degree for the added polynomial

        See RBFInterpolator of Scipy for full documentation
        """
        # --- Select the colums to be used
        # temporal solution to save the coordinates in the array
        coords = np.zeros((self.y.size, 2))
        coords[:, 0] = self.y
        coords[:, 1] = self.z
        self.map_interpolators = {
            'Gyroradius':
                scipy_interp.RBFInterpolator(coords,
                                             self.gyroradius,
                                             kernel=kernel, degree=degree),
            'XI':
                scipy_interp.RBFInterpolator(coords,
                                             self.XI, kernel=kernel,
                                             degree=degree),
        }
        if self.diag == 'FILD':
            # Keep the old name just as a backwards compatibility
            self.map_interpolators['Pitch'] = self.map_interpolators['XI']

    def remap_strike_points(self):
        """
        Remap the StrikePoints

        Jose Rueda: jrrueda@us.es
        """
        # --- See if the interpolators are defined
        if self.map_interpolators is None:
            print('Interpolators not calcualted. Calculating them')
            self.calculate_mapping_interpolators()
        # --- See if the remap already exist in the strikes:
        if 'remap_rl' in self.strike_points.header['info'].keys():
            print('The remapped values are already in the strikes object')
            print('Nothing to do here')
            return
        iix = self.strike_points.header['info']['ys']['i']
        iiy = self.strike_points.header['info']['zs']['i']
        for ir in range(self.ngyr):
            for ip in range(self.nXI):
                if self.strike_points.header['counters'][ip, ir] > 0:
                    n_strikes = self.strike_points.header['counters'][ip, ir]
                    remap_data = np.zeros((n_strikes, 2))
                    remap_data[:, 0] = \
                        self.map_interpolators['Gyroradius'](
                            self.strike_points.data[ip, ir][:, [iix, iiy]])
                    # self.strike_points.data[ip, ir][:, iiy])
                    remap_data[:, 1] = \
                        self.map_interpolators['XI'](
                            self.strike_points.data[ip, ir][:, [iix, iiy]])
                    # self.strike_points.data[ip, ir][:, iiy])
                    # append the remapped data to the object
                    self.strike_points.data[ip, ir] = \
                        np.append(self.strike_points.data[ip, ir],
                                  remap_data, axis=1)
        # Update the headers.
        Old_number_colums = len(self.strike_points.header['info'])
        extra_column = {
            'remap_rl': {
                'i': Old_number_colums,  # Column index in the file
                'units': ' [cm]',  # Units
                'longName': 'Remapped Larmor radius',
                'shortName': '$r_l$',
            },
        }
        if self.diag == 'FILD':
            extra_column['remap_XI'] = {
                'i': Old_number_colums + 1,  # Column index in the file
                'units': ' [$\\degree$]',  # Units
                'longName': 'Remapped pitch angle',
                'shortName': '$\\lambda$',
            }
            # FILDSIM backwards compatibility:
            extra_column['remap_pitch'] = extra_column['remap_XI']
        else:  # INPA case, this is radius
            extra_column['remap_XI'] = {
                'i': Old_number_colums + 1,  # Column index in the file
                'units': ' [m]',  # Units
                'longName': 'Remapped R',
                'shortName': '$R$',
            }
        # Update the header
        self.strike_points.header['info'].update(extra_column)

    def plot_resolutions(self, ax_params: dict = {}, cMap=None, nlev: int = 20,
                         index_gyr=None):
        """
        Plot the resolutions.

        Jose Rueda: jrrueda@us.es

        @todo: Implement label size in colorbar

        @param ax_param: parameters for the axis beauty function. Note, labels
        of the color axis are hard-cored, if you want custom axis labels you
        would need to draw the plot on your own
        @param cMap: is None, Gamma_II will be used
        @param nlev: number of levels for the contour
        @param index_gyr: if present, reslution would be plotted along
        gyroradius given by gyroradius[index_gyr]
        """
        # --- Initialise the settings:
        if cMap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = cMap

        # --- Open the figure and prepare the map:
        fig, ax = plt.subplots(1, 2, figsize=(12, 10),
                               facecolor='w', edgecolor='k')

        if self.diag == 'FILD':
            ax_options = {
                'xlabel': '$\\lambda [\\degree]$',
                'ylabel': '$r_l [cm]$'
            }
            ax_options.update(ax_params)
            if index_gyr is None:
                # Plot the gyroradius resolution
                a1 = ax[0].contourf(self.strike_points.header['XI'],
                                    self.strike_points.header['gyroradius'],
                                    self.resolution['Gyroradius']['sigma'],
                                    levels=nlev, cmap=cmap)
                fig.colorbar(a1, ax=ax[0], label='$\\sigma_r [cm]$')
                ax[0] = ssplt.axis_beauty(ax[0], ax_options)
                # plot the pitch resolution
                a = ax[1].contourf(self.strike_points.header['XI'],
                                   self.strike_points.header['gyroradius'],
                                   self.resolution['Pitch']['sigma'],
                                   levels=nlev, cmap=cmap)
                fig.colorbar(a, ax=ax[1], label='$\\sigma_\\lambda$')
                ax[1] = ssplt.axis_beauty(ax[1], ax_options)
                plt.tight_layout()
            else:
                ax_options = {
                    'xlabel': '$\\lambda [\\degree]$',
                    'ylabel': '$\\sigma_l [cm]$'
                }
                ax[0].plot(self.strike_points.header['XI'],
                           self.resolution['Gyroradius']['sigma'][index_gyr, :])
                ax[0] = ssplt.axis_beauty(ax[0], ax_options)

                ax[1].plot(self.strike_points.header['XI'],
                           self.resolution['Pitch']['sigma'][index_gyr, :])
                ax[1] = ssplt.axis_beauty(ax[1], ax_options)

        if self.diag == 'INPA':
            ax_options = {
                'xlabel': '$R [m]$',
                'ylabel': '$r_l [cm]$'
            }
            ax_options.update(ax_params)
            if index_gyr is None:
                # Plot the gyroradius resolution
                a1 = ax[0].contourf(self.unique_R0,
                                    self.strike_points.header['gyroradius'],
                                    self.resolution['Gyroradius']['sigma'],
                                    levels=nlev, cmap=cmap)
                fig.colorbar(a1, ax=ax[0], label='$\\sigma_r [cm]$')
                ax[0] = ssplt.axis_beauty(ax[0], ax_options)
                # plot the R resolution
                a = ax[1].contourf(self.unique_R0,
                                   self.strike_points.header['gyroradius'],
                                   self.resolution['Xi']['sigma'],
                                   levels=nlev, cmap=cmap)
                fig.colorbar(a, ax=ax[1], label='$\\sigma_\\lambda$')
                ax[1] = ssplt.axis_beauty(ax[1], ax_options)
                plt.tight_layout()
            else:
                ax_options = {
                    'xlabel': '$R [m]$',
                    'ylabel': '$\\sigma_l [cm]$'
                }
                ax[0].plot(self.unique_R0,
                           self.resolution['Gyroradius']['sigma'][index_gyr, :])
                ax[0] = ssplt.axis_beauty(ax[0], ax_options)

                ax[1].plot(self.unique_R0,
                           self.resolution['Pitch']['sigma'][index_gyr, :])
                ax[1] = ssplt.axis_beauty(ax[1], ax_options)

        fig.show()
        return

    def plot_collimator_factor(self, ax_param: dict = {}, cMap=None,
                               nlev: int = 20):
        """
        Plot the collimator factor.

        Jose Rueda: jrrueda@us.es

        @todo: Implement label size in colorbar

        @param ax_param: parameters for the axis beauty function. Note, labels
        of the color axis are hard-cored, if you want custom axis labels you
        would need to draw the plot on your own
        @param cMap: is None, Gamma_II will be used
        @param nlev: number of levels for the contour
        """
        # --- Initialise the settings:
        if cMap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = cMap
        ax_options = {
            'xlabel': '$\\lambda [\\degree]$',
            'ylabel': '$r_l [cm]$'
        }
        ax_options.update(ax_param)

        # --- Open the figure and prepare the map:
        fig, ax = plt.subplots(1, 1, figsize=(6, 10),
                               facecolor='w', edgecolor='k')

        if self.diag == 'FILD':
            # Plot the gyroradius resolution
            a1 = ax.contourf(self.unique_pitch,
                             self.unique_gyroradius,
                             self.collimator_factor_matrix,
                             levels=nlev, cmap=cmap)
            fig.colorbar(a1, ax=ax, label='Collimating factor')
            ax = ssplt.axis_beauty(ax, ax_options)

            plt.tight_layout()
        return

    def plot_resolution_fits(self, var: str = 'Gyroradius',
                             ax_params: dict = {},
                             ax=None, gyr_index=None, pitch_index=None,
                             gyroradius=None, pitch=None,
                             kind_of_plot: str = 'normal',
                             include_legend: bool = False,
                             XI_index=None,
                             normalize: bool = False):
        """
        Plot the fits done to calculate the resolution

        @param var: variable to plot, Gyroradius or Pitch for FILD. Capital
        letters will be ignored
        @param ax_param: dictoniary with the axis parameters axis_beauty()
        @param ax: axis where to plot
        @param gyr_index: index, or arrays of indeces, of gyroradius to plot
        @param pitch_index: index, or arrays of indeces, of pitches to plot,
            this is outdated code, please use XI_index instead
        @param gyroradius: gyroradius value of array of then to plot. If
        present, gyr_index will be ignored
        @param pitch: idem to gyroradius bu for the pitch
        @param kind_of_plot: kind of plot to make:
            - normal: scatter plot of the data and fit like a line
            - bar: bar plot of the data and file like a line
            - uncertainty: scatter plot of the data and shading area for the
                fit (3 sigmas)
            - just_fit: Just a line plot as the fit
        @param include_legend: flag to include a legend
        @param XI_index: equivalent to pitch_index, but with the new criteria
        @param normalize: normalize the output
        """
        # --- Initialise plotting options and axis:
        default_labels = {
            'gyroradius': {
                'xlabel': 'Gyroradius [cm]',
                'ylabel': 'Counts [a.u.]'
            },
            'pitch': {
                'xlabel': 'Pitch [$\\degree$]',
                'ylabel': 'Counts [a.u.]'
            }
        }
        ax_options = {
            'grid': 'both',
        }
        ax_options.update(default_labels[var.lower()])
        ax_options.update(ax_params)
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        if (pitch_index is None) and (XI_index is not None):
            pitch_index = XI_index
        # --- Localise the values to plot
        if gyroradius is not None:
            # test if it is a number or an array of them
            if isinstance(gyroradius, (list, np.ndarray)):
                gyroradius = gyroradius
            else:
                gyroradius = np.array([gyroradius])
            index_gyr = np.zeros(gyroradius.size, dtype=int)
            for i in range(index_gyr.size):
                index_gyr[i] = \
                    np.argmin(np.abs(self.unique_gyroradius - gyroradius[i]))
            print('Found gyroradius: ', self.unique_gyroradius[index_gyr])
        else:
            # test if it is a number or an array of them
            if gyr_index is not None:
                if isinstance(gyr_index, (list, np.ndarray)):
                    index_gyr = gyr_index
                else:
                    index_gyr = np.array([gyr_index])
            else:
                index_gyr = np.arange(self.ngyr, dtype=np.int)

        if pitch is not None:
            # test if it is a number or an array of them
            if isinstance(pitch, (list, np.ndarray)):
                pitch = pitch
            else:
                pitch = np.array([pitch])
            index_pitch = np.zeros(pitch.size, dtype=int)
            for i in range(index_pitch.size):
                index_pitch[i] = \
                    np.argmin(np.abs(self.unique_pitch - pitch[i]))
            print('Found pitches: ', self.unique_pitch[index_pitch])
        else:
            # test if it is a number or an array of them
            if pitch_index is not None:
                if isinstance(pitch_index, (list, np.ndarray)):
                    index_pitch = pitch_index
                else:
                    index_pitch = np.array([pitch_index])
            else:
                index_pitch = np.arange(self.npitch, dtype=np.int)
        # --- Get the maximum value for the normalization

        # --- Plot the desired data
        # This is just to allow the user to ask the variable with capitals
        # letters or not

        for ir in index_gyr:
            for ip in index_pitch:
                # The lmfit model has included a plot function, but is slightly
                # not optimal so we will plot it 'manually'
                if self.resolution['fits'][var.lower()][ir, ip] is not None:
                    x = self.resolution['fits'][var.lower()
                                                ][ir, ip].userkws['x']
                    deltax = x.max() - x.min()
                    x_fine = np.linspace(x.min() - 0.1 * deltax,
                                         x.max() + 0.1 * deltax)
                    name = 'rl: ' + str(round(self.unique_gyroradius[ir], 1))\
                        + ' $\\lambda$: ' + \
                        str(round(self.unique_pitch[ip], 1))
                    normalization = \
                        self.resolution['fits']['normalization_'
                                                + var.lower()][ir, ip]
                    y = self.resolution['fits'][var.lower()][ir, ip].eval(
                        x=x_fine) * normalization
                    if kind_of_plot.lower() == 'normal':
                        # plot the data as scatter plot
                        scatter = ax.scatter(
                            x,
                            normalization * self.resolution['fits'][var.lower(
                                             )][ir, ip].data,
                            label='__noname__')
                        # plot the fit as a line
                        ax.plot(x_fine, y,
                                color=scatter.get_facecolor()[0, :3],
                                label=name)
                    elif kind_of_plot.lower() == 'bar':
                        bar = ax.bar(
                            x,
                            normalization * self.resolution['fits'][var.lower(
                                 )][ir, ip].data,
                            label='__noname__', width=x[1]-x[0],
                            alpha=0.25)
                        ax.plot(x_fine, y,
                                color=bar.patches[0].get_facecolor()[:3],
                                label=name)
                    elif kind_of_plot.lower() == 'just_fit':
                        ax.plot(x_fine, y, label=name)
                    elif kind_of_plot.lower() == 'uncertainty':
                        scatter = ax.scatter(
                            x,
                            normalization * self.resolution['fits'][var.lower(
                                 )][ir, ip].data,
                            label='__noname__')
                        dely = normalization \
                            * self.resolution['fits'][var.lower(
                            )][ir, ip].eval_uncertainty(sigma=3, x=x_fine)
                        ax.fill_between(x_fine, y-dely, y+dely, alpha=0.25,
                                        label='3-$\\sigma$ uncertainty band',
                                        color=scatter.get_facecolor()[0, :3])
                    else:
                        raise errors.NotValidInput(
                            'Not kind of plot not understood')
                else:
                    print('Not fits for rl: '
                          + str(round(self.unique_gyroradius[ir], 1))
                          + 'pitch: '
                          + str(round(self.unique_pitch[ip], 1)))
        if include_legend:
            ax.legend()
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)

    @deprecated('Please use smap.plot_resolution_fits() instead')
    def plot_pitch_histograms(self, diag_params: dict = {},
                              adaptative: bool = True,
                              min_statistics=100,
                              gyroradius=3,
                              plot_fit=True,
                              axarr=None, dpi=100, alpha=0.5):
        """
        Calculate the resolution associated with each point of the map.

        Ajvv

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch
        @param min_statistics: Minimum number of points for a given r p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_statistics: Minimum number of counts to perform the fit
        @param adaptative: If true, the bin width will be adapted such that the
        number of bins in a sigma of the distribution is 4. If this is the
        case, dpitch, dgyr, will no longer have an impact
        """
        print('This function is provisional')
        print('It will be changed in future versions')
        if self.strike_points is None:
            raise Exception('You should load the strike points first!!')
        if self.diag == 'FILD':
            # --- Prepare options:
            diag_options = {
                'dpitch': 1.0,
                'dgyr': 0.1,
                'p_method': 'Gauss',
                'g_method': 'sGauss'
            }
            diag_options.update(diag_params)
            dpitch = diag_options['dpitch']
            p_method = diag_options['p_method']

            npitch = self.strike_points.header['pitch'].size
            ir = np.argmin(abs(self.strike_points.header['gyroradius']
                               - gyroradius))

            for ip in range(npitch):
                # --- Select the data
                data = self.strike_points.data[ip, ir]

                if len(data[:, 0]) < min_statistics:
                    continue
                # Prepare the bin edges according to the desired width
                edges_pitch = \
                    np.arange(start=data[:, 5].min() - dpitch,
                              stop=data[:, 5].max() + dpitch,
                              step=dpitch)

                # --- Reduce (if needed) the bin width, we will set the
                # bin width as 1/4 of the std, to ensure a good fitting
                if adaptative:
                    n_bins_in_sigma = 4
                    sigma_p = np.std(data[:, 5])
                    new_dpitch = sigma_p / n_bins_in_sigma
                    edges_pitch = \
                        np.arange(start=data[:, 5].min() - dpitch,
                                  stop=data[:, 5].max() + dpitch,
                                  step=new_dpitch)
                # --- Proceed to fit
                par_p, resultp = _fit_to_model_(data[:, 5],
                                                bins=edges_pitch,
                                                model=p_method,
                                                normalize=False)

                if axarr is None:
                    fig, axarr = plt.subplots(nrows=1, ncols=1,
                                              figsize=(6, 10),
                                              facecolor='w', edgecolor='k',
                                              dpi=dpi)
                    ax_pitch = axarr  # topdown view, should see pinhole surfac
                    ax_pitch.set_xlabel('Pitch [$\\degree$]')
                    ax_pitch.set_ylabel('Counts')
                    ax_pitch.set_title(
                        'Pitch resolution at gyroradius '
                        + str(self.strike_points.header['gyroradius'][ir])
                        + ' cm')

                    created_ax = True

                cent = 0.5 * (edges_pitch[1:] + edges_pitch[:-1])
                fit_line = ax_pitch.plot(cent, resultp.best_fit,
                                         label='_nolegend_')
                label_plot = \
                    f"{float(self.strike_points.header['pitch'][ip]):g}"\
                    + '$\\degree$'
                ax_pitch.hist(data[:, 7], bins=edges_pitch, alpha=alpha,
                              label=label_plot, color=fit_line[0].get_color())

        ax_pitch.legend(loc='best')

        if created_ax:
            fig.tight_layout()
            fig.show()

        return

    @deprecated('Please use smap.plot_resolution_fits() instead')
    def plot_gyroradius_histograms(self, diag_params: dict = {},
                                   adaptative: bool = True,
                                   min_statistics=100,
                                   pitch=30,
                                   plot_fit=True,
                                   axarr=None, dpi=100, alpha=0.5):
        """
        Calculate the resolution associated with each point of the map

        Ajvv

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch
        @param min_statistics: Minimum number of points for a given r p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_statistics: Minimum number of counts to perform the fit
        @param adaptative: If true, the bin width will be adapted such that the
        number of bins in a sigma of the distribution is 4. If this is the
        case, dpitch, dgyr, will no longer have an impact
        """
        print('This function is provisional')
        print('It will be changed in future versions')
        if self.strike_points is None:
            raise Exception('You should load the strike points first!!')
        if self.diag == 'FILD':
            # --- Prepare options:
            diag_options = {
                'dpitch': 1.0,
                'dgyr': 0.1,
                'p_method': 'Gauss',
                'g_method': 'sGauss'
            }
            diag_options.update(diag_params)
            dgyr = diag_options['dgyr']
            g_method = diag_options['g_method']

            nr = self.strike_points.header['gyroradius'].size

            ip = np.argmin(abs(self.strike_points.header['pitch'] - pitch))

            for ir in range(nr):
                # --- Select the data
                data = self.strike_points.data[ip, ir]

                if len(data[:, 0]) < min_statistics:
                    continue
                # Prepare the bin edges according to the desired width
                edges_gyr = \
                    np.arange(start=data[:, 4].min() - dgyr,
                              stop=data[:, 4].max() + dgyr,
                              step=dgyr)
                # --- Reduce (if needed) the bin width, we will set the
                # bin width as 1/4 of the std, to ensure a good fitting
                if adaptative:
                    n_bins_in_sigma = 4
                    sigma_r = np.std(data[:, 4])
                    new_dgyr = sigma_r / n_bins_in_sigma
                    edges_gyr = \
                        np.arange(start=data[:, 4].min() - new_dgyr,
                                  stop=data[:, 4].max() + new_dgyr,
                                  step=new_dgyr)

                # --- Proceed to fit

                par_g, resultg = _fit_to_model_(data[:, 4],
                                                bins=edges_gyr,
                                                model=g_method,
                                                normalize=False)
                if axarr is None:
                    fig, axarr = \
                        plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                     facecolor='w', edgecolor='k', dpi=dpi)
                    ax_gyroradius = axarr
                    ax_gyroradius.set_xlabel('Gyroradius [cm]')
                    ax_gyroradius.set_ylabel('Counts')
                    title_plot = 'Gyroradius resolution at pitch '\
                        + str(self.strike_points.header['pitch'][ip])\
                        + '$\\degree$'
                    ax_gyroradius.set_title(title_plot)

                    created_ax = True
                else:
                    created_ax = False
                    ax_gyroradius = axarr

                cent = 0.5 * (edges_gyr[1:] + edges_gyr[:-1])
                fit_line = ax_gyroradius.plot(cent, resultg.best_fit,
                                              label='_nolegend_')
                label_plot = \
                    f"{float(self.strike_points.header['gyroradius'][ir]):g}"\
                    + '[cm]'

                ax_gyroradius.hist(data[:, 4], bins=edges_gyr,
                                   alpha=alpha, label=label_plot,
                                   color=fit_line[0].get_color())

        ax_gyroradius.legend(loc='best')

        if created_ax:
            fig.tight_layout()
            fig.show()

        return
        fig.tight_layout()
        fig.show()

        return
        return
        return
