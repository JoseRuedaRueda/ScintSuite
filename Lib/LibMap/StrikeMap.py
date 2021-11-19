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
import Lib.SimulationCodes.SINPA as ssSINPA
import Lib.LibMap.Common as common
from Lib.LibMachine import machine
import Lib.LibPaths as p
from tqdm import tqdm   # For waitbars
pa = p.Path(machine)
del p


class StrikeMap:
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
        """
        ## Associated diagnostic
        if flag == 0 or str(flag).lower() == 'fild':
            self.diag = 'FILD'
        elif flag == 1 or str(flag).lower() == 'inpa':
            self.diag = 'INPA'
        elif flag == 2 or str(flag).lower() == 'ihibp':
            self.diag = 'iHIBP'
            raise Exception('Diagnostic not implemented: Talk with Pablo')
        else:
            print('Flag: ', flag)
            raise Exception('Diagnostic not implemented')
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
                dumm = ssFILDSIM.guess_strike_map_name_FILD(phi,
                                                            theta,
                                                            machine=machine,
                                                            decimals=decimals)
                file = os.path.join(smap_folder, dumm)
                self.file = file
            if not os.path.isfile(file):
                raise Exception('Strike map no found in the database')
            dummy = np.loadtxt(file, skiprows=3)
            # See which rows has collimator factor larger than zero (ie see for
            # which combination of gyroradius and pitch some markers arrived)
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
            self.collimator_factor_matrix = np.zeros((self.ngyr, self.npitch))
            for ir in range(self.ngyr):
                for ip in range(self.npitch):
                    # By definition, flags can only have one True
                    flags = (self.gyroradius == self.unique_gyroradius[ir]) \
                        * (self.pitch == self.unique_pitch[ip])
                    if np.sum(flags) > 0:
                        self.collimator_factor_matrix[ir, ip] = \
                            self.collimator_factor[flags]
        elif self.diag == 'INPA':
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
            ## Colimator facror as a matrix
            # This simplify a lot W calculation and forward modelling:
            self.ngyr = len(self.unique_gyroradius)
            self.nalpha = len(self.unique_alpha)
            self.collimator_factor_matrix = np.zeros((self.ngyr, self.nalpha))
            for ir in range(self.ngyr):
                for ip in range(self.nalpha):
                    # By definition, flags can only have one True
                    flags = (self.gyroradius == self.unique_gyroradius[ir]) \
                        * (self.nalpha == self.unique_alpha[ip])
                    if np.sum(flags) > 0:
                        self.collimator_factor_matrix[ir, ip] = \
                            self.collimator_factor[flags]

    def plot_real(self, ax=None,
                  marker_params: dict = {}, line_params: dict = {},
                  labels: bool = False,
                  rotation_for_gyr_label: float = 90.0,
                  rotation_for_pitch_label: float = 30.0):
        """
        Plot the strike map (x,y = dimensions in the scintillator).

        Jose Rueda: jrrueda@us.es

        @param ax: Axes where to plot
        @param markers_params: parameters for plt.plot() to plot the markers
        @param line_params: parameters for plt.plot() to plot the markers
        @param labels: flag to add the labes (gyroradius, pitch) on the plot
        @param rotation_for_gyr_label: Rotation angle for the gyroradius label
        @param rotation_for_pitch_label: Rotation angle for the pitch label

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
        for i in range(self.ngyr):
            flags = self.gyroradius == self.unique_gyroradius[i]
            ax.plot(self.y[flags], self.z[flags], **line_options)
            # Add the gyroradius label, but just each 2 entries so the plot
            # does not get messy
            if i == 0:
                delta = abs(self.y[flags][1] - self.y[flags][0])
            if (i % 2 == 0):  # add gyro radius labels
                # Delta variable just to adust nicelly the distance (as old
                # fildsim is in cm and new in m)
                ax.text((self.y[flags]).min()-0.5 * delta,
                        (self.z[flags]).min(),
                        f'{float(self.unique_gyroradius[i]):g}',
                        horizontalalignment='right',
                        verticalalignment='center')
        ax.annotate('Gyroradius [cm]',
                    xy=(min(self.y) - delta,
                        (max(self.z) - min(self.z))/2 + min(self.z)),
                    rotation=rotation_for_gyr_label,
                    horizontalalignment='center',
                    verticalalignment='center')
        if self.diag == 'FILD':
            # Draw the lines of constant pitch. 'Vertical' lines
            for i in range(self.npitch):
                flags = self.pitch == self.unique_pitch[i]
                ax.plot(self.y[flags], self.z[flags], **line_options)
                if i == 0:
                    delta = abs(self.z[flags][-1] - self.z[flags][-2])
                ax.text((self.y[flags])[-1],
                        (self.z[flags])[-1] - delta,
                        f'{float(self.unique_pitch[i]):g}',
                        horizontalalignment='center',
                        verticalalignment='top')

            ax.annotate('Pitch [$\\degree$]',
                        xy=((max(self.y) - min(self.y))/2 + min(self.y),
                            min(self.z) - 1.5 * delta),
                        rotation=rotation_for_pitch_label,
                        horizontalalignment='center',
                        verticalalignment='center')
        elif self.diag == 'INPA':
            # Draw the lines of constant alpha. 'Vertical' lines
            for i in range(self.nalpha):
                flags = self.alpha == self.unique_alpha[i]
                if i == 0:
                    delta = abs(self.y[flags][1] - self.y[flags][0])

                ax.plot(self.y[flags], self.z[flags], **line_options)

                ax.text((self.y[flags])[-1],
                        (self.z[flags])[-1] - delta,
                        f'{float(self.unique_alpha[i]):g}',
                        horizontalalignment='center',
                        verticalalignment='top')

            ax.annotate('Alpha [rad]',
                        xy=((max(self.y) - min(self.y))/2 + min(self.y),
                            min(self.z) - 1.5*delta),
                        rotation=rotation_for_pitch_label,
                        horizontalalignment='center',
                        verticalalignment='center')
        else:
            raise Exception('Diagnostic not implemented')

        # Plot some markers in the grid position
        ax.plot(self.y, self.z, **marker_options)

    def plot_pix(self, ax=None, marker_params: dict = {},
                 line_params: dict = {}):
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
        else:
            raise Exception('Not implemented diagnostic')

        # Plot some markers in the grid position
        ## @todo include labels energy/pitch in the plot
        ax.plot(self.xpixel, self.ypixel, **marker_options)

    def calculate_pixel_coordinates(self, calib):
        """
        Transform the real coordinates of the map into pixels.

        Jose Rueda Rueda: jrrueda@us.es

        @param calib: a CalParams() object with the calibration info
        """
        self.xpixel, self.ypixel = \
            common.transform_to_pixel(self.y, self.z, calib)

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
            raise Exception('Not recognized interpolation method')
        if verbose:
            print('Using %s interpolation of the grid' % met)
        if self.diag == 'FILD':
            # Initialise the structure
            self.grid_interp = {
                'gyroradius': None,
                'pitch': None,
                'collimator_factor': None,
                'interpolators': {
                    'gyroradius': None,
                    'pitch': None,
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
            self.grid_interp['pitch'] = dummy2.copy().T
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
                        r_markers = self.grid_interp['interpolators']['gyroradius'](
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

        # --- Plot
        if plot:
            if self.diag == 'FILD':
                fig, axes = plt.subplots(2, 2)
                # Plot the scintillator grid
                self.plot_pix(axes[0, 0], line_params={'color': 'k'})
                # Plot the interpolated gyroradius
                c1 = axes[0, 1].imshow(self.grid_interp['gyroradius'],
                                       cmap=ssplt.Gamma_II(),
                                       vmin=0, vmax=10, origin='lower')
                fig.colorbar(c1, ax=axes[0, 1], shrink=0.9)
                # Plot the interpolated pitch
                c2 = axes[1, 0].imshow(self.grid_interp['pitch'],
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

    def load_strike_points(self, file=None, verbose: bool = True,
                           newFILDSIM=False):
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
                if newFILDSIM:
                    path, filename = os.path.split(self.file)
                    file = os.path.join(path, 'StrikePoints.bin')
                    Object = ssSINPA.Strikes
                else:
                    file = self.file[:-14] + 'strike_points.dat'
                    Object = ssFILDSIM.Strikes
        elif self.diag == 'INPA':
            if file is None:
                path, filename = os.path.split(self.file)
                file = os.path.join(path, 'StrikePoints.bin')
                Object = ssSINPA.Strikes

        self.fileStrikePoints = file
        self.strike_points = Object(file=file, verbose=verbose)

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
        Warning('This function will be removed in future versions.',
                DeprecationWarning)
        print('Please use smap.strike_points.plot2D() instead')
        self.strike_points.plot2D(ax=ax, mar_params=plt_param)

    def calculate_resolutions(self, diag_params: dict = {},
                              min_statistics: int = 100,
                              adaptative: bool = True):
        """
        Calculate the resolution associated with each point of the map.

        Jose Rueda Rueda: jrrueda@us.es

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch. It contains:
            FOR FILD:
                -dpitch: pitch space used by default in the fit. 1.0 default
                -dgyr: giroradius space used by default in the fit. 0.1 default
                -p_method: Function to use in the pitch fit, default Gauss
                -g_method: Function to use in the gyroradius fit, default sGauss
        @param min_statistics: Minimum number of points for a given r, p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_statistics: Minimum number of counts to perform the fit
        @param adaptative: If true, the bin width will be adapted such that the
        number of bins in a sigma of the distribution is 4. If this is the
        case, dpitch, dgyr, will no longer have an impact
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
                npitch = self.strike_points.header['nalpha']
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
                'amplitude': np.zeros((nr, npitch)),
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
                        par_p, fitp[ir, ip], normalization_p[ir, ip] = \
                            common._fit_to_model_(data[:, iip],
                                                  bins=edges_pitch,
                                                  model=p_method)
                        par_g, fitg[ir, ip], normalization_g[ir, ip] = \
                            common._fit_to_model_(data[:, iir],
                                                  bins=edges_gyr,
                                                  model=g_method)
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
                        # gyroradius parameters:
                        parameters_gyr['amplitude'][ir, ip] = \
                            par_g['amplitude']
                        parameters_gyr['center'][ir, ip] = par_g['center']
                        parameters_gyr['sigma'][ir, ip] = par_g['sigma']
                        if g_method == 'Gauss':
                            parameters_gyr['gamma'][ir, ip] = np.nan
                        elif g_method == 'sGauss':
                            parameters_gyr['gamma'][ir, ip] = par_g['gamma']

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
                'pitch_model': p_method
                }
            # --- Prepare the interpolators:
            self.calculate_interpolators()
        else:
            raise Exception('Diagnostic still not implemented')

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
                                     self.strike_points.header['alphas'])
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

    def calculate_mapping_interpolators(self, k=3, s=1):
        """
        Calculate interpolators scintillator position -> phase space.

        Jose Rueda: jrrueda@us.es

        @param k: parameter kx and ky for the BivariantSpline
        """
        # --- Select the colums to be used
        if self.diag == 'FILD':
            # temporal solution to save the coordinates in the array
            coords = np.zeros((self.y.size, 2))
            coords[:, 0] = self.y
            coords[:, 1] = self.z

            self.map_interpolators = {
                # 'Gyroradius':
                #     scipy_interp.SmoothBivariateSpline(self.y, self.z,
                #                                        self.gyroradius,
                #                                        kx=k, ky=k, s=s),
                # 'Pitch':
                #     scipy_interp.SmoothBivariateSpline(self.y, self.z,
                #                                        self.pitch, kx=k, ky=k,
                #                                        s=s)
                'Gyroradius':
                    scipy_interp.RBFInterpolator(coords,
                                                 self.gyroradius),
                'Pitch':
                    scipy_interp.RBFInterpolator(coords,
                                                 self.pitch)

            }
        elif self.diag == 'INPA':
            self.map_interpolators = {
                'Gyroradius':
                    scipy_interp.SmoothBivariateSpline(self.y, self.z,
                                                       self.gyroradius,
                                                       kx=k, ky=k),
                'Alpha':
                    scipy_interp.SmoothBivariateSpline(self.y, self.z,
                                                       self.alpha, kx=k, ky=k)
            }
        else:
            raise Exception('Diagnostic not understood')

    def remap_strike_points(self):
        """
        Remap the StrikePoints

        Jose Rueda: jrrueda@us.es
        """
        # --- See if the interpolators are defined
        if self.map_interpolators is None:
            print('Interpolators not calcualted. Calculating them')
            self.calculate_mapping_interpolators()
        # ---
        if self.diag == 'FILD':
            # --- See if the remap already exist in the strikes:
            if 'remap_rl' in self.strike_points.header['info'].keys():
                print('The remapped values are already in the strikes object')
                print('Nothing to do here')
                return
            iix = self.strike_points.header['info']['ys']['i']
            iiy = self.strike_points.header['info']['zs']['i']
            for ir in range(self.ngyr):
                for ip in range(self.npitch):
                    if self.strike_points.data[ip, ir] is not None:
                        n_strikes = self.strike_points.data[ip,
                                                            ir][:, iix].size
                    else:
                        n_strikes = 0
                    if n_strikes > 0:
                        remap_data = np.zeros((n_strikes, 2))
                        remap_data[:, 0] = \
                            self.map_interpolators['Gyroradius'](
                                self.strike_points.data[ip, ir][:, [iix, iiy]])
                                # self.strike_points.data[ip, ir][:, iiy])
                        remap_data[:, 1] = \
                            self.map_interpolators['Pitch'](
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
                'remap_pitch': {
                    'i': Old_number_colums + 1,  # Column index in the file
                    'units': ' [$\\degree$]',  # Units
                    'longName': 'Remapped pitch angle',
                    'shortName': '$\\lambda$',
                },
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
        ax_options = {
            'xlabel': '$\\lambda [\\degree]$',
            'ylabel': '$r_l [cm]$'
        }
        ax_options.update(ax_params)

        # --- Open the figure and prepare the map:
        fig, ax = plt.subplots(1, 2, figsize=(12, 10),
                               facecolor='w', edgecolor='k')

        if self.diag == 'FILD':
            if index_gyr is None:
                # Plot the gyroradius resolution
                a1 = ax[0].contourf(self.strike_points.header['pitch'],
                                    self.strike_points.header['gyroradius'],
                                    self.resolution['Gyroradius']['sigma'],
                                    levels=nlev, cmap=cmap)
                fig.colorbar(a1, ax=ax[0], label='$\\sigma_r [cm]$')
                ax[0] = ssplt.axis_beauty(ax[0], ax_options)
                # plot the pitch resolution
                a = ax[1].contourf(self.strike_points.header['pitch'],
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
                ax[0].plot(self.strike_points.header['pitch'],
                           self.resolution['Gyroradius']['sigma'][index_gyr, :])
                ax[0] = ssplt.axis_beauty(ax[0], ax_options)

                ax[1].plot(self.strike_points.header['pitch'],
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
            a1 = ax.contourf(self.strike_points.header['pitch'],
                             self.strike_points.header['gyroradius'],
                             self.collimator_factor_matrix,
                             levels=nlev, cmap=cmap)
            fig.colorbar(a1, ax=ax, label='Collimating factor')
            ax = ssplt.axis_beauty(ax, ax_options)

            plt.tight_layout()
        return

    def plot_resolution_fits(self, var: str = 'Gyroradius', ax_params: dict = {},
                             ax=None, gyr_index=None, pitch_index=None,
                             gyroradius=None, pitch=None,
                             kind_of_plot: str = 'normal',
                             include_legend: bool = False):
        """
        Plot the fits done to calculate the resolution

        @param var: variable to plot, Gyroradius or Pitch for FILD. Capital
        letters will be ignored
        @param ax_param: dictoniary with the axis parameters axis_beauty()
        @param ax: axis where to plot
        @param gyr_index: index, or arrays of indeces, of gyroradius to plot
        @param pitch_index: index, or arrays of indeces, of gyroradius to plot
        @param gyroradius: gyroradius value of array of then to plot. If
        present, gyr_index will be ignored
        @param pitch: idem to gyroradius bu for the pitch
        @param kind_of_plot: kind of plot to make:
            - normal: scatter plot of the data and fit like a line
            - bar: bar plot of the data and file like a line
            - uncertainty: scatter plot of the data and shading area for the fit
             (3 sigmas)
            - just_fit: Just a line plot as the fit
        @param include_legend: flag to include a legend
        """
        # --- Initialise plotting options and axis:
        default_labels = {
            'gyroradius': {
                'xlabel': 'Gyroradius [cm]',
                'ylabel': '$\\sigma_r [cm]$'
            },
            'pitch': {
                'xlabel': 'Pitch [$\\degree$]',
                'ylabel': '$\\sigma_p [$\\degree$]'
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
        # --- Localise the values to plot
        if gyroradius is not None:
            # test if it is a number or an array of them
            if isinstance(gyroradius, (list, np.ndarray)):
                gyroradius = gyroradius
            else:
                gyroradius = np.array([gyroradius])
            index_gyr = np.zeros(gyroradius.size)
            for i in range(index_gyr.size):
                index_gyr[i] = \
                    np.argmin(np.abs(self.unique_gyroradius - gyroradius[ir]))
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
            index_pitch = np.zeros(pitch.size)
            for i in range(index_pitch.size):
                index_pitch[i] = \
                    np.argmin(np.abs(self.unique_pitch - pitch[ir]))
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
                        scatter = ax.scatter(x,
                                             normalization * self.resolution['fits'][var.lower(
                                             )][ir, ip].data,
                                             label='__noname__')
                        # plot the fit as a line
                        ax.plot(x_fine, y, color=scatter.get_facecolor()[0, :3],
                                label=name)
                    elif kind_of_plot.lower() == 'bar':
                        bar = ax.bar(x,
                                     normalization * self.resolution['fits'][var.lower()
                                                                             ][ir, ip].data,
                                     label='__noname__', width=x[1]-x[0],
                                     alpha=0.25)
                        ax.plot(x_fine, y, color=bar.patches[0].get_facecolor()[
                                :3], label=name)
                    elif kind_of_plot.lower() == 'just_fit':
                        ax.plot(x_fine, y, label=name)
                    elif kind_of_plot.lower() == 'uncertainty':
                        scatter = ax.scatter(x,
                                             normalization * self.resolution['fits'][var.lower(
                                             )][ir, ip].data,
                                             label='__noname__')
                        dely = normalization * self.resolution['fits'][var.lower(
                            )][ir, ip].eval_uncertainty(sigma=3, x=x_fine)
                        ax.fill_between(x_fine, y-dely, y+dely, alpha=0.25,
                                        label='3-$\\sigma$ uncertainty band',
                                        color=scatter.get_facecolor()[0, :3])
                    else:
                        raise Exception('Not kind of plot not understood')
                else:
                    print('Not fits for rl: '
                          + str(round(self.unique_gyroradius[ir], 1))
                          + 'pitch: '
                          + str(round(self.unique_pitch[ip], 1)))
        if include_legend:
            ax.legend()
        if created:
            ax = ssplt.axis_beauty(ax, ax_options)

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
                par_p, resultp = common._fit_to_model_(data[:, 5],
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

                par_g, resultg = common._fit_to_model_(data[:, 4],
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

                cent = 0.5 * (edges_gyr[1:] + edges_gyr[:-1])
                fit_line = ax_gyroradius.plot(cent, resultg.best_fit,
                                              label='_nolegend_')
                label_plot = \
                    f"{float(self.strike_points.header['gyroradius'][ir]):g}"\
                    + '[cm]'
                ax_gyroradius.hist(data[:, 6], bins=edges_gyr,
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
