"""
Strike map class

Jose Rueda: jrrueda@us.es
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interp
import Lib.LibPlotting as ssplt
import Lib.LibFILDSIM as ssFILDSIM
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

        Notes: machine, theta and phi options introduced in version 0.4.14.
        INPA compatibility included in version 0.6.0
        """
        ## Associated diagnostic
        if flag == 0 or flag == 'FILD':
            self.diag = 'FILD'
        elif flag == 1 or flag == 'INPA':
            self.diag = 'INPA'
        elif flag == 2 or flag == 'iHIBP':
            self.diag = 'iHIBP'
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
        self.intepolators = None
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
                print('Strike map no fpun in the database')
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
            dummy = np.loadtxt(file, skiprows=2)
            # See which rows has collimator factor larger than zero (ie see for
            # which combination of rl and alpha some markers arrived)
            ind = dummy[:, 9] > 0
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
            self.x0 = dummy[ind, 5]
            ## y coordinates of closest point to NBI
            self.y0 = dummy[ind, 6]
            ## z coordinates of closest point to NBI
            self.z0 = dummy[ind, 7]
            ## distance to the NBI central line
            self.d0 = dummy[ind, 8]
            ## Collimator factor as defined in FILDSIM
            self.collimator_factor = dummy[ind, 9]
            ## Number of markers striking in this area
            self.n_strike_points = dummy[ind, 10]
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

        if self.diag == 'FILD':
            # Draw the lines of constant gyroradius (energy). These are the
            # 'horizontal' lines]
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_options)

                if (i % 2 == 0):  # add gyro radius labels
                    ax.text((self.y[flags])[0]-0.2,
                            (self.z[flags])[0], f'{float(uniq[i]):g}',
                            horizontalalignment='right',
                            verticalalignment='center')

            ax.annotate('Gyroradius [cm]',
                        xy=(min(self.y) - 0.5,
                            (max(self.z) - min(self.z))/2 + min(self.z)),
                        rotation=rotation_for_gyr_label,
                        horizontalalignment='center',
                        verticalalignment='center')

            # Draw the lines of constant pitch. 'Vertical' lines
            uniq = np.unique(self.pitch)
            n = len(uniq)
            for i in range(n):
                flags = self.pitch == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_options)

                ax.text((self.y[flags])[-1],
                        (self.z[flags])[-1] - 0.1,
                        f'{float(uniq[i]):g}',
                        horizontalalignment='center',
                        verticalalignment='top')

            ax.annotate('Pitch [$\\degree$]',
                        xy=((max(self.y) - min(self.y))/2 + min(self.y),
                            min(self.z) - 0.1),
                        rotation=rotation_for_pitch_label,
                        horizontalalignment='center',
                        verticalalignment='center')
        elif self.diag == 'INPA':
            # Draw the lines of constant gyroradius (energy). These are the
            # 'horizontal' lines]
            uniq = np.unique(self.gyroradius)
            n = len(uniq)
            for i in range(n):
                flags = self.gyroradius == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_options)

                if (i % 2 == 0):  # add gyro radius labels
                    ax.text((self.y[flags])[0]-0.2,
                            (self.z[flags])[0], f'{float(uniq[i]):g}',
                            horizontalalignment='right',
                            verticalalignment='center')

            ax.annotate('Gyroradius [cm]',
                        xy=(min(self.y) - 0.5,
                            (max(self.z) - min(self.z))/2 + min(self.z)),
                        rotation=rotation_for_gyr_label,
                        horizontalalignment='center',
                        verticalalignment='center')

            # Draw the lines of constant alpha. 'Vertical' lines
            uniq = np.unique(self.alpha)
            n = len(uniq)
            for i in range(n):
                flags = self.alpha == uniq[i]
                ax.plot(self.y[flags], self.z[flags], **line_options)

                ax.text((self.y[flags])[-1],
                        (self.z[flags])[-1] - 0.1,
                        f'{float(uniq[i]):g}',
                        horizontalalignment='center',
                        verticalalignment='top')

            ax.annotate('Alpha [rad]',
                        xy=((max(self.y) - min(self.y))/2 + min(self.y),
                            min(self.z) - 0.1),
                        rotation=rotation_for_pitch_label,
                        horizontalalignment='center',
                        verticalalignment='center')
        else:
            raise Exception('Diagnostic not implemented')

        # Plot some markers in the grid position
        ax.plot(self.y, self.z, **marker_options)
        return

    def plot_pix(self, ax=None, marker_params: dict = {},
                 line_params: dict = {}):
        """
        Plot the strike map (x,y = pixels on the camera).

        Jose Rueda: jrrueda@us.es

        @param ax: Axes where to plot
        @param marker_params: parameters for the centroid plotting
        @param line_params: parameters for the lines plotting
        @return: Strike maps over-plotted in the axis
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
        if method == 1 or method == 'linear':
            met = 'linear'
            interpolator = scipy_interp.LinearNDInterpolator
        elif method == 2 or method == 'cubic':
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
                        r_markers = self.grid_interp['interpolators']\
                            ['gyroradius'](x_markers, y_markers)
                        p_markers = self.grid_interp['interpolators']\
                            ['pitch'](x_markers, y_markers)
                        # make the histogram in the r-pitch space
                        H, xedges, yedges = \
                            np.histogram2d(p_markers, r_markers,
                                           bins=[pitch_edges, gyr_edges])
                        transform[:, :, i, j] = H.copy()
                # Normalise the transformation matrix
                transform /= MC_number
                transform /= (grid_options['dx'] * grid_options['dy'])
                # This last normalization will be removed once we include the
                # jacobian somehow
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
        @param A: the mass number
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
        """
        if self.diag == 'FILD':
            if file is None:
                file = self.file[:-14] + 'strike_points.dat'
            if verbose:
                print('Reading strike points: ', file)
            ssFILDSIM.Strikes(file=file, verbose=verbose)

        return

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
        print('This function will be removed in future versions.')
        print('Please use smap.strike_points.plot2D() instead')
        self.strike_points.plot2D(ax=ax, mar_params=plt_param)

    def calculate_resolutions(self, diag_params: dict = {},
                              min_statistics: int = 100,
                              adaptative: bool = True):
        """
        Calculate the resolution associated with each point of the map.

        Jose Rueda Rueda: jrrueda@us.es

        @param diag_options: Dictionary with the diagnostic specific parameters
        like for example the method used to fit the pitch
        @param min_statistics: Minimum number of points for a given r p to make
        the fit (if we have less markers, this point will be ignored)
        @param min_statistics: Minimum number of counts to perform the fit
        @param adaptative: If true, the bin width will be adapted such that the
        number of bins in a sigma of the distribution is 4. If this is the
        case, dpitch, dgyr, will no longer have an impact
        """
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
            dgyr = diag_options['dgyr']
            p_method = diag_options['p_method']
            g_method = diag_options['g_method']
            npitch = self.strike_points.header['npitch']
            nr = self.strike_points.header['ngyroradius']
            # --- Pre-allocate variables
            npoints = np.zeros((nr, npitch))
            parameters_pitch = {'amplitude': np.zeros((nr, npitch)),
                                'center': np.zeros((nr, npitch)),
                                'sigma': np.zeros((nr, npitch)),
                                'gamma': np.zeros((nr, npitch))}
            parameters_gyr = {'amplitude': np.zeros((nr, npitch)),
                              'center': np.zeros((nr, npitch)),
                              'sigma': np.zeros((nr, npitch)),
                              'gamma': np.zeros((nr, npitch))}
            fitg = []
            fitp = []
            gyr_array = []
            pitch_array = []
            print('Calculating FILD resolutions')
            for ir in tqdm(range(nr)):
                for ip in range(npitch):
                    # --- Select the data
                    data = self.strike_points.data[ip, ir]

                    # --- See if there is enough points:
                    if self.header['counters'][ip, ir] < min_statistics:
                        parameters_gyr['amplitude'][ir, ip] = np.nan
                        parameters_gyr['center'][ir, ip] = np.nan
                        parameters_gyr['sigma'][ir, ip] = np.nan
                        parameters_gyr['gamma'][ir, ip] = np.nan

                        parameters_pitch['amplitude'][ir, ip] = np.nan
                        parameters_pitch['center'][ir, ip] = np.nan
                        parameters_pitch['sigma'][ir, ip] = np.nan
                        parameters_pitch['gamma'][ir, ip] = np.nan
                    else:  # If we have enough points, make the fit
                        # Prepare the bin edges according to the desired width
                        edges_pitch = \
                            np.arange(start=data[:, 5].min() - dpitch,
                                      stop=data[:, 5].max() + dpitch,
                                      step=dpitch)
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
                            sigma_p = np.std(data[:, 5])
                            new_dpitch = sigma_p / n_bins_in_sigma
                            edges_pitch = \
                                np.arange(start=data[:, 5].min() - dpitch,
                                          stop=data[:, 5].max() + dpitch,
                                          step=new_dpitch)
                        # --- Proceed to fit
                        par_p, resultp = common._fit_to_model_(data[:, 5],
                                                               bins=edges_pitch,
                                                               model=p_method)
                        par_g, resultg = common._fit_to_model_(data[:, 4],
                                                               bins=edges_gyr,
                                                               model=g_method)
                        fitp.append(resultp)
                        fitg.append(resultg)
                        gyr_array.append(self.strike_points.header['gyroradius'][ir])
                        pitch_array.append(self.strike_points.header['pitch'][ip])
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
                    'Gyroradius': fitg,
                    'Pitch': fitp,
                    'FILDSIM_gyroradius': np.array(gyr_array),
                    'FILDSIM_pitch': np.array(pitch_array),
                },
                'gyroradius_model': g_method,
                'pitch_model': p_method
                }
            # --- Prepare the interpolators:
            self.calculate_interpolators()
        return

    def calculate_interpolators(self):
        """
        Calc interpolators from phase space to instrument function params

        Jose Rueda: jrrueda@us.es
        """
        if self.diag == 'FILD':
            # --- Prepare the interpolators:
            # Prepare grid
            xx, yy = np.meshgrid(self.strike_points.header['gyroradius'],
                                 self.strike_points.header['pitch'])
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

    def plot_resolutions(self, ax_param: dict = {}, cMap=None, nlev: int = 20):
        """
        Plot the resolutions.

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
        fig, ax = plt.subplots(1, 2, figsize=(12, 10),
                               facecolor='w', edgecolor='k')

        if self.diag == 'FILD':
            # Plot the gyroradius resolution
            a1 = ax[0].contourf(self.strike_points.header['pitch'],
                                self.strike_points.header['gyroradius'],
                                self.resolution['Gyroradius']['sigma'],
                                levels=nlev, cmap=cmap)
            fig.colorbar(a1, ax=ax[0], label='$\\sigma_r [cm]$')
            ax[0] = ssplt.axis_beauty(ax[0], ax_param)
            # plot the pitch resolution
            a = ax[1].contourf(self.strike_points.header['pitch'],
                               self.strike_points.header['gyroradius'],
                               self.resolution['Pitch']['sigma'],
                               levels=nlev, cmap=cmap)
            fig.colorbar(a, ax=ax[1], label='$\\sigma_\\lambda$')
            ax[1] = ssplt.axis_beauty(ax[1], ax_options)
            plt.tight_layout()
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

    def sanity_check_resolutions(self):
        """
        Plot basic quantities of the resolution calculation as a test.

        Jose Rueda: jrrueda@us.es

        Designed to quickly see some figures of merit of the resolution
        calculation, ie, compare the centroids of the fits with the actual
        values the particles were iniciated in FILDSIM
        """
        if self.diag == 'FILD':
            axis_param = {'grid': 'both', 'ratio': 'equal'}
            # Centroids comparison:
            cen_g = []
            cen_p = []
            fild_g = []
            # Arange centroids by pitch (gyroradius)
            for p in np.unique(self.resolution['fits']['FILDSIM_pitch']):
                dummy = []
                dummy_FILDSIM = []
                print(p)
                nfits = len(self.resolution['fits']['FILDSIM_gyroradius'])
                for i in range(nfits):
                    if self.resolution['fits']['FILDSIM_pitch'][i] == p:
                        dummy.append(self.resolution['fits']['Gyroradius'][i]\
                                     .params['center'].value)
                        dummy_FILDSIM.append(self.resolution['fits']\
                                             ['FILDSIM_gyroradius'][i])

                cen_g.append(dummy.copy())
                fild_g.append(dummy_FILDSIM.copy())
            for i in range(len(self.resolution['fits']['FILDSIM_pitch'])):
                cen_p.append(self.resolution['fits']['Pitch'][i]\
                             .params['center'].value)
            figc, axc = plt.subplots(1, 2)
            for i in range(len(fild_g)):
                label_plot = \
                    str(np.unique(self.resolution['fits']['FILDSIM_pitch'])[i])
                axc[0].plot(fild_g[i], cen_g[i], 'o', label=label_plot)
            axc[0].set_xlabel('FILDSIM')
            axc[0].legend()
            axc[0] = ssplt.axis_beauty(axc[0], axis_param)
            axc[1].plot(self.resolution['fits']['FILDSIM_pitch'],
                        cen_p, 'o')
            axc[1].set_xlabel('FILDSIM')
            axc[1] = ssplt.axis_beauty(axc[1], axis_param)

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
                        + str(self.strike_points.header['gyroradius'][ir])\
                        +' cm')

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
