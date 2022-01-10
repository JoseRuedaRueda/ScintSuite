"""Handle the video from the different cameras

This module is created to handle the .cin (.cine) files, binary files
created by the Phantom cameras. In its actual state it can read everything
from the file, but it can't write/create a cin file. It also load data from
PNG files as the old FILD_GUI and will be able to work with tiff files
"""
from Lib.LibVideo.BasicVideoObject import BVO
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tkinter as tk                       # To open UI windows
import Lib.LibPlotting as ssplt
import Lib.LibMap as ssmap
import Lib.LibPaths as p
import Lib.LibIO as ssio
import Lib.GUIs as ssGUI             # For GUI elements
import Lib.LibData as ssdat
import Lib.SimulationCodes.FILDSIM as ssFILDSIM
from Lib.LibMachine import machine
from Lib.version_suite import version
from scipy.io import netcdf                # To export remap data
from tqdm import tqdm                      # For waitbars
pa = p.Path(machine)
del p


class FILDVideo(BVO):
    """
    Video class for the FILD diagnostic.

    Jose Rueda: jrrueda@us.es
    """

    def __init__(self, file: str = None, diag: str = 'FILD', shot=None,
                 diag_ID: int = 1):
        """
        Initialise the class

        @param file: For the initialization, file (full path) to be loaded),
        if the path point to a .cin file, the .cin file will be loaded. If
        the path points to a folder, the program will look for png files or
        tiff files inside (tiff coming soon). If none, a window will be open to
        select a file
        @param shot: Shot number, if is not given, the program will look for it
        in the name of the loaded file
        """
        # initialise the parent class
        BVO.__init__(self, file=file, shot=shot)
        ## Diagnostic used to record the data
        self.diag = diag
        ## Diagnostic ID (FILD number)
        self.diag_ID = diag_ID

    def remap_loaded_frames(self, calibration, shot, options: dict = {},
                            mask=None):
        """
        Remap all loaded frames in the video object

        @param    calibration: Calibration object (see LibMap)
        @type:    type

        @param    shot: Shot number
        @param    mask: binary mask (as for the TimeTraces), to just select a
        region of the scintillator to be remapped
        @param    options: Options for the remapping routine. See
        remap_all_loaded_frames_XXXX in the LibMap package for a full
        description
        @type:    dict

        @return:  write in the object a dictionary containing with:
            -# options: Options used for the remapping
            -# frames: Remaped frames
            -# time: time associated to the remapped points
            -# xaxis: xaxis of the remapped frames
            -# xlabel: name of the xaxis of he remaped frame (pitch for FILD)
            -# yaxis: xaxis of the remapped frames
            -# ylabel: name of the yaxis of he remaped frame (r for FILD)
            -# sprofx: signal integrated over the y range given by options
            -# sprofy: signal integrated over the x range given by options
        """
        self.remap_dat, opt = \
            ssmap.remap_all_loaded_frames_FILD(self, calibration, shot,
                                               mask=mask, **options)
        self.remap_dat['options'] = opt

    def integrate_remap(self, xmin=20.0, xmax=90.0, ymin=1.0, ymax=10.0,
                        mask=None):
        """
        Integrate the remaped frames over a given region of the phase space

        Jose Rueda: jrrueda@us.es

        @param xmin: Minimum value of the x axis to integrate (pitch for FILD)
        @param xmax: Maximum value of the x axis to integrate (pitch for FILD)
        @param ymin: Minimum value of the y axis to integrate (radius in FILD)
        @param ymax: Maximum value of the y axis to integrate (radius in FILD)
        @param mask: bynary mask denoting the desired pixes of the space to
        integate
        @return : Output: Dictionary containing the trace and the settings used
        to caclualte it
        """
        if self.remap_dat is None:
            raise Exception('Please remap before call this function!!!')
        # First calculate the dif x and y to integrate
        dx = self.remap_dat['xaxis'][1] - self.remap_dat['xaxis'][0]
        dy = self.remap_dat['yaxis'][1] - self.remap_dat['yaxis'][0]
        # Find the flags:
        mask_was_none = False
        if mask is None:
            flagsx = (xmin < self.remap_dat['xaxis']) *\
                (self.remap_dat['xaxis'] < xmax)
            flagsy = (ymin < self.remap_dat['yaxis']) *\
                (self.remap_dat['yaxis'] < ymax)
            mask_was_none = True
        # Perform the integration:
        nx, ny, nt = self.remap_dat['frames'].shape
        trace = np.zeros(nt)
        for iframe in tqdm(range(nt)):
            dummy = self.remap_dat['frames'][:, :, iframe].copy()
            dummy = dummy.squeeze()
            if mask_was_none:
                trace[iframe] = np.sum(dummy[flagsx, :][:, flagsy]) * dx * dy
            else:
                trace[iframe] = np.sum(dummy[mask]) * dx * dy
        if mask_was_none:
            output = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
        else:
            output = {'mask': mask}
        output['trace'] = trace
        output['t'] = self.exp_dat['tframes']
        return output

    def plot_frame(self, frame_number=None, ax=None, ccmap=None,
                   strike_map: str = 'off', t: float = None,
                   verbose: bool = True,
                   smap_marker_params: dict = {},
                   smap_line_params: dict = {}, vmin=0, vmax=None,
                   xlim=None, ylim=None, scale: str = 'linear'):
        """
        Plot a frame from the loaded frames

        @param frame_number: Number of the frame to plot, relative to the video
            file, optional
        @param ax: Axes where to plot, is none, just a new axes will be created
        @param ccmap: colormap to be used, if none, Gamma_II from IDL
        @param strike_map: StrikeMap to plot:
            -# 'auto': The code will load the Smap corresponding to the theta,
            phi angles. Note, the angles should be calculated, so the remap,
            should be done. (also, it will load the calibration from the
            performed remap)
            -# StrikeMap(): if a StrikeMap() object is passed, it will be
            plotted. Note, the program will only check if the StrikeMap input
            is a class, it will not check if the calibration was apply etc, so
            it is supposed that the user had given a Smap, ready to be plotted
            -# 'off': (or any other string) No strike map will be plotted
        @param verbose: If true, info of the theta and phi used will be printed
        @param smap_marker_params: dictionary with parameters to plot the
            strike_map centroid (see StrikeMap.plot_pix)
        @param smap_line_params: dictionary with parameters to plot the
            strike_map lines (see StrikeMap.plot_pix)
        @param vmin: Minimum value for the color scale to plot
        @param vmax: Maximum value for the color scale to plot
        @param xlim: tuple with the x-axis limits
        @param ylim: tuple with the y-axis limits
        @param scale: Scale for the plot: 'linear', 'sqrt', or 'log'

        @return ax: the axes where the frame has been drawn

        """
        # --- Check inputs:
        if (frame_number is not None) and (t is not None):
            raise Exception('Do not give frame number and time!')
        if (frame_number is None) and (t is None):
            raise Exception("Didn't you want to plot something?")
        if strike_map == 'auto' and self.remap_dat is None:
            raise Exception('To use the auto mode, you need to remap first')
        # --- Prepare the scale:
        if scale == 'sqrt':
            extra_options = {'norm': colors.PowerNorm(0.5)}
        elif scale == 'log':
            extra_options = {'norm': colors.LogNorm(0.5)}
        else:
            extra_options = {}
        # --- Load the frames
        # If we use the frame number explicitly
        if frame_number is not None:
            if len(self.exp_dat['nframes']) == 1:
                if self.exp_dat['nframes'] == frame_number:
                    dummy = self.exp_dat['frames'].squeeze()
                    tf = float(self.exp_dat['tframes'])
                    frame_index = 0
                else:
                    raise Exception('Frame not loaded')
            else:
                frame_index = self.exp_dat['nframes'] == frame_number
                if np.sum(frame_index) == 0:
                    raise Exception('Frame not loaded')
                dummy = self.exp_dat['frames'][:, :, frame_index].squeeze()
                tf = float(self.exp_dat['tframes'][frame_index])
        # If we give the time:
        if t is not None:
            frame_index = np.argmin(abs(self.exp_dat['tframes'] - t))
            tf = self.exp_dat['tframes'][frame_index]
            dummy = self.exp_dat['frames'][:, :, frame_index].squeeze()
        # --- Check the colormap
        if ccmap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = ccmap
        # --- Check the axes to plot
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        if vmax is None:
            vmax = dummy.max()
        img = ax.imshow(dummy, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        **extra_options)
        # --- trick to make the colorbar of the correct size
        # cax = fig.add_axes([ax.get_position().x1 + 0.01,
        #                     ax.get_position().y0, 0.02,
        #                     ax.get_position().height])
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im_ratio = dummy.shape[0]/dummy.shape[1]
        plt.colorbar(img, label='Counts', fraction=0.042*im_ratio, pad=0.04)
        ax.set_title('t = ' + str(round(tf, 4)) + (' s'))
        # Save axis limits, if not, if the strike map is larger than
        # the frame (FILD4,5) the output plot will be horrible
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # --- Plot the StrikeMap
        if isinstance(strike_map, ssmap.StrikeMap):
            strike_map.plot_pix(ax=ax)
        elif strike_map == 'auto':
            # get parameters of the map
            theta_used = self.remap_dat['theta_used'][frame_index]
            phi_used = self.remap_dat['phi_used'][frame_index]

            # Get the full name of the file
            name__smap = ssFILDSIM.guess_strike_map_name_FILD(
                phi_used, theta_used, machine=machine,
                decimals=self.remap_dat['options']['decimals']
            )
            smap_folder = self.remap_dat['options']['smap_folder']
            full_name_smap = os.path.join(smap_folder, name__smap)
            # Load the map:
            smap = ssmap.StrikeMap(0, full_name_smap)
            # Calculate pixel coordinates
            smap.calculate_pixel_coordinates(
                self.remap_dat['options']['calibration']
            )
            # Plot the map
            smap.plot_pix(ax=ax, marker_params=smap_marker_params,
                          line_params=smap_line_params)
            if verbose:
                theta_calculated = self.remap_dat['theta'][frame_index]
                phi_calculated = self.remap_dat['phi'][frame_index]
                print('Calculated theta: ', theta_calculated)
                print('Used theta: ', theta_used)
                print('Calculated phi: ', phi_calculated)
                print('Used phi: ', phi_used)
        # Set 'original' limits:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Arrange the axes:
        if created:
            fig.show()
            plt.tight_layout()
        return ax

    def plot_frame_remap(self, frame_number=None, ax=None, ccmap=None,
                         t: float = None, vmin: float = 0, vmax: float = None,
                         xlim: float = None, ylim: float = None,
                         scale: str = 'linear',
                         interpolation: str = 'bicubic',
                         cbar_tick_format: str = '%.1E'):
        """
        Plot a frame from the remaped frames

        @param frame_number: Number of the frame to plot, relative to the video
            file, optional
        @param ax: Axes where to plot, is none, just a new axes will be created
        @param ccmap: colormap to be used, if none, Gamma_II from IDL
        @param vmin: Minimum value for the color scale to plot
        @param vmax: Maximum value for the color scale to plot
        @param xlim: tuple with the x-axis limits
        @param ylim: tuple with the y-axis limits
        @param scale: Scale for the plot: 'linear', 'sqrt', or 'log'
        @param interpolation: interpolation method for plt.imshow
        @param cbar_tick_format: format for the colorbar ticks

        @return ax: the axes where the frame has been drawn
        """
        # --- Check inputs:
        if (frame_number is not None) and (t is not None):
            raise Exception('Do not give frame number and time!')
        if (frame_number is None) and (t is None):
            raise Exception("Didn't you want to plot something?")
        # --- Prepare the scale:
        if scale == 'sqrt':
            extra_options = {'norm': colors.PowerNorm(0.5)}
        elif scale == 'log':
            extra_options = {'norm': colors.LogNorm(0.5)}
        else:
            extra_options = {}
        # --- Load the frames
        # If we use the frame number explicitly
        if frame_number is not None:
            if len(self.remap_dat['nframes']) == 1:
                if self.remap_dat['nframes'] == frame_number:
                    dummy = self.remap_dat['frames'].squeeze()
                    tf = float(self.remap_dat['tframes'])
                    frame_index = 0
                else:
                    raise Exception('Frame not loaded')
            else:
                frame_index = self.remap_dat['nframes'] == frame_number
                if np.sum(frame_index) == 0:
                    raise Exception('Frame not loaded')
                dummy = self.remap_dat['frames'][:, :, frame_index].squeeze()
                tf = float(self.remap_dat['tframes'][frame_index])
        # If we give the time:
        if t is not None:
            frame_index = np.argmin(np.abs(self.remap_dat['tframes'] - t))
            tf = self.remap_dat['tframes'][frame_index]
            dummy = self.remap_dat['frames'][:, :, frame_index].squeeze()
        # --- Check the colormap
        if ccmap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = ccmap
        # --- Check the axes to plot
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        if vmax is None:
            vmax = dummy.max()
        img = ax.imshow(dummy.T, extent=[self.remap_dat['xaxis'][0],
                                         self.remap_dat['xaxis'][-1],
                                         self.remap_dat['yaxis'][0],
                                         self.remap_dat['yaxis'][-1]],
                        origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation=interpolation, aspect='auto',
                        **extra_options)
        # --- trick to make the colorbar of the correct size
        # cax = fig.add_axes([ax.get_position().x1 + 0.01,
        #                     ax.get_position().y0, 0.02,
        #                     ax.get_position().height])
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im_ratio = dummy.shape[0]/dummy.shape[1]
        plt.colorbar(img, label='Counts', fraction=0.042*im_ratio, pad=0.04,
                     format=cbar_tick_format)
        ax.set_title('t = ' + str(round(tf, 4)) + (' s'))
        # Save axis limits, if not, if the strike map is larger than
        # the frame (FILD4,5) the output plot will be horrible
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if created:
            ax.set_xlabel('$\\lambda \\ [\\degree]$')
            ax.set_ylabel('$r_l$ [cm]')
            fig.show()
            plt.tight_layout()
        return ax

    def find_orientation(self, t, verbose: bool = True, R: float = None,
                         z: float = None):
        """
        Find the orientation of FILD for a given time.

        José Rueda: jrrueda@us.es

        @param t: time point where we want the angles [s]. It can be 'all' in
        that case, the orientation will be calculated for all time points
        @param verbose: flag to print information or not
        @param R: R coordinate of the detector (in meters) for B calculation
        @param z: z coordinate of the detector (in meters) for B calculation

        @return theta: theta angle [º]
        @return phi: phi angle [º]
        """
        if self.remap_dat is None:
            alpha = ssdat.FILD[self.diag_ID-1]['alpha']
            beta = ssdat.FILD[self.diag_ID-1]['beta']
            print('Remap not done, calculating angles')

            if t == 'all':
                nframes = self.exp_dat['tframes'].size
                if machine == 'AUG':
                    print('Opening shotfile from magnetic field')
                    import map_equ as meq
                    equ = meq.equ_map(self.shot, diag='EQH')
                    br = np.zeros(nframes)
                    bz = np.zeros(nframes)
                    bt = np.zeros(nframes)
                theta = np.zeros(nframes)
                phi = np.zeros(nframes)
                for iframe in tqdm(range(nframes)):
                    # To avoid stupid bugs in the python library of AUG to
                    # read the magnetic field
                    if machine == 'AUG':
                        tframe = self.exp_dat['tframes'][iframe]
                        br[iframe], bz[iframe], bt[iframe], bp =\
                            ssdat.get_mag_field(self.shot, R, z,
                                                time=tframe,
                                                equ=equ)
                    phi[iframe], theta[iframe] = \
                        ssFILDSIM.calculate_fild_orientation(br[iframe],
                                                             bz[iframe],
                                                             bt[iframe],
                                                             alpha, beta)
                time = 'all'
            else:
                br, bz, bt, bp =\
                    ssdat.get_mag_field(self.shot, R, z, time=t)

                phi, theta = \
                    ssFILDSIM.calculate_fild_orientation(br, bz, bt,
                                                         alpha, beta)
                time = t

        else:
            tmin = self.remap_dat['tframes'][0]
            tmax = self.remap_dat['tframes'][-1]
            if t < tmin or t > tmax:
                raise Exception('Time not present in the remap')
            else:
                it = np.argmin(abs(self.remap_dat['tframes'] - t))
                theta = self.remap_dat['theta'][it]
                phi = self.remap_dat['phi'][it]
                time = self.remap_dat['tframes'][it]
        if verbose:
            # I include these 'np.array' in order to be compatible with the
            # case of just one time point and multiple ones. It is not the most
            # elegant way to proceed, but it works ;)
            print('Requested time:', t)
            if self.remap_dat is None:
                print('Found time: ', time)
            print('Average theta:', np.array(theta).mean())
            print('Average phi:', np.array(phi).mean())
            if self.remap_dat is None:
                print('Average B field: ',
                      np.array(np.sqrt(bt**2 + bp**2)[0]).mean())
        return phi, theta

    def GUI_frames(self):
        """Small GUI to explore camera frames"""
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print('-------------------')
        print(text)
        print('-------------------')
        root = tk.Tk()
        root.resizable(height=None, width=None)
        ssGUI.ApplicationShowVid(root, self.exp_dat, self.remap_dat)
        root.mainloop()
        root.destroy()

    def GUI_frames_and_remap(self):
        """Small GUI to explore camera and remapped frames"""
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print(text)
        root = tk.Tk()
        root.resizable(height=None, width=None)
        ssGUI.ApplicationShowVidRemap(root, self.exp_dat, self.remap_dat)
        root.mainloop()
        root.destroy()

    # def GUI_profiles(self):
    #     """Small GUI to explore camera and remapped frames"""
    #     text = 'Press TAB until the time slider is highlighted in red.'\
    #         + ' Once that happend, you can move the time with the arrows'\
    #         + ' of the keyboard, frame by frame'
    #     print(text)
    #     root = tk.Tk()
    #     root.resizable(height=None, width=None)
    #     ssGUI.ApplicationShowProfiles(root, self.exp_dat, self.remap_dat)
    #     root.mainloop()
    #     root.destroy()

    def plot_profiles_in_time(self, ccmap=None, ax_params: dict = {}, t=None,
                              nlev: int = 50, cbar_tick_format: str = '%.1E',
                              normalise=False, max_gyr=None, min_gyr=None,
                              max_pitch=None, min_pitch=None, ax=None,
                              line_params={}, scale: str = 'linear',
                              interpolation: str = 'bicubic'):
        """
        Create a plot with the evolution of the profiles.

        Jose Rueda Rueda: jrrueda@us.es

        @param ccmap: colormap to be used, if none, Gamma_II will be used
        @param ax_params: params for the function axis beauty plt. Notice that,
            the xlabel for the rl and pitch are hardwritten in the inside
            (sorry, this was to avoid to have 2 ax_params inputs)
        @param t: time, if present, just a line plot for the profiles for that
        time will be used
        @param nlev: Number of levels for the contourf plots (deprecated, as we
            now use imshow to plot the data)
        @param cbar_tick_format: format for the colorbar ticks
        @param max_gyr: maximum value for colorbar plot in gyroradius
        @param min_gyr: minimum value for colorbar plot in gyroradius
        @param max_pitch: maximum value for colorbar plot in pitch
        @param min_pitch: minimum value for colorbar plot in pitch
        @param scale: Color scale to plot, up to know only implemeneted for
               the gyroradius plot. it accept 'linear' and 'log'
        @param interpolation: interpolation method for plt.imshow
        """
        # --- Initialise the plotting options
        # Color map
        if ccmap is None:
            cmap = ssplt.Gamma_II()
        else:
            cmap = ccmap
        # scale
        if scale.lower() == 'linear':
            extra_options_plot = {}
        elif scale.lower() == 'log':
            extra_options_plot = {
                'norm': colors.LogNorm()
            }
        elif scale.lower() == 'sqrt':
            extra_options_plot = {
                'norm': colors.PowerNorm(0.5)
            }
        if t is None:  # 2d plots
            # --- Gyroradius profiles
            fig1, ax1 = plt.subplots()   # Open figure and plot
            dummy = self.remap_dat['sprofy'].copy()
            if normalise:
                dummy /= self.remap_dat['sprofy'].max()
            cont = ax1.imshow(dummy, cmap=cmap,
                              vmin=min_gyr, vmax=max_gyr,
                              extent=[self.remap_dat['tframes'][0],
                                      self.remap_dat['tframes'][-1],
                                      self.remap_dat['yaxis'][0],
                                      self.remap_dat['yaxis'][-1]],
                              aspect='auto', origin='lower',
                              interpolation=interpolation,
                              **extra_options_plot)
            cbar = plt.colorbar(cont, format=cbar_tick_format)
            cbar.set_label('Counts [a.u.]')
            # Write the shot number and detector id.
            plt.text(0.05, 0.9, '#' + str(self.shot),
                     horizontalalignment='left',
                     color='w', verticalalignment='bottom',
                     transform=ax1.transAxes)
            plt.text(0.05, 0.9,
                     str(self.remap_dat['options']['pprofmin']) + 'º to '
                     + str(self.remap_dat['options']['pprofmax']) + 'º',
                     horizontalalignment='left',
                     color='w', verticalalignment='top',
                     transform=ax1.transAxes)
            plt.text(0.95, 0.9, self.diag + str(self.diag_ID),
                     horizontalalignment='right',
                     color='w', verticalalignment='bottom',
                     transform=ax1.transAxes)
            ax_params['xlabel'] = 'Time [s]'
            ax_params['ylabel'] = self.remap_dat['ylabel'] + ' [' +\
                self.remap_dat['yunits'] + ']'
            ax1 = ssplt.axis_beauty(ax1, ax_params)
            plt.tight_layout()
            # --- Pitch profiles in time
            fig2, ax2 = plt.subplots()   # Open figure and draw the image
            dummy = self.remap_dat['sprofx'].copy()
            if normalise:
                dummy /= dummy.max()
            cont = ax2.imshow(dummy, cmap=cmap,
                              vmin=min_gyr, vmax=max_gyr,
                              extent=[self.remap_dat['tframes'][0],
                                      self.remap_dat['tframes'][-1],
                                      self.remap_dat['xaxis'][0],
                                      self.remap_dat['xaxis'][-1]],
                              aspect='auto', origin='lower',
                              interpolation=interpolation,
                              ** extra_options_plot)
            cbar = plt.colorbar(cont, format=cbar_tick_format)
            cbar.set_label('Counts [a.u.]')
            # Write the shot number and detector id

            plt.text(0.05, 0.9, '#' + str(self.shot),
                     horizontalalignment='left',
                     color='w', verticalalignment='bottom',
                     transform=ax2.transAxes)
            plt.text(0.05, 0.9,
                     str(self.remap_dat['options']['rprofmin']) + 'cm to '
                     + str(self.remap_dat['options']['rprofmax']) + 'cm',
                     horizontalalignment='left',
                     color='w', verticalalignment='top',
                     transform=ax2.transAxes)
            plt.text(0.95, 0.9, self.diag + str(self.diag_ID),
                     horizontalalignment='right',
                     color='w', verticalalignment='bottom',
                     transform=ax2.transAxes)
            ax_params['xlabel'] = 'Time [s]'
            ax_params['ylabel'] = self.remap_dat['xlabel'] + ' [' +\
                self.remap_dat['xunits'] + ']'
            ax2 = ssplt.axis_beauty(ax2, ax_params)
            plt.tight_layout()
        else:  # The line plots:
            # Set the grid option for plotting
            if 'grid' not in ax_params:
                ax_params['grid'] = 'both'
            # see if the input time is an array:
            if not isinstance(t, (list, np.ndarray)):
                t = np.array([t])
            # Open the figure
            if ax is None:
                fig, ax = plt.subplots(1, 2)
            for tf in t:
                # find the frame we want to plot
                it = np.argmin(abs(self.remap_dat['tframes'] - tf))
                # Plot the gyroradius profile:
                y = self.remap_dat['sprofy'][:, it].copy()
                if normalise:
                    y /= y.max()
                ax[0].plot(self.remap_dat['yaxis'], y,
                           label='t = {0:.3f}s'.format(
                           self.remap_dat['tframes'][it]),
                           **line_params)
                # Plot the pitch profile
                y = self.remap_dat['sprofx'][:, it].copy()
                if normalise:
                    y /= y.max()
                ax[1].plot(self.remap_dat['xaxis'], y,
                           label='t = {0:.3f}s'.format(
                           self.remap_dat['tframes'][it]),
                           **line_params)
            # set the labels
            title = '#' + str(self.shot) + ' ' +\
                str(self.remap_dat['options']['pprofmin']) + 'º to ' +\
                str(self.remap_dat['options']['pprofmax']) + 'º'
            ax_params['xlabel'] = self.remap_dat['ylabel'] + ' [' +\
                self.remap_dat['yunits'] + ']'
            ax_params['ylabel'] = 'Counts [a.u.]'
            ax[0].set_title(title)
            ax[0] = ssplt.axis_beauty(ax[0], ax_params)
            ax[0].legend()
            # Set the labels
            title = '#' + str(self.shot) + ' ' +\
                str(self.remap_dat['options']['rprofmin']) + 'cm to ' +\
                str(self.remap_dat['options']['rprofmax']) + 'cm'
            ax_params['xlabel'] = self.remap_dat['xlabel'] + ' [' +\
                self.remap_dat['xunits'] + ']'
            ax_params['ylabel'] = 'Counts [a.u.]'
            ax[1].set_title(title)
            ax[1].legend()
            ax[1] = ssplt.axis_beauty(ax[1], ax_params)
            plt.tight_layout()
        plt.show()

    def plot_orientation(self, ax_params: dict = {}, line_params: dict = {}):
        """
        Plot the orientaton angles of the diagnostic in each time point

        Jose Rueda Rueda: jrrueda@us.es

        @param ax_param: axis parameters for the axis beauty routine
        """
        # Set plotting options:
        ax_options = {
            'grid': 'both'
        }
        ax_options.update(ax_params)
        line_options = {
            'linewidth': 2
        }
        line_options.update(line_params)
        # Proceed to plot
        fig, ax = plt.subplots(2, sharex=True)
        # Plot the theta angle:
        # Plot a shaded area indicating the points where only an
        # aproximate map was used, taken from the solution given here:
        # https://stackoverflow.com/questions/43233552/
        # how-do-i-use-axvfill-with-a-boolean-series
        ax[0].fill_between(self.remap_dat['tframes'], 0, 1,
                           where=self.remap_dat['existing_smaps'],
                           alpha=0.25, color='g',
                           transform=ax[0].get_xaxis_transform())
        ax[0].fill_between(self.remap_dat['tframes'], 0, 1,
                           where=~self.remap_dat['existing_smaps'],
                           alpha=0.25, color='r',
                           transform=ax[0].get_xaxis_transform())
        # Plot the line
        ax[0].plot(self.remap_dat['tframes'], self.remap_dat['theta'],
                   **line_options, label='Calculated', color='k')
        ax[0].plot(self.remap_dat['tframes'], self.remap_dat['theta_used'],
                   **line_options, label='Used', color='b')
        ax_options['ylabel'] = '$\\Theta$ [degrees]'

        ax[0] = ssplt.axis_beauty(ax[0], ax_options)
        # Plot the phi angle
        ax[1].fill_between(self.remap_dat['tframes'], 0, 1,
                           where=self.remap_dat['existing_smaps'],
                           alpha=0.25, color='g',
                           transform=ax[1].get_xaxis_transform())
        ax[1].fill_between(self.remap_dat['tframes'], 0, 1,
                           where=~self.remap_dat['existing_smaps'],
                           alpha=0.25, color='r',
                           transform=ax[1].get_xaxis_transform())
        ax[1].plot(self.remap_dat['tframes'], self.remap_dat['phi'],
                   **line_options, label='Calculated', color='k')
        ax[1].plot(self.remap_dat['tframes'], self.remap_dat['phi_used'],
                   **line_options, label='Used', color='b')

        ax_options['ylabel'] = '$\\phi$ [degrees]'
        ax_options['xlabel'] = 't [s]'
        ax[1] = ssplt.axis_beauty(ax[1], ax_options)
        plt.legend()

    def plot_number_saturated_counts(self, ax_params: dict = {},
                                     line_params: dict = {},
                                     threshold=None,
                                     ax=None):
        """
        Plot the nuber of camera pixels larger than a given threshold

        Jose Rueda: jrrueda@us.es

        @param ax_params: ax param for the axis_beauty
        @param line_params: line parameters
        @param threshold: If none, it will plot the data calculated when
        reading the camera frames (by the function self.read_frames) if it is
        a value [0,1] it willrecalculate this number
        @param ax: axis where to plot, if none, a new figure will pop-up

        @return ax: axis where the data was plotted
        """
        # Default plot parameters:
        ax_options = {
            'fontsize': 14,
            'grid': 'both',
            'xlabel': 'T [s]',
            'ylabel': '# saturated pixels',
            'yscale': 'log'
        }
        ax_options.update(ax_params)
        line_options = {
            'linewidth': 2
        }
        line_options.update(line_params)
        # Select x,y data
        x = self.exp_dat['tframes']
        if threshold is None:
            y = self.exp_dat['n_pixels_gt_threshold']
            print('Threshold was set to: ',
                  self.exp_dat['threshold_for_counts'] * 100, '%')
        else:
            max_scale_frames = 2 ** self.settings['RealBPP'] - 1
            thres = threshold * max_scale_frames
            print('Counting "saturated" pixels')
            print('The threshold is set to: ', thres, ' counts')
            number_of_frames = len(self.exp_dat['tframes'])
            n_pixels_saturated = np.zeros(number_of_frames)
            for i in range(number_of_frames):
                n_pixels_saturated[i] = \
                    (self.exp_dat['frames'][:, :, i] >= thres).sum()
            y = n_pixels_saturated.astype('int32')
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, y, **line_options)  # Plot the data
        # Plot the maximum posible (the number of pixels)
        npixels = self.imageheader['biWidth'] * self.imageheader['biHeight']
        ax.plot([x[0], x[-1]], [npixels, npixels], '--',
                **line_options)
        ax = ssplt.axis_beauty(ax, ax_options)
        return ax

    def export_remap(self, name=None):
        """
        Export the dictionary containing the remapped data

        Jose Rueda Rueda: jrrueda@us.es
        """
        # Test if the file exist:
        if name is None:
            name = os.path.join(pa.Results, str(self.shot) + '_'
                                + self.diag + str(self.diag_ID) + '_remap.nc')
            name = ssio.check_save_file(name)
            if name == '' or name == ():
                print('You canceled the export')
                return
        print('Saving results in: ', name)
        # Write the data:
        with netcdf.netcdf_file(name, 'w') as f:
            f.history = 'Done with version ' + version

            # Save shot number
            f.createDimension('number', 1)
            shot = f.createVariable('shot', 'i', ('number', ))
            shot[:] = self.shot
            shot.units = ' '
            shot.long_name = 'Shot number'

            # Save the time of the remapped frames
            f.createDimension('tframes', len(self.remap_dat['tframes']))
            time = f.createVariable('tframes', 'float64', ('tframes', ))
            time[:] = self.remap_dat['tframes']
            time.units = 's'
            time.long_name = 'Time'

            # Save the pitches
            f.createDimension('xaxis', len(self.remap_dat['xaxis']))
            xaxis = f.createVariable('xaxis', 'float64', ('xaxis', ))
            xaxis[:] = self.remap_dat['xaxis']
            xaxis.units = self.remap_dat['xunits']
            xaxis.long_name = self.remap_dat['xlabel']

            # Save the gyroradius
            f.createDimension('yaxis', len(self.remap_dat['yaxis']))
            yaxis = f.createVariable('yaxis', 'float64', ('yaxis', ))
            yaxis[:] = self.remap_dat['yaxis']
            yaxis.units = self.remap_dat['yunits']
            yaxis.long_name = self.remap_dat['ylabel']

            # Save the remapped data
            frames = f.createVariable('frames', 'float64',
                                      ('xaxis', 'yaxis', 'tframes'))
            frames[:, :, :] = self.remap_dat['frames']
            frames.units = 'Counts per axis area'
            frames.long_name = 'Remapped frames'

            # Save the modulus of the magnetic field at the FILD positon
            bfield = f.createVariable('bfield', 'float64', ('tframes', ))
            bfield[:] = self.remap_dat['bfield']
            bfield.units = 'T'
            bfield.long_name = 'Field at detector'

            # Save the temporal evolution of the profiles
            sprofx = f.createVariable('sprofx', 'float64',
                                      ('xaxis', 'tframes'))
            sprofx[:, :] = self.remap_dat['sprofx']
            sprofx.units = 'a.u.'
            sprofx.long_name = self.remap_dat['sprofxlabel']

            sprofy = f.createVariable('sprofy', 'float64',
                                      ('yaxis', 'tframes'))
            sprofy[:, :] = self.remap_dat['sprofy']
            sprofy.units = 'a.u.'
            sprofy.long_name = self.remap_dat['sprofylabel']

            # Save the calibration
            xscale = f.createVariable('xscale', 'float64', ('number', ))
            xscale[:] = self.remap_dat['options']['calibration'].xscale
            xscale.units = 'px / cm'
            xscale.long_name = 'x scale of the used calibration'

            yscale = f.createVariable('yscale', 'float64', ('number', ))
            yscale[:] = self.remap_dat['options']['calibration'].yscale
            yscale.units = 'px / cm'
            yscale.long_name = 'y scale of the used calibration'

            xshift = f.createVariable('xshift', 'float64', ('number', ))
            xshift[:] = self.remap_dat['options']['calibration'].xshift
            xshift.units = 'px / cm'
            xshift.long_name = 'x shift of the used calibration'

            yshift = f.createVariable('yshift', 'float64', ('number', ))
            yshift[:] = self.remap_dat['options']['calibration'].yshift
            yshift.units = 'px / cm'
            yshift.long_name = 'y shift of the used calibration'

            deg = f.createVariable('deg', 'float64', ('number', ))
            deg[:] = self.remap_dat['options']['calibration'].deg
            deg.units = 'degrees'
            deg.long_name = 'alpha angle the used calibration'

            # Noise subtraction
            if 't1_noise' in self.exp_dat.keys():
                t1_noise = f.createVariable('t1_noise', 'float64', ('number',))
                t1_noise[:] = self.exp_dat['t1_noise']
                t1_noise.units = 's'
                t1_noise.long_name = 't1 for noise subtraction'

                t2_noise = f.createVariable('t2_noise', 'float64', ('number',))
                t2_noise[:] = self.exp_dat['t2_noise']
                t2_noise.units = 's'
                t2_noise.long_name = 't2 for noise subtraction'

            if 'frame_noise' in self.exp_dat.keys():
                nframex, nframey = self.exp_dat['frame_noise'].shape
                f.createDimension('nx', nframex)
                f.createDimension('ny', nframey)
                frame_noise = f.createVariable('frame_noise', 'float64',
                                               ('nx', 'ny',))
                frame_noise[:] = self.exp_dat['frame_noise']
                frame_noise.units = 'counts'
                frame_noise.long_name = 'frame used for noise subtraction'

            # Save the saturated number of pixels
            n_pixels_gt_threshold = f.createVariable('n_pixels_gt_threshold',
                                                     'int32', ('tframes', ))
            n_pixels_gt_threshold[:] = self.exp_dat['n_pixels_gt_threshold']
            n_pixels_gt_threshold.units = ''
            n_pixels_gt_threshold.long_name = \
                'Number of pixels with more counts than threshold'

            threshold_for_counts = f.createVariable('threshold_for_counts',
                                                    'float64', ('number', ))
            threshold_for_counts[:] = \
                self.exp_dat['threshold_for_counts']
            threshold_for_counts.units = ''
            threshold_for_counts.long_name = \
                'Threshold for n_pixels_gt_threshold'
            # Save the specific FILD variables
            if self.diag == 'FILD':
                # Detector orientation
                theta = f.createVariable('theta', 'float64', ('tframes', ))
                theta[:] = self.remap_dat['theta']
                theta.units = '{}^o'
                theta.long_name = 'theta'

                phi = f.createVariable('phi', 'float64', ('tframes', ))
                phi[:] = self.remap_dat['phi']
                phi.units = '{}^o'
                phi.long_name = 'phi'

                # Options used for the remapping
                rmin = f.createVariable('rmin', 'float64', ('number', ))
                rmin[:] = self.remap_dat['options']['rmin']
                rmin.units = 'cm'
                rmin.long_name = 'Minimum r_l for the remap'

                rmax = f.createVariable('rmax', 'float64', ('number', ))
                rmax[:] = self.remap_dat['options']['rmax']
                rmax.units = 'cm'
                rmax.long_name = 'Maximum r_l for the remap'

                dr = f.createVariable('dr', 'float64', ('number', ))
                dr[:] = self.remap_dat['options']['dr']
                dr.units = 'cm'
                dr.long_name = 'dr_l for the remap'

                dp = f.createVariable('dp', 'float64', ('number', ))
                dp[:] = self.remap_dat['options']['dp']
                dp.units = '{}^o'
                dp.long_name = 'dp for the remap'

                pmin = f.createVariable('pmin', 'float64', ('number', ))
                pmin[:] = self.remap_dat['options']['pmin']
                pmin.units = '{}^o'
                pmin.long_name = 'Minimum pitch for the remap'

                pmax = f.createVariable('pmax', 'float64', ('number', ))
                pmax[:] = self.remap_dat['options']['pmax']
                pmax.units = '{}^o'
                pmax.long_name = 'Maximum pitch for the remap'

                pprofmin = f.createVariable('pprofmin', 'float64', ('number',))
                pprofmin[:] = self.remap_dat['options']['pprofmin']
                pprofmin.units = '{}^o'
                pprofmin.long_name = 'Minimum pitch to integrate the remap'

                pprofmax = f.createVariable('pprofmax', 'float64', ('number',))
                pprofmax[:] = self.remap_dat['options']['pprofmax']
                pprofmax.units = '{}^o'
                pprofmax.long_name = 'Maximum pitch to integrate the remap'

                rprofmin = f.createVariable('rprofmin', 'float64', ('number',))
                rprofmin[:] = self.remap_dat['options']['rprofmin']
                rprofmin.units = 'cm'
                rprofmin.long_name = 'Minimum r_l to integrate the remap'

                rprofmax = f.createVariable('rprofmax', 'float64', ('number',))
                rprofmax[:] = self.remap_dat['options']['rprofmax']
                rprofmax.units = 'cm'
                rprofmax.long_name = 'Maximum r_l to integrate the remap'

                rfild = f.createVariable('rfild', 'float64', ('number',))
                rfild[:] = self.remap_dat['options']['rfild']
                rfild.units = 'm'
                rfild.long_name = 'R FILD position'

                zfild = f.createVariable('zfild', 'float64', ('number',))
                zfild[:] = self.remap_dat['options']['zfild']
                zfild.units = 'm'
                zfild.long_name = 'z FILD position'

                alpha = f.createVariable('alpha', 'float64', ('number',))
                alpha[:] = self.remap_dat['options']['alpha']
                alpha.units = '{}^o'
                alpha.long_name = 'alpha orientation'

                beta = f.createVariable('beta', 'float64', ('number',))
                beta[:] = self.remap_dat['options']['beta']
                beta.units = '{}^o'
                beta.long_name = 'beta orientation'

            # if present, save the bit depth used to save the video
            try:
                a = self.settings['RealBPP']
                bits = f.createVariable('RealBPP', 'i', ('number',))
                bits[:] = a
                bits.units = ' '
                bits.long_name = 'Bits used in the camera'
            except KeyError:
                print('Bits info not present in the video object')
        return
