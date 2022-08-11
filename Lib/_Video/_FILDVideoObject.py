"""
FILD Video object

This module contains the child class from BVO. This child is designed to handle
FILD experimental data. It allows you to load the video files as well as call
the routines to calculate the remap

Jose Rueda Rueda: jrrueda@us.es
Lina Velarde Gallardo: lvelarde@us.es
"""
from Lib._Video._FILD_INPA_Parent import FIV
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tkinter as tk                       # To open UI windows
import Lib._Plotting as ssplt
import Lib._Mapping as ssmap
import Lib._Paths as p
import Lib._IO as ssio
import Lib._GUIs as ssGUI             # For GUI elements
import Lib.LibData as ssdat
import Lib.SimulationCodes.FILDSIM as ssFILDSIM
from Lib.version_suite import version
from Lib._Machine import machine
import Lib._Video._AuxFunctions as _aux
import xarray as xr
pa = p.Path(machine)
del p


class FILDVideo(FIV):
    """
    Video class for the FILD diagnostic.

    Inside there are the necesary routines to load and remapp a FILD video

    Jose Rueda: jrrueda@us.es

    Public Methods (it also contains all present in the BVO):
        - remap_loaded_frames: Remap the loaded frames
        - integrate_remap: integrate the remap in the desired region of rl and
            pitch to get a time trace
        - plot_frame: plot one of the experimental (or averaged) frames
        - plot_frame_remap: plot one of the remapped frames
        - calculateBangles: calculate the orientation of the magnetic field
            in the head reference system
        - GUI_frames_and_remap: Shows a GUI with the experimental and remap
            data. @Todo. Allows this to show the average
        - plot_profiles_in_time: Plot the evolution of the rl or pitch profiles
            It can also plot the profiles for an specifit time
        - plot_orientation: plot the orientation of the magnetic field in the
            head reference system. If this is executed after the remap, it plot
            some shaded areas indicated which angles were found in the database
        - export_remap: Export remap
    """

    def __init__(self, file: str = None, shot: int = None,
                 diag_ID: int = 1, empty: bool = False,
                 logbookOptions: dict = {}, Boptions: dict = {},
                 verbose: bool = True):
        """
        Initialise the class

        There are several ways of initialising this object:
            1: Give shot number and fild number (diag_ID). The code will
              predict the filename to load. (not recomended for old AUG shots,
              where this is a mess). This option is the prefered one
            2: Give the file (or folder with pngs). The code will try to infer
              the shot number from the file name. the FILD number should be
              necessarily given
            3: Give the file and the shot number. The code will ignore the shot
              to load the files and try to load the file. But the shot number
              will not be guessed from the file name. This given shot number
              will be used for the magnetic field (remap calculation)
            +: For all the cases, we need the position, orientation, and
              collimator geometries, this extracted from the logbook. If no
              input is given for FILDlogbook, the code will use the default.

        @param file: file or folder (see above)
        @param shot: Shot number, if is not given, the program will look for it
            in the name of the loaded file (see above)
        @param diag_ID: manipulator number for FILD
        @param empty: Initialise the video object as empty. This flag is added
            to load data from a remap file
        @param logbookOptions: dictionary containing the options to start the
            FILDlogbook. Can be machine dependent
        @param Boptions: dictionary containing the options to load the magnetic
            field, can be machine dependent. Notice that the shot number and
            needed time will be collected from the video object. If you provide
            them also here, the code will fail. Same with R and z
        @param verbose: flag to print information (overheating + comments)
        """
        if not empty:
            # Guess the filename:
            if file is None:
                file = ssdat.guessFILDfilename(shot, diag_ID)
            if shot is None:
                shot = _aux.guess_shot(file, ssdat.shot_number_length)
            # Initialise the logbook
            FILDlogbook = ssdat.FILD_logbook(**logbookOptions)  # Logbook
            try:
                AdqFreq = FILDlogbook.getAdqFreq(shot, diag_ID)
                t_trig = FILDlogbook.gettTrig(shot, diag_ID)
            except AttributeError:
                AdqFreq = None
                t_trig = None
            # initialise the parent class
            FIV.__init__(self, file=file, shot=shot, empty=empty,
                         adfreq=AdqFreq, t_trig=t_trig)
            ## Diagnostic used to record the data
            self.diag = 'FILD'
            ## Diagnostic ID (FILD manipulator number)
            self.diag_ID = diag_ID
            if shot is not None:
                self.position = FILDlogbook.getPosition(shot, diag_ID)
                self.orientation = \
                    FILDlogbook.getOrientation(shot, diag_ID)
                self.geometryID = FILDlogbook.getGeomID(shot, diag_ID)
                self.CameraCalibration = \
                    FILDlogbook.getCameraCalibration(shot, diag_ID)
                try:
                    self.FILDoperatorComment =\
                        FILDlogbook.getComment(self.shot)
                    if len(self.FILDoperatorComment) == 0:
                        self.FILDoperatorComment = ['']
                    self.overheating = \
                        FILDlogbook.getOverheating(self.shot, diag_ID)
                except AttributeError:
                    self.FILDoperatorComment = None
                    self.overheating = None
            else:
                self.position = None
                self.orientation = None
                self.geometryID = None
                self.CameraCalibration = None
                self.FILDoperatorComment = None
                self.overheating = None
                print('Shot not provided, you need to define FILDposition')
                print('You need to define FILDorientation')
                print('You need to define FILDgeometry')
                print('You need to give the camera parameters')
           
            # Particular options for the magnetic field calculation
            self.BFieldOptions = Boptions

            if verbose:
                if self.FILDoperatorComment is not None:
                    print('--- FILD Operator comment:')
                    print(self.FILDoperatorComment[0])
                if self.overheating is not None:
                    print('--- Overheating level: %i' % self.overheating)
        else:
            FIV.__init__(self, empty=empty)

    def _getBangles(self):
        """Get the orientation of the field respec to the head"""
        if self.orientation is None:
            raise Exception('FILD orientation not know')
        phi, theta = \
            ssFILDSIM.calculate_fild_orientation(
                self.BField['BR'], self.BField['Bz'], self.BField['Bt'],
                self.orientation['alpha'], self.orientation['beta']
            )
        self.Bangles = xr.Dataset()
        self.Bangles['phi'] = \
            xr.DataArray(phi, dims=('t'), coords={'t': self.BField['t'].values})
        self.Bangles['theta'] = xr.DataArray(theta, dims=('t'))

    def _checkStrikeMapDatabase():
        pass

    def remap_loaded_frames(self, options: dict = {}):
        """
        Remap all loaded frames in the video object

        Jose Rueda Rueda: jrrueda@us.es

        @param    options: Options for the remapping routine. See
            remapAllLoadedFrames in the LibMap package for a full description

        @return:  write in the object the remap_dat dictionary containing with:
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
        # Check if there was some data
        if self.remap_dat is not None:
            self.remap_dat = None
        # Check if the user want to use the average
        if 'use_average' in options.keys():
            use_avg = options['use_average']
            nt = self.avg_dat['t'].size
        else:
            use_avg = False
            nt = self.exp_dat['t'].size

        # Check if the magnetic field and the angles are ready, only if the map
        # is not given
        if 'map' not in options.keys():
            if self.BField is None:
                self._getB(self.BFieldOptions, use_average=use_avg)
            if self.Bangles is None:
                self._getBangles()
            # Check if we need to recaluculate them because they do not
            # have the proper length (ie they were calculated for the exp_dat
            # not the average)
            if self.BField['BR'].size != nt:
                print('Need to recalculate the magnetic field')
                self._getB(self.BFieldOptions, use_average=use_avg)
            if self.Bangles['phi'].size != nt:
                self._getBangles()
        self.remap_dat = ssmap.remapAllLoadedFrames(self, **options)
        # Calculate the integral of the remap
        ouput = self.integrate_remap(xmin=self.remap_dat['x'].values[0],
                                     xmax=self.remap_dat['x'].values[-1],
                                     ymin=self.remap_dat['y'].values[0],
                                     ymax=self.remap_dat['y'].values[-1])
        self.remap_dat = xr.merge([self.remap_dat, ouput])

    def calculateBangles(self, t=None, verbose: bool = True):
        """
        Find the orientation of FILD for a given time.

        José Rueda: jrrueda@us.es

        @param t: time point where we want the angles [s]. if None the
            orientation will be calculated for all time points
        @param verbose: flag to print information or not
        @param R: R coordinate of the detector (in meters) for B calculation
        @param z: z coordinate of the detector (in meters) for B calculation

        @return phi: phi angle [º]
        @return theta: theta angle [º]
        """
        if self.remap_dat is None:
            if self.orientation is None:
                raise Exception('FILD orientation not know')

            alpha = self.orientation['alpha']
            beta = self.orientation['beta']
            print('Remap not done, calculating angles')

            if t is not None:
                br, bz, bt, bp =\
                    ssdat.get_mag_field(self.shot, self.position['R'],
                                        self.position['z'], time=t)

                phi, theta = \
                    ssFILDSIM.calculate_fild_orientation(br, bz, bt,
                                                         alpha, beta)
                time = t
            else:
                if self.BField is None:
                    self._getB()
                self._getBangles()
                phi = self.Bangles['phi'].values
                theta = self.Bangles['theta'].values
                time = 'all'
        else:
            tmin = self.remap_dat['t'].values[0]
            tmax = self.remap_dat['t'].values[-1]
            if t < tmin or t > tmax:
                raise Exception('Time not present in the remap')
            else:
                it = np.argmin(abs(self.remap_dat['t'].values - t))
                theta = self.remap_dat['theta'].values[it]
                phi = self.remap_dat['phi'].values[it]
                time = self.remap_dat['t'].values[it]
        if verbose:
            # I include these 'np.array' in order to be compatible with the
            # case of just one time point and multiple ones. It is not the most
            # elegant way to proceed, but it works ;)
            print('Requested time:', t)
            if self.remap_dat is not None:
                print('Found time: ', time)
            print('Average theta:', np.array(theta).mean())
            print('Average phi:', np.array(phi).mean())
        return phi, theta

    def GUI_frames_and_remap(self):
        """Small GUI to explore camera and remapped frames"""
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print(text)
        root = tk.Tk()
        root.resizable(height=None, width=None)
        ssGUI.ApplicationShowVidRemap(root, self.exp_dat, self.remap_dat,
                                      self.CameraCalibration,
                                      self.geometryID)
        root.mainloop()
        root.destroy()

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

