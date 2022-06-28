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
from scipy.io import netcdf                # To export remap data
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
            ## Magnetic field at FILD head
            self.BField = None
            ## Particular options for the magnetic field calculation
            self.BFieldOptions = Boptions
            ## Orientation angles
            self.Bangles = None

            if verbose:
                if self.FILDoperatorComment is not None:
                    print('--- FILD Operator comment:')
                    print(self.FILDoperatorComment[0])
                if self.overheating is not None:
                    print('--- Overheating level: %i' % self.overheating)
        else:
            FIV.__init__(self, empty=empty)

    # def _getB(self, extra_options: dict = {}, use_average: bool = False):
    #     """
    #     Get the magnetic field in the FILD position
    #
    #     Jose Rueda - jrrueda@us.es
    #
    #     @param extra_options: Extra options to be passed to the magnetic field
    #         calculation. Ideal place to insert all your machine dependent stuff
    #     @param use_average: flag to use the timebase of the average frames or
    #         the experimental frames
    #
    #     Note: It will overwrite the content of self.Bfield
    #     """
    #     if self.position is None:
    #         raise Exception('FILD position not know')
    #     # Get the proper timebase
    #     if use_average:
    #         time = self.avg_dat['tframes']
    #     else:
    #         time = self.exp_dat['tframes']
    #     # Calculate the magnetic field
    #     print('Calculating magnetic field (this might take a while): ')
    #     br, bz, bt, bp =\
    #         ssdat.get_mag_field(self.shot, self.position['R'],
    #                             self.position['z'],
    #                             time=time,
    #                             **extra_options)
    #     self.BField = {
    #         'BR': np.array(br).squeeze(),
    #         'Bz': np.array(bz).squeeze(),
    #         'Bt': np.array(bt).squeeze(),
    #     }
    #     self.BField['B'] = np.sqrt(self.BField['Bt']**2 + self.BField['BR']**2
    #                                + self.BField['Bz']**2)

    def _getBangles(self):
        """Get the orientation of the field respec to the head"""
        if self.orientation is None:
            raise Exception('FILD orientation not know')
        phi, theta = \
            ssFILDSIM.calculate_fild_orientation(
                self.BField['BR'], self.BField['Bz'], self.BField['Bt'],
                self.orientation['alpha'], self.orientation['beta']
            )
        self.Bangles = {'phi': phi, 'theta': theta}

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
        # Check if the user want to use the average
        if 'use_average' in options.keys():
            use_avg = options['use_average']
            nt = len(self.avg_dat['tframes'])
        else:
            use_avg = False
            nt = len(self.exp_dat['tframes'])

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
        self.remap_dat, opt = \
            ssmap.remapAllLoadedFrames(self, **options)
        self.remap_dat['options'] = opt

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
                phi = self.Bangles['phi']
                theta = self.Bangles['theta']
                time = 'all'
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

            # Create the dimensions for the variables:
            f.createDimension('number', 1)  # For numbers
            f.createDimension('tframes', len(self.remap_dat['tframes']))  # t
            f.createDimension('xaxis', len(self.remap_dat['xaxis']))  # pitch
            f.createDimension('yaxis', len(self.remap_dat['yaxis']))  # rl

            # Suite version
            ver = version.split('.')
            va = int(ver[0])
            vb = int(ver[1])
            vc = int(ver[2])
            versionIDa = f.createVariable('versionIDa', 'i', ('number', ))
            versionIDa[:] = va
            versionIDa.units = ' '
            versionIDa.long_name = 'Version ID a'
            versionIDb = f.createVariable('versionIDb', 'i', ('number', ))
            versionIDb[:] = vb
            versionIDb.units = ' '
            versionIDb.long_name = 'Version ID b'
            versionIDc = f.createVariable('versionIDc', 'i', ('number', ))
            versionIDc[:] = vc
            versionIDc.units = ' '
            versionIDc.long_name = 'Version ID c'

            # Save shot number
            shot = f.createVariable('shot', 'i', ('number', ))
            shot[:] = self.shot
            shot.units = ' '
            shot.long_name = 'Shot number'

            # Save FILD number
            diag_ID = f.createVariable('diag_ID', 'i', ('number', ))
            diag_ID[:] = self.diag_ID
            diag_ID.units = ' '
            diag_ID.long_name = 'FILD number'

            # # Save FILD geometry
            # geom_ID = f.createVariable('geom_ID', 's', )
            # geom_ID[:] = self.geometryID
            # geom_ID.units = ' '
            # geom_ID.long_name = 'FILD geomID'

            # Save the flag which indicate if the remap was from average or
            # real frames
            avg_flag = f.createVariable('use_average', 'i', ('number', ))
            avg_flag[:] = int(self.remap_dat['options']['use_average'])
            avg_flag.units = ' '
            avg_flag.long_name = 'Flag of frames used to remap (1=avg, 0=exp)'

            # Save the time of the remapped frames
            time = f.createVariable('tframes', 'float64', ('tframes', ))
            time[:] = self.remap_dat['tframes']
            time.units = 's'
            time.long_name = 'Time'

            # Save the pitches
            xaxis = f.createVariable('xaxis', 'float64', ('xaxis', ))
            xaxis[:] = self.remap_dat['xaxis']
            xaxis.units = self.remap_dat['xunits']
            xaxis.long_name = self.remap_dat['xlabel']

            # Save the gyroradius
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
            bfield[:] = self.BField['B']
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
            xscale[:] = self.CameraCalibration.xscale
            xscale.units = 'px / cm'
            xscale.long_name = 'x scale of the used calibration'

            yscale = f.createVariable('yscale', 'float64', ('number', ))
            yscale[:] = self.CameraCalibration.yscale
            yscale.units = 'px / cm'
            yscale.long_name = 'y scale of the used calibration'

            xshift = f.createVariable('xshift', 'float64', ('number', ))
            xshift[:] = self.CameraCalibration.xshift
            xshift.units = 'px / cm'
            xshift.long_name = 'x shift of the used calibration'

            yshift = f.createVariable('yshift', 'float64', ('number', ))
            yshift[:] = self.CameraCalibration.yshift
            yshift.units = 'px / cm'
            yshift.long_name = 'y shift of the used calibration'

            deg = f.createVariable('deg', 'float64', ('number', ))
            deg[:] = self.CameraCalibration.deg
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
            # Detector orientation
            theta = f.createVariable('theta', 'float64', ('tframes', ))
            theta[:] = self.Bangles['theta']
            theta.units = '$\\degree$'
            theta.long_name = 'theta'

            theta_used = \
                f.createVariable('theta_used', 'float64', ('tframes', ))
            theta_used[:] = self.remap_dat['theta_used']
            theta_used.units = '$\\degree$'
            theta_used.long_name = 'theta used'

            phi = f.createVariable('phi', 'float64', ('tframes', ))
            phi[:] = self.Bangles['phi']
            phi.units = '$\\degree$'
            phi.long_name = 'phi'

            phi_used = f.createVariable('phi', 'float64', ('tframes', ))
            phi_used[:] = self.remap_dat['phi_used']
            phi_used.units = '$\\degree$'
            phi_used.long_name = 'phi used'

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
            dp.units = '$\\degree$'
            dp.long_name = 'dp for the remap'

            pmin = f.createVariable('pmin', 'float64', ('number', ))
            pmin[:] = self.remap_dat['options']['pmin']
            pmin.units = '$\\degree$'
            pmin.long_name = 'Minimum pitch for the remap'

            pmax = f.createVariable('pmax', 'float64', ('number', ))
            pmax[:] = self.remap_dat['options']['pmax']
            pmax.units = '$\\degree$'
            pmax.long_name = 'Maximum pitch for the remap'

            pprofmin = f.createVariable('pprofmin', 'float64', ('number',))
            pprofmin[:] = self.remap_dat['options']['pprofmin']
            pprofmin.units = '$\\degree$'
            pprofmin.long_name = 'Minimum pitch to integrate the remap'

            pprofmax = f.createVariable('pprofmax', 'float64', ('number',))
            pprofmax[:] = self.remap_dat['options']['pprofmax']
            pprofmax.units = '$\\degree$'
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
            rfild[:] = self.position['R']
            rfild.units = 'm'
            rfild.long_name = 'R FILD position'

            zfild = f.createVariable('zfild', 'float64', ('number',))
            zfild[:] = self.position['z']
            zfild.units = 'm'
            zfild.long_name = 'z FILD position'

            phifild = f.createVariable('phifild', 'float64', ('number',))
            phifild[:] = self.position['phi']
            phifild.units = 'm'
            phifild.long_name = 'phi FILD position'

            alpha = f.createVariable('alpha', 'float64', ('number',))
            alpha[:] = self.orientation['alpha']
            alpha.units = '$\\degree$'
            alpha.long_name = 'alpha orientation'

            beta = f.createVariable('beta', 'float64', ('number',))
            beta[:] = self.orientation['beta']
            beta.units = '$\\degree$'
            beta.long_name = 'beta orientation'

            gamma = f.createVariable('gamma', 'float64', ('number',))
            gamma[:] = self.orientation['gamma']
            beta.units = '$\\degree$'
            beta.long_name = 'gamma orientation'

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
