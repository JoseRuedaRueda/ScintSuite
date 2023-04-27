"""
FILD Video object

This module contains the child class from BVO. This child is designed to handle
FILD experimental data. It allows you to load the video files as well as call
the routines to calculate the remap

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 0.8.0
"""
import os
import math
import logging
import numpy as np
import xarray as xr
import tkinter as tk
import Lib._Paths as p
import Lib._IO as ssio
import Lib._GUIs as ssGUI             # For GUI elements
import Lib.LibData as ssdat
import Lib.errors as errors
import Lib._Mapping as ssmap
import Lib._Plotting as ssplt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Lib._Video._AuxFunctions as _aux
from Lib._Machine import machine
from Lib.version_suite import version
from Lib._Video._FILD_INPA_Parent import FIV
from Lib._Scintillator import Scintillator


# -----------------------------------------------------------------------------
# %% Auxiliary objects
# -----------------------------------------------------------------------------
logger = logging.getLogger('ScintSuite.INPAVideo')
pa = p.Path(machine)
del p


# -----------------------------------------------------------------------------
# %% INPA video class
class INPAVideo(FIV):
    """
    Video class for the INPA diagnostic.

    Inside there are the necesary routines to load and remapp a FILD video

    Jose Rueda: jrrueda@us.es

    Public Methods:
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

    :param  file: file or folder (see above)
    :param  shot: Shot number, if is not given, the program will look for it
        in the name of the loaded file (see above)
    :param  diag_ID: manipulator number for FILD
    :param  empty: Initialise the video object as empty. This flag is added
        to load data from a remap file
    :param  logbookOptions: dictionary containing the options to start the
        FILDlogbook. Can be machine dependent
    :param  Boptions: dictionary containing the options to load the magnetic
        field, can be machine dependent. Notice that the shot number and
        needed time will be collected from the video object. If you provide
        them also here, the code will fail. Same with R and z
    :param  verbose: flag to print logbook comments
    :param  loadPlasmaData: Flag to load plasma data

    :Example of using this object:
        1:
            Give shot number and INPA number (diag_ID, by default 1, by now
            there is only one INPA in AUG). The code will predict the filename
            to load. This option is the prefered one

        >>> import Lib as ss
        >>> vid = ss.vid.INPAVideo(shot=40412)

        2:
            Give the cin file (or folder with pngs). The code will try to infer
            the shot number from the file name.

        >>> import Lib as ss
        >>> vid = ss.vid.INPAVideo('MyPathTothefile')

        3:
            Give the file and the shot number. The code will ignore the shot
            to load the files and try to load the file. But the shot number
            will not be guessed from the file name. This given shot number
            will be used for the magnetic field (remap calculation)

        >>> import Lib as ss
        >>> vid = ss.vid.INPAVideo('MypathTofile', shot=40412)
    """

    def __init__(self, file: str = None, shot: int = None,
                 diag_ID: int = 1, empty: bool = False,
                 logbookOptions: dict = {}, Boptions: dict = {},
                 verbose: bool = True, loadPlasmaData: bool = True,
                 YOLO: bool = True):
        """Initialise the class."""
        if not empty:
            # Guess the filename:
            if file is None:
                file = ssdat.guessINPAfilename(shot, diag_ID)
            if shot is None:
                shot = _aux.guess_shot(file, ssdat.shot_number_length)
            # initialise the parent class
            FIV.__init__(self, file=file, shot=shot, empty=empty, YOLO=YOLO)
            ## Diagnostic used to record the data
            self.diag = 'INPA'
            ## Diagnostic ID (FILD manipulator number)
            self.diag_ID = diag_ID
            # Initialise the logbook
            self.logbookOptions = logbookOptions
            INPAlogbook = ssdat.INPA_logbook(**logbookOptions)  # Logbook
            if shot is not None:
                self.position, self.orientation = \
                    INPAlogbook.getPositionOrientation(shot, diag_ID)
                self.geometryID = INPAlogbook.getGeomID(shot, diag_ID)
                self.CameraCalibration = \
                    INPAlogbook.getCameraCalibration(shot, diag_ID)
                self.operatorComment = INPAlogbook.getComment(shot)
            else:
                self.position = None
                self.orientation = None
                self.geometryID = None
                self.CameraCalibration = None
                self.operatorComment = None
                print('Shot not provided, you need to define INPAgeometry')
                print('You need to define INPApositionOrientation')
                print('You need to give the camera calibration parameters')
            # Try to load the scintillator plate
            if self.geometryID is not None:
                platename = os.path.join(pa.ScintSuite, 'Data', 'Plates',
                                         'INPA', machine,
                                         self.geometryID,
                                         'Scintillator.pl')
                if os.path.isfile(platename):
                    self.scintillator = Scintillator(file=platename)
                    self.scintillator.calculate_pixel_coordinates(
                            self.CameraCalibration)
                    self.ROIscintillator = self.scintillator.get_roi()
            else:
                self.scintillator = None
                self.ROIscintillator = None
            # Save the options for the magnetic field
            self.BFieldOptions = Boptions
            # Load the plasma data
            if loadPlasmaData and self.shot is not None:
                self._getNe()
                self._getNBIpower()
            if verbose:
                if self.operatorComment is not None:
                    print('--- INPA Operator comment:')
                    print(self.operatorComment[0][0])
        else:
            FIV.__init__(self, empty=empty)

    # --------------------------------------------------------------------------
    # ---- Get shot / magnetic data
    # --------------------------------------------------------------------------
    def _getBangles(self):
        """Get the orientation of the field respec to the head."""
        s1_projection = \
            (self.orientation['s1rzt'][0] * self.BField['BR']
             + self.orientation['s1rzt'][1] * self.BField['Bz']
             + self.orientation['s1rzt'][2] * self.BField['Bt']) \
            / self.BField['B']

        s2_projection = \
            (self.orientation['s2rzt'][0] * self.BField['BR']
             + self.orientation['s2rzt'][1] * self.BField['Bz']
             + self.orientation['s2rzt'][2] * self.BField['Bt'])\
            / self.BField['B']

        s3_projection = \
            (self.orientation['s3rzt'][0] * self.BField['BR']
             + self.orientation['s3rzt'][1] * self.BField['Bz']
             + self.orientation['s3rzt'][2] * self.BField['Bt'])\
            / self.BField['B']
        theta = np.arccos(s3_projection) * 180.0 / math.pi
        phi = np.arctan2(s2_projection, s1_projection) * 180.0 / math.pi
        # For AUG shots, this angle is around 180 degrees, so phi is changing
        # around that value and as arctan2 return between -pi and pi, the break
        # is there and is nasty to plot. So I would just add 360 for the phi<0
        # If you are in other machine and your normal phi is around 0, you will
        # get the jump, sorry, life is hard, AUG was first :)
        phi[phi < 0] += 360.0
        # Save the angles in place
        self.Bangles = xr.Dataset()
        self.Bangles['phi'] = \
            xr.DataArray(phi, dims=('t'), coords={'t': self.BField['t'].values})
        self.Bangles['theta'] = xr.DataArray(theta, dims=('t'))

    def _checkStrikeMapDatabase(self,):
        pass

    # --------------------------------------------------------------------------
    # ---- Remap
    # --------------------------------------------------------------------------
    def remap_loaded_frames(self, options: dict = {}):
        """
        Remap all loaded frames in the video object

        Jose Rueda Rueda: jrrueda@us.es

        :param     options: Options for the remapping routine. See
            remapAllLoadedFrames in the LibMap package for a full description

        :return:  write in the object the remap_dat dictionary containing with:
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
        # Destroy the previous dataset, if there was some
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
                logger.info('Need to recalculate the magnetic field')
                self._getB(self.BFieldOptions, use_average=use_avg)
            if self.Bangles['phi'].size != nt:
                self._getBangles()
        self.remap_dat = \
            ssmap.remapAllLoadedFramesINPA(self, **options)
        # Calculate the integral of the remap
        ouput = self.integrate_remap(xmin=self.remap_dat['x'].values[0],
                                     xmax=self.remap_dat['x'].values[-1],
                                     ymin=self.remap_dat['y'].values[0],
                                     ymax=self.remap_dat['y'].values[-1])
        self.remap_dat = xr.merge([self.remap_dat, ouput])

    def calculateBangles(self, t='all', verbose: bool = True):
        """
        Find the orientation of INPA for a given time.

        José Rueda: jrrueda@us.es

        :param  t: time point where we want the angles[s]. It can be 'all' in
        that case, the orientation will be calculated for all time points
        :param  verbose: flag to print information or not
        :return theta: theta angle[º]
        :return phi: phi angle[º]

        :Example of use:

        >>> # Prepare the video
        >>> import Lib as ss
        >>> vid = ss.vid.INPAVideo(shot=40412)
        >>> vid.read_frame()
        >>> # Now yes, calculate the angles
        >>> phi, theta = vid.calculateBangles(t=2.5)
        """
        if self.remap_dat is None:
            if t == 'all':
                if self.BField is None:
                    self._getB(extra_options=self.BFieldOptions)
                self._getBangles()
                phi = self.Bangles['phi']
                theta = self.Bangles['theta']
                time = 'all'
            else:
                br, bz, bt, bp = ssdat.get_mag_field(
                    self.shot, self.position['R_scintillator'],
                    self.position['z_scintillator'], time=t,
                    **self.BFieldOptions)
                BR = np.array(br).squeeze()
                Bz = np.array(bz).squeeze()
                Bt = np.array(bt).squeeze()
                B = np.sqrt(BR**2 + Bz**2 + Bt**2)
                s1_projection = \
                    (self.orientation['s1rzt'][0] * BR
                     + self.orientation['s1rzt'][1] * Bz
                     + self.orientation['s1rzt'][2] * Bt) / B

                s2_projection = \
                    (self.orientation['s2rzt'][0] * BR
                     + self.orientation['s2rzt'][1] * Bz
                     + self.orientation['s2rzt'][2] * Bt) / B

                s3_projection = \
                    (self.orientation['s3rzt'][0] * BR
                     + self.orientation['s3rzt'][1] * Bz
                     + self.orientation['s3rzt'][2] * Bt) / B

                theta = np.arccos(s3_projection) * 180.0 / math.pi
                phi = np.arctan2(s2_projection, s1_projection) * 180.0 / math.pi
                if phi < 0:
                    phi += 360.0
                time = t
        else:
            tmin = self.remap_dat['t'][0]
            tmax = self.remap_dat['t'][-1]
            if t < tmin or t > tmax:
                raise Exception('Time not present in the remap')
            else:
                it = np.argmin(abs(self.remap_dat['t'].values - t))
                theta = self.remap_dat['theta'][it]
                phi = self.remap_dat['phi'][it]
                time = self.remap_dat['t'][it]
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

    # --------------------------------------------------------------------------
    # ---- GUIs/plot
    # --------------------------------------------------------------------------
    def GUI_frames_and_remap(self):
        """GUI to explore camera and remapped frames."""
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happened, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print(text)
        root = tk.Tk()
        root.resizable(height=None, width=None)
        ssGUI.ApplicationShowVidRemap(root, self.exp_dat, self.remap_dat,
                                      self.CameraCalibration,
                                      self.geometryID)
        root.mainloop()
        root.destroy()

    def GUI_frames_and_traces(self):
        """GUI to explore camera and remapped frames."""
        text = 'Press TAB until the time slider is highlighted in red.' \
               + ' Once that happened, you can move the time with the arrows' \
               + ' of the keyboard, frame by frame'
        print(text)
        root = tk.Tk()
        root.resizable(height=None, width=None)
        ssGUI.ApplicationShowVidAndTraces(root, self)
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

        :param  ccmap: colormap to be used, if none, Gamma_II will be used
        :param  ax_params: params for the function axis beauty plt. Notice that,
            the xlabel for the rl and pitch are hardwritten in the inside
            (sorry, this was to avoid to have 2 ax_params inputs)

        :param  t: time, if present, just a line plot for the profiles for that
        time will be used

        :param  nlev: Number of levels for the contourf plots(deprecated, as we
            now use imshow to plot the data)
        :param  cbar_tick_format: format for the colorbar ticks
        :param  max_gyr: maximum value for colorbar plot in gyroradius
        :param  min_gyr: minimum value for colorbar plot in gyroradius
        :param  max_pitch: maximum value for colorbar plot in pitch
        :param  min_pitch: minimum value for colorbar plot in pitch
        :param  scale: Color scale to plot, up to know only implemeneted for
               the gyroradius plot. it accept 'linear' and 'log'

        :param  interpolation: interpolation method for plt.imshow
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

    # --------------------------------------------------------------------------
    # ---- Camera calibration
    # --------------------------------------------------------------------------
    def reloadCameraCalibration(self):
        """
        Reload the camera calibration.

        Useful if you are iterating wuth the calibration file

        Jose Rueda: jrrueda@us.es
        """
        INPAlogbook = ssdat.INPA_logbook(**self.logbookOptions)  # Logbook
        self.CameraCalibration = \
            INPAlogbook.getCameraCalibration(self.shot, self.diag_ID)

    # --------------------------------------------------------------------------
    # ---- Pitch profile
    # --------------------------------------------------------------------------
    def estimatePitchProfile(self, t: float, NBIline: dict = {}, rmin = 1.4,
                             rmax = 2.2, IPBtSign: float = -1.0):
        """
        Estimate the pitch profile, based in geometry and equilibrium.

        Jose Rueda - jrrueda@us.es

        :param t: DESCRIPTION
        :type t: float
        :param NBIline: DESCRIPTION, defaults to {}
        :type NBIline: dict, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        # ---- Check inputs
        if self.position is None:
            errors.NotValidInput('position should be in the video object')
        if t is None:
            t = self.exp_dat.t.values.mean()
        # ---- Get the points along the line
        npoints = 150
        points = np.zeros((npoints, 3))
        points[0, :] = NBIline['origin']
        for i in range(1, npoints):
            points[i, :] = points[i-1, :] + NBIline['v']/100.0
        Rpoints = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        # Get only the points inside the view
        flags = (Rpoints > rmin) * (Rpoints < rmax)
        Rpoints = Rpoints[flags]
        points = points[flags, :]
        phipoints = np.arctan2(points[:, 1], points[:, 0])
        zpoints = points[:, 1]
        # ---- Get the magnetic field
        br, bz, bt, bp =\
            ssdat.get_mag_field(self.shot,
                                Rpoints,
                                zpoints,
                                time=t,
                                **self.BFieldOptions)
        bx = np.array(br*np.cos(phipoints) - bt*np.sin(phipoints)).squeeze()
        by = np.array(- br*np.cos(phipoints) + bt*np.cos(phipoints)).squeeze()
        bz = np.array(bz).squeeze()
        b = np.sqrt(bx**2 + by**2 + bz**2).squeeze()
        # ---- Calculate the pitch
        pitch = np.zeros(Rpoints.size)
        phi_pinhole = self.position['phi_pinhole'] * np.pi / 180.0
        pinhole = np.array((
            self.position['R_pinhole'] * np.cos(phi_pinhole),
            self.position['R_pinhole'] * np.sin(phi_pinhole),
            self.position['z_pinhole']))
        for i in range(pitch.size):
            v = pinhole - points[i, :]
            v /= np.linalg.norm(v)
            pitch[i] = v[0] * bx[i] + v[1] * by[i] + v[2] * bz[i]
            pitch[i] /= b[i]
        self.pitchProfile = {
            'R': Rpoints,
            'phi': phipoints,
            'z': zpoints,
            'pitch': pitch * IPBtSign
            }
        return
