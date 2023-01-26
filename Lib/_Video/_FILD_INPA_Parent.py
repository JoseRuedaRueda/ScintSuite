"""
Parent bject for FILD and INPA video, which share many points in common

Introduced in version 0.9.0
"""
import os
import math
import logging
import tarfile
import json
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from Lib._Video._BasicVideoObject import BVO
from Lib._TimeTrace._timetrace import TimeTrace
import Lib._GUIs as ssGUI
import Lib._Mapping as ssmap
import Lib.LibData as ssdat
import Lib._Plotting as ssplt
import Lib.SimulationCodes.FILDSIM as ssFILDSIM
import Lib.SimulationCodes.SINPA as ssSINPA
import Lib._Parameters as sspar
import Lib.errors as errors
import Lib._IO as ssio
import Lib._StrikeMap as ssmapnew
import Lib.version_suite as version
import xarray as xr
from tqdm import tqdm   # For waitbars
import Lib._Paths as p
from Lib._Machine import machine
pa = p.Path(machine)
del p

logger = logging.getLogger('ScintSuite.Video')


class FIV(BVO):
    """
    Class containing common methods for INPA and FILD videos

    Jose Rueda: jrrueda@us.es

    Introduced in version 0.9.0

    Note FIV stands for FILD INPA Video

    - Public methods (*indicate methods coming from the parent class):
        - *read_frame: Load a given range of frames
        - *subtract noise: Use a range of times to average de noise and subtract
            it to all frames
        - *filter_frames: apply filters such as median, gaussian, etc
        - *average_frames: average frames under certain windows
        - *generate_average_window: generate the windows to average the frames
        - *return_to_original_frames: remove the noise subtraction etc
        - *plot_number_saturated_counts: plot the total number of saturated
            counts in each frame
        - plot_ frame: plot a given frame. Include the posibility of plotting a
            StrikeMap
        - *GUI_frames: display a GUI to explore the video
        - *getFrameIndex: get the frame number associated to a given time
        - *getFrame: return the frame associated to a given time
        - *getTime: return the time associated to a frame index
        - *getTimeTrace: calculate a video timetrace
        - *exportVideo: save the dataframes to a netCDF
        - get_smap_name: Get the name of the strike map for a given frame
        - plot_frame_remap: Plot the frame from the remap
        - plotBangles: Plot the angles of the B field respect to the head
        - integrate_remap: Perform the integration over a region of the
            phase space
        - translate_remap_to_energy: deprecated in this version of the suite
        - GUI_profile_analysis: GUI to analyse the profiles
        - GUI_remap_analysis: GUI to analyse the remap
        - export_Bangles: export the Bangles to netCDF files
        - export_Bfield: export the B field at the head to netCDF files
        - export_remap: export the remap into a series of netCDF files

    - Private methods:
        - _getB: Get the magnetic field at the detector position


    """
    def __init__(self, **kargs):
        """Init the class"""
        # Init the parent class
        BVO.__init__(self, **kargs)
        ## Diag name
        self.diag = None
        ## Diag number
        self.diag_ID = None
        ## Magnetic field at FILD head
        self.BField = None
        ## Particular options for the magnetic field calculation
        self.BFieldOptions = {}
        ## Orientation angles
        self.Bangles = None
        ## Position of the diagnostic
        self.position = None
        ## NBI traces
        self._NBIpower = None
        ## density traces
        self._ne = None
        ## ROI of the scintillator
        self.ROIscintillator = None
        ## Scintilaltor plate
        self.scintillator = None

    # --------------------------------------------------------------------------
    # --- Get shot / magnetic data
    # --------------------------------------------------------------------------
    def _getB(self, extra_options: dict = {}, use_average: bool = False):
        """
        Get the magnetic field in the FILD position

        Jose Rueda - jrrueda@us.es

        :param  extra_options: Extra options to be passed to the magnetic field
            calculation. Ideal place to insert all your machine dependent stuff
        :param  use_average: flag to use the timebase of the average frames or
            the experimental frames

        Note: It will overwrite the content of self.Bfield
        """
        if self.position is None:
            raise Exception('Detector position not know')
        # Get the proper timebase
        if use_average:
            time = self.avg_dat['t'].values
        else:
            time = self.exp_dat['t'].values
        # Calculate the magnetic field
        print('Calculating magnetic field (this might take a while): ')
        if 'R_scintillator' in self.position.keys():  # INPA case
            key1 = 'R_scintillator'
            key2 = 'z_scintillator'
        else:
            key1 = 'R'
            key2 = 'z'
        br, bz, bt, bp =\
            ssdat.get_mag_field(self.shot,
                                self.position[key1],
                                self.position[key2],
                                time=time,
                                **extra_options)
        # Save the data in the array
        self.BField = xr.Dataset()
        self.BField['BR'] = xr.DataArray(np.array(br).squeeze(), dims=('t'),
                                         coords={'t': time})

        self.BField['Bz'] = xr.DataArray(np.array(bz).squeeze(), dims=('t'))
        self.BField['Bt'] = xr.DataArray(np.array(bt).squeeze(), dims=('t'))
        self.BField['B'] = xr.DataArray(
            np.sqrt(np.array(bp)**2 + np.array(bt)**2).squeeze(), dims=('t'))
        self.BField.attrs['units'] = 'T'
        self.BField.attrs['R'] = self.position[key1]
        self.BField.attrs['z'] = self.position[key2]
        self.BField.attrs.update(extra_options)

    def _getNBIpower(self):
        """Store the power of the NBI"""
        self._NBIpower = ssdat.getNBI_timeTraces(self.shot,
                                                 xArrayOutput=True)

    def _getNe(self):
        """Store plasma density"""
        self._ne = ssdat.get_ne(self.shot, xArrayOutput=True)

    # --------------------------------------------------------------------------
    # --- Strike map library handling
    # --------------------------------------------------------------------------
    def get_smap_name(self, frame_number=None, t: float = None,
                      verbose: bool = False):
        """
        Get name of the strike map

        Jose Rueda Rueda: jrrueda@us.es
        Lina Velarde: lvelarde@us.es

        :param  frame_number: Number of the frame to get the smap name, relative
                to the video file, optional
        :param  t: Time instant of the frame to get the smap name, optional
        :param  verbose: If true, info of the theta and phi used will be printed

        :return full_name_smap: Name of the used strike map
        """
        # # Get Bangles
        # if self.Bangles is None:
        #     self._getB()
        #     self._getBangles()
        # Get the frame number
        if t is not None:
            frame_index = np.argmin(abs(self.exp_dat['t'].values - t))
            theta_used = self.Bangles['theta_used'].values[frame_index]
            phi_used = self.Bangles['phi_used'].values[frame_index]
        else:
            frame_index = self.exp_dat['nframes'].values == frame_number
            theta_used = self.Bangles['theta_used'].values[frame_index][0]
            phi_used = self.Bangles['phi_used'].values[frame_index][0]

        # Get the full name of the file
        if self.diag == 'FILD' and self.strikemap.attrs['CodeUsed'].lower() == 'fildsim':
            name__smap = ssFILDSIM.guess_strike_map_name(
                phi_used, theta_used, geomID=self.geometryID,
                decimals=self.Bangles['theta_used'].attrs['decimals'])
        elif self.diag == 'FILD':
            name__smap = ssSINPA.execution.guess_strike_map_name(
                phi_used, theta_used, geomID=self.geometryID,
                decimals=self.Bangles['theta_used'].attrs['decimals']
                )
        elif self.diag == 'INPA':
            name__smap = ssSINPA.execution.guess_strike_map_name(
                phi_used, theta_used, geomID=self.geometryID,
                decimals=self.Bangles['theta_used'].attrs['decimals']
                )
        else:
            raise Exception('Diagnostic not understood')
        smap_folder = self.strikemap.attrs['smap_folder']
        full_name_smap = os.path.join(smap_folder, name__smap)

        if verbose:
            theta_calculated = self.Bangles['theta'].values[frame_index]
            phi_calculated = self.Bangles['phi'].values[frame_index]
            print('Calculated theta: ', theta_calculated)
            print('Used theta: ', theta_used)
            print('Calculated phi: ', phi_calculated)
            print('Used phi: ', phi_used)

        return full_name_smap

    # --------------------------------------------------------------------------
    # --- Time Traces
    # --------------------------------------------------------------------------
    def getTimeTrace(self, t: float = None, mask=None, ROIname: str = None):
        """
        Calculate the timeTrace of the video. Extended method from parent class

        Difference from parent class: if no argument is given, we will use the
        Scintillator roi and calculate the trace of the whole video

        Jose Rueda Rueda: jrrueda@us.es

        :param  t: time of the frame to be plotted for the selection of the roi
        :param  mask: bolean mask of the ROI

        If mask is present, the t argument will be ignored

        :returns timetrace: a timetrace object
        """
        if mask is not None or t is not None:
            trace, mask = super().getTimeTrace(t=t, mask=mask,
                                           ROIname=ROIname)
        else:
            mask = \
                self.ROIscintillator.getMask(self.exp_dat['frames'][:, :,
                                             0].squeeze())
            trace = TimeTrace(self, mask, ROIname='ScintROI')

        return trace, mask

    # --------------------------------------------------------------------------
    # --- Plotting block
    # --------------------------------------------------------------------------
    def plot_frame(self, frame_number=None, ax=None, ccmap=None,
                   strike_map: str = 'off', t: float = None,
                   verbose: bool = True,
                   smap_marker_params: dict = {},
                   smap_line_params: dict = {}, vmin: int = 0,
                   vmax: int = None, xlim: float = None, ylim: float = None,
                   scale: str = 'linear',
                   alpha: float = 1.0, IncludeColorbar: bool = True,
                   RemoveAxisTicksLabels: bool = False,
                   flagAverage:bool = False,
                   normalise=None,
                   smap_labels: bool = False):
        """
        Plot a frame from the loaded frames

        Jose Rueda Rueda: jrrueda@us.es

        Notice that this function just call the plot_frame of the BVO and then
        adds the strike map

        :param  frame_number: Number of the frame to plot (option 1)
        :param  ax: Axes where to plot, is none, just a new axes will be created
        :param  ccmap: colormap to be used, if none, Gamma_II from IDL
        :param  strike_map: StrikeMap to plot:
            -  # 'auto': The code will load the Smap corresponding to the theta
            phi angles. Note, the angles should be calculated, so the remap,
            should be done. (also, it will load the calibration from the
            performed remap)
            -  # StrikeMap(): if a StrikeMap() object is passed, it will be
            plotted. Note, the program will only check if the StrikeMap input
            is a class, it will not check if the calibration was apply etc, so
            it is supposed that the user had given a Smap, ready to be plotted
            -  # 'off': (or any other string) No strike map will be plotted
        :param  t: time point to select the frame (option 2)
        :param  verbose: If true, info of the theta and phi used will be printed
        :param  smap_marker_params: dictionary with parameters to plot the
            strike_map centroid(see StrikeMap.plot_pix)
        :param  smap_line_params: dictionary with parameters to plot the
            strike_map lines(see StrikeMap.plot_pix)
        :param  vmin: Minimum value for the color scale to plot
        :param  vmax: Maximum value for the color scale to plot
        :param  xlim: tuple with the x-axis limits
        :param  ylim: tuple with the y-axis limits
        :param  scale: Scale for the plot: 'linear', 'sqrt', or 'log'
        :param  alpha: transparency factor, 0.0 is 100 % transparent
        :param  IncludeColorbar: flag to include a colorbar
        :param  RemoveAxisTicksLabels: boolean flag to remove the numbers in the
            axis
        :param  flagAverage: flag to pick the axis from the experimental or the
            averaged frames
        :param  normalise: parameter to normalise the frame when plotting:
            if normalise == 1 it would be normalised to the maximum
            if normalise == <number> it would be normalised to this value
            if normalise == None, nothing will be done
        :param  smap_labels: boolean flag to plot the labels of the strike map

        :return ax: the axes where the frame has been drawn
        """
        # --- Call the parent function
        ax = super().plot_frame(
            frame_number=frame_number, ax=ax, ccmap=ccmap, t=t,
            verbose=verbose, vmin=vmin, vmax=vmax, xlim=xlim,
            ylim=ylim, scale=scale, alpha=alpha,
            IncludeColorbar=IncludeColorbar,
            RemoveAxisTicksLabels=RemoveAxisTicksLabels,
            flagAverage=flagAverage,
            normalise=normalise
        )
        # Get the frame number
        if t is not None:
            frame_index = np.argmin(abs(self.exp_dat['t'].values - t))
        else:
            frame_index = self.exp_dat['nframes'].values == frame_number

        # --- Plot the strike map

        # Save axis limits, if not, if the strike map is larger than
        # the frame (FILD4,5) the output plot will be horrible
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # --- Plot the StrikeMap
        if isinstance(strike_map, (ssmapnew.Ismap, ssmapnew.Fsmap)):
            strike_map.plot_pix(ax=ax)
        elif strike_map == 'auto':
            full_name_smap = self.get_smap_name(frame_number=frame_number, t=t,
                                                verbose=verbose)
            if self.diag.lower() == 'fild':
                id = 0
            else:
                id = 1
            # Load the map:
            smap = ssmap.StrikeMap(id, full_name_smap)
            # Calculate pixel coordinates
            smap.calculate_pixel_coordinates(self.CameraCalibration)
            # Plot the map
            smap.plot_pix(ax=ax, marker_params=smap_marker_params,
                          line_params=smap_line_params, labels=smap_labels)

        # Set 'original' limits:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    def plot_frame_remap(self, frame_number=None, ax=None, ccmap=None,
                         t: float = None, vmin: float = 0, vmax: float = None,
                         xlim: float = None, ylim: float = None,
                         scale: str = 'linear',
                         interpolation: str = 'bicubic',
                         cbar_tick_format: str = '%.1E',
                         IncludeColorbar: bool = True,
                         color_labels_in_plot: str = 'w',
                         normalise: bool = False,
                         translation: tuple = None):
        """
        Plot a frame from the remaped frames

        :param  frame_number: Number of the frame to plot, relative to the video
            file, optional

        :param  ax: Axes where to plot, is none, just a new axes will be created
        :param  ccmap: colormap to be used, if none, Gamma_II from IDL
        :param  vmin: Minimum value for the color scale to plot
        :param  vmax: Maximum value for the color scale to plot
        :param  xlim: tuple with the x-axis limits
        :param  ylim: tuple with the y-axis limits
        :param  scale: Scale for the plot: 'linear', 'sqrt', or 'log'
        :param  interpolation: interpolation method for plt.imshow
        :param  cbar_tick_format: format for the colorbar ticks
        :param  IncludeColorbar: Boolean flag to include the colorbar
        :param  color_labels_in_plot: Color for the labels in the plot
        :param  translation: tuple with the desired specie and translation to
            plot. Example ('D', 1)

        :return ax: the axes where the frame has been drawn
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
                    if translation is None:
                        dummy = self.remap_dat['frames'].values.squeeze()
                    else:
                        dummy = self.remap_dat['translations'][translation[0]]\
                            [translation[1]]['frames'].squeeze()
                    tf = float(self.remap_dat['t'].values)
                    frame_index = 0
                else:
                    raise Exception('Frame not loaded')
            else:
                frame_index = self.remap_dat['nframes'].values == frame_number
                if np.sum(frame_index) == 0:
                    raise Exception('Frame not loaded')
                if translation is None:
                    dummy = \
                        self.remap_dat['frames'].values[:, :, frame_index].squeeze()
                else:
                    dummy = self.remap_dat['translations'][translation[0]]\
                        [translation[1]]['frames'][:, :, frame_index].squeeze()
                tf = float(self.remap_dat['t'].values[frame_index])
        # If we give the time:
        if t is not None:
            frame_index = np.argmin(np.abs(self.remap_dat['t'].values - t))
            tf = self.remap_dat['t'].values[frame_index]
            if translation is None:
                dummy = \
                    self.remap_dat['frames'].values[:, :, frame_index].squeeze()
            else:
                dummy = self.remap_dat['translations'][translation[0]]\
                    [translation[1]]['frames'][:, :, frame_index].squeeze()
        # --- Normalise
        if normalise:
            dummy = dummy.copy()/dummy.max()  # To avoid modifying the video
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
        if translation is None:
            ext = [self.remap_dat['x'].values[0], self.remap_dat['x'].values[-1],
                   self.remap_dat['y'].values[0], self.remap_dat['y'].values[-1]]
        else:
            ext = [self.remap_dat['translations'][translation[0]][translation[1]]['xaxis'][0],
                   self.remap_dat['translations'][translation[0]][translation[1]]['xaxis'][-1],
                   self.remap_dat['translations'][translation[0]][translation[1]]['yaxis'][0],
                   self.remap_dat['translations'][translation[0]][translation[1]]['yaxis'][-1]]
        img = ax.imshow(dummy.T, extent=ext,
                        origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation=interpolation, aspect='auto',
                        **extra_options)
        # --- trick to make the colorbar of the correct size
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if IncludeColorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, label='Counts [a.u.]', cax=cax,
                         format=cbar_tick_format)
        # Set the labels with t and shot
        ax.text(0.05, 0.9, '#' + str(self.shot),
                horizontalalignment='left',
                color=color_labels_in_plot, verticalalignment='bottom',
                transform=ax.transAxes)
        plt.text(0.95, 0.9, 't = ' + str(round(tf, 4)) + (' s'),
                 horizontalalignment='right',
                 color=color_labels_in_plot, verticalalignment='bottom',
                 transform=ax.transAxes)

        if created:
            if translation is None:
                ax.set_ylabel('%s [%s]' %
                              (self.remap_dat['y'].attrs['long_name'].capitalize(),
                               self.remap_dat['y'].attrs['units']))
                ax.set_xlabel('%s [%s]' %
                              (self.remap_dat['x'].attrs['long_name'].capitalize(),
                               self.remap_dat['x'].attrs['units']))
            elif translation[1] == 1:
                ax.set_ylabel('E [keV]')
            fig.show()
            plt.tight_layout()
        return ax

    def plotBangles(self, ax_params: dict = {}, line_params: dict = {},
                    ax=None):
        """
        Plot the orientaton angles of the diagnostic in each time point

        If the remap is done, it plot the calculated and used orientation
        of the magnetic field as well as some shaded areas to guide the eye

        Jose Rueda Rueda: jrrueda@us.es

        :param  ax_param: axis parameters for the axis beauty routine
        :param  ax: array (with size 2) of axes where to plot
        """
        # --- Plotting options:
        ax_options = {
            'grid': 'both'
        }
        ax_options.update(ax_params)
        line_options = {
            'linewidth': 2
        }
        line_options.update(line_params)
        # --- Get the data to plot if remap dat is present
        if self.remap_dat is not None:
            time = self.remap_dat['t'].values
            phi = self.remap_dat['phi'].values
            phi_used = self.remap_dat['phi_used'].values
            theta = self.remap_dat['theta'].values
            theta_used = self.remap_dat['theta_used'].values
        else:
            phi = self.Bangles['phi'].values
            phi_used = None
            theta = self.Bangles['theta'].values
            theta_used = None
            time = self.Bangles['t'].values
        # proceed to plot
        if ax is None:
            fig, ax = plt.subplots(2, sharex=True)
        # Plot the theta angle:
        # Plot a shaded area indicating the points where only an
        # aproximate map was used, taken from the solution given here:
        # https://stackoverflow.com/questions/43233552/
        # how-do-i-use-axvfill-with-a-boolean-series
        if theta_used is not None:
            ax[0].fill_between(time, 0, 1,
                               where=self.remap_dat['existing_smaps'].values,
                               alpha=0.25, color='g',
                               transform=ax[0].get_xaxis_transform())
            ax[0].fill_between(time, 0, 1,
                               where=~self.remap_dat['existing_smaps'].values,
                               alpha=0.25, color='r',
                               transform=ax[0].get_xaxis_transform())
            ax[0].plot(time, theta_used,
                       **line_options, label='Used', color='b')
        # Plot the line
        ax[0].plot(time, theta,
                   **line_options, label='Calculated', color='k')

        ax_options['ylabel'] = '$\\Theta$ [degrees]'
        ax[0] = ssplt.axis_beauty(ax[0], ax_options)
        # Plot the phi angle
        if phi_used is not None:
            ax[1].fill_between(time, 0, 1,
                               where=self.remap_dat['existing_smaps'].values,
                               alpha=0.25, color='g',
                               transform=ax[1].get_xaxis_transform())
            ax[1].fill_between(time, 0, 1,
                               where=~self.remap_dat['existing_smaps'].values,
                               alpha=0.25, color='r',
                               transform=ax[1].get_xaxis_transform())
            ax[1].plot(time, phi_used,
                       **line_options, label='Used', color='b')

        ax[1].plot(time, phi,
                   **line_options, label='Calculated', color='k')
        ax_options['ylabel'] = '$\\phi$ [degrees]'
        ax_options['xlabel'] = 't [s]'
        ax[1] = ssplt.axis_beauty(ax[1], ax_options)
        plt.legend()
        return ax

    # --------------------------------------------------------------------------
    # --- Remap block
    # --------------------------------------------------------------------------
    def integrate_remap(self, xmin: float = 20.0, xmax: float = 90.0,
                        ymin: float = 1.0, ymax: float = 10.0,
                        mask=None, specie: str = 'D',
                        translationNumber: int = 0):
        """
        Integrate the remaped frames over a given region of the phase space

        Jose Rueda: jrrueda@us.es

        :param  xmin: Minimum value of the x axis to integrate (pitch for FILD)
        :param  xmax: Maximum value of the x axis to integrate (pitch for FILD)
        :param  ymin: Minimum value of the y axis to integrate (rl for FILD)
        :param  ymax: Maximum value of the y axis to integrate (rl for FILD)
        :param  mask: bynary mask denoting the desired cells of the space to
            integate. If present, xmin-xmax, ymin-ymax will be ignored
        :param  specie: Not used in this version of the suite
        :param  translationNumber: Not used in this version of the suite

        :return output: dataset containing the integral:
            - integral_over_y: integral over the y axis
            - integral_over_x: integral over the x axis
            - integral_over_xy: integral over the xy axis

        @Todo: Include integrals in x and y when we use a mask
        """
        if self.remap_dat is None:
            raise Exception('Please remap before call this function!!!')
        # First look what do we ned to integrate
        if translationNumber == 0:  # Usual remap
            data = self.remap_dat
        else:
            data = self.remap_dat['translations'][specie][translationNumber]
        # First calculate the dif x and y to integrate
        dx = (data['x'][1] - data['x'][0]).values
        dy = (data['y'][1] - data['y'][0]).values
        # Find the flags:
        mask_was_none = False
        if mask is None:
            flagsx = (xmin < data['x'].values) * (data['x'].values < xmax)
            flagsy = (ymin < data['y'].values) * (data['y'].values < ymax)
            mask_was_none = True
        # Perform the integration:
        if mask_was_none:
            # Get the trace (need a better method, but at least this work)
            dummy = np.sum(data['frames'].values[flagsx, :, :], axis=0) * dy \
                    * dx
            trace = np.sum(dummy[flagsy, :], axis=0)
            integral_over_y = np.sum(data['frames'][:, flagsy, :], axis=1) * dy
            integral_over_x = np.sum(data['frames'][flagsx, :, :], axis=0) * dx
        else:
            trace = np.sum(self.remap_dat['frames'][mask, :], axis=0) * dy * dx
            integral_over_x = 0
            integral_over_y = 0

        # -- Save the result
        output = xr.Dataset()
        output['integral_over_y'] = \
            xr.DataArray(integral_over_y, dims=('x', 't'),
                         coords={'x': data['x'], 't': data['t']})
        output['integral_over_x'] = \
            xr.DataArray(integral_over_x, dims=('y', 't'),
                         coords={'y': data['y'], 't': data['t']})
        # -- Save the metadata
        output['integral_over_x'].attrs['xmin'] = xmin
        output['integral_over_x'].attrs['xmax'] = xmax
        output['integral_over_x'].attrs['long_name'] = 'Signal'
        output['integral_over_x'].attrs['units'] = \
            '#/' + data['y'].attrs['units']
        output['integral_over_y'].attrs['ymin'] = ymin
        output['integral_over_y'].attrs['ymax'] = ymax
        output['integral_over_y'].attrs['long_name'] = 'Signal'
        output['integral_over_y'].attrs['units'] = \
            '#/' + data['x'].attrs['units']

        output['integral_over_xy'] = xr.DataArray(trace, dims=('t'))
        output['integral_over_xy'].attrs['long_name'] = 'Signal'
        output['integral_over_xy'].attrs['units'] = '#'

        if mask is not None:
            output['integral_over_xy'].attrs['mask'] = mask.astype('int')
        else:
            output['integral_over_xy'].attrs['mask'] = 0
        return output

    def translate_remap_to_energy(self, Emin: float = 10.0, Emax: float = 99.0,
                                  dE: int = 1.0, useAverageB: bool = True,
                                  specie: str = 'D'):
        """
        Transform the remap from Gyroradius to Energy

        Introduced in version 0.9.5

        :param  Emin: Minimum energy for the new axis
        :param  Emax: Maximum energy for the new axis
        :param  dE: spacing for the new axis
        :param  useAverageB: flag to use the average value of the field or the
            time-dependent field
        :param  specie: assumed specie of the incident ion (H, D, T, He...)
        """
        # See if there is remap data
        if (self.remap_dat is None) or (self.BField is None):
            raise Exception('You need to remap first!')
        # See if there remap data has the same length of the B field (the user
        # may have use the average in one of them...)
        nx, ny, nt = self.remap_dat['frames'].shape
        nt2 = self.BField['B'].size
        if nt != nt2:
            raise Exception('The B points do not agree with the remap!!!')
        # Get the specie Mass and charge
        par = sspar.species_info[specie.upper()]
        # Prepare the new energy axis
        ne = int((Emax-Emin)/dE) + 1
        Eedges = Emin - dE/2 + np.arange(ne+1) * dE
        E = 0.5 * (Eedges[0:-1] + Eedges[1:])
        rl = self.remap_dat['yaxis']
        if useAverageB:
            B = self.BField['B'].mean()
            K = ssFILDSIM.get_energy(rl, B, A=par['A'], Z=par['Z']) / 1000.0
            factor = math.sqrt(par['A'] * sspar.amu2kg / 2.0) / B / par['Z'] \
                / sspar.ec / np.sqrt(K)
            frames = self.remap_dat['frames'] / factor[None, :, None]
            # Now comes the slow part, this need to be optimized, but let's go:
            print('Interpolating in the new energy grid')
            new_frames = np.zeros((nx, ne, nt))
            for it in tqdm(range(nt)):
                for ixi in range(nx):
                    # Interpolate the signal in the K array:
                    f2 = interp1d(K, frames[ixi, :, it].squeeze(),
                                  kind='cubic', fill_value=0.0,
                                  bounds_error=False)
                    new_frames[ixi, :, it] = f2(E)
        else:
            # Now comes the slow part, this need to be optimized, but let's go:
            frames = self.remap_dat['frames'].copy()
            new_frames = np.zeros((nx, ne, nt))
            print('Interpolating in the new energy grid')
            for it in tqdm(range(nt)):
                B = self.BField['B'][it]
                K = ssFILDSIM.get_energy(rl, B, A=par['A'], Z=par['Z'])/1000.0
                factor = math.sqrt(par['A'] * sspar.amu2kg / 2.0) / B  \
                    / par['Z'] / sspar.ec / np.sqrt(K)
                frames[:, :, it] /= factor[None, :]
                for ixi in range(nx):
                    # Interpolate the signal in the K array:
                    f2 = interp1d(K, frames[ixi, :, it].squeeze(),
                                  kind='cubic', fill_value=0.0,
                                  bounds_error=False)
                    new_frames[ixi, :, it] = f2(E)
        # Now save the remap in the proper place
        if 'translations' not in self.remap_dat.keys():
            self.remap_dat['translations'] = {}
        if specie.upper() not in self.remap_dat['translations'].keys():
            self.remap_dat['translations'][specie.upper()] = {}
        self.remap_dat['translations'][specie.upper()][1] = {
            'xaxis': self.remap_dat['xaxis'],
            'xaxisLabel': '%s [%s]' % (self.remap_dat['xlabel'],
                                       self.remap_dat['xunits']),
            'yaxis': E,
            'yaxisLabel': 'Energy [keV]',
            'frames': new_frames
        }

    # --------------------------------------------------------------------------
    # --- GUIs block
    # --------------------------------------------------------------------------
    def GUI_profile_analysis(self, translation: tuple = None):
        """Small GUI to explore camera frames"""
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print('--. ..- ..')
        print(text)
        print('-... . ..- - -.--')
        root = tk.Tk()
        root.resizable(height=None, width=None)
        ssGUI.ApplicationRemapAnalysis(root, self, translation)
        root.mainloop()
        root.destroy()

    def GUI_remap_analysis(self, traces=None):
        """
        Small GUI to explore camera frames

        Jose Rueda: jrrueda@us.es

        :param  traces: traces to plot in the upper plot of the GUI, should
            contain 't1', 'y1', 'l1', etc
        """
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print('--. ..- ..')
        print(text)
        print('-... . ..- - -.--')
        root = tk.Tk()
        root.resizable(height=None, width=None)
        if traces is None:
            traces = self._ne
        ssGUI.ApplicationRemap2DAnalyser(root, self, traces)
        root.mainloop()
        root.destroy()

    # --------------------------------------------------------------------------
    # --- Export Block
    # --------------------------------------------------------------------------
    def export_Bangles(self, filename):
        """
        Export the B angles into a netCDF files

        :param  filename: filename

        Notice, in principle this function should not be called, the method
            self.export_remap() will take care of calling this one
        """
        logger.info('Saving Bangles in file %s' % filename)
        self.Bangles.to_netcdf(filename)

    def export_Bfield(self, filename):
        """
        Export the B angles into a netCDF files

        :param  filename: filename

        Notice, in principle this function should not be called, the method
            self.export_remap() will take care of calling this one
        """
        logger.info('Saving BField in file %s' % filename)
        self.BField.to_netcdf(filename)

    def export_remap(self, folder: str = None, clean: bool = False,
                     overwrite: bool = False):
        """
        Export video file

        Notice: This will create a netcdf with the exp_dat xarray, this is not
        intended as a replace of the data base, as camera settings and
        metadata will not be exported. But allows to quickly export the video
        to netCDF format to be easily shared among computers

        :param  folder: Path to the folder where to save the results. It is
            recommended to leave it as None
        :param  clean: delete the netCDF files and leave only the .tar file
        :param  overwrite: ignore old files, if present
        """
        if folder is None:
            folder = os.path.join(pa.Results, str(self.shot), self.diag,
                                  str(self.diag_ID))

        # Now get the name of the files:
        magField = os.path.join(folder, 'Bfield.nc')
        magFieldAngles = os.path.join(folder, 'BfieldAngles.nc')
        remap = os.path.join(folder, 'remap.nc')
        calibration = os.path.join(folder, 'CameraCalibration.nc')
        versionFile = os.path.join(folder, 'version.txt')
        noiseFrame = os.path.join(folder, 'noiseFrame.nc')
        metaDataFile = os.path.join(folder, 'metadata.txt')
        orientationFile = os.path.join(folder, 'orientation.json')
        positionFile = os.path.join(folder, 'position.json')
        tarFile = os.path.join(folder, str(self.shot) + '_' + self.diag +
                               str(self.diag_ID) + '_' + 'remap.tar')
        if os.path.isfile(tarFile) and not overwrite:
            raise Exception('The file exist!')
        logger.info('Saving results in: %s', folder)
        os.makedirs(folder, exist_ok=True)

        # Create the individual netCDF files
        self.export_Bangles(magFieldAngles)
        self.export_Bfield(magField)
        self.remap_dat.to_netcdf(remap)
        self.CameraCalibration.save2netCDF(calibration)
        if 'frame_noise' in self.exp_dat:
            self.exp_dat['frame_noise'].to_netcdf(noiseFrame)
        version.exportVersion(versionFile)
        with open(metaDataFile, 'w') as f:
            f.write('Shot: %i\n'%self.shot)
            f.write('diag_ID: %i\n'%self.diag_ID)
            f.write('geom_ID: %s\n'%self.geometryID)
            f.write('CameraFileBPP: %s\n'%self.settings['RealBPP'])

        json.dump(self.position, open(positionFile, 'w' ) )
        json.dump({k:np.array(v).tolist() for k,v in self.orientation.items()},
                  open(orientationFile, 'w' ) )
        # Create the tar file
        tar = tarfile.open(name=tarFile, mode='w')
        tar.add(magField, arcname='Bfield.nc')
        tar.add(magFieldAngles, arcname='BfieldAngles.nc')
        tar.add(remap, arcname='remap.nc')
        tar.add(calibration, arcname='CameraCalibration.nc')
        tar.add(versionFile, arcname='version.txt')
        tar.add(metaDataFile, arcname='metadata.txt')
        tar.add(positionFile, arcname='position.json')
        tar.add(orientationFile, arcname='orientation.json')
        if 'frame_noise' in self.exp_dat:
            tar.add(noiseFrame, arcname='noiseFrame.nc')
        tar.close()

        # Clean if asked
        if clean:
            os.remove(magField)
            os.remove(magFieldAngles)
            os.remove(remap)
            os.remove(calibration)
            os.remove(versionFile)
