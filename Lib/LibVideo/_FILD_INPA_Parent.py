"""
Parent bject for FILD and INPA video, which share many points in common

Introduced in version 0.9.0
"""
import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Lib.LibVideo._BasicVideoObject import BVO
import Lib.GUIs as ssGUI
import Lib.LibMap as ssmap
import Lib.LibPlotting as ssplt
import Lib.SimulationCodes.FILDSIM as ssFILDSIM
import Lib.SimulationCodes.SINPA as ssSINPA


class FIV(BVO):
    """
    Class containing common methods for INPA and FILD videos

    Jose Rueda: jrrueda@us.es

    Introduced in version 0.9.0

    Note FIV stands for FILD INPA Video

    - Public methods:
        - plot_frame: Plot a camera frame
        - plot_frame_remap: Plot a frame from the remap data
        - integrate_remap: Perform the integration over a region of the
            phase space
    """

    # Plotting routines

    def plot_frame(self, frame_number=None, ax=None, ccmap=None,
                   strike_map: str = 'off', t: float = None,
                   verbose: bool = True,
                   smap_marker_params: dict = {},
                   smap_line_params: dict = {}, vmin: int = 0,
                   vmax: int = None, xlim: float = None, ylim: float = None,
                   scale: str = 'linear',
                   alpha: float = 1.0, IncludeColorbar: bool = True,
                   RemoveAxisTicksLabels: bool = False):
        """
        Plot a frame from the loaded frames

        Jose Rueda Rueda: jrrueda@us.es

        Notice that this function just call the plot_frame of the BVO and then
        adds the strike map

        @param frame_number: Number of the frame to plot, relative to the video
            file, optional

        @param ax: Axes where to plot, is none, just a new axes will be created
        @param ccmap: colormap to be used, if none, Gamma_II from IDL
        @param strike_map: StrikeMap to plot:
            -  # 'auto': The code will load the Smap corresponding to the theta
            phi angles. Note, the angles should be calculated, so the remap,
            should be done. (also, it will load the calibration from the
            performed remap)
            -  # StrikeMap(): if a StrikeMap() object is passed, it will be
            plotted. Note, the program will only check if the StrikeMap input
            is a class, it will not check if the calibration was apply etc, so
            it is supposed that the user had given a Smap, ready to be plotted
            -  # 'off': (or any other string) No strike map will be plotted
        @param verbose: If true, info of the theta and phi used will be printed
        @param smap_marker_params: dictionary with parameters to plot the
            strike_map centroid(see StrikeMap.plot_pix)
        @param smap_line_params: dictionary with parameters to plot the
            strike_map lines(see StrikeMap.plot_pix)
        @param vmin: Minimum value for the color scale to plot
        @param vmax: Maximum value for the color scale to plot
        @param xlim: tuple with the x-axis limits
        @param ylim: tuple with the y-axis limits
        @param scale: Scale for the plot: 'linear', 'sqrt', or 'log'
        @param alpha: transparency factor, 0.0 is 100 % transparent
        @param RemoveAxisTicksLabels: boolean flag to remove the numbers in the
            axis

        @return ax: the axes where the frame has been drawn
        """
        # --- Call the parent function
        ax = BVO.plot_frame(frame_number=frame_number, ax=ax, ccmap=ccmap,
                            t=t, verbose=verbose, vmin=vmin, vmax=vmax,
                            xlim=xlim, ylim=ylim, scale=scale,
                            alpha=alpha, IncludeColorbar=IncludeColorbar,
                            RemoveAxisTicksLabels=RemoveAxisTicksLabels)
        # Get the frame number
        if t is not None:
            frame_index = np.argmin(abs(self.exp_dat['tframes'] - t))
        else:
            frame_index = self.exp_dat['nframes'] == frame_number
        # --- Plot the strike map

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
            if self.diag == 'FILD' and self.remap_dat['options']['CodeUsed'].lower() == 'fildsim':
                name__smap = ssFILDSIM.guess_strike_map_name(
                    phi_used, theta_used, geomID=self.FILDgeometry,
                    decimals=self.remap_dat['options']['decimals'])
                id = 0  # To identify the kind of strike map
            elif self.diag == 'FILD':
                name__smap = ssSINPA.execution.guess_strike_map_name(
                    phi_used, theta_used, geomID=self.FILDgeometry,
                    decimals=self.remap_dat['options']['decimals']
                    )
                id = 0  # To identify the kind of strike map
            elif self.diag == 'SINPA':
                name__smap = ssSINPA.execution.guess_strike_map_name(
                    phi_used, theta_used, geomID=self.INPAgeometry,
                    decimals=self.remap_dat['options']['decimals']
                    )
                id = 1  # To identify the kind of strike map
            else:
                raise Exception('Diagnostic not understood')
            smap_folder = self.remap_dat['options']['smap_folder']
            full_name_smap = os.path.join(smap_folder, name__smap)
            # Load the map:
            smap = ssmap.StrikeMap(id, full_name_smap)
            # Calculate pixel coordinates
            smap.calculate_pixel_coordinates(self.CameraCalibration)
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
            if self.diag == 'FILD':
                ax.set_xlabel('$\\lambda \\ [\\degree]$')
                ax.set_ylabel('$r_l$ [cm]')
            else:
                ax.set_xlabel('R [m]')
                ax.set_ylabel('$r_l$ [cm]')
            fig.show()
            plt.tight_layout()
        return ax

    def integrate_remap(self, ximin=20.0, ximax=90.0, rlmin=1.0, rlmax=10.0,
                        pitchmin=None, pitchmax=None,
                        mask=None):
        """
        Integrate the remaped frames over a given region of the phase space

        Jose Rueda: jrrueda@us.es

        @param ximin: Minimum value of the x axis to integrate (pitch for FILD)
        @param ximax: Maximum value of the x axis to integrate (pitch for FILD)
        @param rlmin: Minimum value of the y axis to integrate (radius in FILD)
        @param rlmax: Maximum value of the y axis to integrate (radius in FILD)
        @param mask: bynary mask denoting the desired cells of the space to
            integate. If present, ximin-ximax, rlmin-rlmax will be ignored
        @param pitchmin: dummy variables for FILD operators which may not like
            xi. If present, ximin will be ignored
        @param pitchmax: dummy variables for FILD operators which may not like
            xi. If present, ximax will be ignored

        @return : Output: Dictionary containing the trace and the settings used
            to caclualte it
            output = {
                'rlmin': rlmin,  Minimum rl used in the integration
                'rlmax': rlmax,  Maximum rl used in the integration
                'rl': self.remap_dat['yaxis'],    Array of rl
                'ximin': ximin,  Minimum xi used in the integration
                'ximax': ximax,  Maximum xi used in the integration
                'xi': self.remap_dat['xaxis'],    Array of xi
                't': self.repmap_dat['tframes'],  Array of times
                'trace': trace,  Integral in Xi and rl
                'mask': mask,    Mask for the integration, if used
                'integral_in_rl': integral_in_rl,  Signal as function of Xi
                'integral_in_xi': integral_in_xi,  Signal as function of rl
            }
        """
        if self.remap_dat is None:
            raise Exception('Please remap before call this function!!!')
        if pitchmin is not None:
            ximin = pitchmin
            print('Using pitchmin')
        if pitchmax is not None:
            ximax = pitchmax
            print('Using pitchmax')
        print(ximin, ximax)
        # First calculate the dif x and y to integrate
        dx = self.remap_dat['xaxis'][1] - self.remap_dat['xaxis'][0]
        dy = self.remap_dat['yaxis'][1] - self.remap_dat['yaxis'][0]
        # Find the flags:
        mask_was_none = False
        if mask is None:
            flagsx = (ximin < self.remap_dat['xaxis']) *\
                (self.remap_dat['xaxis'] < ximax)
            flagsy = (rlmin < self.remap_dat['yaxis']) *\
                (self.remap_dat['yaxis'] < rlmax)
            mask_was_none = True
        # Perform the integration:
        if mask_was_none:
            dummy = np.sum(self.remap_dat['frames'][flagsx, :, :],
                           axis=0) * dy * dx
            trace = np.sum(dummy[flagsy, :],
                           axis=0)
            integral_in_rl = \
                np.sum(self.remap_dat['frames'][:, flagsy, :], axis=1) * dy
            integral_in_xi = \
                np.sum(self.remap_dat['frames'][flagsx, :, :], axis=0) * dx
        else:
            trace = np.sum(self.remap_dat['frames'][mask, :], axis=0) * dy * dx
            integral_in_rl = 0
            integral_in_xi = 0

        # save the result
        output = {
            'rlmin': rlmin,
            'rlmax': rlmax,
            'rl': self.remap_dat['yaxis'],
            'ximin': ximin,
            'ximax': ximax,
            'xi': self.remap_dat['xaxis'],
            't': self.remap_dat['tframes'],
            'trace': trace,
            'mask': mask,
            'integral_in_rl': integral_in_rl,
            'integral_in_xi': integral_in_xi
        }

        return output

    def GUI_profile_analysis(self):
        """Small GUI to explore camera frames"""
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print('--. ..- ..')
        print(text)
        print('-... . ..- - -.--')
        root = tk.Tk()
        root.resizable(height=None, width=None)
        ssGUI.ApplicationRemapAnalysis(root, self)
        root.mainloop()
        root.destroy()
