"""
Object to work with the VRT cameras.

"""

from Lib._Video._BasicVideoObject import BVO
import xml.etree.ElementTree as et
import numpy as np
import tkinter as tk                       # To open UI windows
import Lib._GUIs as ssGUI                   # For GUI elements
import matplotlib.pyplot as plt
import Lib._TimeTrace as tt
import matplotlib.colors as colors
import Lib._Plotting as ssplt
import Lib._IO as ssio
import os
from Lib._Machine import machine
import Lib._VRT as vrt

# import numpy as np
# import Lib._Mapping as ssmap
# import Lib.LibData as ssdat
# import Lib.SimulationCodes.FILDSIM as ssFILDSIM
# from Lib.version_suite import version
# from scipy.io import netcdf                # To export remap data
# from tqdm import tqdm                      # For waitbars
import Lib._Paths as p
pa = p.Path(machine)
del p


class VRTVideo(BVO):
    """
    Video class for the VRT cameras

    Heavily based on FILDVideoObject.py by Jose Rueda (jrrueda@us.es)

    Javier Hidalgo-Salaverri (jhsalaverri@us.es)

    """
    def __init__(self, camera: str, shot: int):
        """
        Initialise the class

        :param  camera: Camera name
        :param  shot: Shot number

        """
        # Initialise the parent class
        folder= '/afs/ipp/u/augd/rawfiles/VRT/'+str(shot)[0:2]+'/S'+str(shot)
        self.path_video = folder+'/S'+str(shot)+'_'+camera+'.mp4'
        self.path_time = folder+'/Prot/FrameProt/'+camera+'_FrameProt.xml'
        self.properties = {}

        # Get the time base and the corresponding frame number
        root = et.parse(self.path_time).getroot()
        TS6 = int(root.attrib['ts6Time'],0)
        time = []
        nframe = []
        for r in root.findall('Entry'):
            time.append((int(r.attrib['time'],0)-TS6)*1e-9)
            nframe.append(int(r.attrib['frameNumber']))
        self.timebase = np.array(time, dtype=float)

        BVO.__init__(self, file = self.path_video, shot = shot)
        self.exp_dat['frames'] = self.exp_dat['frames'][::-1,:,:]

        self.exp_dat['tframes'] = np.array(time)
        self.exp_dat['nframes'] = np.array(nframe)
        self.shot = shot
        self.camera = camera


    def GUI_frames(self, calibrated: bool = False):
        """Small GUI to explore camera frames

        :param  calibrated: return the GUI in terms of temperature -> currently
        not working properly
        Changing colormap scale is also bugged
        """
        text = 'Press TAB until the time slider is highlighted in red.'\
            + ' Once that happend, you can move the time with the arrows'\
            + ' of the keyboard, frame by frame'
        print('-------------------')
        print(text)
        print('-------------------')
        root = tk.Tk()
        root.resizable(height=None, width=None)
        if calibrated:
            # Get the gain and shutter position for that shot
            camrange = 1024
            camera_list = vrt.get_cameras(self.shot)
            GA = camera_list[self.camera]['GA']
            SH = camera_list[self.camera]['SH']
            Tcal = vrt.get_calibration(self.camera, self.shot, GA, SH)
            exp_dat = self.exp_dat

            exp_dat['frames'] = np.interp(exp_dat['frames'],
                                          np.arange(camrange), Tcal)
            ssGUI.ApplicationShowVid(root, exp_dat, None)
        else:
            ssGUI.ApplicationShowVid(root, self.exp_dat, None)
        root.mainloop()
        root.destroy()

    def plot_frame(self, frame_number=None, ax=None, ccmap=None,
                   t: float = None,
                   verbose: bool = True,
                   vmin=0, vmax=None,
                   xlim=None, ylim=None, scale: str = 'linear'):
        """
        Plot a frame from the loaded frames

        :param  frame_number: Number of the frame to plot, relative to the video
            file, optional
        :param  ax: Axes where to plot, is none, just a new axes will be created
        :param  ccmap: colormap to be used, if none, Gamma_II from IDL. To be
            changed to the Hot colormap
        :param  verbose: If true, info of the theta and phi used will be printed
        :param  vmin: Minimum value for the color scale to plot
        :param  vmax: Maximum value for the color scale to plot
        :param  xlim: tuple with the x-axis limits
        :param  ylim: tuple with the y-axis limits
        :param  scale: Scale for the plot: 'linear', 'sqrt', or 'log'

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
            frame_index = np.argmin(np.abs(self.exp_dat['tframes'].values - t))
            tf = self.exp_dat['tframes'].values[frame_index]
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
        # Arrange the axes:
        if created:
            fig.show()
            plt.tight_layout()
        return ax

    def create_roi(self, time: float = 0.0, save: bool = False,
                   filename: str = ''):
        """
        Define a ROI to be used afterwards. It can be saved

        This is a wrapper for the create_roi function of the LibTimeTraces
        library by Jose Rueda (joserrueda@us.es)
        @
        """
        # Plot the chosen time
        self.plot_frame(t = time)
        fig = plt.gcf()
        fig,roi = tt.create_roi(fig)

        if save:
            if filename == '':
                folder = os.path.join(pa.Results, 'VRT/masks/')
                filename = folder+'S'+str(self.shot)+'_'+self.camera
                file_id = 1
                exists = True
                while exists == True:
                    if os.path.isfile(filename+'_'+str(file_id)+'.nc'):
                        file_id += 1
                    else:
                        exists = False

                filename = filename+'_'+str(file_id)+'.nc'

            # Get the corresponding frame
            frame_index = np.argmin(abs(self.exp_dat['tframes'] - time))
            # tf = self.exp_dat['tframes'][frame_index]
            frame = self.exp_dat['frames'][:, :, frame_index].squeeze()
            mask = roi.get_mask(frame)
            ssio.save_mask(mask = mask,filename = filename, shot = self.shot,
                           frame = frame)
        return roi
