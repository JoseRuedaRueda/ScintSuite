"""
i-HIBPgui to plot the videos of iHIBP.

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import os
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import Lib._Plotting as libplt
import Lib._Video as libvideo
import Lib._Paths as lpaths
import Lib.SimulationCodes.iHIBPsim.strikes as ihibpstrikes

paths = lpaths.Path('AUG')

class app_ihibp_vid:
    """
    Class containing the data to create a GUI to plot the iHIBP videos and
    overplot the strikeline, when computed.
    """

    def __init__(self, tkwindow, shotnumber: int, path: str=None, **kwargs):
        """
        Initializes the class with the neccessary data to create the GUI and
        smooth plot the i-HIBP videos.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        :param  tkwindow: window handler.
        :param  shotnumber: shotnumber of the video to load.
        :param  path: path to the simulation with iHIBPsim to plot the lines.
        """
        self.TKwindow   = tkwindow
        self.shotnumber = shotnumber
        self.strikeline_on = path is not None

        self.video = libvideo.iHIBPvideo(shot=shotnumber,
                                         **kwargs)

        # Setting the colormap options.
        self.cmaps = {
            'Cai': libplt.Cai(),
            'Greys': plt.get_cmap('Greys_r'),
            'Gamma_II': libplt.Gamma_II(),
            'Plasma': plt.get_cmap('plasma'),
        }
        names_cmaps = ['Cai', 'Gamma_II', 'Greys', 'Plasma']
        defalut_cmap = 'Plasma'

        self.frame = tk.Frame(tkwindow)
        # Allows to the figure, to resize
        tk.Grid.columnconfigure(tkwindow, 0, weight=1)
        tk.Grid.rowconfigure(tkwindow, 0, weight=1)

        # --- Create the time slider
        # dt for the slider
        dt = self.video.exp_dat['t'].values[1] - self.video.exp_dat['t'].values[0]
        # Slider creation
        self.tSlider = tk.Scale(tkwindow,
                                from_=self.video.exp_dat['t'].values[0],
                                to=self.video.exp_dat['t'].values[-1],
                                resolution=dt,
                                command=self.plot_frame,
                                highlightcolor='red',
                                length=400,
                                takefocus=1,
                                label='Time [s]')

        # Put the slider in place
        self.tSlider.grid(row=0, column=3)

        # Plotting the first video.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.image1 = ax.imshow(self.video.exp_dat['frames'].values[:, :, 0].squeeze(),
                                 origin='lower', cmap=self.cmaps[defalut_cmap],
                                aspect='equal')

        # Place the figure in a canvas
        self.canvas1 = tkagg.FigureCanvasTkAgg(fig, master=tkwindow)
        # --- Include the tool bar to zoom and export
        # We need a new frame because the toolbar uses 'pack' internally and we
        # are using 'grid' to place our elements, so they are not compatible
        self.toolbarFrame1 = tk.Frame(master=tkwindow)
        self.toolbarFrame1.grid(row=1, column=0, columnspan=2)
        self.toolbar1 = \
            tkagg.NavigationToolbar2Tk(self.canvas1, self.toolbarFrame1)
        self.toolbar1.update()

        # Reading in the strike line maps.
        if path is not None:
            self.strline = ihibpstrikes.strikeLine(path,
                                                   shotnumber=self.shotnumber)
        fig2 = plt.figure()
        self.canvas2 = tkagg.FigureCanvasTkAgg(fig2, master=tkwindow)
        self.ax2 = fig2.add_subplot(111)
        if path is not None:
            self.image2 = self.strline.plotStrikeLine(self.video.exp_dat['t'].values[0],
                                                      ax=self.ax2)
        self.toolbarFrame2 = tk.Frame(master=tkwindow)
        self.toolbarFrame2.grid(row=1, column=2, columnspan=1)
        self.toolbar2 = \
            tkagg.NavigationToolbar2Tk(self.canvas2, self.toolbarFrame2)
        self.toolbar2.update()


        # Adding quit button
        self.quit_button = tk.Button(tkwindow, text="Quit",
                                     command=tkwindow.quit)
        self.quit_button.grid(row=1, column=3)

        # Adding the colormap
        self.selected_cmap = defalut_cmap
        self.cmap_list_label = tk.Label(tkwindow, text='Colormap')
        self.cmap_list_label.grid(row=2, column=0)
        self.cmaps_list = tk.ttk.Combobox(tkwindow, values=names_cmaps,
                                      textvariable=self.selected_cmap,
                                      state='readonly')
        self.cmaps_list.set(defalut_cmap)
        self.cmaps_list.bind("<<ComboboxSelected>>",
                             self.change_cmap)
        self.cmaps_list.grid(row=3, column=0)

        # --- Include the boxes for minimum and maximum of the colormap
        clim = self.image1.get_clim()
        self.cmap_min_label = tk.Label(tkwindow, text='Min.')
        self.cmap_min_label.grid(row=2, column=1)
        self.cmap_min = tk.Entry(tkwindow)
        self.cmap_min.insert(0, str(round(clim[0])))
        self.cmap_min.grid(row=2, column=2)

        self.cmap_max_label = tk.Label(tkwindow, text='Max.')
        self.cmap_max_label.grid(row=3, column=1)
        self.cmap_max = tk.Entry(tkwindow)
        self.cmap_max.insert(0, str(round(clim[1])))
        self.cmap_max.grid(row=3, column=2)
        # --- Include the button for minimum and maximum of the colormap
        self.cmap_scale_button = tk.Button(tkwindow, text='Set Scale',
                                           command=self.set_scale)
        self.cmap_scale_button.grid(row=2, column=3)

        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(row=0, column=0, columnspan=2,
                                          sticky=tk.N+tk.S+tk.E+tk.W)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(row=0, column=2, columnspan=1,
                                          sticky=tk.N+tk.S+tk.E+tk.W)

    def set_scale(self):
        """Set color scale"""
        cmin = int(self.cmap_min.get())
        cmax = int(self.cmap_max.get())
        self.image1.set_clim(cmin, cmax)
        self.canvas1.draw()

    def change_cmap(self, cmap_cname):
        """Change frame cmap"""
        self.image1.set_cmap(self.cmaps[self.cmaps_list.get()])
        self.canvas1.draw()

    def plot_frame(self, t: str):
        """Update the plot"""
        t0 = float(t)
        it = np.abs(self.video.exp_dat['t'].values - t0).argmin()
        dummy = self.video.exp_dat['frames'].values[:, :, it].squeeze()
        self.image1.set_data(dummy)

        # Updating the strikeline.
        if self.strikeline_on:
            if t0 > self.strline.time.max():
                self.image2[0].set_xdata([])
                self.image2[0].set_ydata([])
            else:
                it = np.abs(self.strline.time-t0).argmin()
                self.image2[0].set_xdata(self.strline.maps[it]['x1']*100)
                self.image2[0].set_ydata(self.strline.maps[it]['x2']*100)

        # If needed, plot the smap

        self.canvas1.draw()
        self.canvas2.draw()
