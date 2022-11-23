"""
Graphical elements to explore the remap

Under development, use under your own risk
"""
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import Lib._Plotting as ssplt
from matplotlib.figure import Figure
from tkinter import ttk


class ApplicationShowProfiles:
    """Class to show the camera frames and the remap"""

    def __init__(self, master, data, remap_dat):
        """
        Create the window with the sliders

        :param  master: Tk() opened
        :param  data: the dictionary of experimental frames
        :param  remap_dat: the dictionary containing the remap_dat
        """
        print('This GUI is under development')
        # --- List of supported colormaps
        self.cmaps = {
            'Gamma_II': ssplt.Gamma_II(),
            'Plasma': plt.get_cmap('plasma'),
            'Cai': ssplt.Cai()
        }
        names_cmaps = ['Gamma_II', 'Plasma', 'Cai']
        defalut_cmap = 'Plasma'
        # --- List of supported interpolators for the remap
        self.interpolators = [
            'none', 'nearest', 'bilinear',
            'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
            'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
            'sinc', 'lanczos', 'blackman'
        ]
        default_interpolator = 'bilinear'
        # --- Initialise the data container
        self.data = data
        self.remap_dat = remap_dat
        t = data['t'].values
        # --- Create a tk container
        frame = tk.Frame(master)
        # Allows to the figures, to resize
        tk.Grid.columnconfigure(master, 0, weight=1)
        tk.Grid.rowconfigure(master, 0, weight=1)
        tk.Grid.columnconfigure(master, 6, weight=1)
        # --- Create the time slider
        # dt for the slider
        dt = t[1] - t[0]
        # Slider creation
        self.tSlider = tk.Scale(master, from_=t[0], to=t[-1], resolution=dt,
                                command=self.plot_frame,
                                highlightcolor='red',
                                length=400,
                                takefocus=1,
                                label='Time [s]')
        # Put the slider in place
        self.tSlider.grid(row=0, column=5)
        # --- Open the figure and show the camera frame
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel('Signal')
        ax.set_xlabel('Gyroradius [cm]')
        self.ax_gyr = ax
        self.gyr, = \
            ax.plot(remap_dat['yaxis'], remap_dat['sprofy'][:, 0].squeeze())
        self.ax_gyr.set_ylim(0.0, remap_dat['sprofy'].max())
        # Place the figure in a canvas
        self.canvas = tkagg.FigureCanvasTkAgg(fig, master=master)
        # --- Include the tool bar to zoom and export
        # We need a new frame because the toolbar uses 'pack' internally and we
        # are using 'grid' to place our elements, so they are not compatible
        self.toolbarFrame = tk.Frame(master=master)
        self.toolbarFrame.grid(row=4, column=0, columnspan=5)
        self.toolbar = \
            tkagg.NavigationToolbar2Tk(self.canvas, self.toolbarFrame)
        self.toolbar.update()
        # --- Include the list with colormaps
        # self.selected_cmap = defalut_cmap
        # self.cmaps_list = ttk.Combobox(master, values=names_cmaps,
        #                                textvariable=self.selected_cmap,
        #                                state='readonly')
        # self.cmaps_list.set(defalut_cmap)
        # self.cmaps_list.bind("<<ComboboxSelected>>",
        #                      self.change_cmap)
        # self.cmaps_list.grid(row=3, column=2)
        # self.cmap_list_label = tk.Label(master, text='Camera CMap')
        # self.cmap_list_label.grid(row=3, column=1)
        # --- Include the boxes for minimum and maximum of the colormap
        # clim = self.image.get_clim()
        # self.cmap_min_label = tk.Label(master, text='Min CMap')
        # self.cmap_min_label.grid(row=2, column=1)
        # self.cmap_min = tk.Entry(master)
        # self.cmap_min.insert(0, str(round(clim[0])))
        # self.cmap_min.grid(row=2, column=2)
        #
        # self.cmap_max_label = tk.Label(master, text='Max CMap')
        # self.cmap_max_label.grid(row=2, column=3)
        # self.cmap_max = tk.Entry(master)
        # self.cmap_max.insert(0, str(round(clim[1])))
        # self.cmap_max.grid(row=2, column=4)
        # # --- Include the button for minimum and maximum of the colormap
        # self.cmap_scale_button = tk.Button(master, text='Set Scale',
        #                                    command=self.set_scale)
        # self.cmap_scale_button.grid(row=2, column=5)
        # --- Button for the strike map:
        # If there is not remap data, deactivate the button
        if self.remap_dat is None:
            state = tk.DISABLED
        else:
            state = tk.NORMAL
        # Initialise the variable of the button
        self.checkVar1 = tk.BooleanVar()
        self.checkVar1.set(False)
        # Create the button
        # self.smap_button = tk.Button(master, text="Draw SMap",
        #                              command=self.smap_Button_change,
        #                              takefocus=0, state=state)
        # self.smap_button.grid(row=3, column=3)
        # Draw and show
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=5,
                                         sticky=tk.N+tk.S+tk.E+tk.W)
        # In the last line, the sticky argument, combined with the
        # Grid.columnconfiguration and Grid.rowconfiguration from above
        # allows to the canvas to resize when I resize the window
        # ---------------------------------------------------------------------
        # --- Remap data
        # ---------------------------------------------------------------------
        # --- Open the figure and show the remap
        fig2 = Figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_ylabel('Signal')
        ax2.set_xlabel('Pitch [$\\degree$]')
        # --- Draw the second image
        # vmax = remap_dat['frames'].max() * 0.8
        self.ax_pitch = ax2
        self.ax_pitch.set_ylim(0.0, remap_dat['sprofx'].max())
        self.pitch, = \
            ax2.plot(remap_dat['xaxis'], remap_dat['sprofx'][:, 0].squeeze())
        # Place the figure in a canvas
        self.canvas2 = tkagg.FigureCanvasTkAgg(fig2, master=master)
        # --- Include the tool bar to zoom and export
        # We need a new frame because the toolbar uses 'pack' internally and we
        # are using 'grid' to place our elements, so they are not compatible
        self.toolbarFrame2 = tk.Frame(master=master)
        self.toolbarFrame2.grid(row=4, column=6, columnspan=5)
        self.toolbar2 = \
            tkagg.NavigationToolbar2Tk(self.canvas2, self.toolbarFrame2)
        self.toolbar2.update()
        # --- Include the list with colormaps
        # self.selected_cmap2 = defalut_cmap
        # self.cmaps_list2 = ttk.Combobox(master, values=names_cmaps,
        #                                 textvariable=self.selected_cmap2,
        #                                 state='readonly')
        # self.cmaps_list2.set(defalut_cmap)
        # self.cmaps_list2.bind("<<ComboboxSelected>>",
        #                       self.change_cmap2)
        # self.cmaps_list2.grid(row=3, column=7)
        # self.cmap_list_label2 = tk.Label(master, text='Remap CMap')
        # self.cmap_list_label2.grid(row=3, column=6)
        # # --- Include the boxes for minimum and maximum of the colormap
        # clim2 = self.image2.get_clim()
        # self.cmap_min_label2 = tk.Label(master, text='Min CMap')
        # self.cmap_min_label2.grid(row=2, column=6)
        # self.cmap_min2 = tk.Entry(master)
        # self.cmap_min2.insert(0, str(int(clim2[0])))
        # self.cmap_min2.grid(row=2, column=7)
        #
        # self.cmap_max_label2 = tk.Label(master, text='Max CMap')
        # self.cmap_max_label2.grid(row=2, column=8)
        # self.cmap_max2 = tk.Entry(master)
        # self.cmap_max2.insert(0, str(int(clim2[1])))
        # self.cmap_max2.grid(row=2, column=9)
        # # --- Include the button for minimum and maximum of the colormap
        # self.cmap_scale_button2 = tk.Button(master, text='Set Scale',
        #                                     command=self.set_scale2)
        # self.cmap_scale_button2.grid(row=2, column=10)
        # # --- Include the list with interpolators
        # self.selected_interpolator = default_interpolator
        # self.inter_list = ttk.Combobox(master, values=self.interpolators,
        #                                state='readonly',
        #                                textvariable=self.selected_interpolator)
        # self.inter_list.set(default_interpolator)
        # self.inter_list.bind("<<ComboboxSelected>>",
        #                      self.change_interpolator)
        # self.inter_list.grid(row=3, column=9)
        # self.inter_list_label = tk.Label(master, text='Interp Remap')
        # self.inter_list_label.grid(row=3, column=8)
        # --- Draw and show the canvas
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(row=0, column=6, columnspan=5,
                                          sticky=tk.N+tk.S+tk.E+tk.W)
        # In the last line, the sticky argument, combined with the
        # Grid.columnconfiguration and Grid.rowconfiguration from above
        # allows to the canvas to resize when I resize the window

        # --- Quit button
        self.qButton = tk.Button(master, text="Quit", command=master.quit)
        self.qButton.grid(row=3, column=5)
        frame.grid()

    def change_cmap(self, cmap_cname):
        """Change frame cmap"""
        self.image.set_cmap(self.cmaps[self.cmaps_list.get()])
        self.canvas.draw()

    def change_cmap2(self, cmap_cname):
        """Change remap cmap"""
        self.image2.set_cmap(self.cmaps[self.cmaps_list2.get()])
        self.canvas2.draw()

    def change_interpolator(self, interp_name):
        """Change interpolator of the remap plot"""
        self.image2.set_interpolation(self.inter_list.get())
        self.canvas2.draw()

    def plot_frame(self, t):
        """Plot the new frame"""
        t0 = np.float64(t)
        it = np.argmin(abs(self.data['t'].values - t0))
        self.gyr.set_ydata(self.remap_dat['sprofy'][:, it].squeeze())
        # If needed, plot the smap
        self.canvas.draw()
        # Now chane the remap
        self.pitch.set_ydata(self.remap_dat['sprofx'][:, it].squeeze())
        self.canvas2.draw()

    def set_scale(self):
        """Set color scale"""
        cmin = int(self.cmap_min.get())
        cmax = int(self.cmap_max.get())
        self.image.set_clim(cmin, cmax)
        self.canvas.draw()

    def set_scale2(self):
        """Set color scale remap"""
        cmin = int(self.cmap_min2.get())
        cmax = int(self.cmap_max2.get())
        self.image2.set_clim(cmin, cmax)
        self.canvas2.draw()

    def smap_Button_change(self):
        """Decide to plot or not Smap"""
        # If it was true and we push the button, the smap should be deleted:
        if self.checkVar1.get():
            ssplt.remove_lines(self.canvas.figure.axes[0])
        # Now update the value
        self.checkVar1.set(not self.checkVar1.get())
        print('Draw Smap :', self.checkVar1.get())
        self.canvas.draw()
