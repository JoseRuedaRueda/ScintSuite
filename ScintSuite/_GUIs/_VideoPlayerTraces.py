"""Graphical elements to explore the camera frames"""
import os
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import ScintSuite._Plotting as ssplt
import ScintSuite.SimulationCodes.SINPA as sssinpa
import ScintSuite._Mapping as ssmap
from matplotlib.figure import Figure
from tkinter import ttk


class ApplicationShowVidAndTraces:
    """Class to show the camera frames, thought for INPA video"""

    def __init__(self, master, video):
        """
        Create the window with the sliders

        :param  master: Tk() opened
        :param  video: The INPA video
        """
        # --- List of supported colormaps
        self.cmaps = {
            'Cai': ssplt.Cai(),
            'Greys': plt.get_cmap('Greys_r'),
            'Gamma_II': ssplt.Gamma_II(),
            'Plasma': plt.get_cmap('plasma'),
            'BWR': plt.get_cmap('bwr')
        }
        names_cmaps = ['Cai', 'Gamma_II', 'Greys', 'Plasma', 'BWR']
        defalut_cmap = 'Greys'
        # --- Initialise the data container
        self.video = video

        t = self.video.exp_dat['t'].values
        # --- Create a tk container
        frame = tk.Frame(master)
        # Allows to the figure, to resize
        tk.Grid.columnconfigure(master, 0, weight=1)
        tk.Grid.rowconfigure(master, 2, weight=1)
        # --- Create the time slider
        # dt for the slider
        if t is None:
            raise Exception('You might need to run vid.read_frames() first')
        dt = t[1] - t[0]
        # Slider creation
        self.tSlider = tk.Scale(master, from_=t[0], to=t[-1], resolution=dt,
                                command=self.plot_frame,
                                highlightcolor='red',
                                length=400,
                                takefocus=1,
                                label='Time [s]')
        # Put the slider in place
        self.tSlider.grid(row=2, column=5)
        # --- Open the figure and plot the profiles
        figData = Figure()
        self.neAxis = figData.add_subplot(211)
        self.NBIAxis = figData.add_subplot(212)
        dummy = self.video._ne['data'].sel(rho=0.9, method='nearest').values
        self.neAxis.plot(self.video._ne.t, dummy, label='n_e(0.9)')
        self.neAxis.legend()
        dummy = self.video._NBIpower.sum(dim='number').values
        self.NBIAxis.plot(self.video._NBIpower.t, dummy, label='Total')
        dummy = self.video._NBIpower.values[2,:]
        self.NBIAxis.plot(self.video._NBIpower.t, dummy, label='NBI3')
        self.NBIAxis.set_xlim(t[0], t[-1])
        self.neAxis.set_xlim(t[0], t[-1])
        self.NBIAxis.legend()
        # Add the vertical lines
        self.v_line = self.neAxis.axvline(x=self.video.exp_dat['t'][0],
                                          color='g')
        self.v_line2 = self.NBIAxis.axvline(x=self.video.exp_dat['t'][0],
                                            color='g')
        self.canvasData = tkagg.FigureCanvasTkAgg(figData, master=master)

        self.canvasData.draw()
        self.canvasData.get_tk_widget().grid(row=0, column=0, columnspan=5,
                                             rowspan=2,
                                             sticky=tk.N+tk.S+tk.E+tk.W)

        # --- Open the Frame figure and show the image
        fig = Figure()
        ax = fig.add_subplot(111)
        self.image = ax.imshow(video.exp_dat['frames'].values[:, :,
                               0].squeeze(),
                               origin='lower', cmap=self.cmaps[defalut_cmap],
                               aspect='equal')
        self.time = \
            ax.text(0.85, 0.9, '%5.3f s'%video.exp_dat['t'].values[0],
                    transform=ax.transAxes, color='w')
        if video.shot is not None:
            self.shotLabel = \
                ax.text(0.05, 0.9, '#%i'%video.shot,
                        transform=ax.transAxes, color='w')
        else:
            self.shotLabel = None
        # Place the figure in a canvas
        self.canvas = tkagg.FigureCanvasTkAgg(fig, master=master)
        # --- Include the tool bar to zoom and export
        # We need a new frame because the toolbar uses 'pack' internally and we
        # are using 'grid' to place our elements, so they are not compatible
        self.toolbarFrame = tk.Frame(master=master)
        self.toolbarFrame.grid(row=6, column=1, columnspan=5)
        self.toolbar = \
            tkagg.NavigationToolbar2Tk(self.canvas, self.toolbarFrame)
        self.toolbar.update()
        # --- Include the list with colormaps
        self.selected_cmap = defalut_cmap
        self.cmaps_list = ttk.Combobox(master, values=names_cmaps,
                                       textvariable=self.selected_cmap,
                                       state='readonly')
        self.cmaps_list.set(defalut_cmap)
        self.cmaps_list.bind("<<ComboboxSelected>>",
                             self.change_cmap)
        self.cmaps_list.grid(row=5, column=2)
        self.cmap_list_label = tk.Label(master, text='Select CMap')
        self.cmap_list_label.grid(row=5, column=1)
        # --- Include the boxes for minimum and maximum of the colormap
        clim = self.image.get_clim()
        self.cmap_min_label = tk.Label(master, text='Min CMap')
        self.cmap_min_label.grid(row=4, column=1)
        self.cmap_min = tk.Entry(master)
        self.cmap_min.insert(0, str(round(clim[0])))
        self.cmap_min.grid(row=4, column=2)

        self.cmap_max_label = tk.Label(master, text='Max CMap')
        self.cmap_max_label.grid(row=4, column=3)
        self.cmap_max = tk.Entry(master)
        self.cmap_max.insert(0, str(round(clim[1])))
        self.cmap_max.grid(row=4, column=4)
        # --- Include the button for minimum and maximum of the colormap
        self.cmap_scale_button = tk.Button(master, text='Set Scale',
                                           command=self.set_scale)
        self.cmap_scale_button.grid(row=4, column=5)
        # --- Button for the strike map:
        # If there is not remap data, deactivate the button
        if self.video.remap_dat is None:
            state = tk.DISABLED
        else:
            state = tk.NORMAL
        # Initialise the variable of the button
        self.checkVar1 = tk.BooleanVar()
        self.checkVar1.set(False)
        # Create the button
        self.smap_button = tk.Button(master, text="Draw SMap",
                                     command=self.smap_Button_change,
                                     takefocus=0, state=state)
        self.smap_button.grid(row=5, column=3)
        # --- Button for the Scintillator:
        # If there is not scintillator data, deactivate the button
        if self.video.scintillator is None:
            state = tk.DISABLED
        else:
            state = tk.NORMAL
        # Initialise the variable of the button
        self.checkVar2 = tk.BooleanVar()
        self.checkVar2.set(False)
        # Create the button
        self.scint_button = tk.Button(master, text="Draw Scint",
                                      command=self.scint_Button_change,
                                      takefocus=0, state=state)
        self.scint_button.grid(row=5, column=4)
        # Draw and show
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=5,
                                         sticky=tk.N+tk.S+tk.E+tk.W)
        # In the last line, the sticky argument, combined with the
        # Grid.columnconfiguration and Grid.rowconfiguration from above
        # allows to the canvas to resize when I resize the window
        # --- Quit button
        self.qButton = tk.Button(master, text="Quit", command=master.quit)
        self.qButton.grid(row=5, column=5)
        frame.grid()

    def change_cmap(self, cmap_cname):
        """Change frame cmap"""
        self.image.set_cmap(self.cmaps[self.cmaps_list.get()])
        self.canvas.draw()

    def plot_frame(self, t):
        """Update the plot"""
        t0 = float(t)
        it = np.argmin(abs(self.video.exp_dat['t'].values - t0))
        dummy = self.video.exp_dat['frames'].values[:, :, it].squeeze().copy()
        self.time.set_text(str(round(self.video.exp_dat['t'].values[it], 3)) + ' s')
        self.image.set_data(dummy)
        self.v_line.set_xdata([t0, t0])
        self.v_line2.set_xdata([t0, t0])
        # If needed, plot the smap
        if self.checkVar1.get():
            # remove the old one
            ssplt.remove_lines(self.canvas.figure.axes[0])
            # choose the new one:
            # get parameters of the map
            theta_used = self.video.remap_dat['theta_used'].values[it]
            phi_used = self.video.remap_dat['phi_used'].values[it]

            # Get the full name of the file
            name__smap = sssinpa.execution.guess_strike_map_name(
                phi_used, theta_used, geomID=self.video.geometryID,
                decimals=self.video.remap_dat['frames'].attrs['decimals'])
            smap_folder = self.video.remap_dat['frames'].attrs['smap_folder']
            full_name_smap = os.path.join(smap_folder, name__smap)
            # Load the map:
            smap = ssmap.StrikeMap(0, full_name_smap)
            # Calculate pixel coordinates
            smap.calculate_pixel_coordinates(self.video.CameraCalibration)
            # Plot the map
            self.xlim = self.canvas.figure.axes[0].get_xlim()
            self.ylim = self.canvas.figure.axes[0].get_ylim()
            smap.plot_pix(ax=self.canvas.figure.axes[0], labels=False)
            self.canvas.figure.axes[0].set_xlim(self.xlim[0], self.xlim[1])
            self.canvas.figure.axes[0].set_ylim(self.ylim[0], self.ylim[1])
            # Plot the scintillator:
        if self.checkVar2.get():
            self.video.scintillator.plot_pix(ax=self.canvas.figure.axes[0])
        self.canvas.draw()
        self.canvasData.draw()

    def set_scale(self):
        """Set color scale"""
        cmin = float(self.cmap_min.get())
        cmax = float(self.cmap_max.get())
        self.image.set_clim(cmin, cmax)
        self.canvas.draw()

    def smap_Button_change(self):
        """Decide to plot or not Smap"""
        # If it was true and we push the button, the smap should be deleted:
        if self.checkVar1.get():
            ssplt.remove_lines(self.canvas.figure.axes[0])
        # Now update the value
        self.checkVar1.set(not self.checkVar1.get())
        print('Draw Smap :', self.checkVar1.get())
        self.canvas.draw()

    def scint_Button_change(self):
        """Decide to plot or not the scintillator"""
        # If it was true and we push the button, the smap should be deleted:
        if self.checkVar2.get():
            ssplt.remove_lines(self.canvas.figure.axes[0])
        # Now update the value
        self.checkVar2.set(not self.checkVar2.get())
        print('Draw scintillator :', self.checkVar2.get())
        self.canvas.draw()
