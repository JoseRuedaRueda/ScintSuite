"""Graphical elements to explore the camera frames"""
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import ScintSuite._Plotting as ssplt
from matplotlib.figure import Figure
from tkinter import ttk


class ApplicationRemap2DAnalyser:
    """Class to show the remap and compare them"""

    def __init__(self, master, vid, traces=None):
        """
        Create the window with the sliders

        :param  master: Tk() opened
        :param  data: the dictionary of experimental frames
        :param  remap_dat: the dictionary of remapped data
        :param  traces: dictionary containing 't1', 'y1', 'l1', and so on, the
            set of traces to be plotted ('l' key is for the label)
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
        self.remap_dat = vid.remap_dat
        t = vid.remap_dat.t.values
        # --- Create a tk container
        frame = tk.Frame(master)
        # Allows to the figure, to resize
        tk.Grid.columnconfigure(master, 0, weight=1)
        tk.Grid.rowconfigure(master, 0, weight=1)
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
        self.tSlider.grid(row=1, column=5)
        # --- Open the figure and show the image
        fig = Figure()
        ax = fig.add_subplot(111)
        self.dummy_frames = vid.remap_dat['frames'].copy()
        self.image = ax.imshow(self.dummy_frames[:, :, 0].squeeze().T,
                               origin='lower', cmap=self.cmaps[defalut_cmap],
                               aspect='auto',
                               extent=[vid.remap_dat['x'].values[0],
                                       vid.remap_dat['x'].values[-1],
                                       vid.remap_dat['y'].values[0],
                                       vid.remap_dat['y'].values[-1]])
        self.time = \
            ax.text(0.8, 0.9, str(round(vid.remap_dat.t.values[0], 3))+' s',
                    transform=ax.transAxes, color='w')
        # Place the figure in a canvas
        self.canvas = tkagg.FigureCanvasTkAgg(fig, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=5,
                                         sticky=tk.N+tk.S+tk.E+tk.W)

        # --- Open the secon figue and show the traces
        fig2 = Figure()
        ax2 = fig2.add_subplot(111)
        # ax2.set_aspect(0.33, adjustable='box')
        ax2.set_xlim(vid.remap_dat.t.values[0], vid.remap_dat.t.values[-1])
        if traces is not None:
            dummy = traces['data'].sel(rho=0.9, method='nearest').values
            ax2.plot(traces.t, dummy, label='n_e(0.9)')
            ax2.legend()

        self.v_line = ax2.axvline(x=vid.remap_dat['t'].values[0], color='g')
        self.canvas2 = tkagg.FigureCanvasTkAgg(fig2, master=master)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(row=0, column=0, columnspan=5)
        # --- Include the tool bar to zoom and export
        # We need a new frame because the toolbar uses 'pack' internally and we
        # are using 'grid' to place our elements, so they are not compatible
        self.toolbarFrame = tk.Frame(master=master)
        self.toolbarFrame.grid(row=7, column=1, columnspan=5)
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
        self.cmaps_list.grid(row=4, column=2)
        self.cmap_list_label = tk.Label(master, text='Select CMap')
        self.cmap_list_label.grid(row=4, column=1)
        # --- Include the boxes for minimum and maximum of the colormap
        clim = self.image.get_clim()
        self.cmap_min_label = tk.Label(master, text='Min CMap')
        self.cmap_min_label.grid(row=3, column=1)
        self.cmap_min = tk.Entry(master)
        self.cmap_min.insert(0, str(round(clim[0])))
        self.cmap_min.grid(row=3, column=2)

        self.cmap_max_label = tk.Label(master, text='Max CMap')
        self.cmap_max_label.grid(row=3, column=3)
        self.cmap_max = tk.Entry(master)
        self.cmap_max.insert(0, str(round(clim[1])))
        self.cmap_max.grid(row=3, column=4)
        # --- Include the button for minimum and maximum of the colormap
        self.cmap_scale_button = tk.Button(master, text='Set Scale',
                                           command=self.set_scale)
        self.cmap_scale_button.grid(row=3, column=5)
        # --- reference times
        # If there is not remap data, deactivate the button
        # Initialise the variable of the button
        self.checkVar1 = tk.BooleanVar()
        self.checkVar1.set(False)
        # Create the button
        self.reference_button = tk.Button(master, text="Set Reference",
                                          command=self.reference_button_change,
                                          takefocus=0)
        self.reference_button.grid(row=5, column=3)
        # Create the button
        self.reset_button = tk.Button(master, text="Reset", command=self.reset,
                                      takefocus=0)
        self.reset_button.grid(row=5, column=4)
        # Create the tbox
        self.t_ref_label = tk.Label(master, text='t ref')
        self.t_ref_label.grid(row=5, column=1)
        self.t_ref = tk.Entry(master)
        self.t_ref.insert(0, str(round(0.0)))
        self.t_ref.grid(row=5, column=2)
        # referencias
        self.Gyr_ref_label = tk.Label(master, text='Gyr ref')
        self.Gyr_ref_label.grid(row=6, column=1)
        self.Gyr_ref = tk.Entry(master)
        self.Gyr_ref.insert(0, str(2.5))
        self.Gyr_ref.grid(row=6, column=2)

        self.XI_ref_label = tk.Label(master, text='XI ref')
        self.XI_ref_label.grid(row=6, column=3)
        self.XI_ref = tk.Entry(master)
        self.XI_ref.insert(0, str(1.77))
        self.XI_ref.grid(row=6, column=4)
        # Draw and show

        # In the last line, the sticky argument, combined with the
        # Grid.columnconfiguration and Grid.rowconfiguration from above
        # allows to the canvas to resize when I resize the window
        # --- Quit button
        self.qButton = tk.Button(master, text="Quit", command=master.quit)
        self.qButton.grid(row=4, column=5)
        frame.grid()

    def change_cmap(self, cmap_cname):
        """Change frame cmap"""
        self.image.set_cmap(self.cmaps[self.cmaps_list.get()])
        self.canvas.draw()

    def plot_frame(self, t):
        """Update the plot"""
        t0 = float(t)
        it = np.argmin(abs(self.remap_dat['t'].values - t0))
        if self.checkVar1.get():
            dummy = self.dummy_frames[:, :, it].squeeze().copy().astype(float)
            # dummy /= dummy[self.ixi, self.igyr]
            dummy /= self.scale
            dummy -= self.ref_frame
        else:
            dummy = self.dummy_frames[:, :, it].squeeze().copy()
        self.time.set_text(str(round(self.remap_dat['t'].values[it], 3)) + ' s')
        self.image.set_data(dummy.T)
        self.canvas.draw()
        self.v_line.set_xdata([t0, t0])
        self.canvas2.draw()

    def set_scale(self):
        """Set color scale"""
        cmin = float(self.cmap_min.get())
        cmax = float(self.cmap_max.get())
        self.image.set_clim(cmin, cmax)
        self.canvas.draw()

    def reference_button_change(self):
        """Set the reference"""
        # If it was false, set the bolean to true
        if not self.checkVar1.get():
            self.checkVar1.set(not self.checkVar1.get())
        t0 = float(self.t_ref.get())
        it = np.argmin(abs(self.remap_dat['t'].values - t0))
        self.ref_frame = \
            self.remap_dat['frames'].values[:, :, it].squeeze().copy()
        # --- Get the normalization
        g0 = float(self.Gyr_ref.get())
        x0 = float(self.XI_ref.get())
        self.igyr = np.argmin(np.abs(self.remap_dat['y'].values - g0))
        self.ixi = np.argmin(np.abs(self.remap_dat['x'].values - x0))
        self.scale = self.ref_frame[self.ixi, self.igyr]
        self.ref_frame /= self.scale
        print('Updated Ref frame')

    def reset(self):
        """Reset the situation"""
        if self.checkVar1.get():
            self.checkVar1.set(not self.checkVar1.get())
        self.dummy_frames = self.remap_dat['frames'].copy()
