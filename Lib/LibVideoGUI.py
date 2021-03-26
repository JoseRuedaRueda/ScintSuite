"""Graphical elements to explore signal and remaps"""
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import LibPlotting as ssplt
from matplotlib.figure import Figure
from tkinter import ttk


class ApplicationShowVid:
    """Class to show the frames"""

    def __init__(self, master, data):
        """
        Create the window with the sliders

        @param master: Tk() opened
        @param data: the dictionary of experimental frames or remapped ones
        """
        # List of supported colormaps
        self.cmaps = {
            'Gamma_II': ssplt.Gamma_II(),
            'Plasma': plt.get_cmap('plasma')
        }
        names_cmaps = ['Gamma_II', 'Plasma']
        defalut_cmap = 'Plasma'
        # Save here the data
        self.data = data
        # Create a container
        frame = tk.Frame(master)
        # Create the time slider
        t = data['tframes']
        print(t.shape)
        print(t.size)
        dt = t[1] - t[0]
        self.tSlider = tk.Scale(master, from_=t[0], to=t[-1], resolution=dt,
                                command=self.plot_frame,
                                highlightcolor='red',
                                length=400,
                                takefocus=1,
                                label='Time [s]')
        self.tSlider.pack(side='right')
        # create the quit button
        self.qButton = tk.Button(master, text="Quit", command=master.quit)
        self.qButton.pack(side='bottom')
        # Open the figure and show the image
        fig = Figure()
        ax = fig.add_subplot(111)
        self.image = ax.imshow(data['frames'][:, :, 0].squeeze(),
                               origin='lower', cmap=self.cmaps[defalut_cmap])
        self.canvas = tkagg.FigureCanvasTkAgg(fig, master=master)
        # Include the tool bar to zoom and export
        self.toolbar = tkagg.NavigationToolbar2Tk(self.canvas, master)
        self.toolbar.update()
        self.toolbar.pack()
        # include a list with colormaps
        self.selected_cmap = defalut_cmap
        self.cmaps_list = ttk.Combobox(master, values=names_cmaps,
                                       textvariable=self.selected_cmap,
                                       state='readonly')
        self.cmaps_list.set(defalut_cmap)
        self.cmaps_list.bind("<<ComboboxSelected>>",
                             self.change_cmap)
        self.cmaps_list.pack(side='bottom')
        # Draw and show
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()

    def change_cmap(self, cmap_cname):
        """Change frame cmap"""
        self.image.set_cmap(self.cmaps[self.cmaps_list.get()])

    def plot_frame(self, t):
        """Get a plot the frame"""
        t0 = np.float64(t)
        it = np.argmin(abs(self.data['tframes'] - t0))
        dummy = self.data['frames'][:, :, it].squeeze()
        self.image.set_data(dummy)
        self.canvas.draw()
