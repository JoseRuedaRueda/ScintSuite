"""Graphical elements to explore the camera frames"""
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
from tkinter import ttk


class ApplicationRemapAnalysis:
    """Class to compare the profiles from the remap"""

    def __init__(self, master, vid, translation: tuple = None):
        """
        Create the window with the sliders

        @param master: Tk() opened
        @param data: the dictionary of experimental frames
        @param video: the INPA/FILD video object
        """
        # --- Initialise the data container
        self.vid = vid
        t = vid.remap_dat['tframes']
        # --- Create a tk container
        frame = tk.Frame(master)
        # Allows to the figure, to resize
        tk.Grid.columnconfigure(master, 0, weight=1)
        tk.Grid.rowconfigure(master, 0, weight=1)
        # --- Variable selection
        self.selected_variable = 'xi'
        self.variable_list = ttk.Combobox(master, values=['xi', 'rl'],
                                          textvariable=self.selected_variable,
                                          state='readonly')
        self.variable_list.set('xi')
        # self.variable_list.bind("<<ComboboxSelected>>",
        #                         self.change_variable)
        self.variable_list.grid(row=3, column=6)
        self.variable_list_label = tk.Label(master, text='Select Var.')
        self.variable_list_label.grid(row=3, column=5)
        # --- Create the time slider and the refence time box
        # dt for the slider
        dt = t[1] - t[0]
        # Slider creation
        self.tSlider = tk.Scale(master, from_=t[0], to=t[-1], resolution=dt,
                                command=self.update_plot,
                                highlightcolor='red',
                                length=400,
                                takefocus=1,
                                label='Time [s]')
        # Put the slider in place
        self.tSlider.grid(row=0, column=6)
        # Reference time box
        self.t_ref_label = tk.Label(master, text='t ref')
        self.t_ref_label.grid(row=7, column=6)
        self.t_ref = tk.Entry(master)
        self.t_ref.insert(0, str(t[1]))
        self.t_ref.grid(row=8, column=6)
        self.t_ref_buttom = \
            tk.Button(master, text="Update ref", command=self.plot_reference)
        self.t_ref_buttom.grid(row=9, column=6)
        # --- Profile selection and integration limits
        # Minimum gyroradius
        self.r_min_label = tk.Label(master, text='rl min')
        self.r_min_label.grid(row=2, column=1)
        self.r_min = tk.Entry(master)
        self.r_min.insert(0, str(1.0))
        self.r_min.grid(row=2, column=2)
        # Maximum gyroradius
        self.r_max_label = tk.Label(master, text='rl max')
        self.r_max_label.grid(row=2, column=3)
        self.r_max = tk.Entry(master)
        self.r_max.insert(0, str(5.0))
        self.r_max.grid(row=2, column=4)
        # Minimum XI
        self.xi_min_label = tk.Label(master, text='XI min')
        self.xi_min_label.grid(row=3, column=1)
        self.xi_min = tk.Entry(master)
        self.xi_min.insert(0, str(1.4))
        self.xi_min.grid(row=3, column=2)
        # Maximum XI
        self.xi_max_label = tk.Label(master, text='XI max')
        self.xi_max_label.grid(row=3, column=3)
        self.xi_max = tk.Entry(master)
        self.xi_max.insert(0, str(2.2))
        self.xi_max.grid(row=3, column=4)
        # Normalise flag
        self.NormaliseToRef = tk.BooleanVar()
        self.NormaliseToRef.set(False)
        # Normalise flag
        self.NormaliseAbsolute = tk.BooleanVar()
        self.NormaliseAbsolute.set(False)
        # --- Scale normalisation
        self.normalization_factor = 1.0
        # Create the button
        self.Normalise_button = tk.Button(master, text="Norm to ref",
                                          command=self.normalise_change_change,
                                          takefocus=0)
        self.Normalise_button.grid(row=2, column=5)
        self.Normalise_button2 = tk.Button(master, text="Norm absolute",
                                           command=self.norm_abs_change_change,
                                           takefocus=0)
        self.Normalise_button2.grid(row=2, column=6)

        # --- remap Update button
        self.update_button = tk.Button(master, text='Update',
                                       command=self.update_integral)
        self.update_button.grid(row=4, column=4)

        # --- Quit button
        self.qButton = tk.Button(master, text="Quit", command=master.quit)
        self.qButton.grid(row=4, column=6)

        # --- Open the figure and show the plot
        # First calculate the integral
        self.update_integral()
        # Open the figure
        fig = Figure()
        ax = fig.add_subplot(111)
        self.ax = ax
        # Place the figure in a canvas
        self.canvas = tkagg.FigureCanvasTkAgg(fig, master=master)
        # and plot a dummy line
        self.line_reference, = ax.plot([0, 1], [0, 1], '--k',)
        self.line, = ax.plot([0, 1], [0, 1], 'r',)

        # --- Include the tool bar to zoom and export
        # We need a new frame because the toolbar uses 'pack' internally and we
        # are using 'grid' to place our elements, so they are not compatible
        self.toolbarFrame = tk.Frame(master=master)
        self.toolbarFrame.grid(row=1, column=0, columnspan=5)
        self.toolbar = \
            tkagg.NavigationToolbar2Tk(self.canvas, self.toolbarFrame)
        self.toolbar.update()

        # Draw and show
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=5,
                                         sticky=tk.N+tk.S+tk.E+tk.W)
        # In the last line, the sticky argument, combined with the
        # Grid.columnconfiguration and Grid.rowconfiguration from above
        # allows to the canvas to resize when I resize the window
        frame.grid()
        # Plot the real lines
        self.plot_reference()
        self.update_plot(float(self.t_ref.get()))

    def update_integral(self):
        """Update the integral"""
        self.integral = self.vid.integrate_remap(
            ymin=float(self.r_min.get()),
            ymax=float(self.r_max.get()),
            xmax=float(self.xi_max.get()),
            xmin=float(self.xi_min.get()),
        )

    def plot_reference(self):
        """Plot the reference profile"""
        # --- Get the time to plot
        t0 = float(self.t_ref.get())
        it = np.argmin(abs(self.vid.remap_dat['tframes'] - t0))
        # --- Get the proper variable to plot
        if self.variable_list.get() == 'xi':
            y = self.integral['integral_over_y'][:, it].copy()
            x = self.integral['xaxis']
        else:
            y = self.integral['integral_over_x'][:, it].copy()
            x = self.integral['yaxis']

        # Noramlise if needed
        if self.NormaliseToRef.get() or self.NormaliseAbsolute.get():
            self.normalization_factor = y.max()
            y /= self.normalization_factor
        # Update the plot
        self.line_reference.set_data(x, y)
        self.ax.set_xlim(x.min(), x.max())
        self.ax.set_ylim(y.min(), y.max()*1.1)
        self.canvas.draw()

    def update_plot(self, t):
        """Plot the reference profile"""
        t0 = float(t)
        it = np.argmin(abs(self.vid.remap_dat['tframes'] - t0))

        # --- Get the proper variable to plot
        if self.variable_list.get() == 'xi':
            y = self.integral['integral_over_y'][:, it].copy()
            x = self.integral['xaxis']
        else:
            y = self.integral['integral_over_x'][:, it].copy()
            x = self.integral['yaxis']
        if self.NormaliseAbsolute.get():
            y /= y.max()
        else:
            y /= self.normalization_factor
        # Update the plot
        self.line.set_data(x, y)
        self.canvas.draw()

    def normalise_change_change(self):
        """Decide to normalise or not"""
        # If it was true and we push the button, the smap should be deleted:
        if not self.NormaliseToRef.get():
            self.normalization_factor = 1.0
        self.NormaliseToRef.set(not self.NormaliseToRef.get())

    def norm_abs_change_change(self):
        """Decide to normalise or not"""
        # If it was true and we push the button, the smap should be deleted:
        self.NormaliseAbsolute.set(not self.NormaliseAbsolute.get())
