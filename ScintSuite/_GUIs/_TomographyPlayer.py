"""Graphical elements to explore the tomographic reconstruction"""
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import ScintSuite._Plotting as ssplt
from matplotlib.figure import Figure
import numpy as np


class ApplicationShowTomography:
    def __init__(self, master, data, defalut_cmap='Plasma'):
        """
        Create the window with the sliders

        :param  master: Tk() opened
        :param  data: the dictionary of experimental frames
        :param  remap_dat: the dictionary of remapped data
        """
        # --- List of supported colormaps
        self.cmaps = {
            'Gamma_II': ssplt.Gamma_II(),
            'Plasma': plt.get_cmap('plasma'),
            'Cai': ssplt.Cai()
        }
        # --- Initialise the data container
        self.data = data
        # --- Create a tk container
        frame = tk.Frame(master)
        # Allows to the figure, to resize
        tk.Grid.columnconfigure(master, 0, weight=1)
        tk.Grid.columnconfigure(master, 1, weight=1)
        tk.Grid.rowconfigure(master, 1, weight=1)
        tk.Grid.rowconfigure(master, 2, weight=1)
        tk.Grid.rowconfigure(master, 5, weight=1)
        tk.Grid.rowconfigure(master, 6, weight=1)
        # --- Create the tex boxes with labels:
        # Labels
        self.labels = {
            'exp_frame': tk.Label(master, text='Original Scintillator'),
            'remap': tk.Label(master, text='Remaped frame'),
            'inversion': tk.Label(master, text='Inversion'),
            'L_curve': tk.Label(master, text='L curve'),
            'forward': tk.Label(master, text='Forward'),
            'profile': tk.Label(master, text='Gyroradius profile'),
            'MSE': tk.Label(master, text='MSE'),
            'residual': tk.Label(master, text='Residual'),
            'Hyper param': tk.Label(master, text='Hyperparam')
        }
        positions_labels = {
            'exp_frame': [0, 0],
            'remap': [0, 1],
            'inversion': [4, 1],
            'L_curve': [4, 0],
            'forward': [0, 2],
            'profile': [4, 2],
            'MSE': [4, 3],
            'residual': [5, 3],
            'Hyper param': [6, 3]
        }
        for i in self.labels.keys():
            self.labels[i].grid(row=positions_labels[i][0],
                                column=positions_labels[i][1])
        # text boxes.
        self.boxes_var = {
            'MSE': tk.StringVar(),
            'residual': tk.StringVar(),
            'HypParam': tk.StringVar()
        }
        self.boxes = {
            'MSE': tk.Entry(master, text='MSE', state='readonly',
                            textvariable=self.boxes_var['MSE']),
            'residual': tk.Entry(master, text='Residual', state='readonly',
                                 textvariable=self.boxes_var['residual']),
            'HypParam': tk.Entry(master, text='Hyperparam', state='readonly',
                                 textvariable=self.boxes_var['HypParam']),
        }
        positions_entries = {
            'MSE': [4, 4],
            'residual': [5, 4],
            'HypParam': [6, 4]
        }
        for i in self.boxes.keys():
            self.boxes[i].grid(row=positions_entries[i][0],
                               column=positions_entries[i][1])
        # --- Create the slider
        # dt for the slider
        dt = 1
        # Slider creation
        self.Slider = tk.Scale(master, from_=0, to=len(data['alpha'])-1,
                               resolution=dt,
                               command=self.plot,
                               highlightcolor='red',
                               takefocus=1,
                               label='# param')
        # Put the slider in place
        self.Slider.grid(row=1, column=3, rowspan=3)
        # --- Scintillator figure
        # Create the figure
        fig_scint = Figure()
        ax = fig_scint.add_subplot(111)
        self.orig_scint = ax.imshow(data['frame'],
                                    origin='lower', aspect='equal',
                                    cmap=self.cmaps[defalut_cmap])
        # Place the figure in a canvas
        self.canvasScint = tkagg.FigureCanvasTkAgg(fig_scint, master=master)
        # Draw and show
        self.canvasScint.draw()
        self.canvasScint.get_tk_widget().grid(row=1, column=0, rowspan=2,
                                              sticky=tk.N+tk.S+tk.E+tk.W)
        # Include the tool bar to zoom and export
        self.toolbarScintFram = tk.Frame(master=master)
        self.toolbarScintFram.grid(row=3, column=0)
        self.toolbarScint = \
            tkagg.NavigationToolbar2Tk(self.canvasScint, self.toolbarScintFram)
        self.toolbarScint.update()
        # --- Remap figure
        # Create the figure
        fig_remap = Figure()
        ax = fig_remap.add_subplot(111)
        ax.set_xlabel('Pitch [º]')
        ax.set_ylabel('Gyroradius [cm]')
        self.remap = ax.imshow(data['remap'], origin='lower',
                               extent=[self.data['sg']['p'][0],
                                       self.data['sg']['p'][-1],
                                       self.data['sg']['r'][0],
                                       self.data['sg']['r'][-1]],
                               aspect='auto',
                               cmap=self.cmaps[defalut_cmap])
        # Place the figure in a canvas
        self.canvasRemap = tkagg.FigureCanvasTkAgg(fig_remap, master=master)
        # Draw and show
        self.canvasRemap.draw()
        self.canvasRemap.get_tk_widget().grid(row=1, column=1, rowspan=2,
                                              sticky=tk.N+tk.S+tk.E+tk.W)
        # Include the tool bar to zoom and export
        self.toolbarRemapFram = tk.Frame(master=master)
        self.toolbarRemapFram.grid(row=3, column=1)
        self.toolbarRemap = \
            tkagg.NavigationToolbar2Tk(self.canvasRemap, self.toolbarRemapFram)
        self.toolbarRemap.update()

        # --- L curve
        # Create the figure
        fig_Lcurve = Figure()
        ax = fig_Lcurve.add_subplot(111)
        ax.set_xlabel('Residual')
        ax.set_ylabel('|F|')
        self.Lcurve = ax.plot(data['residual'], data['norm'], 'k',
                              linewidth=2.)
        [self.point_selected] = ax.plot(data['residual'][0], data['norm'][0],
                                        'or', markersize=8)
        # Place the figure in a canvas
        self.CaLcurve = tkagg.FigureCanvasTkAgg(fig_Lcurve, master=master)
        # Draw and show
        self.CaLcurve.draw()
        self.CaLcurve.get_tk_widget().grid(row=5, column=0, rowspan=2,
                                           sticky=tk.N+tk.S+tk.E+tk.W)
        # Include the tool bar to zoom and export
        self.toolbarLcurveFram = tk.Frame(master=master)
        self.toolbarLcurveFram.grid(row=7, column=0)
        self.toolbarLcurve = \
            tkagg.NavigationToolbar2Tk(self.CaLcurve,
                                       self.toolbarLcurveFram)
        self.toolbarLcurve.update()

        # --- Tomographic inversion
        # Create the figure
        fig_inversion = Figure()
        ax = fig_inversion.add_subplot(111)
        ax.set_xlabel('Pitch [º]')
        ax.set_ylabel('Gyroradius [cm]')
        self.tomog = ax.imshow(data['tomoFrames'][:, :, 0].squeeze(),
                               origin='lower',
                               extent=[self.data['pg']['p'][0],
                                       self.data['pg']['p'][-1],
                                       self.data['pg']['r'][0],
                                       self.data['pg']['r'][-1]],
                               aspect='auto',
                               cmap=self.cmaps[defalut_cmap])
        # Place the figure in a canvas
        self.CaTomo = tkagg.FigureCanvasTkAgg(fig_inversion, master=master)
        # Draw and show
        self.CaTomo.draw()
        self.CaTomo.get_tk_widget().grid(row=5, column=1, rowspan=2,
                                         sticky=tk.N+tk.S+tk.E+tk.W)
        # Include the tool bar to zoom and export
        self.toolbarTomo = tk.Frame(master=master)
        self.toolbarTomo.grid(row=7, column=1)
        self.toolbarTomo = \
            tkagg.NavigationToolbar2Tk(self.CaTomo, self.toolbarTomo)
        self.toolbarTomo.update()

        # --- Forward modelling figure
        # Create the figure
        fig_for = Figure()
        ax = fig_for.add_subplot(111)
        ax.set_xlabel('Pitch [º]')
        ax.set_ylabel('Gyroradius [cm]')
        self.forwa = ax.imshow(data['forwardFrames'][:, :, 0].squeeze(),
                               origin='lower',
                               extent=[self.data['sg']['p'][0],
                                       self.data['sg']['p'][-1],
                                       self.data['sg']['r'][0],
                                       self.data['sg']['r'][-1]],
                               aspect='auto',
                               cmap=self.cmaps[defalut_cmap])
        # Place the figure in a canvas
        self.canvasforwa = tkagg.FigureCanvasTkAgg(fig_for, master=master)
        # Draw and show
        self.canvasforwa.draw()
        self.canvasforwa.get_tk_widget().grid(row=1, column=2, rowspan=2,
                                              sticky=tk.N+tk.S+tk.E+tk.W)
        # Include the tool bar to zoom and export
        self.toolbarforwaFram = tk.Frame(master=master)
        self.toolbarforwaFram.grid(row=3, column=2)
        self.toolbarforwa = \
            tkagg.NavigationToolbar2Tk(self.canvasforwa, self.toolbarforwaFram)
        self.toolbarforwa.update()

        # --- Profiles figure
        # Create the figure
        fig_prof = Figure()
        ax = fig_prof.add_subplot(111)
        ax.set_xlabel('Gyroradius')
        ax.set_ylabel('Signal')
        # plot the foward profile
        [self.profile] = ax.plot(self.data['sg']['r'],
                                 self.data['profiles'][:, 0] /
                                 self.data['profiles'][:, 0].max(), 'r',
                                 linewidth=1.5,
                                 label='Foward modelling')
        # plot the inversion
        ax.plot(self.data['sg']['r'], np.sum(self.data['remap'], axis=1) /
                np.sum(self.data['remap'], axis=1).max(),
                '--k', linewidth=1.5, label='Remap')
        [self.profile_pinhole] = \
            ax.plot(self.data['pg']['r'], self.data['profiles_pinhole'][:, 0] /
                    self.data['profiles_pinhole'][:, 0].max(),
                    '--b', linewidth=1.5, label='Remap')
        # Place the figure in a canvas
        self.Caprof = tkagg.FigureCanvasTkAgg(fig_prof, master=master)
        # Draw and show
        self.Caprof.draw()
        self.Caprof.get_tk_widget().grid(row=5, column=2, rowspan=2,
                                         sticky=tk.N+tk.S+tk.E+tk.W)
        # Include the tool bar to zoom and export
        self.toolbarprofFram = tk.Frame(master=master)
        self.toolbarprofFram.grid(row=7, column=2)
        self.toolbarprof = \
            tkagg.NavigationToolbar2Tk(self.Caprof,
                                       self.toolbarprofFram)
        self.toolbarprof.update()

        # --- Quit button
        self.qButton = tk.Button(master, text="Quit", command=master.quit)
        self.qButton.grid(row=0, column=4)
        frame.grid()

    def plot(self, i):
        """Update the plots"""
        ii = int(i)
        # Plot the frame
        tomo_frame = self.data['tomoFrames'][:, :, ii].squeeze()
        self.tomog.set_data(tomo_frame)
        cmin = tomo_frame.min()
        cmax = tomo_frame.max() * 1.1
        self.tomog.set_clim(cmin, cmax)
        # Plot the foward scintillator:
        for_frame = self.data['forwardFrames'][:, :, ii].squeeze()
        self.forwa.set_data(for_frame)
        cmin = for_frame.min()
        cmax = for_frame.max() * 1.1
        self.forwa.set_clim(cmin, cmax)
        # plot the profile:
        self.profile.set_data(self.data['sg']['r'],
                              self.data['profiles'][:, ii] /
                              self.data['profiles'][:, ii].max())
        self.profile_pinhole.set_data(
            self.data['pg']['r'],
            self.data['profiles_pinhole'][:, ii] /
            self.data['profiles_pinhole'][:, ii].max()
        )
        # Plot the selected point in the L curve
        self.point_selected.set_data(self.data['residual'][ii],
                                     self.data['norm'][ii])
        # Update the boxes with numbers
        self.boxes_var['MSE'].set("{:.2e}".format(self.data['MSE'][ii]))
        self.boxes_var['residual'].set(
            "{:.2e}".format(self.data['residual'][ii])
        )
        self.boxes_var['HypParam'].set(
            "{:.2e}".format(self.data['alpha'][ii])
        )
        self.CaTomo.draw()
        self.CaLcurve.draw()
        self.Caprof.draw()
        self.canvasforwa.draw()
