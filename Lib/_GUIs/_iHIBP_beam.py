"""
Simple GUI to see and change the injection geometry of the iHIBP as it will be
set in iHIBPsim.

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
from Lib.SimulationCodes.iHIBPsim.crossSections import alkMasses
from Lib.SimulationCodes.iHIBPsim.geom import gaussian_beam

class appHIBP_beam:
    """
    Class to hold all the data relative to the plot of the iHIBPsim geometry.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """

    def __init__(self, tkwindow, origin: float,
                 beta: float=0.0, theta: float=0.0,
                 meanE: float=70.0, stdE: float=0.0,
                 mass: float=alkMasses['Rb85'],
                 divergency: float=0.0, pinhole_size: float=0.0,
                 shotnumber: int=None, exp: str='AUGD', ed: int=0,
                 diag: str='EQI', time: float=None):
        """
        Initializes the GUI for the iHIBP beam. See iHIBPsim/geom.py ->
        gaussian_beam class for info about the parameters.
        """
        self.TKwindow   = tkwindow
        self.origin = origin

        # Initiate the class
        self.beam = gaussian_beam(origin=origin, beta=beta, theta=theta,
                                  meanE=meanE, stdE=stdE, mass=mass,
                                  divergency=divergency,
                                  pinhole_size=pinhole_size)

        # --- Adding the figure into the canvas
        self.ax, self.line, self.div = self.beam.plot(view='pol')

        # Adding the separatrix.
        if shotnumber is not None:
            self.equ = meq.equ_map(shotnumber, diag=diag, exp=exp, ed=ed)
            self.equ.read_pfm()
            self.Rsep, self.zsep = self.equ.rho2rz([1.0])

            [self.sepline] = self.ax.plot(self.Rsep[0][0],
                                        self.zsep[0][0], 'r-',
                                        linewidth=1.0)
            self.t0 = self.equ.t_eq[0]
            self.t1 = self.equ.t_eq[-1]
            self.dt = self.equ.t_eq[1] - self.equ.t_eq[0]
        else:
            self.t0 = 0.0
            self.t1 = 1.0
            self.dt = 0.0

        self.fig = plt.gcf()
        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=self.TKwindow)

        # --- Setting up the toolbar to save the data.
        self.toolbarFrame = tk.Frame(master=self.TKwindow)
        self.toolbarFrame.grid(row=0, column=1, columnspan=1)
        self.toolbar = tkagg.NavigationToolbar2Tk(self.canvas,
                                                  self.toolbarFrame)
        self.toolbar.update()

        # --- Setting up the window.
        self.frame = tk.Frame(tkwindow)
        tk.Grid.columnconfigure(self.TKwindow, 0, weight=1)
        tk.Grid.rowconfigure(self.TKwindow, 0, weight=1)

        # We will have three sliders: one for each of the important angles.
        self.slider_beta = tk.Scale(self.TKwindow, from_=-90.,
                                    to=90., resolution = 0.5,
                                    command=self.update_beta,
                                    highlightcolor='red', length=400,
                                    takefocus=1, label='beta [º]',
                                    orient=tk.HORIZONTAL)

        self.slider_beta.grid(row=1, column=0)

        # Poloidal tilting angle.
        self.slider_theta = tk.Scale(self.TKwindow, from_=-90.,
                                     to=90., resolution = 0.5,
                                     command=self.update_theta,
                                     highlightcolor='blue', length=400,
                                     takefocus=1, label='Theta [º]',
                                     orient=tk.HORIZONTAL)

        self.slider_theta.grid(row=2, column=0)

        # Divergency angle.
        self.slider_alpha = tk.Scale(self.TKwindow, from_=0.0,
                                     to=40., resolution = 0.5,
                                     command=self.update_alpha,
                                     highlightcolor='green', length=400,
                                     takefocus=1, label='alpha [º]',
                                     orient=tk.HORIZONTAL)
        self.slider_alpha.grid(row=3, column=0)

        # Time slider.
        self.slider_time = tk.Scale(self.TKwindow, from_=self.t0,
                                    to=self.t1, resolution = self.dt,
                                    command=self.update_shot,
                                    highlightcolor='blue', length=400,
                                    takefocus=1, label='Shot time [s]',
                                    orient=tk.HORIZONTAL)
        self.slider_time.grid(row=4, column=0)


        # Adding boxes for the origin point.
        self.Label_Origin_title = tk.Label(tkwindow, text='Origin coordinates')
        self.Label_Origin_title.grid(row=1, column=1)

        self.Label_Origin_x = tk.Label(tkwindow, text='x [m]')
        self.Label_Origin_x.grid(row=2, column=1)
        self.Label_Origin_y = tk.Label(tkwindow, text='y [m]')
        self.Label_Origin_y.grid(row=3, column=1)
        self.Label_Origin_z = tk.Label(tkwindow, text='z [m]')
        self.Label_Origin_z.grid(row=4, column=1)

        self.Entry_OriginChar_x = tk.StringVar()
        self.Entry_OriginChar_y = tk.StringVar()
        self.Entry_OriginChar_z = tk.StringVar()
        self.Entry_OriginChar_x.set(str(self.origin[0]))
        self.Entry_OriginChar_y.set(str(self.origin[1]))
        self.Entry_OriginChar_z.set(str(self.origin[2]))

        self.Entry_Origin_x = tk.Entry(tkwindow, bd=5,
                                       textvariable=self.Entry_OriginChar_x)
        self.Entry_Origin_x.grid(row=2, column=2)
        self.Entry_Origin_y = tk.Entry(tkwindow, bd=5,
                                       textvariable=self.Entry_OriginChar_y)
        self.Entry_Origin_y.grid(row=3, column=2)
        self.Entry_Origin_z = tk.Entry(tkwindow, bd=5,
                                       textvariable=self.Entry_OriginChar_z)
        self.Entry_Origin_z.grid(row=4, column=2)

        # Adding update and plot buttons
        self.quit_button = tk.Button(tkwindow, text="Quit",
                                     command=tkwindow.quit)
        self.quit_button.grid(row=5, column=1)

        self.update_button = tk.Button(tkwindow, text="Set origin",
                                     command=self.update_origin)
        self.update_button.grid(row=5, column=0)


        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=1,
                                         sticky=tk.N+tk.S+tk.E+tk.W)

        self.canvas.draw()
        self.frame.grid()

    def update_beta(self, beta):
        """
        Updates the plot with the new injection angle.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.beam.update(beta=float(beta))

        self.line[0].set_xdata(self.beam._beam_data['Rbeam'])
        self.line[0].set_ydata(self.beam._beam_data['zbeam'])

        if not self.beam.infsmall:
            xpoly = np.hstack([np.array(self.beam._beam_data['Rbeam']),
                               np.flip(self.beam._beam_data['Rbeam'])])
            ypoly = np.hstack([np.array(self.beam._beam_pol_up['zbeam']),
                               np.flip(self.beam._beam_pol_down['zbeam'])])

            a = [np.vstack((xpoly, ypoly)).T]

            self.div.set_verts(a)
        self.canvas.draw()

    def update_theta(self, theta: str):
        """
        Updates the plot with the new injection angle.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.beam.update(theta=float(theta))

        self.line[0].set_xdata(self.beam._beam_data['Rbeam'])
        self.line[0].set_ydata(self.beam._beam_data['zbeam'])

        if not self.beam.infsmall:
            xpoly = np.hstack([np.array(self.beam._beam_data['Rbeam']),
                               np.flip(self.beam._beam_data['Rbeam'])])
            ypoly = np.hstack([np.array(self.beam._beam_pol_up['zbeam']),
                               np.flip(self.beam._beam_pol_down['zbeam'])])

            a = [np.vstack((xpoly, ypoly)).T]

            self.div.set_verts(a)
        self.canvas.draw()

    def update_alpha(self, alpha: str):
        """
        Updates the plot with the new injection angle.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.beam.update(divergency=float(alpha))

        self.line[0].set_xdata(self.beam._beam_data['Rbeam'])
        self.line[0].set_ydata(self.beam._beam_data['zbeam'])

        if not self.beam.infsmall:
            xpoly = np.hstack([np.array(self.beam._beam_data['Rbeam']),
                               np.flip(self.beam._beam_data['Rbeam'])])
            ypoly = np.hstack([np.array(self.beam._beam_pol_up['zbeam']),
                               np.flip(self.beam._beam_pol_down['zbeam'])])

            a = [np.vstack((xpoly, ypoly)).T]

            self.div.set_verts(a)
        self.canvas.draw()

    def update_origin(self):
        """
        Updates the plot with the new injection angle.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """

        origin = [float(self.Entry_OriginChar_x.get()),
                  float(self.Entry_OriginChar_y.get()),
                  float(self.Entry_OriginChar_z.get())]

        self.beam.update(origin=origin)

        self.line[0].set_xdata(self.beam._beam_data['Rbeam'])
        self.line[0].set_ydata(self.beam._beam_data['zbeam'])

        if not self.beam.infsmall:
            xpoly = np.hstack([np.array(self.beam._beam_data['Rbeam']),
                               np.flip(self.beam._beam_data['Rbeam'])])
            ypoly = np.hstack([np.array(self.beam._beam_pol_up['zbeam']),
                               np.flip(self.beam._beam_pol_down['zbeam'])])

            a = [np.vstack((xpoly, ypoly)).T]

            self.div.set_verts(a)
        self.canvas.draw()

    def update_shot(self, time: str):
        """
        Function to update the separatrix plot to the time set.
        """
        ttime = float(time)
        itime = min(self.equ.t_eq.size-1,
                    np.searchsorted(self.equ.t_eq, ttime))

        self.sepline.set_xdata(self.Rsep[itime][0])
        self.sepline.set_ydata(self.zsep[itime][0])
        self.canvas.draw()
