"""
Simple GUI to see and change the injection geometry of the iHIBP as it will be
set in iHIBPsim.

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import os
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import Lib.LibPlotting as ssplt
import Lib
from Lib.ihibp.xsection import alkMasses
from Lib.ihibp.geom import gaussian_beam

class appHIBP_beam:
    """
    Class to hold all the data relative to the plot of the iHIBPsim geometry.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """
    
    def __init__(self, tkwindow, origin: float, 
                 beta: float=0.0, theta: float=0.0,
                 meanE: float=70.0, stdE: float=0.0, 
                 mass: float=alkMasses['Rb85'],
                 divergency: float=0.0, pinhole_size: float=0.0):
        """
        Initializes the GUI for the iHIBP beam. See iHIBPsim/geom.py -> 
        gaussian_beam class for info about the parameters.
        """
        self.TKwindow   = tkwindow
        
        
        # Initiate the class
        self.beam = gaussian_beam(origin=origin, beta=beta, theta=theta,
                                  meanE=meanE, stdE=stdE, mass=mass,
                                  divergency=divergency, 
                                  pinhole_size=pinhole_size)
        
        # --- Adding the figure into the canvas
        self.ax, self.line, self.div = self.beam.plot(view='pol')
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
                                    takefocus=1, label='$\\beta$ [º]')
        
        self.slider_beta.grid(row=1, column=0)
        
        # Poloidal tilting angle.
        self.slider_theta = tk.Scale(self.TKwindow, from_=-90.,
                                     to=90., resolution = 0.5,
                                     command=self.update_theta,
                                     highlightcolor='blue', length=400, 
                                     takefocus=1, label='$\\Theta$ [º]')
        
        self.slider_theta.grid(row=2, column=0)
        
        # Divergency angle.
        self.slider_alpha = tk.Scale(self.TKwindow, from_=-40.,
                                     to=40., resolution = 0.5,
                                     command=self.update_theta,
                                     highlightcolor='green', length=400, 
                                     takefocus=1, label='$\\alpha$ [º]')
        
        self.slider_alpha.grid(row=3, column=0)
        
        # Adding boxes for the origin point.
        self.Label_Origin_title = tk.Label(tkwindow, text='Origin coordinates')
        self.Label_Origin_title.grid(row=1, column=1)
        
        self.Label_Origin_x = tk.Label(tkwindow, text='x [m]')
        self.Label_Origin_x.grid(row=2, column=1)
        self.Label_Origin_y = tk.Label(tkwindow, text='x [m]')
        self.Label_Origin_y.grid(row=3, column=1)
        self.Label_Origin_z = tk.Label(tkwindow, text='x [m]')
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
        self.quit_button.grid(row=4, column=1)
        
        self.update_button = tk.Button(tkwindow, text="Set origin",
                                     command=tkwindow.update_origin)
        self.update_button.grid(row=4, column=1.5)
        
    def update_beta(self, beta: str):
        """
        Updates the plot with the new injection angle.
        
        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.beam.update(beta=float(beta))
        
        self.line.set_xdata(self.beam._beam_data['Rbeam'])
        self.line.set_ydata(self.beam._beam_data['zbeam'])
        
        if not self.beam.infsmall:
            xpoly = [self.beam._beam_data['Rbeam'], 
                     np.flip(self.beam._beam_data['Rbeam'])]
            ypoly = [self.beam._pol_up['zbeam'],self.beam._pol_down['zbeam']]
            
            self.div.set_verts([[xpoly, ypoly]])
        
    def update_theta(self, theta: str):
        """
        Updates the plot with the new injection angle.
        
        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.beam.update(theta=float(theta))
        
        self.line.set_xdata(self.beam._beam_data['Rbeam'])
        self.line.set_ydata(self.beam._beam_data['zbeam'])
        
        if not self.beam.infsmall:
            xpoly = [self.beam._beam_data['Rbeam'], 
                     np.flip(self.beam._beam_data['Rbeam'])]
            ypoly = [self.beam._pol_up['zbeam'],self.beam._pol_down['zbeam']]
            
            self.div.set_verts([[xpoly, ypoly]])
        
    def update_alpha(self, alpha: str):
        """
        Updates the plot with the new injection angle.
        
        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.beam.update(alpha=float(alpha))
        
        self.line.set_xdata(self.beam._beam_data['Rbeam'])
        self.line.set_ydata(self.beam._beam_data['zbeam'])
        
        if not self.beam.infsmall:
            xpoly = [self.beam._beam_data['Rbeam'], 
                     np.flip(self.beam._beam_data['Rbeam'])]
            ypoly = [self.beam._pol_up['zbeam'],self.beam._pol_down['zbeam']]
            
            self.div.set_verts([[xpoly, ypoly]])        
        
    def update_origin(self):
        """
        Updates the plot with the new injection angle.
        
        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        
        origin = [float(self.Entry_OriginChar_x.get()),
                  float(self.Entry_OriginChar_y.get()),
                  float(self.Entry_OriginChar_z.get())]
        
        self.beam.update(origin=origin)
        
        self.line.set_xdata(self.beam._beam_data['Rbeam'])
        self.line.set_ydata(self.beam._beam_data['zbeam'])
        
        if not self.beam.infsmall:
            xpoly = [self.beam._beam_data['Rbeam'], 
                     np.flip(self.beam._beam_data['Rbeam'])]
            ypoly = [self.beam._pol_up['zbeam'],self.beam._pol_down['zbeam']]
            
            self.div.set_verts([[xpoly, ypoly]])                
        