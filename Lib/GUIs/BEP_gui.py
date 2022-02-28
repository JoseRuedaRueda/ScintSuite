"""
Simple BEP - GUI

Simple graphical interface to plot the BEP spectra and make some checks.


"""
import os
import numpy as np
import tkinter as tk                       # To open UI windows
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import Lib.LibPlotting as ssplt
import Lib.BEP.LibBEP as rbep
from matplotlib.figure import Figure
from Lib.LibMachine import machine
from Lib.LibData.AUG.VesselNBI import getNBIwindow


signal_name = ('Sigma', 'Mixed', 'Pi')


class AppBEP_plot:
    def __init__(self, tkwindow, shotnumber: int,
                 timeWindowOverlap: float = 0.100,
                 calibrated: bool=True):

        """
        This will initialize the GUI viewer for the BEP signals.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param shotnumber: shotnumber to plot.
        @param timeWindowOverlap: length in time to overlap the data and plot.
        By default, set to 100 ms. (Input must be given in seconds.)
        @param calibrated: plots the calibrated signals (by default) or False
        if the RAW signal is to be plotted instead.
        """

        self.calib_flag = calibrated
        self.twin       = timeWindowOverlap
        self.shot       = shotnumber
        self.TKwindow   = tkwindow
        self.shotnumber = shotnumber
        self.dt         = 0.0
        self.t0         = 0.0
        self.t1         = 0.0
        self.Delta_time = 0  # Number of points to average over.
        self.update_plot_internal = None
        self.changed = False

        # --- Update the BEP data.
        self.bepdata = self.updateBEPdata()

        # --- Setting up the window.
        self.frame = tk.Frame(tkwindow)
        tk.Grid.columnconfigure(self.TKwindow, 0, weight=1)
        tk.Grid.rowconfigure(self.TKwindow, 0, weight=1)


        # --- Creating the slider.
        self.slider = tk.Scale(self.TKwindow, from_=self.t0,
                               to=self.t1, resolution = self.twin,
                               command=self.update_plot,
                               highlightcolor='red', length= 400,
                               takefocus=1, label='Time [s]')

        self.slider.grid(row=0, column=6)

        # --- Opening the figure and plotting.
        self.fig, self.axis = self.plotBEP_1st()

        # --- Adding the figure into the canvas
        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=self.TKwindow)

        # --- Setting up the toolbar to save the data.
        self.toolbarFrame = tk.Frame(master=self.TKwindow)
        self.toolbarFrame.grid(row=4, column=1, columnspan=6)
        self.toolbar = tkagg.NavigationToolbar2Tk(self.canvas,
                                                  self.toolbarFrame)
        self.toolbar.update()

        # --- Adding the 'Quit' button.
        self.qButton = tk.Button(tkwindow, text="Quit", command=tkwindow.quit)
        self.qButton.grid(row=3, column=6)

        # Check variable for using the calibrated signal
        if self.calib_flag is None:
            state = tk.DISABLED
        else:
            state = tk.NORMAL
        self.checkVar1 = tk.BooleanVar()
        self.checkVar1.set(True)
        self.updateButton = tk.Button(tkwindow, text="Update",
                                     command=self.change_calibrated_status,
                                     takefocus=0, state=state)
        self.updateButton.grid(row=3, column=3)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=5,
                                         sticky=tk.N+tk.S+tk.E+tk.W)


        self.canvas.draw()
        self.frame.grid()

    def updateBEPdata(self):
        """
        Reads from the database the BEP data and returns its content into a
        dictionary.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """

        if self.calib_flag:
            bepdata = rbep.readBEP(shotnumber=self.shotnumber,
                                time=(0, 3600.0))
            self.update_plot_internal = rbep.plotBEP
            self.xlabel = 'Wavelength $\\lambda$ [nm]'
            self.ylabel = 'Counts'
            self.name_bepdata = 'spectra'
        else:
            bepdata = rbep.BEPfromSF(shotnumber=self.shotnumber,
                                  time=(0, 3600.0))
            self.update_plot_internal = rbep.plotBEP_fromSF
            self.xlabel = 'Pixel number'
            self.name_bepdata = 'data'
            self.ylabel = 'Counts'

        # --- Update the time step:
        self.t0 = bepdata['time'].min()
        self.t1 = min(11.0, bepdata['time'].max())

        # --- Setting the dt:
        self.dt = bepdata['time'][1] - bepdata['time'][0]

        if self.twin < self.dt:
            print('The time window is smaller than the exposure time. Reducing\
                  the time-window = %.3f [ms]'%self.dt*1.0e3)

        self.Delta_time = min(int(1), int(self.twin/self.dt))

        # --- Getting NBI data.
        # Take only the time windows in which the Q6 is ON and Q5 is OFF.
        nbidata = getNBIwindow((self.t0, self.t1),
                                    shotnumber=self.shotnumber,
                                    nbion=(6,), nbioff=(5,))

        self.t0 = max(self.t0, nbidata['time'].min())
        self.t1 = min(self.t1, nbidata['time'].max())
        self.nbitimes = nbidata['time']

        tindx0, tindx1 = bepdata['time'].searchsorted((self.t0, self.t1))

        bepdata[self.name_bepdata] = bepdata[self.name_bepdata][:, tindx0:tindx1, :]
        bepdata['time'] = bepdata['time'][tindx0:tindx1]
        return bepdata


    def plotBEP_1st(self):

        """
        Plots the first time into the axis and returns the axis.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """

        fig, ax = plt.subplots(5, 3, sharex=True)
        time0 = self.t0
        time1 = self.t0+self.twin
        time = (time0, time1)

        for ii in range(5):
            for jj in range(3):
                losname='BEP-%d-%d'%(ii+1, jj+1)
                if self.shotnumber >= 36900:
                    if jj == 0:
                        losname='BEP-%d-%d'%(ii+1, 3)
                    elif jj == 2:
                        losname='BEP-%d-%d'%(ii+1, 1)

                opts  = { 'label': '%05d - %s'%(self.shotnumber, losname)}
                ax[ii, jj] = self.update_plot_internal(self.bepdata,
                                                       losname=losname,
                                                       ax=ax[ii, jj],
                                                       line_options=opts,
                                                       time=time)
                ax[ii, jj].legend()

        for ii in range(5):
            ax[ii, 0].set_ylabel(self.ylabel)

        for ii in range(3):
            ax[-1, ii].set_xlabel(self.xlabel)
            ax[0, ii].set_title(signal_name[ii])

        return fig, ax


    def update_plot(self, timepoint):
        """
        Update the plots according to the point where the scale is.
        """

        t0 = float(timepoint)
        t1 = t0 + self.twin


        time = (t0, t1)
        tindx0, tindx1 = self.bepdata['time'].searchsorted(time)
        for ii in range(5):
            for jj in range(3):
                losname='BEP-%d-%d'%(ii+1, jj+1)
                if self.shotnumber >= 36900:
                    if jj == 0:
                        losname='BEP-%d-%d'%(ii+1, 3)
                    elif jj == 2:
                        losname='BEP-%d-%d'%(ii+1, 1)

                idx_channel = self.bepdata['losnam'].index(losname)
                pltdata = np.mean(self.bepdata[self.name_bepdata]\
                                  [idx_channel, tindx0:tindx1, :],
                                  axis=0)

                if self.calib_flag:
                    xpltdata = self.bepdata['lambda'][idx_channel, :]
                else:
                    xpltdata = np.arange(pltdata)
                if t0 not in self.nbitimes:
                    pltdata *= 0.0
                self.axis[ii, jj].lines[0].set_data(xpltdata, pltdata)

        if self.changed:
            for ii in range(5):
                self.axis[ii, 0].set_ylabel(self.ylabel)

            for ii in range(3):
                self.axis[-1, ii].set_xlabel(self.xlabel)
                self.axis[0, ii].set_title(signal_name[ii])
            self.changed = False

        self.canvas.draw()

    def change_calibrated_status(self):
        """
        Checks if there has been some change in the calibrated status and
        update the data of the class.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """
        self.calib_flag = ~self.calib_flag
        print(self.calib_flag)
        self.bepdata = self.updateBEPdata()
        self.changed = True
