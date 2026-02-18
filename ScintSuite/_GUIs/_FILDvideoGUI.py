"""
GUI for FILDSIM vanilla users
author: areyner@us.es
"""
import ScintSuite as ss
import ScintSuite._Plotting as ssplt
import ScintSuite._StrikeMap as ssmap
import ScintSuite.SimulationCodes.SINPA as sssinpa

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from tkinter import ttk
from matplotlib.path import Path
import xarray as xr

import os
import copy
import pickle
import time
import yaml
import tarfile
import json
import tempfile

import logging
logger = logging.getLogger('ScintSuite.FILDvideoGUI')

from ScintSuite._Machine import machine as mach


##

class FILDvideoGUI:
    '''
    Build a GUI to analyse data from FILD videos.
    1. Select shot, fild number and time interva
    2. Define t interval for background subtraction and select size of 
        median and gaussian filters
        - This will apply the filters to the raw data and overwrite the
            previous treated data.
        - Will remove the remap, in case of been done before.
    3. Select mesh for remapping, what smaps to use, the precision of the
        magnetic field and the remapping method
        - This will change the plot to remap format directly
    - The plotting options include being able to change colorbar, colorbar 
            limits ([0,None] is the default), plot the smap (if remap is done),
            and scintillator.
    - Capacity to export the data
    - Capacity to extract the time trace of various ROI in the same plot, to be
        able to compare.
    '''

    def __init__(self, shot = 41256, diag = 1, tini = 0, tfin = 10):
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("FILD data explorer GUI")

        self.shot = shot
        self.diag = diag
        self.tini = tini
        self.tfin = tfin
        self.save_folder = ss.paths.ScintSuite + '/Data/VideosRemaps/FILD'

        self.vid = None
        self.vid_raw = None
        self.current_frame = 0

        self.fig = Figure(figsize=(14, 5), constrained_layout = False)
        self.ax1 = self.fig.add_axes([0.05, 0.15, 0.45, 0.75])
        self.ax2 = self.fig.add_axes([0.60, 0.15, 0.30, 0.75])

        self.frames = None
        self.data_vals1 = []
        self.vmax_all1 = []
        self.im1 = None
        self.cbar1 = None
        self.cax1 = None
        self.t_text1 = None
        self.shot_text = None

        self.remaps = None
        self.data_vals2 = []
        self.vmax_all2 = []
        self.im2 = None
        self.cax2 = None
        self.cbar2 = None
        self.t_text2 = None

        self.smap_state = False
        self.scint_state = False

        self.collecting = False
        self.roi_points = []
        self.roi_line = None
        self.roi_scatter = None
        self.cid_click = None
        self.ax3 = None
        self.ax4 = None
        self.fig3 = None

        # ---- Colors
        self.cmaps = {
            'Greys': plt.get_cmap('Greys_r'),
            'Gamma_II': ssplt.Gamma_IIb(),
            'Gamma_III': ssplt.Gamma_III(),
            'Plasma': plt.get_cmap('plasma'),
            'BWR': plt.get_cmap('bwr'),
            'hot': plt.get_cmap('hot_r'),
        }
        self.cmap_names = list(self.cmaps.keys())
        self.cmap_default1 = 'Gamma_III'
        self.cmap_default2 = 'Gamma_II'

        self.formater = FuncFormatter(lambda x, _: f"{x:.1e}")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)

        # LAYOUT
        # -------------------------------------------------------------------
        # ---- Title
        crow = 0
        tk.Label(self.root, text=f"EXP: {mach}", font=("Arial", 14, "bold"))\
            .grid(row=crow, column=0, columnspan=4)

        # ---- Shot
        crow += 1
        tk.Label(self.root, text="Shot:").grid(row=crow, column=0, sticky='e')
        self.entry_shot = tk.Entry(self.root, width=6)
        self.entry_shot.insert(0, str(self.shot))
        self.entry_shot.grid(row=crow, column=1)
        tk.Label(self.root, text="FILD:").grid(row=crow, column=2, sticky='e')
        self.entry_diag = tk.Entry(self.root, width=6)
        self.entry_diag.insert(0, str(self.diag))
        self.entry_diag.grid(row=crow, column=3)
        # ---- Time interval
        crow += 1
        tk.Label(self.root, text="Time interval (s):")\
            .grid(row=crow, column=0, columnspan=2, sticky='e')
        self.entry_t1 = tk.Entry(self.root, width=6)
        self.entry_t1.insert(0, str(tini))
        self.entry_t1.grid(row=crow, column=2)
        self.entry_t2 = tk.Entry(self.root, width=6)
        self.entry_t2.insert(0, str(tfin))
        self.entry_t2.grid(row=crow, column=3)
        # ---- Load video button
        crow += 1
        self.btn_load = tk.Button(self.root, text="Load", bg = 'black',
                                  command=self.load_video)
        self.btn_load.grid(row=crow, column=0, columnspan=2, sticky='we')
        self.btn_import = tk.Button(self.root, text="LoadH5", bg = 'blue',
                                     activebackground="#007BFF",
                                  command=self.import_video)
        self.btn_import.grid(row=crow, column=2, columnspan=2, sticky='we')
        # ---- Background subtraction
        crow += 1
        tk.Label(self.root, text="BKG sub. (s):")\
            .grid(row=crow, column=0, columnspan=2, sticky="e")
        self.entry_tn1 = tk.Entry(self.root, width=6)
        self.entry_tn1.insert(0, "0")
        self.entry_tn1.grid(row=crow, column=2)
        self.entry_tn2 = tk.Entry(self.root, width=6)
        self.entry_tn2.insert(0, "0.2")
        self.entry_tn2.grid(row=crow, column=3)
        # ---- Median filter
        crow += 1
        tk.Label(self.root, text="Median:")\
            .grid(row=crow, column=0, sticky='e')
        self.entry_median = tk.Entry(self.root, width=6)
        self.entry_median.insert(0, "3")
        self.entry_median.grid(row=crow, column=1)
        # ---- Gaussian filter
        tk.Label(self.root, text="Gauss:")\
            .grid(row=crow, column=2, sticky='e')
        self.entry_gauss = tk.Entry(self.root, width=6)
        self.entry_gauss.insert(0, "2")
        self.entry_gauss.grid(row=crow, column=3)
        # ---- Filter video button
        crow += 1
        self.btn_filter = tk.Button(self.root, text="Filter", bg = 'black',
                                    command=self.process_video, 
                                    state=tk.DISABLED)
        self.btn_filter.grid(row=crow, column=0, columnspan=2, sticky='we')
        self.btn_loadfilter = tk.Button(self.root, text="Load + Filter", 
                                        bg = 'green',
                                        activebackground="#319F31",
                                        command=self.load_plus_filter,
                                        state=tk.NORMAL)
        self.btn_loadfilter.grid(row=crow, column=2, columnspan=2, sticky='we')
        # ---- Remap parameters
        crow += 1
        parameters = {'xmin': 20, 'xmax': 90, 'dx': 1, 
                      'ymin': 1, 'ymax': 8, 'dy': 0.1}
        labels = {'xmin': ' p min', 'xmax': 'p max', 'dx': 'dp', 
                  'ymin': ' r min', 'ymax': 'r max', 'dy': 'dr'}
        self.entry_params = {}
        for i, (var, default) in enumerate(parameters.items()):
            row = i % 3 + crow
            col = (i // 3) * 2
            name = labels[var]
            tk.Label(self.root, text=f"{name}:")\
                .grid(row=row, column=col, sticky='e')
            e = tk.Entry(self.root, width=6)
            e.insert(0, str(default))
            e.grid(row=row, column=col+1)
            self.entry_params[var] = e
        crow += 3
        # ---- Strikemap options
        tk.Label(self.root, text="Smaps:")\
            .grid(row=crow, column=0, columnspan=1, sticky='w')
        self.opts_smap = ["Computed", "Existing"]
        self.opt_smap = tk.StringVar(value="Computed")
        self.menu_smap = tk.OptionMenu(self.root, 
                                       self.opt_smap, *self.opts_smap)
        self.menu_smap.grid(row=crow, column=1, columnspan=3, sticky='we')
        self.menu_smap.configure(state = tk.NORMAL)
        # ---- Strikemap precision
        crow +=1
        tk.Label(self.root, text="Smap precision:")\
            .grid(row=crow, column=0, columnspan=2, sticky='w')
        self.entry_precision = tk.Entry(self.root, width=6)
        self.entry_precision.insert(0, "1")
        self.entry_precision.grid(row=crow, column=2)
        # ---- Remapping method
        crow +=1
        tk.Label(self.root, text="Method:")\
            .grid(row=crow, column=0, columnspan=1)
        self.opts_remap = ["Centers", "Fwrap_simple"]
        self.opt_remap = tk.StringVar(value="Centers")
        self.menu_remap = tk.OptionMenu(self.root, 
                                        self.opt_remap, *self.opts_remap)
        self.menu_remap.grid(row=crow, column=1, columnspan=3, sticky='we')
        self.menu_remap.configure(state = tk.NORMAL)
        # ---- Remap video
        crow +=1
        self.btn_remap = tk.Button(self.root, text="Remap", bg = 'black', 
                                   command=self.remap_video, 
                                   state=tk.DISABLED)
        self.btn_remap.grid(row=crow, column=0, columnspan=2, sticky='we')
        self.btn_all = tk.Button(self.root, text="DO ALL", bg = 'green',
                                     activebackground="#319F31",
                                   command=self.do_all_actions, 
                                   state=tk.NORMAL)
        self.btn_all.grid(row=crow, column=2, columnspan=2, sticky='we')

        # --------------------------------------------------------------------
        crow +=1
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.grid(row=crow, column=0, columnspan=4, sticky='we', pady=5)
        # ---- Camera plot
        crow +=1
        tk.Label(self.root, text="Camera plot", font=("Arial", 10, "bold"))\
            .grid(row=crow, column=0, columnspan=2)
                # ---- Time trace button
        self.btn_TT1 = tk.Button(text="ROI time trace",
            command = lambda: self.extract_time_trace(),
            width=12, state=tk.DISABLED)
        self.btn_TT1.grid(row=crow, column=2, columnspan=2, sticky='we')
        # ---- Color menu
        crow += 1
        self.combo_cmap_c = ttk.Combobox(self.root, values=self.cmap_names,
            state="readonly", width=12)
        self.combo_cmap_c.set(self.cmap_default1)
        self.combo_cmap_c.bind("<<ComboboxSelected>>", self.change_cmap)
        self.combo_cmap_c.grid(row=crow, column=0, columnspan=2)
        self.combo_cmap_c.configure(state=tk.DISABLED)
        # ---- Colorbar limits
        self.entry_vmin_c = tk.Entry(self.root, width=6)
        self.entry_vmin_c.insert(0, "0")
        self.entry_vmin_c.grid(row=crow, column=2)
        self.entry_vmax_c = tk.Entry(self.root, width=6)
        self.entry_vmax_c.grid(row=crow, column=3)
        # ---- Smap module
        crow += 1
        self.btn_smap = tk.Button(self.root, text="SMAP", 
                                  command=self.plot_smap_button, 
                                  state=tk.DISABLED)
        self.btn_smap.grid(row=crow, column=0, columnspan=2, sticky='we')
        # ---- Scinillator module
        self.btn_scint = tk.Button(self.root, text="SCINT", 
                                   command=self.plot_scint_button,
                                   state=tk.DISABLED)
        self.btn_scint.grid(row=crow, column=2, columnspan=2, sticky='we')

        # ---- Remap plot
        crow +=1
        tk.Label(self.root, text="Remap plot", font=("Arial", 10, "bold"))\
            .grid(row=crow, column=0, columnspan=2)
        self.btn_TT2 = tk.Button(text="ROI time trace",
            command = lambda: self.extract_time_trace(remap=True),
            width=12, state=tk.DISABLED)
        self.btn_TT2.grid(row=crow, column=2, columnspan=2, sticky='we')
        # ---- Color menu
        crow += 1
        self.combo_cmap_r = ttk.Combobox(self.root, values=self.cmap_names,
            state="readonly", width=12)
        self.combo_cmap_r.set(self.cmap_default2)
        self.combo_cmap_r.bind("<<ComboboxSelected>>", self.change_cmap)
        self.combo_cmap_r.grid(row=crow, column=0, columnspan=2)
        self.combo_cmap_r.configure(state=tk.DISABLED)
        # ---- Colorbar limits
        self.entry_vmin_r = tk.Entry(self.root, width=6)
        self.entry_vmin_r.insert(0, "0")
        self.entry_vmin_r.grid(row=crow, column=2)
        self.entry_vmax_r = tk.Entry(self.root, width=6)
        self.entry_vmax_r.grid(row=crow, column=3)
        # ---- Slider
        crow -=1
        self.slider = ttk.Scale(self.root, from_=0, to=0, orient="horizontal", 
                                command=self.update_plot)
        self.slider.grid(row=crow, column=4, columnspan=8, sticky='we')
        # ---- Toolbar
        crow +=1
        toolbar_top = tk.Frame(self.root)
        toolbar_top.grid(row=crow, column=4, columnspan=4, sticky='we')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_top)
        self.toolbar.update()

        buttons_frame = tk.Frame(self.root)
        buttons_frame.grid(row=crow, column=8, columnspan=3, sticky='e')
        # ---- Export button
        self.btn_export = tk.Button(buttons_frame, text="Export data", 
                               command=self.export_data, 
                               width=12, state=tk.DISABLED)
        self.btn_export.pack(side="left", padx=5)
        # ---- Quit button
        btn_quit = tk.Button(buttons_frame, text="Quit", bg = 'red',
                             activebackground="#FF5050",
                             command=self.root.destroy,
                             width=6, state=tk.NORMAL)
        btn_quit.pack(side="left", padx=5)
        # ---- Canvas
        self.canvas.get_tk_widget()\
            .grid(row=0, column=4, rowspan=crow-1, columnspan=6, sticky='nsew')
        self.canvas.draw_idle()
        # ---- Keys
        self.root.bind('<Left>', self.key_press)
        self.root.bind('<Right>', self.key_press)

    def run(self):
        self.root.mainloop()


    # FUNTIONS
    # -----------------------------------------------------------------------
    def key_press(self, event):
        if self.data_vals1.size == 0:
            return
        
        if event.keysym == 'Right' and self.current_frame < len(self.data_vals1) - 1:
            self.current_frame += 1
        elif event.keysym == 'Left' and self.current_frame > 0:
            self.current_frame -= 1
        else:
            return
        self.slider.set(self.current_frame)
        self.update_plot(self.current_frame)

    # ---- Plot updating
    def update_plot(self, idx):
        '''
        Updates the plot with the data from the new frame (time-wise).
            1. Change data
            2. Change colorbar
            3. Plot lines (if wanted)
        '''
        idx = int(float(idx))
        if idx >= len(self.data_vals1):
            return
        self.current_frame = idx
        self.im1.set_data(self.data_vals1[idx])
        self.t_text1.set_text(f"{self.frames.t[idx].values:.3f} s")
        self.change_cbar1()
        self.plot_lines()
        if self.im2 is not None:
            self.im2.set_data(self.data_vals2[idx])
            self.t_text2.set_text(f"{self.frames.t[idx].values:.3f} s")
            self.change_cbar2()

        # self.fig.tight_layout()
        self.canvas.draw_idle()

    def change_cbar1(self, event=None):
        '''
        Changes colorbar limits.
        Triggered in update_plot
        '''
        try: vmin = float(self.entry_vmin_c.get())
        except: vmin = 0.0
        try: vmax = float(self.entry_vmax_c.get())
        except: vmax = self.vamx_all1[self.current_frame]
        self.im1.set_clim(vmin=vmin, vmax=vmax)
        self.cbar1.update_normal(self.im1)
        self.cbar1.formatter = self.formater

    def change_cbar2(self, event=None):
        '''
        Changes colorbar limits.
        Triggered in update_plot
        '''
        try: vmin = float(self.entry_vmin_r.get())
        except: vmin = 0.0
        try: vmax = float(self.entry_vmax_r.get())
        except: vmax = self.vamx_all2[self.current_frame]
        self.im2.set_clim(vmin=vmin, vmax=vmax)
        self.cbar2.update_normal(self.im2)
        self.cbar2.formatter = self.formater

    def plot_lines(self):
        '''
        Function that plots lines in the image (smap or scint)
        '''
        logging.disable(logging.CRITICAL)
        ssplt.remove_lines(self.ax1)
        ssplt.remove_lines(self.ax2)

        if self.smap_state:
            theta_used = self.vid.remap_dat['theta_used'].values[self.current_frame]
            phi_used = self.vid.remap_dat['phi_used'].values[self.current_frame]
            name_smap = sssinpa.execution.guess_strike_map_name(
                phi_used, theta_used, geomID=self.vid.geometryID,
                decimals=self.vid.remap_dat['frames'].attrs['decimals']
            )
            smap_folder = self.vid.remap_dat['frames'].attrs['smap_folder']
            full_name_smap = os.path.join(smap_folder, name_smap)

            smap = ssmap.Fsmap(full_name_smap)
            smap.calculate_pixel_coordinates(self.vid.CameraCalibration)
            xlim = self.ax1.get_xlim()
            ylim = self.ax1.get_ylim()
            smap.plot_pix(ax=self.ax1, labels=False)
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim)

        if self.scint_state:
            xlim = self.ax1.get_xlim()
            ylim = self.ax1.get_ylim()
            self.vid.scintillator.plot_pix(ax=self.ax1)
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim)

        logging.disable(logging.NOTSET)
        self.canvas.draw_idle()


    # ---- Buttons
    def load_video(self):
        '''
        Load a new video data.
            1. Get shot, diagnostic and time data
            2. Load and store data
            3. Set basic variables
            4. Clear remap plot (if exists)
            5. Update video plot
            6. Set slider again
            7. Enable buttons
        '''
            
        self.shot = int(self.entry_shot.get())
        self.diag = int(self.entry_diag.get())
        self.tini = float(self.entry_t1.get())
        self.tfin = float(self.entry_t2.get())
        
        self.vid_raw = ss.vid.FILDVideo(shot=self.shot, diag_ID=self.diag)
        self.vid_raw.read_frame(t1=self.tini, t2=self.tfin)
        self.vid = copy.deepcopy(self.vid_raw)

        self.smap_state = False
        self.scint_state = False
        self.current_frame = 0 #go back to first

        self.ax2.clear()
        if self.cax2 is not None:
            self.cax2.remove()  
            self.cax2 = None
            self.im2 = None
            self.cbar2 = None
        
        self.update_video()
        self.slider.config(from_=0, to=len(self.data_vals1)-1)
        self.slider.set(self.current_frame)
        self.enabling_after_loading()    

    def import_video(self):
        '''
        Work on progress. Missing the import of the data to a vid object.
        '''

        base_config = self.get_gui_config()
        shot_label = str(self.shot) + 'FILD' + str(self.diag)
        ubi = os.path.join(self.save_folder, mach, shot_label)

        # Here load the GUI config
        stored_config = self.load_gui_config_from_yaml(folder=ubi)
        self.apply_gui_config(stored_config)
        if base_config['t1'] > stored_config ['t2']:
            logger.warning('Outside of stored video. Setting stored limits.')
        else:
            if base_config['t1'] < stored_config['t1']:
                logger.warning('New initial time previous than stored')
            else:
                logger.info('Initial time inside stored video') 
                self.entry_t1.delete(0, "end")
                self.entry_t1.insert(0, str(base_config['t1']))
            if base_config['t2'] > stored_config['t2']:
                logger.warning('New final time larger than stored') 
            else:
                logger.info('Final time inside stored video') 
                self.entry_t2.delete(0, "end")
                self.entry_t2.insert(0, str(base_config['t2']))

        self.shot = int(self.entry_shot.get())
        self.diag = int(self.entry_diag.get())
        self.tini = float(self.entry_t1.get())
        self.tfin = float(self.entry_t2.get())
        
        # Here load the video
        self.vid = ss.vid.FILDVideo(shot=self.shot, diag_ID=self.diag)
        self.vid.import_remap(folder = ubi)


        self.smap_state = False
        self.scint_state = False
        self.current_frame = 0 #go back to first

        self.ax2.clear()
        if self.cax2 is not None:
            self.cax2.remove()  
            self.cax2 = None
            self.im2 = None
            self.cbar2 = None
        
        self.update_video()
        self.update_remap()
        self.slider.config(from_=0, to=len(self.data_vals1)-1)
        self.slider.set(self.current_frame)
        self.enabling_after_loading()  
        self.enabling_after_remaping()

    def process_video(self):
        '''
        Apply filters and subtract noise from the video
            1. Loads de raw video
            2. Subtracts noise, apply filters
            3. Since remap is removed, change to VIDEO plot mode (no reset)
            4. Enable buttons
        '''        
        self.vid = copy.deepcopy(self.vid_raw)

        # Background substraction
        try:
            tn1 = float(self.entry_tn1.get())
            tn2 = float(self.entry_tn2.get())
            if tn2 >= tn1:
                self.vid.subtract_noise(t1=tn1, t2=tn2, speed_flag=True) #from BVO
            else: logger.warning('No background substracted')
        except: logger.warning('No background substracted')
        # Median filter
        m_filt = self.entry_median.get() if self.entry_median is not None else None
        if m_filt:
            m_val = int(self.entry_median.get())
            self.vid.filter_frames(method = 'median', 
                                   options = {'size': m_val},
                                   speed_flag=True)
        else:
            logger.warning('No median filter')
        # Gaussian filter
        g_filt = self.entry_gauss.get() if self.entry_gauss is not None else None
        if g_filt:
            g_val = int(self.entry_gauss.get())
            self.vid.filter_frames(method = 'gaussian', 
                                   options = {'sigma': g_val},
                                   speed_flag=True)
        else:
            logger.warning('No gaussian filter')

        # Update the plotting
        self.update_video()      
    
    def remap_video(self):
        '''
        Remaps a video
            1. Remap
            2. Update remap plot
            3. Enable buttons
        '''
        smap_precision = int(self.entry_precision.get())
        smap_opt = self.opt_smap.get()
        remap_method = self.opt_remap.get()
        if remap_method == 'Centers':
            method = 'centers'
        elif remap_method == 'Fwrap_simple':
            method = 'forward_warping_simple'
        if smap_opt == "Computed":
            allIn = 2
        else:
            allIn = 1
        par = {
            'method': 2,  # 2 Spline, 1 Linear
            'decimals': smap_precision,
            'allIn': allIn,
            'remap_method': method
            }
        for key, entry in self.entry_params.items():
                par[key] = float(entry.get())

        log_smap = logging.getLogger('ScintSuite.StrikeMap')
        log_FVid = logging.getLogger('ScintSuite.FILDVideo')
        log_smap.setLevel(logging.INFO)
        log_FVid.setLevel(logging.INFO)
        self.vid.remap_loaded_frames(par)
        log_smap.setLevel(logging.DEBUG)
        log_FVid.setLevel(logging.DEBUG)



        self.update_remap()
        self.enabling_after_remaping()

    def load_plus_filter(self):
        old_params = {'shot':self.shot,
                      'diag':self.diag,
                      'tini':self.tini,
                      'tfin':self.tfin}
        new_params = {'shot':int(self.entry_shot.get()),
                      'diag':int(self.entry_diag.get()),
                      'tini':float(self.entry_t1.get()),
                      'tfin':float(self.entry_t2.get())}
        if old_params != new_params or self.vid is None:
            self.load_video()
        self.process_video()

    def do_all_actions(self):
        old_params = {'shot':self.shot,
                      'diag':self.diag,
                      'tini':self.tini,
                      'tfin':self.tfin}
        new_params = {'shot':int(self.entry_shot.get()),
                      'diag':int(self.entry_diag.get()),
                      'tini':float(self.entry_t1.get()),
                      'tfin':float(self.entry_t2.get())}
        if old_params != new_params or self.vid is None:
            self.load_video()
        self.process_video()
        self.remap_video()

    def change_cmap(self, event=None):
        '''
        Just changes the colorbar
        '''

        self.im1.set_cmap(self.cmaps[self.combo_cmap_c.get()])
        self.cbar1.update_normal(self.im1)
        if self.im2 is not None:
            self.im2.set_cmap(self.cmaps[self.combo_cmap_r.get()])
            self.cbar2.update_normal(self.im2)
        self.canvas.draw_idle()

    def plot_smap_button(self):
        '''
        Button control the strikemap plot
        '''
        self.smap_state = not self.smap_state  # Change button state
        if self.smap_state:
            self.btn_smap.config(text="SMAP")
        else:
            self.btn_smap.config(text="SMAP")
        self.plot_lines()
        self.canvas.draw_idle()

    def plot_scint_button(self):
        '''
        Button control the strikemap plot
        '''
        self.scint_state = not self.scint_state  # Change button state
        if self.scint_state:
            self.btn_scint.config(text="SCINT")
        else:
            self.btn_scint.config(text="SCINT")
        self.plot_lines()
        self.canvas.draw_idle()

    ## ---- Data export
    def export_data(self):
        '''
        Export data to a folder
        '''
        self.shot = int(self.entry_shot.get())
        self.diag = int(self.entry_diag.get())
        shot_label = str(self.shot) + 'FILD' + str(self.diag)
        ubi = os.path.join(self.save_folder, mach, shot_label)
        self.vid.export_remap(folder = ubi, clean = True, overwrite = True)
        self.save_gui_config(folder = ubi)
        logger.info('Data saved')

    def get_gui_config(self):
        params_values = {k: float(e.get()) for k, e in self.entry_params.items()}
        config = {
            "shot": int(self.entry_shot.get()),
            "diag": int(self.entry_diag.get()),
            "t1": float(self.entry_t1.get()),
            "t2": float(self.entry_t2.get()),
            "tn1": float(self.entry_tn1.get()),
            "tn2": float(self.entry_tn2.get()),
            "median": int(self.entry_median.get()),
            "gaussian": int(self.entry_gauss.get()),
            "parameters": params_values,
            "smaps": str(self.opt_smap.get()),
            "precision": int(self.entry_precision.get()),
            "method": str(self.opt_remap.get()),
        }
        logger.info('GUI configuration obtained')
        return config
    
    def save_gui_config(self, folder: str = None, 
                        filename: str = "gui_config.yaml"):
        
        if folder is None:
            folder = os.path.join(pa.Results, str(self.shot), self.diag,
                                str(self.diag_ID))
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        config = self.get_gui_config()
        with open(filepath, "w") as f:
            yaml.dump(config, f)
        logger.info('GUI configuration saved')

    def load_gui_config_from_yaml(self, folder: str = None, 
                        filename: str = "gui_config.yaml"):
        
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No config file found at {filepath}")
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        logger.info('Stored GUI configuration loaded')
            
        return config

    def apply_gui_config(self, config: dict):

        mapping = {
            "shot": self.entry_shot,
            "diag": self.entry_diag,
            "t1": self.entry_t1,
            "t2": self.entry_t2,
            "tn1": self.entry_tn1,
            "tn2": self.entry_tn2,
            "median": self.entry_median,
            "gaussian": self.entry_gauss,
            "precision": self.entry_precision,
        }
        for key, widget in mapping.items():
            if key in config:
                widget.delete(0, "end")
                widget.insert(0, str(config[key]))

        if "parameters" in config:
            for k, val in config["parameters"].items():
                if k in self.entry_params:
                    self.entry_params[k].delete(0, "end")
                    self.entry_params[k].insert(0, str(val))
        if "smaps" in config:
            self.opt_smap.set(str(config["smaps"]))

        if "method" in config:
            self.opt_remap.set(str(config["method"]))
        logger.info('New GUI configuration applied')

    ## ---- Time Traces        
    def extract_time_trace(self, remap = False):
        '''
        Start ROI selection for time trace
        '''        
        ax = self.ax2 if remap else self.ax1
        im = self.im2 if remap else self.im1

        self.reset_roi()
        self.collecting = True
        self.roi_points = []
        self.roi_line, = ax.plot([], [], c='lime', lw=1, ls='--')
        self.roi_scatter = ax.scatter([], [], c='lime', marker='+', s=50, lw=2)

        self.cid_click = self.fig.canvas.mpl_connect(
            'button_press_event', lambda event: self.on_click(event, 
                                                              remap=remap))
        logger.info('Select the vertex with L click. Undo with R click')
        logger.info('Once you finished, click the middle button')   

    def reset_roi(self):
        # disconnect events
        if self.cid_click is not None:
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
        # eliminate mask overlay
        if hasattr(self, 'mask_artist') and self.roi_mask is not None:
            self.roi_mask.remove()
            self.roi_mask = None
        # reset data
        self.roi_points = []
        self.collecting = False

        self.canvas.draw_idle()

    def on_click(self, event, remap):
        if not self.collecting:
            return
        ax = self.ax2 if remap else self.ax1
        im = self.im2 if remap else self.im1
        if event.inaxes != ax:
            logger.warning("Click out of axis. ROI ignored.")
            self.off_click()
            return
        # Left click
        if event.button == 1:
            self.roi_points.append((event.xdata, event.ydata))
            self.plot_roi_line_scatter()
        # Right click
        elif event.button == 3:
            if self.roi_points:
                self.roi_points.pop()
                self.plot_roi_line_scatter()
        # Middle click
        elif event.button == 2:
            if len(self.roi_points) >= 3:
                logger.info('Computing mask and time trace')
                self.generate_mask(remap=remap)
                self.off_click()
                self.plot_time_trace(remap=remap)

    def off_click(self):
        if self.cid_click is not None:
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
        self.collecting = False
        if self.roi_scatter is not None:
            self.roi_scatter.remove()
            self.roi_scatter = None
        self.canvas.draw_idle()

    def plot_roi_line_scatter(self):
        if self.roi_points:
            xs, ys = zip(*self.roi_points)
        else:
            xs, ys = [], []
        self.roi_line.set_data(xs, ys)
        self.roi_scatter.set_offsets(np.c_[xs, ys])
        self.canvas.draw_idle()       

    def generate_mask(self, remap):
        ax = self.ax2 if remap else self.ax1
        im = self.im2 if remap else self.im1
        ny, nx = im.get_array().shape[:2]
        xmin, xmax, ymin, ymax = im.get_extent()
        x_vals = np.linspace(xmin, xmax, nx)
        y_vals = np.linspace(ymin, ymax, ny)[::-1] # invert
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        poly = Path(self.roi_points)
        coords = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        mask = poly.contains_points(coords)
        self.roi_mask = mask.reshape((ny, nx)).astype(np.uint8)

        if self.roi_line is not None:
            xs, ys = zip(*(self.roi_points + [self.roi_points[0]]))
            self.roi_line.set_data(xs, ys)

    def plot_time_trace(self, remap):
        if self.roi_mask is None:
            return
        fig_alive = (self.fig3 is not None and self.ax3 is not None 
                    and plt.fignum_exists(self.fig3.number))
        
        if not fig_alive:
            self.fig3, (self.ax3, self.ax4) = plt.subplots(2, 1, 
                                                           figsize=(8, 6),
                                                           sharex=True)
        if remap:
            mask_da = xr.DataArray(self.roi_mask, dims=['y','x'],
                                   coords = {'y':self.vid.remap_dat.frames.y,
                                             'x':self.vid.remap_dat.frames.x})
            mask_da = mask_da.transpose('y','x')
            masked_remaps = self.remaps * mask_da
            time_trace = masked_remaps.mean(dim=['y','x'])
            time_trace.plot(ax=self.ax4, ls='-')
        else:
            mask_da = xr.DataArray(self.roi_mask, dims=['px','py'],
                                   coords = {'py':self.vid.exp_dat.frames.py,
                                             'px':self.vid.exp_dat.frames.px})
            masked_frames = self.frames * mask_da
            time_trace = masked_frames.mean(dim=['px','py'])
            time_trace.plot(ax=self.ax3, ls='-')

        for ax in (self.ax3, self.ax4):
            ax.set_title(' '.join([mach,
                                        '#'+str(self.entry_shot.get()),
                                        'FILD'+str(self.diag)]))
            ax.set_xlabel("Time [s]")
            ax.grid(True)
            ax.set_xlim(self.tini,self.tfin)
            ax.set_ylim(0, max(self.ax3.get_ylim()[1], 
                                     np.max(time_trace.max())*1.2))
            ax.xaxis.set_tick_params(labelbottom=True)
            
        self.ax3.set_ylabel("Mean of ROI (video)")
        self.ax4.set_ylabel("Mean of ROI (remap)")

        self.fig3.tight_layout()
        self.fig3.align_ylabels()
        self.fig3.show()
        self.fig3.canvas.draw()
        self.fig3.canvas.flush_events()

    ## ---- Data update
    def update_video(self):
        '''
        Updates data used for plotting.
            1. Get frames
            2. Compute values and maximums
            4. Plots frame (current)
            5. Generate secondary ax and cbar
            5. Sets canvas parameters
        '''
        self.frames = self.vid.exp_dat.frames\
            .sel(t=slice(self.tini,self.tfin))\
            .transpose('t','px','py')
        xlabel, ylabel = 'xpix','ypix'
        pad, right = 0.1, 0.5
        self.smap_state = False
        self.scint_state = False

        self.data_vals1 = self.frames.data
        self.vamx_all1 = self.frames.quantile(0.999, dim=['px','py']).values

        self.ax1.clear()
        self.im1 = self.frames.isel(t=self.current_frame).plot.imshow(
            ax=self.ax1, add_colorbar=False, cmap=self.cmaps[self.combo_cmap_c.get()])
        self.ax1.set_title("")
        self.ax1.set_xlabel(xlabel)
        self.ax1.set_ylabel(ylabel)
        self.ax1.set_aspect(1)
        self.t_text1 = self.ax1.text(0.98, 1.01, 
                f"{float(self.frames.t[self.current_frame].values):.3f}"+' s',
                ha='right', va='bottom', transform=self.ax1.transAxes, color='k')
        self.shot_text = self.ax1.text(0.02, 1.01,
                mach+ ' #'+str(self.entry_shot.get())+' FILD'+str(self.diag),
                ha='left', va='bottom', transform=self.ax1.transAxes, color='k')
        try:
            self.cbar1.remove()
        except:
            pass

        self.divider = make_axes_locatable(self.ax1)
        self.cax1 = self.divider.append_axes("right", size="3%", pad=pad)
        self.cbar1 = self.fig.colorbar(self.im1, cax=self.cax1)
        self.update_plot(self.current_frame)

        self.canvas.draw_idle()

    def update_remap(self):
        '''
        Updates data used for plotting.
            1. Get remaps
            2. Compute values and maximums
            4. Plots frame (current)
            5. Generate secondary ax and cbar
            5. Sets canvas parameters
        '''
        self.remaps = self.vid.remap_dat.frames\
            .sel(t=slice(self.tini,self.tfin))\
            .transpose('t','y','x')
        xlabel, ylabel = 'Pitch angle [º]', 'Gyroradius [cm]'
        pad, right = 0.05, 0.5

        self.data_vals2 = self.remaps.data
        self.vamx_all2 = self.remaps.quantile(0.999, dim=['y','x']).values
        self.ax2.clear()
        self.im2 = self.remaps.isel(t=self.current_frame).plot.imshow(
            ax=self.ax2, add_colorbar=False, cmap=self.cmaps[self.combo_cmap_r.get()])
        self.ax2.set_title("")
        self.ax2.set_xlabel(xlabel)
        self.ax2.set_ylabel(ylabel)
        self.ax2.set_box_aspect(1)
        self.t_text2 = self.ax2.text(0.98, 1.01, 
                f"{float(self.frames.t[self.current_frame].values):.3f}"+' s',
                ha='right', va='bottom', transform=self.ax2.transAxes, color='k')
        self.shot_text = self.ax2.text(0.02, 1.01,
                mach+ ' #'+str(self.entry_shot.get())+' FILD'+str(self.diag),
                ha='left', va='bottom', transform=self.ax2.transAxes, color='k')
        try:
            self.cbar2.remove()
        except:
            pass

        self.divider = make_axes_locatable(self.ax2)
        self.cax2 = self.divider.append_axes("right", size="4%", pad=pad)
        self.cbar2 = self.fig.colorbar(self.im2, cax=self.cax2)
        self.update_plot(self.current_frame)

        self.canvas.draw_idle()

    ## ---- Button enabling
    def enabling_after_loading(self):
        '''
        Activate/deactivate widgets after loading a video
        '''
        self.btn_TT1.configure(state=tk.NORMAL)
        self.btn_filter.configure(state=tk.NORMAL)
        self.btn_remap.configure(state=tk.NORMAL)

        self.combo_cmap_c.configure(state=tk.NORMAL)
        self.entry_vmin_c.bind("<Return>", lambda e: 
                             self.update_plot(self.current_frame))
        self.entry_vmax_c.bind("<Return>", lambda e: 
                             self.update_plot(self.current_frame))
        self.btn_smap.configure(state=tk.DISABLED)
        if self.vid.scintillator is not None and \
            hasattr(self.vid.scintillator, "plot_pix"):
            self.btn_scint.configure(state=tk.NORMAL)
        else:
            self.btn_scint.configure(state=tk.DISABLED)

        self.btn_TT2.configure(state=tk.DISABLED)


    def enabling_after_remaping(self):
        '''
        Activate/deactivate widgets after remapping a video
        '''
        self.btn_smap.configure(state=tk.NORMAL)
        self.btn_TT2.configure(state=tk.NORMAL)

        self.combo_cmap_r.configure(state=tk.NORMAL)
        self.entry_vmin_r.bind("<Return>", lambda e: 
                             self.update_plot(self.current_frame))
        self.entry_vmax_r.bind("<Return>", lambda e: 
                             self.update_plot(self.current_frame)) 
        self.btn_export.configure(state=tk.NORMAL)