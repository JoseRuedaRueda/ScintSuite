"""GUI for FILDSIM vanilla users"""
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from tkinter import ttk
from matplotlib.path import Path
import xarray as xr

import os
import copy
import pickle

import logging
logger = logging.getLogger('ScintSuite.FILDvideoGUI')
logging.basicConfig(level=logging.INFO)


class FILDvideoGUI:
    '''
    Build a GUI to analyse data from FILD videos.
    '''

    def __init__(self):
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("FILD data explorer GUI")

        self.vid = None
        self.vid_raw = None
        self.frames = None
        self.current_frame = 0
        self.data_vals = []
        self.vmax_all = []

        self.smap_state = False
        self.scint_state = False

        self.save_folder = ss.paths.ScintSuite + '/Data/VideosRemaps/FILD' 

        self.collecting = False
        self.roi_points = []
        self.roi_line = None
        self.roi_scatter = None
        self.cid_click = None
        self.ax2 = None

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
        self.cmap_default = 'Gamma_III'

        self.fig = Figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(111)
        placeholder = np.zeros((200, 200))
        self.im = self.ax.imshow(
            placeholder, cmap=self.cmaps[self.cmap_default],
            origin='lower', vmin=0, vmax=1
        )
        self.ax.set_axis_off()
        self.formater = FuncFormatter(lambda x, _: f"{x:.1e}")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)

        # LAYOUT
        # -------------------------------------------------------------------
        # ---- Title
        crow = 0
        tk.Label(self.root, text="EXP:", font=("Arial", 14, "bold"))\
            .grid(row=crow, column=0, columnspan=2)
        self.opts_exp = ["AUG", "D3D"]
        self.opt_exp = tk.StringVar(value="AUG")
        self.menu_exp = tk.OptionMenu(self.root, 
                                       self.opt_exp, *self.opts_exp)
        self.menu_exp.grid(row=crow, column=2, columnspan=2, sticky='news')
        # ---- Shot
        crow += 1
        tk.Label(self.root, text="Shot:").grid(row=crow, column=0, sticky='e')
        self.entry_shot = tk.Entry(self.root, width=6)
        self.entry_shot.insert(0, "43440")
        self.entry_shot.grid(row=crow, column=1)
        tk.Label(self.root, text="FILD:").grid(row=crow, column=2, sticky='e')
        self.entry_diag = tk.Entry(self.root, width=6)
        self.entry_diag.insert(0, "4")
        self.entry_diag.grid(row=crow, column=3)
        # ---- Time interval
        crow += 1
        tk.Label(self.root, text="Time interval (s):")\
            .grid(row=crow, column=0, columnspan=2, sticky='e')
        self.entry_t1 = tk.Entry(self.root, width=6)
        self.entry_t1.insert(0, "0")
        self.entry_t1.grid(row=crow, column=2)
        self.entry_t2 = tk.Entry(self.root, width=6)
        self.entry_t2.insert(0, "10")
        self.entry_t2.grid(row=crow, column=3)
        # ---- Load video button
        crow += 1
        self.btn_load = tk.Button(self.root, text="Read Video", 
                                  command=self.load_video)
        self.btn_load.grid(row=crow, column=0, columnspan=4, sticky='we')
        # ---- Background subtraction
        crow += 1
        tk.Label(self.root, text="BKG sub. (s):")\
            .grid(row=crow, column=0, columnspan=2, sticky="e")
        self.tn1_entry = tk.Entry(self.root, width=6)
        self.tn1_entry.insert(0, "0")
        self.tn1_entry.grid(row=crow, column=2)
        self.tn2_entry = tk.Entry(self.root, width=6)
        self.tn2_entry.insert(0, "0.2")
        self.tn2_entry.grid(row=crow, column=3)
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
        self.btn_filter = tk.Button(
            self.root, text="Filter Video",
            command=self.process_video, state=tk.DISABLED
        )
        self.btn_filter.grid(row=crow, column=0, columnspan=4, sticky='we')
        # ---- Remap parameters
        crow += 1
        parameters = {'xmin': 30, 'xmax': 90, 'dx': 1, 
                      'ymin': 1, 'ymax': 8, 'dy': 0.2}
        self.entry_params = {}
        for i, (name, default) in enumerate(parameters.items()):
            row = i % 3 + crow
            col = (i // 3) * 2
            tk.Label(self.root, text=f"{name}:")\
                .grid(row=row, column=col, sticky='e')
            e = tk.Entry(self.root, width=6)
            e.insert(0, str(default))
            e.grid(row=row, column=col+1)
            self.entry_params[name] = e
        crow += 3
        # ---- Strikemap options
        tk.Label(self.root, text="Smaps:")\
            .grid(row=crow, column=0, columnspan=1, sticky='w')
        self.opts_smap = ["Compute", "Existing"]
        self.opt_smap = tk.StringVar(value="Compute")
        self.menu_smap = tk.OptionMenu(self.root, 
                                       self.opt_smap, *self.opts_smap)
        self.menu_smap.grid(row=crow, column=1, columnspan=3, sticky='we')
        self.menu_smap.configure(state = tk.DISABLED)
        # ---- Strikemap precision
        crow +=1
        tk.Label(self.root, text="Smap precision:")\
            .grid(row=crow, column=0, columnspan=2, sticky='w')
        self.precision_entry = tk.Entry(self.root, width=6)
        self.precision_entry.insert(0, "1")
        self.precision_entry.grid(row=crow, column=2)
        # ---- Remapping method
        crow +=1
        tk.Label(self.root, text="Method:")\
            .grid(row=crow, column=0, columnspan=1)
        self.opts_remap = ["Centers", "Fwrap_simple"]
        self.opt_remap = tk.StringVar(value="Centers")
        self.menu_remap = tk.OptionMenu(self.root, 
                                        self.opt_remap, *self.opts_remap)
        self.menu_remap.grid(row=crow, column=1, columnspan=3, sticky='we')
        self.menu_remap.configure(state = tk.DISABLED)
        # ---- Remap video
        crow +=1
        self.btn_remap = tk.Button(self.root, text="Remap Video", 
                                   command=self.remap_video, 
                                   state=tk.DISABLED)
        self.btn_remap.grid(row=crow, column=0, columnspan=4, sticky='we')    
        # ---- Second title
        crow +=1
        tk.Label(self.root, text="PLOT", font=("Arial", 16, "bold"))\
            .grid(row=crow, column=0, columnspan=4)
        # ---- Plot changes
        crow += 1
        self.opts_plot = ["VIDEO", "REMAP"]
        self.opt_plot = tk.StringVar(value="VIDEO")
        self.menu_plot = tk.OptionMenu(self.root, 
                                       self.opt_plot, *self.opts_plot)
        self.menu_plot.grid(row=crow, column=0, columnspan=4, sticky='we')
        self.menu_plot.configure(state=tk.DISABLED)
        self.opt_plot.trace_add("write", lambda *_: self.change_data_plot())
        # ---- Color menu
        crow += 1
        self.combo_cmap = ttk.Combobox(self.root, values=self.cmap_names,
            state="readonly", width=12)
        self.combo_cmap.set(self.cmap_default)
        self.combo_cmap.bind("<<ComboboxSelected>>", self.change_cmap)
        self.combo_cmap.grid(row=crow, column=0, columnspan=2)
        self.combo_cmap.configure(state=tk.DISABLED)
        # ---- Colorbar limits
        self.entry_vmin = tk.Entry(self.root, width=6)
        self.entry_vmin.insert(0, "0")
        self.entry_vmin.grid(row=crow, column=2)
        self.entry_vmax = tk.Entry(self.root, width=6)
        self.entry_vmax.grid(row=crow, column=3)
        # ---- Smap module
        crow += 1
        self.btn_smap = tk.Button(self.root, text="Plot smap", 
                                  command=self.plot_smap_button, 
                                  state=tk.DISABLED)
        self.btn_smap.grid(row=crow, column=0, columnspan=2, sticky='we')
        # ---- Scinillator module
        self.btn_scint = tk.Button(self.root, text="Plot scint", 
                                   command=self.plot_scint_button,
                                   state=tk.DISABLED)
        self.btn_scint.grid(row=crow, column=2, columnspan=2, sticky='we')

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
        # ---- Time trace button
        self.btn_TT = tk.Button(buttons_frame, text="Extract TT",
                            command=self.extract_time_trace,
                            width=12, state=tk.DISABLED)
        self.btn_TT.pack(side="left", padx=5)
        # ---- Quit button
        btn_quit = tk.Button(buttons_frame, text="Quit",
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
        if not self.data_vals:
            return

        if event.keysym == 'Right' and self.current_frame < len(self.data_vals) - 1:
            self.current_frame += 1
        elif event.keysym == 'Left' and self.current_frame > 0:
            self.current_frame -= 1
        else:
            return

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
        if idx >= len(self.data_vals):
            return
        self.current_frame = idx
        self.im.set_data(self.data_vals[idx])
        self.time_text.set_text(f"{self.frames.t[idx].values:.3f} s")
        self.change_cbar()
        self.plot_lines()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def change_cbar(self, event=None):
        '''
        Changes colorbar limits.
        Triggered in update_plot
        '''
        try: vmin = float(self.entry_vmin.get())
        except: vmin = 0.0
        try: vmax = float(self.entry_vmax.get())
        except: vmax = self.vmax_all[self.current_frame]
        self.im.set_clim(vmin=vmin, vmax=vmax)
        self.cbar.update_normal(self.im)
        self.cbar.formatter = self.formater

    def plot_lines(self):
        '''
        Function that plots lines in the image (smap or scint)
        '''
        logging.disable(logging.CRITICAL)
        ssplt.remove_lines(self.ax)

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
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            smap.plot_pix(ax=self.ax, labels=False)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        if self.scint_state:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.vid.scintillator.plot_pix(ax=self.ax)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        logging.disable(logging.NOTSET)
        self.canvas.draw_idle()


    # ---- Buttons
    def load_video(self):
        '''
        Load a new video data.
            1. Get shot, diagnostic and time data
            2. Load data
            3. Reset GUI interface
            4. Enable buttons
        '''               
        shot = int(self.entry_shot.get())
        diag = int(self.entry_diag.get())
        t1 = float(self.entry_t1.get())
        t2 = float(self.entry_t2.get())

        # Build video object
        if diag == 1:
            self.vid_raw = ss.vid.FILDVideo(shot=shot, diag_ID=diag)
        else:
            filename = f"/shares/departments/AUG/users/alrevi/ScintSuite/MyRoutines/ASDEX/FILD{diag}/{shot}"
            self.vid_raw = ss.vid.FILDVideo(file=filename, diag_ID=diag)
        
        self.vid_raw.read_frame(t1=t1, t2=t2)
        self.vid = copy.deepcopy(self.vid_raw)

        self.smap_state = False
        self.scint_state = False
        self.reset_video()
        self.enabling_after_loading()    

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
            tn1 = float(self.tn1_entry.get())
            tn2 = float(self.tn2_entry.get())
            if tn2 >= tn1:
                self.vid.subtract_noise(t1=tn1, t2=tn2, fast=True) #from BVO
            else: logger.warning('No background substracted')
        except: logger.warning('No background substracted')
        # Median filter
        m_filt = self.entry_median.get() if self.entry_median is not None else None
        if m_filt:
            m_val = int(self.entry_median.get())
            self.vid.filter_frames(method = 'median', options = {'size': m_val})
        else:
            logger.warning('No median filter')
        # Gaussian filter
        g_filt = self.entry_gauss.get() if self.entry_gauss is not None else None
        if g_filt:
            g_val = int(self.entry_gauss.get())
            self.vid.filter_frames(method = 'gaussian', options = {'sigma': g_val})
        else:
            logger.warning('No gaussian filter')

        # Update the plotting
        self.opt_plot.set("VIDEO") # change_data_plot is triggered
        self.enabling_after_loading()        
    
    def remap_video(self):
        '''
        Remaps a video
            1. Get all the parameters
            2. Disables logger to not saturate the terminal
            3. Update data
            4. Update plot
            5. Enable the rest of the widgets
        '''

        smap_precision = int(self.precision_entry.get())
        smap_opt = self.opt_smap.get()
        remap_method = self.opt_remap.get()
        if remap_method == 'Centers':
            method = 'centers'
        elif remap_method == 'Fwrap_simple':
            method = 'forward_warping_simple'
        if smap_opt == "Compute":
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

        logging.disable(logging.CRITICAL)
        self.vid.remap_loaded_frames(par)
        logging.disable(logging.NOTSET)

        self.enabling_after_remaping()
        self.opt_plot.set("REMAP") # change_data_plot is triggered

    def change_cmap(self, event=None):
        '''
        Just changes the colorbar
        '''

        self.im.set_cmap(self.cmaps[self.combo_cmap.get()])
        self.cbar.update_normal(self.im)
        self.canvas.draw_idle()

    def plot_smap_button(self):
        '''
        Button control the strikemap plot
        '''
        self.smap_state = not self.smap_state  # Change button state
        if self.smap_state:
            self.btn_smap.config(text="rm smap")
        else:
            self.btn_smap.config(text="Plot smap")
        self.plot_lines()
        self.canvas.draw_idle()

    def plot_scint_button(self):
        '''
        Button control the strikemap plot
        '''
        self.scint_state = not self.scint_state  # Change button state
        if self.scint_state:
            self.btn_scint.config(text="rm scint")
        else:
            self.btn_scint.config(text="Plot scint")
        self.plot_lines()
        self.canvas.draw_idle()

    def export_data(self):
        '''
        Export data to a folder
        '''
        shot = int(self.entry_shot.get())
        diag = int(self.entry_diag.get())
        globals()[f'fild{diag}_{shot}'] = copy.deepcopy(self.vid)
        with open(self.save_folder+f'AUG_fild{diag}_{shot}'+".obj", "wb") as f:
            pickle.dump(self.vid, f)
        logger.info('------------------ DATA SAVED ------------------')

    def extract_time_trace(self):
        '''
        Start ROI selection for time trace
        '''        
        self.reset_roi()

        self.collecting = True
        self.roi_points = []

        self.roi_line, = self.ax.plot([], [], c='lime', lw=1, ls='--')
        self.roi_scatter = self.ax.scatter([], [], c='lime', marker='+', s=50, lw=2)


        self.cid_click = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_click)
        logger.info('Please select the vertex of the roi in the figure')
        logger.info('Select each vertex with left click')
        logger.info('Undo your selection with right click')
        logger.info('Once you finished, click the middle button')   

    def reset_roi(self):
        # disconnect events
        if self.cid_click is not None:
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
        if self.roi_line is not None:
            self.roi_line.remove()
            self.roi_line = None
        if self.roi_scatter is not None:
            self.roi_scatter.remove()
            self.roi_scatter = None
        # eliminate mask overlay
        if hasattr(self, 'mask_artist') and self.roi_mask is not None:
            self.roi_mask.remove()
            self.roi_mask = None
        # reset data
        self.roi_points = []
        self.collecting = False

        self.canvas.draw_idle()

    def on_click(self, event):
        if not self.collecting:
            return
        if event.inaxes != self.ax:
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
                self.generate_mask()
                self.plot_time_trace()

    def off_click(self):
        if self.cid_click is not None:
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
        self.collecting = False
        self.roi_line = None   
        self.roi_scatter = None    

    def plot_roi_line_scatter(self):
        if self.roi_points:
            xs, ys = zip(*self.roi_points)
        else:
            xs, ys = [], []
        self.roi_line.set_data(xs, ys)
        self.roi_scatter.set_offsets(np.c_[xs, ys])
        self.canvas.draw_idle()       

    def generate_mask(self):
        ny, nx = self.im.get_array().shape
        poly = Path(self.roi_points)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.vstack((x.ravel(), y.ravel())).T
        mask = poly.contains_points(coords)
        self.roi_mask = mask.reshape((ny, nx)).astype(np.uint8)

        if self.roi_line is not None:
            xs, ys = zip(*(self.roi_points + [self.roi_points[0]]))
            self.roi_line.set_data(xs, ys)
        self.canvas.draw_idle()

    def plot_time_trace(self):
        if self.roi_mask is None:
            return

        mask_da = xr.DataArray(self.roi_mask, dims=self.spatial_dims)
        masked_frames = self.frames * mask_da
        time_trace = masked_frames.sum(dim=self.spatial_dims)

        if self.ax2 is not None:
            time_trace.plot(ax=self.ax2)
        else: 
            self.fig2, self.ax2 = plt.subplots(figsize=(8, 4))
            time_trace.plot(ax=self.ax2)
        plt.xlabel("Frame")
        plt.ylabel("Sum of ROI")
        plt.title("Time trace of ROI")
        plt.grid(True)
        plt.show()



    # ---- Internals
    def reset_video(self):
        '''
        Function to go back to initial postion of the GUI.
            1. Set time to 0
            2. Update slider to new limits
            3. Set data to plot to VIDEO
        '''
        self.current_frame = 0 #go back to first
        self.slider.set(self.current_frame)
        self.opt_plot.set("VIDEO")  # change_data_plot is triggered
        self.slider.config(from_=0, to=len(self.data_vals)-1)

    def change_data_plot(self):
        '''
        Function to change between VIDEO or REMAP data.
            1. Updates data
            2. Resets colorbar
            3. Update plot (current)
        '''
        self.update_data() # prepares data to plot. changes origin if necessasry
        self.entry_vmin.delete(0, tk.END)
        self.entry_vmin.insert(0, '0')
        self.entry_vmax.delete(0, tk.END) # resets colorbar values
        self.update_plot(self.current_frame) #updates de plot in the same frame

    def update_data(self):
        '''
        Updates data used for plotting.
            1. Identifies which data wants to be plotted 
            2. Sets data for plotting
            3. Computes vmax
            4. Plots frame (current)
            5. Sets canvas parameters
        '''
        what_data = self.opt_plot.get()

        if what_data == 'VIDEO':
            self.frames = self.vid.exp_dat.frames.transpose('t','px','py')
            self.spatial_dims = ['px','py']
            xlabel, ylabel = '',''
            pad, right = 0.1, 0.5
            aspect = None
            hide_ticks =  True
            self.btn_smap.configure(state=tk.NORMAL)
            self.btn_scint.configure(state=tk.NORMAL)
        elif what_data == 'REMAP':
            self.frames = self.vid.remap_dat.frames.transpose('t','y','x')
            self.spatial_dims = ['x','y']
            xlabel, ylabel = 'Pitch angle [º]', 'Gyroradius [cm]'
            pad, right = 0.1, 0.5
            aspect = 1
            hide_ticks =  False
            self.btn_smap.configure(state=tk.DISABLED)
            self.btn_scint.configure(state=tk.DISABLED)
            self.smap_state = False
            self.scint_state = False

        frames_da = self.frames.data
        self.data_vals[:] = frames_da
        self.vmax_all[:] = self.frames.quantile(0.999, dim=self.spatial_dims).values
        
        self.ax.clear()
        self.im = self.frames.isel(t=self.current_frame).plot.imshow(
            ax=self.ax, add_colorbar=False, cmap=self.cmaps[self.combo_cmap.get()])
        self.ax.set_title("")
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if hide_ticks:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self.ax.set_box_aspect(aspect)
        self.ax.set_aspect(1 if aspect is None else 'auto')
        self.time_text = self.ax.text(0.98, 1.01, 
                            f"{float(self.frames.t[self.current_frame].values):.3f}"+' s',
                            ha='right', va='bottom', transform=self.ax.transAxes, color='k')
        self.shot_text = self.ax.text(0.02, 1.01, 
                            '#'+str(self.entry_shot.get()),
                            ha='left', va='bottom', transform=self.ax.transAxes, color='k')
        try:
            self.cbar.remove()
        except:
            pass

        self.divider = make_axes_locatable(self.ax)
        self.cax = self.divider.append_axes("right", size="3%", pad=pad)
        self.cbar = self.fig.colorbar(self.im, cax=self.cax)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def enabling_after_loading(self):
        '''
        Activate/deactivate widgets after loading a video
        '''
        self.btn_filter.configure(state=tk.NORMAL)
        self.menu_smap.configure(state=tk.NORMAL)
        self.menu_remap.configure(state=tk.NORMAL)
        self.btn_remap.configure(state=tk.NORMAL)
        self.menu_plot.configure(state=tk.DISABLED)
        self.combo_cmap.configure(state=tk.NORMAL)
        self.entry_vmin.bind("<Return>", lambda e: 
                             self.update_plot(self.current_frame))
        self.entry_vmax.bind("<Return>", lambda e: 
                             self.update_plot(self.current_frame))
        self.btn_smap.configure(state=tk.DISABLED)
        self.btn_scint.configure(state=tk.NORMAL)
        self.btn_export.configure(state=tk.DISABLED)
        self.btn_TT.configure(state=tk.NORMAL)

    def enabling_after_remaping(self):
        '''
        Activate/deactivate widgets after remapping a video
        '''
        self.menu_plot.configure(state=tk.NORMAL)
        self.btn_smap.configure(state=tk.DISABLED)
        self.btn_scint.configure(state=tk.DISABLED)
        self.btn_export.configure(state=tk.NORMAL)
        self.btn_TT.configure(state=tk.NORMAL)