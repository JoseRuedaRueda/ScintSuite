"""
Remap video from FILD cameras

Lesson 4 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise

jose Rueda: jrrueda@us.es

Note; Written for version 0.3.0. Checked for version 1.0.0
"""
import ScintSuite as ss
import ScintSuite._Mapping as ssmap
from ScintSuite._Mapping._Calibration import CalParams
import matplotlib.pyplot as plt
import numpy as np
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 79185
diag_ID = 1  # 6 for rFILD
t1 = 1.29    # Initial time to be loaded, [s]
t2 = 1.7     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.19    # Initial time to average the frames for noise subtraction [s]
tn2 = 0.22     # Final time to average the frames for noise subtraction [s]
flag_copy = False  # If true, a copy of the frames will be done while
#                  # substracting the noise

# - Filter options:
apply_filter = False  # Flag to apply filter to the frames
kind_of_filter = 'median'
options_filter = {
    'size': 1        # Size of the window to apply the filter
}
# If you want a gaussian one
# kind_of_filter = 'gaussian'
# options_filter = {
#     'sigma': 1        # sigma of the gaussian for the convolution (in pixels)
# }

#smap = ssmap.StrikeMap(0, '/home/poley/NoTivoli/SINPA/runs/Benchmark_75620_ur/results/Benchmark_75620_ur.map')
smap = ssmap.StrikeMap(0, '/home/poley/NoTivoli/SINPA/runs/79185@1.400_ur/results/79185@1.400_ur.map')
#smap = ssmap.StrikeMap(0, '/home/jansen/SINPA/runs/77971@0.4_ur/results/77971@0.4_ur.map')
cal = CalParams() 
cal.xscale = 18190 
cal.yscale = 18190
cal.xshift = 352
cal.yshift = 816
cal.deg = 0              
cal.camera = 'CCD'     
smap.calculate_pixel_coordinates(cal)
Br, Bt, Bz = 0.0141, -1.1328, 0.1532 #(Br, Bphi, Bz) #75620@1.020s
modB = 1.2080342638545698
smap.calculate_energy(modB)
smap.convert_to_normalized_pitch()
smap.setRemapVariables(('p0', 'e0'), verbose=False)

# - Remapping options:
save_remap = False  # If true, the remap will be saved in a netCDF file
par = {
    'ymin': 4.0,      # Minimum gyroradius [in cm]
    'ymax': 65.0,     # Maximum gyroradius [in cm]
    'dy': 5,        # Interval of the gyroradius [in cm]
    'xmin': -1,     # Minimum pitch angle [in degrees]
    'xmax': 0,     # Maximum pitch angle [in degrees]
    'dx': 0.05,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 1,
    'smap_folder': '/home/poley/NoTivoli/SINPA/runs/79185@1.270_ur/results/',
    'map': smap
    }  # Precision for the strike map (1 is more than enough)
# Note, if the smap_folder variable is not present, the program will look for
# the strike maps in the path given by ss.paths.StrikeMaps
# - Plotting options:
plot_profiles_in_time = False   # Plot the time evolution of pitch and r
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
#filename='/home/poley/NoTivoli/ScintSuite/11111.mat'
filename='/tmp/poley/79300_b.mat'
vid = ss.vid.FILDVideo(shot=79185)

# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise and filter frames
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2, flag_copy=flag_copy)

if apply_filter:
    vid.filter_frames(kind_of_filter, options_filter)
# -----------------------------------------------------------------------------
# --- Section 4: Remap
# -----------------------------------------------------------------------------
# - Remap frames:
vid.read_frame(t1=t1,t2=t2)
vid.remap_loaded_frames(par)

# - Plot:
if plot_profiles_in_time:
    b = vid.integrate_remap(xmax=par['xmax'],ymax=par['ymax']) # rL max is 18
    # Integral in XI
    fig, ax = plt.subplots()
    b['integral_over_x'].plot()
    # Integral in rL
    fig2, ax2 = plt.subplots()
    b['integral_over_y'].plot()
# - Export remapped data
if save_remap:
    vid.export_remap()
    
    
    
    ######
    
    
    """
Remap video from FILD cameras
Lesson 4 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise
jose Rueda: jrrueda@us.es
Note; Written for version 0.3.0. Checked for version 1.0.0
"""
import ScintSuite as ss
import ScintSuite._Mapping as ssmap
from ScintSuite._Mapping._Calibration import CalParams
import matplotlib.pyplot as plt
from time import time
import ScintSuite.LibData.TCV.Equilibrium as TCV_equilibrium
import numpy as np
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 75620
diag_ID = 1  # 6 for rFILD
t1 = 1.015     # Initial time to be loaded, [s]
t2 = 1.017     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources
# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.005     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.05     # Final time to average the frames for noise subtraction [s]
flag_copy = False  # If true, a copy of the frames will be done while
#                  # substracting the noise
# - Filter options:
apply_filter = False  # Flag to apply filter to the frames
kind_of_filter = 'median'
options_filter = {
    'size': 1        # Size of the window to apply the filter
}
# If you want a gaussian one
# kind_of_filter = 'gaussian'
# options_filter = {
#     'sigma': 1        # sigma of the gaussian for the convolution (in pixels)
# }
smap = ssmap.StrikeMap(0, '/home/jansen/NoTivoli/SINPA/runs/75620@1.020_ur_r3/results/75620@1.020_ur_r3.map')
cal = CalParams()
cal.xscale = 18165
cal.yscale = 18165
cal.xshift = 126
cal.yshift = 775
cal.deg = 0
cal.camera = 'CCD'
smap.calculate_pixel_coordinates(cal)
modB = 1.1462
if False:
    Rin = -17 *0.001
    Br, Bz, Bt, bp =  TCV_equilibrium.get_mag_field(shot, Rin, time)
    modB = np.sqrt(Br**2 + Bz**2 + Bt**2)
smap.calculate_energy(modB)
smap.convert_to_normalized_pitch()
smap.setRemapVariables(('p0', 'e0'), verbose=False)
# - Remapping options:
save_remap = False  # If true, the remap will be saved in a netCDF file
par = {
    'ymin': 5.0,      # Minimum gyroradius [in cm]
    'ymax': 50.0,     # Maximum gyroradius [in cm]
    'dy': 0.5,        # Interval of the gyroradius [in cm]
    'xmin': -1,     # Minimum pitch angle [in degrees]
    'xmax': 0,     # Maximum pitch angle [in degrees]
    'dx': 0.005,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 5,
    'smap_folder': '',
    'map': smap,
    'variables_to_remap':('p0', 'e0'),#'pitch', 'energy'),
    'remap_method':  'centers',
    #'MC_number': 150,
    #'transformationMatrixLimit': 30
    }
# Note, if the smap_folder variable is not present, the program will look for
# the strike maps in the path given by ss.paths.StrikeMaps
# - Plotting options:
plot_profiles_in_time = False   # Plot the time evolution of pitch and r
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
#vid = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID)
vid = ss.vid.FILDVideo(file='/tmp/poley/%i.mat'%shot)#shot=shot, diag_ID=diag_ID)
# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise and filter frames
# -----------------------------------------------------------------------------
if subtract_noise:
    it1 = np.argmin(np.abs(vid.exp_dat.t.values - tn1))
    it2 = np.argmin(np.abs(vid.exp_dat.t.values - tn2))
    frame = vid.exp_dat['frames'].isel(t=slice(it1, it2)).mean(dim='t')
    vid.read_frame(t1=t1, t2=t2) #first get rid of the frames to be faster
    vid.subtract_noise(frame=frame, flag_copy=False)
if apply_filter:
    vid.filter_frames(kind_of_filter, options_filter)
# -----------------------------------------------------------------------------
# --- Section 4: Remap
# -----------------------------------------------------------------------------
# - Remap frames:
flag_MC = False
if flag_MC:
    MC_number = 150  # number of MC markers per pixel
    # Calculate the relation pixel - gyr and pitch
    grid = {'ymin': par['ymin'], 'ymax': par['ymax'], 'dy': par['dy'],
            'xmin': par['xmin'], 'xmax': par['xmax'], 'dx': par['dx']}
    smap.interp_grid(vid.exp_dat['frames'].shape[0:2], method=par['method'],
                    MC_number=MC_number, grid_params=grid, limitation=30)
# Include this map in the remapping parameters:
par['map'] = smap
vid.remap_loaded_frames(par)
# - Plot:
if plot_profiles_in_time:
    b = vid.integrate_remap(xmax=par['xmax'],ymax=par['ymax']) # rL max is 18
    # Integral in XI
    fig, ax = plt.subplots()
    b['integral_over_x'].plot()
    # Integral in rL
    fig2, ax2 = plt.subplots()
    b['integral_over_y'].plot()
# - Export remapped data
if save_remap:
    vid.export_remap()
