"""
Remap video from FILD cameras using just one strike map

Lesson 6 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise and timetraces remap of the whole video, but
this time using a given strike map. As just one Smap will be used, the MC
remapping will be employed

Please write the proper path to your file in the Smap_file

jose Rueda: jrrueda@us.es

Note: Written for version 0.5.3. Revised for version 0.8.0
"""
import ScintSuite.as ss
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 40412
diag_ID = 1  # 6 for rFILD
t1 = 0.0    # Initial time to be loaded, [s]
t2 = 10.0     # Final time to be loaded [s]
limitation = False  # If true, the suite will not allow to load more than
limit = 2048        # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.0     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.5     # Final time to average the frames for noise subtraction [s]

# - Remapping options:
save_remap = False
par = {
    'rmin': 1.2,      # Minimum gyroradius [in cm]
    'rmax': 5.5,     # Maximum gyroradius [in cm]
    'dr': 0.05,        # Interval of the gyroradius [in cm]
    'ximin': 1.4,     # Minimum pitch angle [in degrees]
    'ximax': 2.2,     # Maximum pitch angle [in degrees]
    'dxi': 0.01,    # Pitch angle interval
    # Parameters for the pitch-gryroradius profiles
    'rprofmin': 0,     # Minimum gyroradius for the pitch profile calculation
    'rprofmax': 200,    # Maximum gyroradius for the pitch profile calculation
    'xiprofmin': 0,    # Minimum pitch for the gyroradius profile calculation
    'xiprofmax': 200,    # Maximum pitch for the gyroradius profile calculation
    # methods for the interpolation
    'method': 2,    # 2 Spline, 1 Linear (smap interpolation)
    'decimals': 0,  # Precision for the strike map (1 is more than enough)
    'remap_method': 'centers',  # Remap algorithm
    }
MC_number = 0  # number of MC markers per pixel
# - Plotting options:
plot_profiles_in_time = True   # Plot the time evolution of pitch and r
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
vid = ss.vid.INPAVideo(shot=shot, diag_ID=diag_ID)
# - read the frames:
tdummy = time()
print('Reading camera frames: ', shot, '...')
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)
print('Elapsed time [s]: ', time() - tdummy)
# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)

# -----------------------------------------------------------------------------
# --- Section 3: Proceed with the remap
# -----------------------------------------------------------------------------
# - Remap frames:
vid.remap_loaded_frames(par)
# - Plot:
if plot_profiles_in_time:
    vid.plot_profiles_in_time()

# -----------------------------------------------------------------------------
# --- Section 4: Export data
# -----------------------------------------------------------------------------
# - Export remapped data
if save_remap:
    vid.export_remap()
