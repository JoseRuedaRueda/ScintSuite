"""
Remap video from FILD cameras using just one strike map

Lesson 6 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise and timetraces remap of the whole video, but
this time using a given strike map. As just one Smap will be used, the MC
remapping will be employed

jose Rueda: jrrueda@us.es

Note: Written for version 0.5.3. Revised for version 0.7.2
"""
import Lib as ss
import matplotlib.pyplot as plt
from time import time
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 39612
diag_ID = 1  # 6 for rFILD
t1 = 0.1     # Initial time to be loaded, [s]
t2 = 3.2     # Final time to be loaded [s]
limitation = False  # If true, the suite will not allow to load more than
limit = 2048        # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.1     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.15     # Final time to average the frames for noise subtraction [s]

# - Remapping options:
calibration_database = ss.paths.ScintSuite \
    + '/Data/Calibrations/FILD/AUG/calibration_database.txt'
Smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/39612_1p99s_strike_map.dat'
camera = ss.dat.FILD[diag_ID-1]['camera']

save_remap = False
par = {
    'rmin': 1.2,      # Minimum gyroradius [in cm]
    'rmax': 10.5,     # Maximum gyroradius [in cm]
    'dr': 0.05,        # Interval of the gyroradius [in cm]
    'pmin': 15.0,     # Minimum pitch angle [in degrees]
    'pmax': 90.0,     # Maximum pitch angle [in degrees]
    'dp': 1.0,    # Pitch angle interval
    # Parameters for the pitch-gryroradius profiles
    'rprofmin': 1.5,     # Minimum gyroradius for the pitch profile calculation
    'rprofmax': 4.0,    # Maximum gyroradius for the pitch profile calculation
    'pprofmin': 20.0,    # Minimum pitch for the gyroradius profile calculation
    'pprofmax': 90.0,    # Maximum pitch for the gyroradius profile calculation
    # methods for the interpolation
    'method': 2,    # 2 Spline, 1 Linear (smap interpolation)
    'decimals': 1,  # Precision for the strike map (1 is more than enough)
    'remap_method': 'MC',  # Remap algorithm
    }
MC_number = 150  # number of MC markers per pixel
# - Plotting options:
plot_profiles_in_time = True   # Plot the time evolution of pitch and r
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - Get the proper file name
filename = ss.vid.guess_filename(shot, ss.dat.FILD[diag_ID-1]['path'],
                                 ss.dat.FILD[diag_ID-1]['extension'])

# - open the video file:
vid = ss.vid.FILDVideo(filename, diag_ID=diag_ID)
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
# --- Section 3: Read FILD logbook
# -----------------------------------------------------------------------------
FILD = ss.dat.FILD_logbook(shot, diag_ID)
# Add FILD positions to the remap options:
par['rfild'] = FILD.position['R']
par['zfild'] = FILD.position['z']
par['alpha'] = FILD.orientation['alpha']
par['beta'] = FILD.orientation['beta']
# -----------------------------------------------------------------------------
# --- Section 4: Load the database
# -----------------------------------------------------------------------------
# - Initialize the calibration database object
database = ss.mapping.CalibrationDatabase(calibration_database)
# - Get the calibration for our shot
cal = database.get_calibration(shot, camera, 'PIX', diag_ID)

# -----------------------------------------------------------------------------
# --- Section 5: Load and prepare the strike map
# -----------------------------------------------------------------------------
# Load the strike map
smap = ss.mapping.StrikeMap('FILD', Smap_file)
# Calculate pixel coordinates of the map
smap.calculate_pixel_coordinates(cal)
# Calculate the relation pixel - gyr and pitch
grid = {'ymin': par['rmin'], 'ymax': par['rmax'], 'dy': par['dr'],
        'xmin': par['pmin'], 'xmax': par['pmax'], 'dx': par['dp']}
smap.interp_grid(vid.exp_dat['frames'].shape[0:2], plot=False,
                 method=par['method'], MC_number=MC_number,
                 grid_params=grid)
# Include this map in the remapping parameters:
par['map'] = smap

# -----------------------------------------------------------------------------
# --- Section 6: Proceed with the remap
# -----------------------------------------------------------------------------
# - Remap frames:
vid.remap_loaded_frames(cal, shot, par)
# - Plot:
if plot_profiles_in_time:
    vid.plot_profiles_in_time()

# -----------------------------------------------------------------------------
# --- Section 7: Export data
# -----------------------------------------------------------------------------
# - Export remapped data
if save_remap:
    vid.export_remap()
