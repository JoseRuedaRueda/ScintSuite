"""
Load a frame, load the calibration and remap the scintillator signal

Created as an example to use the routines without the graphical user interface.
MonteCarlo Remap or 'standard' centroid remap will be compared

The main recomendation is to use the vide object, load just one time point and
use the remap routine of the video object, which automatically load for the
appropiate smap and so on. But this is just a pedestrian way to perform the
remap directly calling to the remap function, just in case one wants to check
something

Created for version 0.5.0
"""

# General python packages
import numpy as np
import matplotlib.pyplot as plt
import Lib as ss


# -----------------------------------------------------------------------------
# Section 0: Settings
# -----------------------------------------------------------------------------
# -- Paths (change them to your own paths)
cin_file_name = '/p/IPP/AUG/rawfiles/FIT/32/32312_v710.cin'
calibration_database = ss.paths.ScintSuite \
    + '/Data/Calibrations/FILD/calibration_database.txt'
strike_map = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILDSIM/results/' +\
    'AUG_map_-000.60000_007.50000_strike_map.dat'
# -- Discharge settings
shot = 32312
t0 = 0.29
# -- FILD settings
camera = 'PHANTOM'
cal_type = 'PIX'
diag_ID = 1     # FILD Number
# -- Remap settings:
par = {
    'rmin': 1.2,      # Minimum gyroradius [in cm]
    'rmax': 10.5,     # Maximum gyroradius [in cm]
    'dr': 0.05,        # Interval of the gyroradius [in cm]
    'pmin': 20.0,     # Minimum pitch angle [in degrees]
    'pmax': 90.0,     # Maximum pitch angle [in degrees]
    'dp': 1.0,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
}
MC_markers = 300  # Number of MC markers to use. 0 Deactivate the MC remap
# -- Noise subtraction
tn1 = 0.15
tn2 = 0.19
# -- Plot settings
p1 = True  # Plot the original frame
p2 = True  # Plot comparison between both methods
# -----------------------------------------------------------------------------
# --- Section 1: Load calibration
# -----------------------------------------------------------------------------
database = ss.mapping.CalibrationDatabase(calibration_database)
cal = database.get_calibration(shot, camera, cal_type, diag_ID)
# -----------------------------------------------------------------------------
# --- Section 2: Load the frame
# -----------------------------------------------------------------------------

# Open the video object
cin = ss.vid.Video(cin_file_name)
# Load the frames for the noise calculation
cin.read_frame(t1=tn1, t2=tn2)
# get the noise frame
noise_frame = cin.subtract_noise(t1=tn1, t2=tn2, return_noise=True)
# Extract the frame. Notice, you could also use the read_frame method with the
# flag: read_from_loaded, if the desired time in the loaded window. In this
# case the readed frame will already have the noise subtracted!!!
dummy = np.array([np.argmin(abs(cin.timebase-t0))])
ref_frame = cin.read_frame(dummy, internal=False)
ref_frame = ref_frame.astype(np.float) - noise_frame
frame_shape = ref_frame.shape
if p1:
    fig_ref, ax_ref = plt.subplots()
    ax_ref.imshow(ref_frame, origin='lower')
    fig_ref.show()
# -----------------------------------------------------------------------------
# --- Section 3: Load and remap strike map
# -----------------------------------------------------------------------------
smap = ss.mapping.StrikeMap(0, strike_map)
smap.calculate_pixel_coordinates(cal)
grid_params = {   # parameters for the montacarlo inversion
    'ymin': par['rmin'],
    'ymax': par['rmax'],
    'dy': par['dr'],
    'xmin': par['pmin'],
    'xmax': par['pmax'],
    'dx': par['dp']
}

smap.interp_grid(ref_frame.shape, plot=False, method=2, MC_number=MC_markers,
                 grid_params=grid_params)
if p1:
    smap.plot_pix(ax_ref)
    fig_ref.show()
# -----------------------------------------------------------------------------
# --- Section 3: Remapping
# -----------------------------------------------------------------------------
# With the new version, we need to prepare the gyr and pitch profiles outside
ngyr = int((par['rmax']-par['rmin'])/par['dr']) + 1
npit = int((par['pmax']-par['pmin'])/par['dp']) + 1
p_edges = par['pmin'] - par['dp']/2 + np.arange(npit + 1) * par['dp']
g_edges = par['rmin'] - par['dr']/2 + np.arange(ngyr + 1) * par['dr']
gyr = 0.5 * (g_edges[0:-1] + g_edges[1:])
pitch = 0.5 * (p_edges[0:-1] + p_edges[1:])

# Perform the remapping with the default options
MC = {'remap': ss.mapping.remap(smap, ref_frame, method='MC')}
centroid = {'remap': ss.mapping.remap(smap, ref_frame, x_edges=p_edges,
                                      y_edges=g_edges, method='centers')}

# Sum the remap to obtain the profiles in pitch or gyroradius
MC['gyr_profile'] = np.sum(MC['remap'], axis=0) * par['dp']
MC['pitch_profile'] = np.sum(MC['remap'], axis=1) * par['dr']
centroid['gyr_profile'] = np.sum(centroid['remap'], axis=0) * par['dp']
centroid['pitch_profile'] = np.sum(centroid['remap'], axis=1) * par['dr']

# Plot the remapped frame
if p2:
    cmap = ss.plt.Gamma_II()
    fig, ax = plt.subplots(2, 2)
    # MC remap 2D
    remap = ax[0, 0].imshow(MC['remap'].T, cmap=cmap, origin='lower',
                            extent=[par['pmin'], par['pmax'],
                                    par['rmin'], par['rmax']],
                            aspect='auto')
    ax[0, 0].set_xlabel('Pitch [$\\degree$]')
    ax[0, 0].set_ylabel('Gyroradius [cm]')
    # centroid remap 2d
    remap2 = ax[0, 1].imshow(centroid['remap'].T, cmap=cmap, origin='lower',
                             extent=[par['pmin'], par['pmax'],
                                     par['rmin'], par['rmax']],
                             aspect='auto')
    ax[0, 1].set_xlabel('Pitch [$\\degree$]')
    ax[0, 1].set_ylabel('Gyroradius [cm]')
    # Gyroradius profiles
    ax[1, 0].plot(gyr, MC['gyr_profile'], label='MC')
    ax[1, 0].plot(gyr, centroid['gyr_profile'], label='Centroid')
    ax[1, 0].set_xlabel('Gyroradius [cm]')
    ax[1, 0].set_ylabel('Signal per unit of radius')
    # Pitch profiles
    ax[1, 1].plot(pitch, MC['pitch_profile'], label='MC')
    ax[1, 1].plot(pitch, centroid['pitch_profile'], label='Centroid')
    ax[1, 1].set_xlabel('Pitch [$\\degree$]')
    ax[1, 1].set_ylabel('Signal per unit of pitch')
    fig.show()
