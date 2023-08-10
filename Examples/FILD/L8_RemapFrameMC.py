"""
Load a frame, load the calibration and remap the scintillator signal

Created as an example to use the routines without the graphical user interface.
MonteCarlo Remap or 'standard' centroid remap will be compared

The main recomendation is to use the vide object, load just one time point and
use the remap routine of the video object, which automatically load for the
appropiate smap and so on. But this is just a pedestrian way to perform the
remap directly calling to the remap function, just in case one wants to check
something

Created for version 0.5.0. Revised for version 0.8.0
"""

# General python packages
import numpy as np
import matplotlib.pyplot as plt
import ScintSuite.as ss


# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 44732
diag_ID = 1  # 6 for rFILD
tn1 = 0.1     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.35     # Final time to average the frames for noise subtraction [s]
t0 = 0.24
# - Remapping options:
# Smap_file = '/afs/ipp/home/r/ruejo/FILDSIM/results/39612_1p99s_strike_map.dat'  #!#
strike_map = '/home/jqw5960/SINPA/runs/ajustefino/results/ajustefino.map'
# -- Remap settings:
par = {
    'ymin': 1.2,      # Minimum gyroradius [in cm]
    'ymax': 16.5,     # Maximum gyroradius [in cm]
    'dy': 0.5,        # Interval of the gyroradius [in cm]
    'xmin': 20.0,     # Minimum pitch angle [in degrees]
    'xmax': 90.0,     # Maximum pitch angle [in degrees]
    'dx': 2.0,    # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 1}  # Precision for the strike map (1 is more than enough)

MC_markers = 300  # Number of MC markers to use. 0 Deactivate the MC remap

# -- Plot settings
p1 = True  # Plot the original frame
p2 = True  # Plot comparison between both methods
# -----------------------------------------------------------------------------
# --- Section 2: Load the frame
# -----------------------------------------------------------------------------
# Open the video object
cin = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID)
# Load the frames for the noise calculation
cin.read_frame(t1=tn1, t2=tn2)
# get the noise frame
noise_frame = cin.subtract_noise(t1=tn1, t2=tn2)
# Extract the frame. Notice, you could also use the read_frame method with the
# flag: read_from_loaded, if the desired time in the loaded window. In this
# case the readed frame will already have the noise subtracted!!!
dummy = np.array([np.argmin(abs(cin.timebase-t0))])
ref_frame = cin.read_frame(dummy, internal=False).squeeze()
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
smap.calculate_pixel_coordinates(cin.CameraCalibration)
grid_params = {   # parameters for the montacarlo inversion
    'ymin': par['ymin'],
    'ymax': par['ymax'],
    'dy': par['dy'],
    'xmin': par['xmin'],
    'xmax': par['xmax'],
    'dx': par['dx']
}

smap.interp_grid(ref_frame.shape, method=2, MC_number=MC_markers,
                 grid_params=grid_params)
if p1:
    smap.plot_pix(ax_ref)
    fig_ref.show()
# -----------------------------------------------------------------------------
# --- Section 4: Remapping
# -----------------------------------------------------------------------------
# We need to prepare the gyr and pitch profiles outside, this calculation is
# done outside to avoid losing time in the remaping of the whole shot
ngyr = int((par['ymax']-par['ymin'])/par['dy']) + 1
npit = int((par['xmax']-par['xmin'])/par['dx']) + 1
p_edges = par['xmin'] - par['dx']/2 + np.arange(npit + 1) * par['dx']
g_edges = par['ymin'] - par['dy']/2 + np.arange(ngyr + 1) * par['dy']
gyr = 0.5 * (g_edges[0:-1] + g_edges[1:])
pitch = 0.5 * (p_edges[0:-1] + p_edges[1:])

# Perform the remapping with the default options
MC = {'remap': ss.mapping.remap(smap, ref_frame, method='MC')}
centroid = {'remap': ss.mapping.remap(smap, ref_frame, x_edges=p_edges,
                                      y_edges=g_edges, method='centers')}

# Sum the remap to obtain the profiles in pitch or gyroradius
MC['gyr_profile'] = np.sum(MC['remap'], axis=0) * par['dx']
MC['pitch_profile'] = np.sum(MC['remap'], axis=1) * par['dy']
centroid['gyr_profile'] = np.sum(centroid['remap'], axis=0) * par['dx']
centroid['pitch_profile'] = np.sum(centroid['remap'], axis=1) * par['dy']

# Plot the remapped frame
if p2:
    cmap = ss.plt.Gamma_II()
    fig, ax = plt.subplots(2, 2)
    # MC remap 2D
    remap = ax[0, 0].imshow(MC['remap'].T, cmap=cmap, origin='lower',
                            extent=[par['xmin'], par['xmax'],
                                    par['ymin'], par['ymax']],
                            aspect='auto')
    ax[0, 0].set_xlabel('Pitch [$\\degree$]')
    ax[0, 0].set_ylabel('Gyroradius [cm]')
    # centroid remap 2d
    remap2 = ax[0, 1].imshow(centroid['remap'].T, cmap=cmap, origin='lower',
                             extent=[par['xmin'], par['xmax'],
                                     par['ymin'], par['ymax']],
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
