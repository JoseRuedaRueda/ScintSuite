"""
Load a video and perform the remaping for each frame

Created as an example to use the routines without the graphical user interface

DISCLAIMER: This was created on the 11/11/2020. Since them several
improvements may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jose.rueda@ipp.mpg.de) by email

You should run paths.py before!!!

"""

# General python packages
import os
import numpy as np
import matplotlib.pyplot as plt
import LibPlotting as ssplt
import LibVideoFiles as ssvid
import LibMap as ssmap
import LibDataAUG as ssdat
import LibFILDSIM as ssFILDSIM
import map_equ as meq
import time
from matplotlib.widgets import Slider
from paths import p

# ------------------------------------------------------------------------------
# Section 0: Settings
# Paths:
calibration_database = './Calibrations/FILD/calibration_database.txt'
pa = p()  # 'Standard paths' of the suite
# calibration data
camera = 'PHANTOM'
cal_type = 'PIX'
diag_ID = 1     # FILD Number
# Shot a time interval
t1 = 0.3        # Initial time to remap [in s]
t2 = 0.52        # Final time to remap [in s]
shot = 32326
# remapping parameters
gmin = 1.2      # Minimum gyroradius [in cm]
gmax = 10.5     # Maximum gyroradius [in cm]
deltag = 0.1    # Interval of the gyroradius [in cm]

pmin = 20.0     # Minimum pitch angle [in degrees]
pmax = 90.0     # Maximum pitch angle [in degrees]
deltap = 1.0    # Pitch angle interval
# Parameters for the pitch-gryroradius profiles
gpmin = 4.0     # Minimum gyroradius for the pitch profile calculation
gpmax = 9.0     # Maximum gyroradius for the pitch profile calculation
ppmin = 40.0    # Minimum pitch for the gyroradius profile calculation
ppmax = 70.0    # Maximum pitch for the gyroradius profile calculation
# Position of the FILD
rfild = 2.0
zfild = 0.30
alpha = 0.0
beta = -12.0


# -----------------------------------------------------------------------------

# %% Section 1: Load calibration
database = ssmap.CalibrationDatabase(calibration_database)
cal = database.get_calibration(shot, camera, cal_type, diag_ID)
# -----------------------------------------------------------------------------

# %% Section 2: Load video file and the necesary frames
# Prepare the name of the cin file to be loaded
dummy = str(shot)
file = pa.CinFiles + dummy[0:2] + '/' + dummy + '_v710.cin'
cin = ssvid.Video(file)
it1 = np.array([np.argmin(abs(cin.timebase-t1))])
it2 = np.array([np.argmin(abs(cin.timebase-t2))])
frames = cin.read_frame(np.arange(start=it1, stop=it2+1, step=1))
frame_shape = frames[:, :, 1].shape
nframes = frames.shape[2]
# If the last line fails because you tried to load just one time point... you
# are using the wrong Example file... plase read the name
# ------------------------------------------------------------------------------

# %% Section 3: remap the shot

# Initialise the matix to save the results. I allways get confused with the
# fact that an array ini:end:delta x has end-ini/delta or end-ini/delta+1
# elements... It is late and I am hungry, this is just and example so YOLO, I
# will create the arrays and take: length
tic = time.time()
gyrdum = np.arange(start=gmin, stop=gmax, step=deltag)
pitdum = np.arange(start=pmin, stop=pmax, step=deltap)

ngyr = len(gyrdum) - 1
npit = len(pitdum) - 1

remap = np.zeros((npit, ngyr, nframes))

# open the magnetic field shot file
equ = meq.equ_map(shot, diag='EQH')

name_old = ' '
for iframe in range(nframes):
    # Load the magnetic field
    tframe = cin.timebase[it1 + iframe]
    br, bz, bt, bp = ssdat.get_mag_field(shot, rfild, zfild, time=tframe,
                                         equ=equ)
    # Find/calculate the correct strike map
    phi, theta = ssFILDSIM.calculate_fild_orientation(br, bz, bt, alpha, beta)
    name = ssFILDSIM.find_strike_map(rfild, zfild,
                                     phi, theta, pa.StrikeMaps, pa.FILDSIM)
    print('StrikeMap name', name)
    # Only reload the strike map if it is needed
    if name != name_old:
        map = ssmap.StrikeMap(0, os.path.join(pa.StrikeMaps, name))
        map.calculate_pixel_coordinates(cal)
        print('Interpolating grid')
        map.interp_grid(frame_shape, plot=False, method=2)
    name_old = name
    remap[:, :, iframe], pitch, gyr = ssmap.remap(map, frames[:, :, iframe],
                                                  x_min=pmin, x_max=pmax,
                                                  delta_x=deltap, y_min=gmin,
                                                  y_max=gmax, delta_y=deltag)
toc = time.time()
print('Whole time interval remaped in: ', toc-tic, ' s')
# -----------------------------------------------------------------------------
# Gyroradius / pitch profiles
# Obtain the profiles for the last time point
remaped = remap[:, :, -1]
remaped = remaped.squeeze()
# Obtain a gyroradius profile
gyr_profile = ssmap.gyr_profile(remaped, pitch, ppmin, ppmax, verbose=True)

# Obtain a pitch profile
pit_profile = ssmap.pitch_profile(remaped, gyr, gpmin, gpmax, verbose=True)


# -----------------------------------------------------------------------------
# Plotting the results
# --- Open the figure and initialise the colormap
fig, ax = plt.subplots(2, 2)
plt.subplots_adjust(left=0.25, bottom=0.25)
cmap = ssplt.Gamma_II()
# Get the extreme for the interval to be plotted
max_rep = np.max(remap)
min_rep = np.min(remap)
max_fra = np.max(frames)
min_fra = np.min(frames)
# --- Initialise the four plots
# - Camera
plot_frame = ax[0, 0].imshow(frames[:, :, -1], origin='lower', cmap=cmap)
# map.plot_pix(ax[0, 0])  # Scintillator map
cbar_frame = fig.colorbar(plot_frame, ax=ax[0, 0])
# - Remaped
plot_remap = ax[0, 1].contourf(pitch, gyr, remaped.T, levels=20, cmap=cmap)
cbar_remap = fig.colorbar(plot_remap, ax=ax[0, 1])
# - pitch
plot_pitch = ax[1, 0].plot(pitch, pit_profile)
# - gyr
plot_gyr = ax[1, 1].plot(gyr, gyr_profile)

# --- Set the slider
axcolor = 'k'
# - Time slider
dt = cin.timebase[1] - cin.timebase[0]
axtime = plt.axes([0.15, 0.97, 0.75, 0.02], facecolor=axcolor)
stime = Slider(axtime, 'Time [s]: ', t1, t2, valinit=t2, valstep=dt,
               valfmt='%.3f')
# - Colormap for the frame
axmax_frame = plt.axes([0.15, 0.94, 0.30, 0.02], facecolor=axcolor)
axmin_frame = plt.axes([0.15, 0.91, 0.30, 0.02], facecolor=axcolor)
smin_frame = Slider(axmax_frame, 'Min [#]: ', 0.75 * min_fra, 1.25 * max_fra,
                    valinit=min_fra)
smax_frame = Slider(axmin_frame, 'Max [#]: ',  0.75 * min_fra, 1.25 * max_fra,
                    valinit=max_fra)
# - Colormap for the remap
axmax_rep = plt.axes([0.60, 0.94, 0.30, 0.02], facecolor=axcolor)
axmin_rep = plt.axes([0.60, 0.91, 0.30, 0.02], facecolor=axcolor)
smin_rep = Slider(axmax_rep, 'Min [#]: ', 0.75 * min_rep, 1.25 * max_rep,
                  valinit=min_rep)
smax_rep = Slider(axmin_rep, 'Max [#]: ',  0.75 * min_rep, 1.25 * max_rep,
                  valinit=max_rep)
# # - Giroradius
# axgyr = plt.axes([0.15, 0.03, 0.35, 0.02], facecolor=axcolor)
# sgyr = Slider(axgyr, 'Max [#]: ',  0.75 * mmin, 1.25 * mmax, valinit=mmax)


def update(val):
    """Update the sliders"""
    global ax
    global cmap
    global cbar_frame
    global cbar_remap
    global gyr
    global pitch
    ####
    # Get the time
    t = stime.val
    it = np.array([np.argmin(abs(cin.timebase-t))])

    # Get the limit of the colorbars
    frame_min = smin_frame.val
    frame_max = smax_frame.val
    rep_min = smin_rep.val
    rep_max = smax_rep.val
    # - Plot the frame
    ax[0, 0].clear()
    dummy = frames[:, :, it-it1]
    dummy = dummy.squeeze()
    plot_frame = ax[0, 0].imshow(dummy, origin='lower',
                                 cmap=cmap, vmin=frame_min, vmax=frame_max)
    cbar_frame.update_normal(plot_frame)
    # - Plot the remap
    ax[0, 1].clear()
    dummy = remap[:, :, it - it1]
    dummy = dummy.squeeze()
    plot_remap = ax[0, 1].contourf(pitch, gyr, dummy.T, levels=20,
                                   cmap=cmap, vmin=rep_min, vmax=rep_max)
    cbar_remap.update_normal(plot_remap)
    # - Update the profiles
    # Obtain a gyroradius profile
    gyr_profile = ssmap.gyr_profile(dummy, pitch, ppmin, ppmax, verbose=False)
    # Obtain a pitch profile
    pit_profile = ssmap.pitch_profile(dummy, gyr, gpmin, gpmax, verbose=False)
    # Plot the profiles
    ax[1, 0].clear()
    ax[1, 0].plot(pitch, pit_profile)
    ax[1, 1].clear()
    ax[1, 1].plot(gyr, gyr_profile)

    fig.canvas.draw_idle()


stime.on_changed(update)
smin_frame.on_changed(update)
smax_frame.on_changed(update)
smin_rep.on_changed(update)
smax_rep.on_changed(update)
plt.show()
