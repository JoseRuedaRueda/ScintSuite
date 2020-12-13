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
import LibPlotting as ssplt
import map_equ as meq
import time
from matplotlib.widgets import Slider
from paths_suite import p

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
t1 = 0.25        # Initial time to remap [in s]
t2 = 0.30        # Final time to remap [in s]
shot = 32312
# remapping parameters
gmin = 1.2      # Minimum gyroradius [in cm]
gmax = 10.5     # Maximum gyroradius [in cm]
deltag = 0.1    # Interval of the gyroradius [in cm]

pmin = 20.0     # Minimum pitch angle [in degrees]
pmax = 90.0     # Maximum pitch angle [in degrees]
deltap = 1.0    # Pitch angle interval
# Parameters for the pitch-gryroradius profiles
gpmin = 2.0     # Minimum gyroradius for the pitch profile calculation
gpmax = 4.7     # Maximum gyroradius for the pitch profile calculation
ppmin = 20.0    # Minimum pitch for the gyroradius profile calculation
ppmax = 30.0    # Maximum pitch for the gyroradius profile calculation
# Position of the FILD
rfild = 2.186   # 2.186 for shot 32326
zfild = 0.32
alpha = 0.0
beta = -12.0
# Plotting options
p1 = True  # Plot a window with some sliders to see the evolution of the shot
p2 = True   # Plot the evolution of the signal in gyroradius space
p3 = True   # Plot the evolution of the signal in the pitch space
p4 = True   # Plot the timetraces (see section4 for parameters)
pEner = True  # Use energy instead of gyroradius to plot in p1
FS = 16     # Font size
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
# it1 = np.array([np.argmin(abs(cin.timebase-t1))], dtype=int)
# it2 = np.array([np.argmin(abs(cin.timebase-t2))], dtype=int)
it1 = np.argmin(abs(cin.timebase-t1))
it2 = np.argmin(abs(cin.timebase-t2))
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
signal_in_gyr = np.zeros((ngyr, nframes))
signal_in_pit = np.zeros((npit, nframes))
b_field = np.zeros(nframes)
theta = np.zeros(nframes)
phi = np.zeros(nframes)

# open the magnetic field shot file
equ = meq.equ_map(shot, diag='EQH')

name_old = ' '
for iframe in range(nframes):
    # Load the magnetic field
    tframe = cin.timebase[it1 + iframe]
    br, bz, bt, bp = ssdat.get_mag_field(shot, rfild, zfild, time=tframe,
                                         equ=equ)
    b_field[iframe] = np.hypot(bt, bp)
    # Find/calculate the correct strike map
    phi[iframe], theta[iframe] = \
        ssFILDSIM.calculate_fild_orientation(br, bz, bt, alpha, beta)
    name = ssFILDSIM.find_strike_map(rfild, zfild, phi[iframe], theta[iframe],
                                     pa.StrikeMaps, pa.FILDSIM)
    print('StrikeMap name', name)
    # Only reload the strike map if it is needed
    if name != name_old:
        map = ssmap.StrikeMap(0, os.path.join(pa.StrikeMaps, name))
        map.calculate_pixel_coordinates(cal)
        # print('Interpolating grid')
        map.interp_grid(frame_shape, plot=False, method=2)
    name_old = name
    remap[:, :, iframe], pitch, gyr = ssmap.remap(map, frames[:, :, iframe],
                                                  x_min=pmin, x_max=pmax,
                                                  delta_x=deltap, y_min=gmin,
                                                  y_max=gmax, delta_y=deltag)
    dummy = remap[:, :, iframe]
    dummy = dummy.squeeze()
    signal_in_gyr[:, iframe] = ssmap.gyr_profile(dummy, pitch, ppmin, ppmax)
    signal_in_pit[:, iframe] = ssmap.pitch_profile(dummy, gyr, gpmin, gpmax)

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
# %% Section 4: time traces of selected gyroradius intervals
g_lim1 = 2.0
g_lim2 = 3.5
g_lim3 = 9.0
flags12 = (gyr > g_lim1) * (gyr < g_lim2)
flags23 = (gyr > g_lim2) * (gyr < g_lim3)
trace12 = np.sum(signal_in_gyr[flags12, :], axis=0)
trace23 = np.sum(signal_in_gyr[flags23, :], axis=0)

# -----------------------------------------------------------------------------
# %% Section 5: Plotting the results
# Interactive figure with slider to see the evolution of the shot
if p1:
    # Open the figure and initialise the colormap
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
    plt_param = {'marker': 'None'}
    map.plot_pix(ax[0, 0], plt_param=plt_param)  # Scintillator map
    cbar_frame = fig.colorbar(plot_frame, ax=ax[0, 0])
    # - Remaped
    if pEner:
        ener = ssmap.get_energy_FILD(gyr, b_field[-1]) / 1000.0
        plot_remap = ax[0, 1].contourf(pitch, ener, remaped.T, levels=20,
                                       cmap=cmap)
        cbar_remap = fig.colorbar(plot_remap, ax=ax[0, 1])
    else:
        plot_remap = ax[0, 1].contourf(pitch, gyr, remaped.T, levels=20,
                                       cmap=cmap)
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
    smin_frame = Slider(axmax_frame, 'Min [#]: ', 0.75 * min_fra,
                        1.25 * max_fra, valinit=min_fra)
    smax_frame = Slider(axmin_frame, 'Max [#]: ',  0.75 * min_fra,
                        1.25 * max_fra, valinit=max_fra)
    # - Colormap for the remap
    axmax_rep = plt.axes([0.60, 0.94, 0.30, 0.02], facecolor=axcolor)
    axmin_rep = plt.axes([0.60, 0.91, 0.30, 0.02], facecolor=axcolor)
    smin_rep = Slider(axmax_rep, 'Min [#]: ', 0.75 * min_rep, 1.25 * max_rep,
                      valinit=min_rep)
    smax_rep = Slider(axmin_rep, 'Max [#]: ',  0.75 * min_rep, 1.25 * max_rep,
                      valinit=max_rep)

    def update(val):
        """Update the sliders"""
        global ax
        global cmap
        global cbar_frame
        global cbar_remap
        global gyr
        global pitch
        global rfild
        global zfild
        global pa
        global phi
        global theta
        global pEner
        global plt_param
        ####
        # Get the time
        t = stime.val
        it = np.argmin(abs(cin.timebase-t)) - it1
        # print(t)
        # print(cin.timebase[it+it1])
        # Get the limit of the colorbars
        frame_min = smin_frame.val
        frame_max = smax_frame.val
        rep_min = smin_rep.val
        rep_max = smax_rep.val
        # - Plot the frame
        ax[0, 0].clear()
        dummy = frames[:, :, it]
        dummy = dummy.squeeze()
        plot_frame = ax[0, 0].imshow(dummy, origin='lower',
                                     cmap=cmap, vmin=frame_min, vmax=frame_max)
        cbar_frame.update_normal(plot_frame)
        # Get and plot the strike map
        name = ssFILDSIM.find_strike_map(rfild, zfild, phi[it],
                                         theta[it], pa.StrikeMaps,
                                         pa.FILDSIM)
        map = ssmap.StrikeMap(0, os.path.join(pa.StrikeMaps, name))
        map.calculate_pixel_coordinates(cal)
        map.plot_pix(ax[0, 0], plt_param=plt_param)
        # - Plot the remap
        ax[0, 1].clear()
        dummy = remap[:, :, it]
        dummy = dummy.squeeze()
        if pEner:
            ener = ssmap.get_energy_FILD(gyr, b_field[it]) / 1000.0
            plot_remap = ax[0, 1].contourf(pitch, ener, dummy.T, levels=20,
                                           cmap=cmap, vmin=rep_min,
                                           vmax=rep_max)
        else:
            plot_remap = ax[0, 1].contourf(pitch, gyr, dummy.T, levels=20,
                                           cmap=cmap, vmin=rep_min,
                                           vmax=rep_max)
        cbar_remap.update_normal(plot_remap)
        # - Update the profiles
        # Obtain a gyroradius profile
        gyr_profile = ssmap.gyr_profile(dummy, pitch, ppmin, ppmax)
        # Obtain a pitch profile
        pit_profile = ssmap.pitch_profile(dummy, gyr, gpmin, gpmax)
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
# end if p1

if p2:
    # plot contourf
    time = cin.timebase[it1:it2+1]
    cmap = ssplt.Gamma_II()
    fig_gyr, ax_gyr = plt.subplots()
    signal_gyr_norm = signal_in_gyr / np.max(signal_in_gyr)
    contourf = ax_gyr.contourf(time, gyr, signal_gyr_norm, cmap=cmap)
    # Set a nice axes
    options = {'fontsize': FS, 'ylabel': '$r_\perp$', 'xlabel': 'Time [s]'}
    ax_gyr = ssplt.axis_beauty(ax_gyr, options)
    ax_gyr.set_xticks(ax_gyr.get_xticks()[::2])
    plt.xlim(time[0], time[-1])
    # insert a colorbar
    cbar = plt.colorbar(contourf)
    cbar.set_label('Counts [a.u.]', fontsize=FS)
    cbar.ax.tick_params(labelsize=FS)
    # Write the shot number and FILD
    gyr_level = gyr[-1]-0.1*(gyr[-1] - gyr[0])  # Jut a nice position
    tpos1 = time[0] + 0.05 * (time[-1] - time[0])
    tpos2 = time[0] + 0.95 * (time[-1] - time[0])
    plt.text(tpos1, gyr_level, '#' + str(shot),
             horizontalalignment='left', fontsize=0.9*FS, color='w',
             verticalalignment='bottom')
    plt.text(tpos1, gyr_level, str(ppmin) + 'º to ' + str(ppmax) + 'º',
             horizontalalignment='left', fontsize=0.9*FS, color='w',
             verticalalignment='top')
    plt.text(tpos2, gyr_level, 'FILD' + str(diag_ID),
             horizontalalignment='right', fontsize=0.9*FS, color='w',
             verticalalignment='bottom')

if p3:
    # plot contourf
    time = cin.timebase[it1:it2+1]
    cmap = ssplt.Gamma_II()
    fig_pit, ax_pit = plt.subplots()
    signal_pit_norm = signal_in_pit / np.max(signal_in_pit)
    contourf = ax_pit.contourf(time, pitch, signal_pit_norm, cmap=cmap)
    # Set a nice axes
    options = {'fontsize': FS, 'ylabel': 'Pitch [º]', 'xlabel': 'Time [s]'}
    ax_pit = ssplt.axis_beauty(ax_pit, options)
    ax_pit.set_xticks(ax_pit.get_xticks()[::2])
    plt.xlim(time[0], time[-1])
    # insert a colorbar
    cbar = plt.colorbar(contourf)
    cbar.set_label('Counts [a.u.]', fontsize=FS)
    cbar.ax.tick_params(labelsize=FS)
    # Write the shot number and FILD
    pitch_level = pitch[-1]-0.1*(pitch[-1] - pitch[0])  # Jut a nice position
    tpos1 = time[0] + 0.05 * (time[-1] - time[0])
    tpos2 = time[0] + 0.95 * (time[-1] - time[0])
    plt.text(tpos1, pitch_level, '#' + str(shot),
             horizontalalignment='left', fontsize=0.9*FS, color='w',
             verticalalignment='bottom')
    plt.text(tpos1, pitch_level, str(gpmin) + ' to ' + str(gpmax) + ' cm',
             horizontalalignment='left', fontsize=0.9*FS, color='w',
             verticalalignment='top')
    plt.text(tpos2, pitch_level, 'FILD' + str(diag_ID),
             horizontalalignment='right', fontsize=0.9*FS, color='w',
             verticalalignment='bottom')

if p4:
    tit = '#' + str(shot) + ' ' + 'FILD' + str(diag_ID)
    time = cin.timebase[it1:it2+1]
    fig_tt, ax_tt = plt.subplots()
    label1 = str(g_lim1) + ' - ' + str(g_lim2) + ' cm'
    label2 = str(g_lim2) + ' - ' + str(g_lim3) + ' cm'
    ax_tt.plot(time, trace12, linewidth=1.5, color='r', label=label1)
    ax_tt.plot(time, trace23, linewidth=1.5, color='b', label=label2)
    plt.legend(fontsize=0.9*FS)
    options = {'fontsize': FS, 'ylabel': 'Counts', 'xlabel': 'Time [s]',
               'grid': 'both'}
    plt.title(tit)
    ssplt.axis_beauty(ax_tt, options)
    label3 = label2 + ' / ' + label1
    fig_tt_rat, ax_tt_rat = plt.subplots()
    ax_tt_rat.plot(time, 100 * trace23 / trace12, linewidth=1.5,
                   color='k', label=label3)
    plt.legend(fontsize=0.9*FS)
    options = {'fontsize': FS, 'ylabel': '[%]', 'xlabel': 'Time [s]',
               'grid': 'both'}
    plt.title(tit)
    ssplt.axis_beauty(ax_tt_rat, options)
