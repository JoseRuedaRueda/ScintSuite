"""
Obtain the calibration parameters

Jose Rueda: jrrueda@us.es

Note, a small GUI will come out and you will have some sliders to determine the
calibration parameters for your scintillator. To help with the situation, a
remap will be done on the fly (so the slider moves a bit slow!)

Lines marked with #### should be change for it to work in your instalation

IMPORTANT: If you select SINPA as code format, the scintillaor coordinate must
be in the scintillator reference system, if not, there would be a shift between
SINPA coordinates and the calibration
"""
import math
import Lib as ss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Lib._Video._TIFfiles import load_tiff
plt.ion()

# -----------------------------------------------------------------------------
# %% Settings
# -----------------------------------------------------------------------------
# File with the scintillator
Scint_file = \
    '/afs/ipp/home/r/ruejo/ScintSuite/Data/Plates/INPA/AUG/iAUG01/Scintillator.pl'
SmapFile = None   # If none, the one from the library will be used
FoilFile = \
    '/afs/ipp/home/r/ruejo/ScintSuite/Data/Plates/INPA/AUG/iAUG01/Foil.pl'
#    # 'test_attenuation/results/test_attenuation.map'
format = 'FILDSIM'  # Code for which the geometry file is written
# File with the calibration image (tif)
# calib_image = '/afs/ipp/home/r/ruejo/INPA_Calibration_images/' \
#     + '3_PIXEFLY_OBJETIVE_comissioning_2022_02.bmp'                      # ####
# Video options
diag_ID = 1  # INPA number
shot = 41090  # Shot number
tMinNoise = 0.1  # Min time for noise subtraction
tMaxNoise = 0.4  # Max time for noise subtraction
t = 4.02  # Times to overplot a frame from the video
tmax = 4.8  # Maximum time to load
# x-scale to y scale
XtoY = 1.0
# modify section 3 if you have a custom format for the calibration image
# Staring points for the calibration
xshift = 48.77
yshift = 141.0
xscale = 2632.22
xc = 292.2
yc = 238.1
c = -1.77555e-3
deg = 1.5


# Scale maximum
xshiftmax = 50
yshiftmax = 200
xscalemax = 2800
xcmax = 400
ycmax = 300
cmax = -0.5e-4
degmax = 5.0

# Scale minimum
xshiftmin = 0
yshiftmin = 20
xscalemin = 2400
xcmin = 150
ycmin = 100
cmin = -20e-4
degmin = -5.0

# Remapping options
par = {
    'ymin': 10.0,      # Minimum energy [in keV]
    'ymax': 100.0,     # Maximum energy [in keV]
    'dy': 2.0,         # Interval of the energy [in keV]
    'xmin': 1.4,       # Minimum pitch angle [in m]
    'xmax': 2.2,       # Maximum pitch angle [in m]
    'dx': 0.01,        # Radial interval
    # methods for the interpolation
    'method': 2,    # 2 Spline, 1 Linear (smap interpolation)
    'decimals': 0,  # Precision for the strike map (1 is more than enough)
    'remap_method': 'centers',  # Remap algorithm
    'MC_number': 0,
    }
geomID = 'iAUG01'
# -----------------------------------------------------------------------------
# %% Scintillator load and first alignement
# -----------------------------------------------------------------------------
scintillator = ss.mapping.Scintillator(Scint_file, format)
foil = ss.mapping.Scintillator(FoilFile, format)

cal = ss.mapping.CalParams()
cal.xshift = xshift
cal.yshift = yshift
cal.xscale = xscale
cal.yscale = xscale * XtoY
cal.deg = deg
scintillator.calculate_pixel_coordinates(cal)
foil.calculate_pixel_coordinates(cal)


# -----------------------------------------------------------------------------
# %% Image load and plot
# -----------------------------------------------------------------------------
# if calib_image.endswith('tif'):
#     img = load_tiff(calib_image)
# elif calib_image.endswith('png'):
#     raise Exception('To be implemented')
# else:
#     img = plt.imread(calib_image)
#     img = img[::-1, :]
vid = ss.vid.INPAVideo(shot=shot, diag_ID=diag_ID)
vid.read_frame(t1=t-0.5, t2=tmax)
fig, ax = plt.subplots()
# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.4, bottom=0.5)
# ax.imshow(img, origin='lower', cmap=plt.get_cmap('gray'))
ss.plt.axis_beauty(ax, {'grid': 'both'})
# Plot the frame
vid.plot_frame(ax=ax, alpha=0.75, t=t, IncludeColorbar=False)

# -----------------------------------------------------------------------------
# %% Strike map loading
# -----------------------------------------------------------------------------
if SmapFile is not None:
    smap = ss.smap.Ismap(file=SmapFile)
    smap.calculate_pixel_coordinates(cal)
else:
    phi, theta = vid.calculateBangles(t, verbose=False)
    if phi < 0:
        phi += 360.0
    # phi = np.round(phi.values)
    # theta = np.round(theta.values)
    phi = round(phi)
    theta = round(theta)
    smap = ss.smap.Ismap(theta=theta, phi=phi)
    smap.calculate_pixel_coordinates(cal)
# -----------------------------------------------------------------------------
# %% Plot the scintillator plate
# -----------------------------------------------------------------------------
scintillator.plot_pix(ax)
smap.plot_pix(ax, labels=False)
foil.plot_pix(ax)
# -----------------------------------------------------------------------------
# %% Remap te frame and plot the profiles
# -----------------------------------------------------------------------------
# Get the mangnetic field to translate the map into energy
Geom = ss.simcom.geometry.Geometry(geomID)
rPinXYZ = Geom.ExtraGeometryParams['rPin']
Rpin = math.sqrt(rPinXYZ[0]**2 + rPinXYZ[1]**2)
br, bz, bt, bp = \
    ss.dat.get_mag_field(shot, Rpin, rPinXYZ[2], time=t)
Bmod = np.sqrt(bt**2 + bp**2)[0]
# now yes, calculate the energy
smap.calculate_energy(Bmod)
smap.setRemapVariables(('R0', 'e0'))
smap.calculate_pixel_coordinates(vid.CameraCalibration)
par['map'] = smap
vid.remap_loaded_frames(par)
# - Calculate the profiles
R_profile = vid.remap_dat['frames'].sum(dim='y')
energy_profile = vid.remap_dat['frames'].sum(dim='x')
flags_R = vid.remap_dat['x'].values > 1.87
flags_E = vid.remap_dat['y'].values > 62.0
dummy = vid.remap_dat['frames'].sel(t=t, method='nearest')
# - Axis for the profiles.,
axR0 = plt.axes([0.5, 0.12, 0.25, 0.25])  # Total energy distribution
profile_high_e = np.sum(dummy.values[:, flags_E], axis=(1))
profile_high_R = np.sum(dummy.values[flags_R, :], axis=(0))
axR0.plot(R_profile['x'], R_profile.sel(t=t, method='nearest'))
axR0.plot(R_profile['x'], profile_high_e)
axEs = plt.axes([0.75, 0.12, 0.25, 0.25])
axEs.plot(energy_profile['y'], energy_profile.sel(t=t, method='nearest'))
axEs.plot(energy_profile['y'], profile_high_R)
# -----------------------------------------------------------------------------
# %% GUI
# -----------------------------------------------------------------------------
# Make a vertically oriented slider to control the undistorted calibration
axxs = plt.axes([0.05, 0.25, 0.01, 0.63])
axxs_slider = Slider(
    ax=axxs,
    label='xshift',
    valmin=xshiftmin,
    valmax=xshiftmax,
    valinit=xshift,
    orientation="vertical"
)
axys = plt.axes([0.12, 0.25, 0.01, 0.63])
axys_slider = Slider(
    ax=axys,
    label='yshift',
    valmin=yshiftmin,
    valmax=yshiftmax,
    valinit=yshift,
    orientation="vertical"
)

axxsc = plt.axes([0.19, 0.25, 0.01, 0.63])
axxsc_slider = Slider(
    ax=axxsc,
    label="xscale",
    valmin=xscalemin,
    valmax=xscalemax,
    valinit=xscale,
    orientation="vertical"
)
axdeg = plt.axes([0.26, 0.25, 0.01, 0.63])
axdeg_slider = Slider(
    ax=axdeg,
    label="deg",
    valmin=degmin,
    valmax=degmax,
    valinit=deg,
    orientation="vertical"
)
# Make a horizontal oriented slider to control the distortion
axxc = plt.axes([0.05, 0.05, 0.25, 0.01])
axxc_slider = Slider(
    ax=axxc,
    label='xc',
    valmin=xcmin,
    valmax=xcmax,
    valinit=xc,
)
axyc = plt.axes([0.05, 0.12, 0.25, 0.01])
axyc_slider = Slider(
    ax=axyc,
    label='yc',
    valmin=ycmin,
    valmax=ycmax,
    valinit=yc,
)
axc = plt.axes([0.60, 0.05, 0.25, 0.01])
axc_slider = Slider(
    ax=axc,
    label='c1',
    valmin=cmin,
    valmax=cmax,
    valinit=c,
    valfmt='%1.2E'
)
# The function to be called anytime a slider's value changes


def update(val):
    """Read sliders and update the plot"""
    cal.xshift = axxs_slider.val
    cal.yshift = axys_slider.val
    cal.xscale = axxsc_slider.val
    cal.yscale = XtoY * axxsc_slider.val
    cal.deg = axdeg_slider.val
    cal.xcenter = axxc_slider.val
    cal.ycenter = axyc_slider.val
    cal.c1 = axc_slider.val
    scintillator.calculate_pixel_coordinates(cal)
    foil.calculate_pixel_coordinates(cal)
    smap.calculate_pixel_coordinates(cal)
    ss.plt.remove_lines(ax)
    ax.plot([axxc_slider.val], [axyc_slider.val], '.r')

    scintillator.plot_pix(ax)
    foil.plot_pix(ax)
    smap.plot_pix(ax, labels=False)
    # -- remap again
    par.pop('map')
    smap.calculate_pixel_coordinates(cal)
    smap.interp_grid(vid.exp_dat['frames'].shape[:2], method=par['method'],
                     MC_number=par['MC_number'],
                     grid_params={'ymin': par['ymin'],
                                  'ymax': par['ymax'],
                                  'dy': par['dy'],
                                  'xmin': par['xmin'],
                                  'xmax': par['xmax'],
                                  'dx': par['dx']})
    par['map'] = smap
    vid.remap_loaded_frames(par)
    # - Calculate the profiles
    R_profile = vid.remap_dat['frames'].sum(dim='y')
    energy_profile = vid.remap_dat['frames'].sum(dim='x')
    # - Re-plot the profiles
    ss.plt.remove_lines(axR0)
    ss.plt.remove_lines(axEs)
    axR0.plot(R_profile['x'], R_profile.sel(t=t, method='nearest'))
    axEs.plot(energy_profile['y'], energy_profile.sel(t=t, method='nearest'))
    dummy = vid.remap_dat['frames'].sel(t=t, method='nearest')
    profile_high_e = np.sum(dummy.values[:, flags_E], axis=(1))
    profile_high_R = np.sum(dummy.values[flags_R, :], axis=(0))
    axR0.plot(R_profile['x'], profile_high_e)
    axEs.plot(energy_profile['y'], profile_high_R)
    plt.draw_all()


# register the update function with each slider
axxs_slider.on_changed(update)
axys_slider.on_changed(update)
axxsc_slider.on_changed(update)
axdeg_slider.on_changed(update)
axc_slider.on_changed(update)
axxc_slider.on_changed(update)
axyc_slider.on_changed(update)
plt.show()
