"""
Obtain the calibration parameters

Jose Rueda: jrrueda@us.es

Note, a small GUI will come out and you will have some sliders to determine the
calibration parameters for your scintillator

Lines marked with #### should be change for it to work in your instalation

IMPORTANT: If you select SINPA as code format, the scintillaor coordinate must
be in the scintillator reference system, if not, there would be a shift between
SINPA coordinates and the calibration
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ScintSuite._Video._TIFfiles import load_tiff
import ScintSuite as ss

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
# File with the scintillator
Scint_file = \
    '/afs/ipp/home/r/ruejo/ScintSuite/Data/Plates/INPA/AUG/iAUG01/Scintillator.pl'
FoilFile = \
    '/afs/ipp/home/r/ruejo/ScintSuite/Data/Plates/INPA/AUG/iAUG01/Foil.pl'
#    # 'test_attenuation/results/test_attenuation.map'
format = 'FILDSIM'  # Code for which the geometry file is written
# File with the calibration image (tif)
calib_image = '/afs/ipp/home/r/ruejo/INPA_Calibration_images/' \
    + '3_PIXEFLY_OBJETIVE_comissioning_2022_02.bmp'                      # ####
# x-scale to y scale
XtoY = 1.0
# modify section 3 if you have a custom format for the calibration image
# Staring points for the calibration
xshift = 16.90
yshift = 73.19
xscale = 2615.41
xc = 270
yc = 170
c = -1.825e-3
deg = 0.0

# Scale maximum
xshiftmax = 1500
yshiftmax = 100
xscalemax = 5000
xcmax = 1280
ycmax = 800
cmax = -0.5e-4
degmax = 10

# Scale minimum
xshiftmin = -1200
yshiftmin = 50
xscalemin = -5000
xcmin = 0
ycmin = 0
cmin = -20e-4
degmin = -10

# -----------------------------------------------------------------------------
# --- Scintillator load and first alignement
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
# --- Image load and plot
# -----------------------------------------------------------------------------
if calib_image.endswith('tif'):
    img = load_tiff(calib_image)
elif calib_image.endswith('png'):
    raise Exception('To be implemented')
else:
    img = plt.imread(calib_image)
    img = img[::-1, :]
fig, ax = plt.subplots()
# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.5, bottom=0.4)
ax.imshow(img, origin='lower', cmap=plt.get_cmap('gray'))
ss.plt.axis_beauty(ax, {'grid': 'both'})
# -----------------------------------------------------------------------------
# --- Plot the scintillator plate
# -----------------------------------------------------------------------------
scintillator.plot_pix(ax, line_params={'color': 'g'})
foil.plot_pix(ax)
# -----------------------------------------------------------------------------
# --- GUI
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

axxsc = plt.axes([0.21, 0.25, 0.01, 0.63])
axxsc_slider = Slider(
    ax=axxsc,
    label="xscale",
    valmin=xscalemin,
    valmax=xscalemax,
    valinit=xscale,
    orientation="vertical"
)
axdeg = plt.axes([0.40, 0.25, 0.01, 0.63])
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
    ss.plt.remove_lines(ax)
    ax.plot([axxc_slider.val], [axyc_slider.val], '.r')

    scintillator.plot_pix(ax, line_params={'color': 'g'})
    foil.plot_pix(ax)
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
