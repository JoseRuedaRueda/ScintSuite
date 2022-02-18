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
import Lib as ss

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
# File with the scintillator
Scint_file = '/afs/ipp/home/r/ruejo/SINPA/Geometry/AUG02/Element2.txt'   # ####
format = 'SINPA'  # Code for which the geometry file is written
# File with the calibration image (png)
calib_image = '/afs/ipp/home/r/ruejo/FILD_Calibration_images/FILD1/' + \
    'FILD_reference_800x600_2021_07_05_inserted.png'                     # ####
# modify section 3 if you have a custom format for the calibration image
# Staring points for the calibration
xshift = 427
yshift = 278
xscale = -5100
yscale = 5100
deg = 20.0

# Scale maximum
xshiftmax = 1200
yshiftmax = 800
xscalemax = 10000
yscalemax = 10000
degmax = 45

# Scale minimum
xshiftmin = -1200
yshiftmin = -800
xscalemin = -10000
yscalemin = -10000
degmin = -45

# -----------------------------------------------------------------------------
# --- Scintillator load and first alignement
# -----------------------------------------------------------------------------
scintillator = ss.mapping.Scintillator(Scint_file, format)
cal = ss.mapping.CalParams()
cal.xshift = xshift
cal.yshift = yshift
cal.xscale = xscale
cal.yscale = yscale
cal.deg = deg
scintillator.calculate_pixel_coordinates(cal)

# -----------------------------------------------------------------------------
# --- Image load and plot
# -----------------------------------------------------------------------------
img = ss.vid.PNGfiles.load_png(calib_image)
fig, ax = plt.subplots()
# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.5, bottom=0.4)
ax.imshow(img, origin='lower', cmap=plt.get_cmap('gray'))

# -----------------------------------------------------------------------------
# --- Plot the scintillator plate
# -----------------------------------------------------------------------------
scintillator.plot_pix(ax)
# Get the line, it is just the last thing added in the axis


# -----------------------------------------------------------------------------
# --- GUI
# -----------------------------------------------------------------------------
# Make a horizontal sliders to control shifts
axxs = plt.axes([0.4, 0.05, 0.65, 0.03])
axxs_slider = Slider(
    ax=axxs,
    label='xshift',
    valmin=xshiftmin,
    valmax=xshiftmax,
    valinit=xshift,
)
axys = plt.axes([0.4, 0.15, 0.65, 0.03])
axys_slider = Slider(
    ax=axys,
    label='yshift',
    valmin=yshiftmin,
    valmax=yshiftmax,
    valinit=yshift,
)

# Make a vertically oriented slider to control the amplitude
axxsc = plt.axes([0.05, 0.25, 0.0225, 0.63])
axxsc_slider = Slider(
    ax=axxsc,
    label="xscale",
    valmin=xscalemin,
    valmax=xscalemax,
    valinit=xscale,
    orientation="vertical"
)
axysc = plt.axes([0.25, 0.25, 0.0225, 0.63])
axysc_slider = Slider(
    ax=axysc,
    label="yscale",
    valmin=yscalemin,
    valmax=yscalemax,
    valinit=yscale,
    orientation="vertical"
)
axdeg = plt.axes([0.40, 0.25, 0.0225, 0.63])
axdeg_slider = Slider(
    ax=axdeg,
    label="deg",
    valmin=degmin,
    valmax=degmax,
    valinit=deg,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    cal.xshift = axxs_slider.val
    cal.yshift = axys_slider.val
    cal.xscale = axxsc_slider.val
    cal.yscale = axysc_slider.val
    cal.deg = axdeg_slider.val
    scintillator.calculate_pixel_coordinates(cal)
    ss.plt.remove_lines(ax)
    scintillator.plot_pix(ax)


# register the update function with each slider
axxs_slider.on_changed(update)
axys_slider.on_changed(update)
axxsc_slider.on_changed(update)
axysc_slider.on_changed(update)
axdeg_slider.on_changed(update)


plt.show()
