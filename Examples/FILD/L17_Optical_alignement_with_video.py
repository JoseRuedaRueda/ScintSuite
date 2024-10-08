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
import ScintSuite.as ss
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


## ----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
shot = 41252
time = 1.124
vmax = 125
# modify section 3 if you have a custom format for the calibration image
# Staring points for the calibration
xshift = -11.5
yshift = 134.6
xscale = 5718
deg = 44.49
# x-scale to y scale
XtoY = -1.0

# Scale maximum
xshiftmax = 80
yshiftmax = 500
xscalemax = 10000
degmax = 55

# Scale minimum
xshiftmin = -80
yshiftmin = -100
xscalemin = 2000
degmin = 25
## ----------------------------------------------------------------------------
# --- Load video
# -----------------------------------------------------------------------------
vid = ss.vid.FILDVideo(shot=shot)
vid.read_frame(t1=time-0.3, t2=time+0.3)
## -----------------------------------------------------------------------------
# --- Scintillator load and first alignement
# -----------------------------------------------------------------------------
cal = ss.mapping.CalParams()
cal.xshift = xshift
cal.yshift = yshift
cal.xscale = xscale
cal.yscale = xscale * XtoY
cal.deg = deg
vid.scintillator.calculate_pixel_coordinates(cal)


## ----------------------------------------------------------------------------
# --- Image load and plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots()
# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.30, bottom=0.3)
ax = vid.plot_frame(t=time, ccmap=plt.get_cmap('gray'), vmax=vmax, ax=ax)
vid.scintillator.plot_pix(ax)

# -----------------------------------------------------------------------------
# --- GUI
# -----------------------------------------------------------------------------
# Make a horizontal sliders to control shifts
axxs = plt.axes([0.2, 0.05, 0.65, 0.03])
axxs_slider = Slider(
    ax=axxs,
    label='xshift',
    valmin=xshiftmin,
    valmax=xshiftmax,
    valinit=xshift,
)
axys = plt.axes([0.2, 0.15, 0.65, 0.03])
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

axdeg = plt.axes([0.20, 0.25, 0.0225, 0.63])
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
    cal.yscale = XtoY * axxsc_slider.val
    cal.deg = axdeg_slider.val
    vid.scintillator.calculate_pixel_coordinates(cal)
    ss.plt.remove_lines(ax)
    vid.scintillator.plot_pix(ax)


# register the update function with each slider
axxs_slider.on_changed(update)
axys_slider.on_changed(update)
axxsc_slider.on_changed(update)
axdeg_slider.on_changed(update)


plt.show()
