"""
Compares FILD signals with NBI injection power

Jose Rueda Rueda: jrrueda@us.es

Note: This use routines for ASDEX directly, it will not work on other machine.
Date: 20/02/2021

Issues: Up to now, only NBI injection is included

Lines finishing with  !#! Must be modified acordingly to your local files
"""
import sys
import Lib as ss
import matplotlib.pyplot as plt
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib')
import dd 					# load latest (!) dd library

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------

FILD_TT = ['MyAwesomeTimetraceFile']  # Paths to timetrace files            !#!

labels = ('FILD4')                    # Label for each timetrace            !#!
shot = 38626        # Shot number (for the NBI timetraces)
pmin = 0.7e6        # Minimum power to consider as on an NBI [W]
# Plotting options:
FS = 14             # FontSize
grid = 'both'       # activate bck grid or not
linewidth = 2       # LineWidth
alpha = 0.5

# -----------------------------------------------------------------------------
# --- Load and plot FILD data
# -----------------------------------------------------------------------------
FILD = []
fig, ax = plt.subplots()
for i in range(len(FILD_TT)):
    TT = ss.io.read_timetrace(FILD_TT[i])
    FILD.append(FILD)
    line_par = {'linewidth': linewidth, 'labels': labels[i]}
    ax = TT.plot_single(ax=ax, normalised=True)

# -----------------------------------------------------------------------------
# --- Load and plot NBI data
# -----------------------------------------------------------------------------
nbi = dd.shotfile('NIS', shot)
PNIQ = nbi.getObjectData(b'PNIQ')
tnbi = nbi.getTimeBase(b'PNIQ')
for i in range(8):
    if i < 4:
        ibox = 0
        isource = i
    else:
        ibox = 1
        isource = i - 4
    power = PNIQ[ibox, isource, :].squeeze()
    if power.max() > pmin:
        lab = 'P .NBI#' + str(i + 1) + ' [/2.5MW]'
        ax.plot(tnbi, power/2.5e6, label=lab, linewidth=linewidth, alpha=alpha)

# -----------------------------------------------------------------------------
# --- Axis beauty
# -----------------------------------------------------------------------------
param = {'fontsize': FS, 'grid': grid}
ax = ss.plt.axis_beauty(ax, param)
plt.legend(fontsize=0.8*FS)
plt.show()
