import Lib as ss
import matplotlib.pyplot as plt
import sys
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib')
import dd 					# load latest (!) dd library
import numpy as np
# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------

shot = 39612

# -----------------------------------------------------------------------------
# --- Load and plot temporal evolution
# -----------------------------------------------------------------------------

# --- Plasma current
Ip_shotfile = dd.shotfile('MAG', shot)
Ip = Ip_shotfile(b'Ipa')
Ip_shotfile.close()
# --- Plasma density
DCN = dd.shotfile('DCN', shot)
H2 = DCN(b'H-2')
H5 = DCN(b'H-5')
DCN.close()
# --- Electron temperature
IDA = dd.shotfile('IDA', shot)
Te = IDA(b'Te')
ir10 = np.argmin(abs(Te.area.data[0, :] - 0.1))
ir25 = np.argmin(abs(Te.area.data[0, :] - 0.25))
ir95 = np.argmin(abs(Te.area.data[0, :] - 0.95))
te10 = Te.data[:, ir10]
te25 = Te.data[:, ir25]
te95 = Te.data[:, ir95]
IDA.close()
# --- Injected power
NIS = dd.shotfile('NIS', shot)
PNI = NIS(b'PNI')
NIS.close()
try:
    ICP = dd.shotfile('ICP', shot)
    PICRN = ICP(b'PICRN')
    ICP.close()
    ICRH = True
except:
    print('No ICRH')
    ICRH = False
BDP = dd.shotfile('BPD', shot)
Prad = BDP(b'Pradtot')
BDP.close()

# ELMS
MAC = dd.shotfile('MAC', shot)
Ipolsola = MAC(b'Ipolsola')
MAC.close()
# --- Plot
fig, ax = plt.subplots(5, sharex=True)
# - Ip
ax[0].plot(Ip.time, Ip.data/1.0e6)
ax[0].set_ylabel('Ip [MA]')
# - ne
ax[1].plot(H2.time, H2.data, label='H-2')
ax[1].plot(H5.time, H5.data, label='H-5')
ax[1].set_ylabel('[1e19 $m^{-2}$]')
ax[1].legend()
# Te
ax[2].plot(Te.time, te10/1000., label='$\\rho_p = 0.10$')
ax[2].plot(Te.time, te25/1000., label='$\\rho_p = 0.25$')
ax[2].plot(Te.time, te95/1000., label='$\\rho_p = 0.95$')
ax[2].set_ylabel('Te [keV]')
ax[2].legend()
# NBI
ax[3].plot(PNI.time, PNI.data/1.e6, label='NBI')
if ICRH:
    ax[3].plot(PICRN.time, PICRN.data/1.e6, label='ICRH')
ax[3].plot(Prad.time, Prad.data/1.e6, label='Prad')
ax[3].set_ylabel('[MW]')
ax[3].legend()
# ELMS
ax[4].plot(Ipolsola.time, Ipolsola.data)
ax[4].set_xlabel('Time [s]')
ax[4].set_xlim(0, 7.0)

fig.show()
