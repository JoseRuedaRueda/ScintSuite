"""
Plot the evolution of plasma volume and compare it with density (H-1, H-5) and
q95, q50

Jose Rueda: jrrueda@us.es
"""
import ScintSuite.as ss
import matplotlib.pyplot as plt
import sys
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib')
import dd

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
shot = 39612   # Shot number
t_ini = 0.3    # Initial time to load [in s]
t_fin = 3.2    # Final time to load [in s]
diag = 'EQH'


# -----------------------------------------------------------------------------
# --- Load data
# -----------------------------------------------------------------------------
# volumes and qs
data = ss.dat.get_shot_basics(shot, diag=diag)
# densities
DCN = dd.shotfile('DCN', shot)
H1 = DCN(b'H-1')
H5 = DCN(b'H-5')
DCN.close()
# -----------------------------------------------------------------------------
# --- Plot
# -----------------------------------------------------------------------------
# Volume and density
fig, ax = plt.subplots()

plt.plot(data['time'], data['Vol'], 'k', label='Volume')
ax.set_ylabel('Volume [$m^{-3}$]')
ax.legend()
ax.set_xlabel('Time [s]')
ax2 = ax.twinx()
ax2.plot(H1.time, H1.data, 'b', label='H1')
ax2.plot(H5.time, H5.data, 'r', label='H5')
ax2.set_ylabel('[$m^{-2}$]')
ax2.legend()
fig.show()

# Volume and Qprofile
fig2, axx = plt.subplots()

plt.plot(data['time'], data['Vol'], 'k', label='Volume')
axx.set_ylabel('Volume [$m^{-3}$]')
axx.set_xlabel('Time [s]')
axx.legend()
axx2 = axx.twinx()
axx2.plot(data['time'], data['q50'], 'b', label='q50')
axx2.plot(data['time'], data['q95'], 'r', label='q95')
axx2.legend()
fig2.show()
