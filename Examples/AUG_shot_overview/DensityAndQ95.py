"""
Plot the evolution of plasma volume and compare it with density (H-1, H-5) and
q95, q50

Jose Rueda: jrrueda@us.es
"""
import Lib as ss
import matplotlib.pyplot as plt
import sys
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib')
import dd
from scipy import interpolate as interp

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
shot = 39612   # Shot number
t_ini = 1.5    # Initial time to load [in s]
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
# Respect to time
fig, ax = plt.subplots()

plt.plot(data['time'], data['q95'], 'k', label='q95')
ax.set_ylabel('q95')
ax.legend()
ax.set_xlabel('Time [s]')
ax2 = ax.twinx()
ax2.plot(H1.time, H1.data, 'b', label='H1')
ax2.plot(H5.time, H5.data, 'r', label='H5')
ax2.set_ylabel('[$m^{-2}$]')
ax2.legend()
fig.show()

# respect to q profile
it1 = int(1000 * t_ini)
it2 = int(1000 * t_fin)
f = interp.interp1d(data['time'], data['q95'], fill_value=0.0)
q95_h1_base = f(H1.time[it1:it2])
fig2, axx = plt.subplots()

axx.plot(q95_h1_base, H1.data[it1:it2], 'b', label='H1')
axx.plot(q95_h1_base, H5.data[it1:it2], 'r', label='H5')
axx.legend()
axx.set_xlabel('q95')
axx.set_ylabel('H [$m^{-2}$]')
fig2.show()
