"""
Example of the use of the tracker to folow NBI markers

Jose Rueda: jrrueda@us.es

Done for version 0.4.10

lines marked with #### at the end should be addapted to your paths for the
example to work
"""
import Lib as ss
import os
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
# --- Magnetic field equilibrium:
shot = 34570   # shot number
time = 2.5     # time point [s]
diag = 'EQH'   # diagnostic for the magnetic field
# --- NBI and markers settings:
nNBI = 8       # NBI number
A = 2.0        # mass of the particle in amu
Nions = 999    # Number of ions to launch
E = 93000      # Energy of the markers, in eV
sE = 2000      # Standard deviation of the markers' energy
# --- Simulation general settings:
tmax = 0.001   # Maximum time to follow the particles
dt = 1.0e-10   # dt for the integration algorithm
# --- custom nml and paths
directory = '/afs/ipp/home/r/ruejo/ihibpsim/jrr/NBI_case'                 # ###
files = {
    'Bfield': os.path.join(directory, 'B.bin'),
    'Rho': os.path.join(directory, 'rho.bin'),
    'Orbits': os.path.join(directory, 'orb.bin'),
    'Deposition': os.path.join(directory, 'deposition.bin'),
    'Output': os.path.join(directory, 'output.bin'),
    'Namelist': os.path.join(directory, 'nml.cfg')
}
nml = {
    'FIELD_FILES': {
        'Bfield_name': files['Bfield'],
    },
    'INTEGRATION': {
        'dt': dt,
        'max_step': int(tmax / dt),
        'file_out': files['Output'],
    },
    'ORBITS_CONF': {
        'save_orbits': False,
        'num_orbits': 1.0,
        'file_orbits': files['Orbits'],
        'dt_orbit': 1.0e-10,
    },
    'DEPOSITION': {
        'markerNumber': Nions,
        'depos_file': files['Deposition'],
    },
    'SCINTILLATOR': {
        'triangle_file': '/afs/ipp/home/r/ruejo/iHIBPsim/bin/plate.pod'   # ###
    }
}
# --- plot options
p1 = True   # Plot the energy distribution
p2 = True   # Plot the

# -----------------------------------------------------------------------------
# --- Field preparation
# -----------------------------------------------------------------------------
field = ss.iHIBP.fields.ihibpEMfields()   # open the field object
field.readBfromDB(time=time, shot=shot, nR=1000, nZ=2000, diag=diag)
# -----------------------------------------------------------------------------
# --- Markers preparation
# -----------------------------------------------------------------------------
NBI = ss.dat.NBI(nNBI)
markers = NBI.generate_tarcker_markers(Nions, E, sE)
# -----------------------------------------------------------------------------
# --- Prepare files for tracker
# -----------------------------------------------------------------------------
# --- magnetic field
f = open(files['Bfield'], 'wb')
field.tofile(f, bflag=True)
f.close()
# --- Namelist
ss.tracker.write_tracker_namelist(nml, files['Namelist'])
# --- markers
ss.tracker.write_markers(markers, files['Deposition'])
# --- Launch tracker
os.system(ss.paths.tracker + ' ' + files['Namelist'])
# --- Read the final position of the markers
fpoints = ss.iHIBP.strikes.readStrikeFile(files['Output'])
# --- calculate the energy distribution:
# There is a small inconsistency of naming in the lib routines, will be solved
# in version 4.13, up to now, we have to deal with the tho names
# -- First check if some marker was written in other position in the file
dummy = markers['ID'] - fpoints['ID']
if np.sum(abs(dummy)) > 0:
    print('Order of the markers in the output file changed.')
flags = fpoints['time'] > 0.999 * tmax  # to separate between lost or not
# -- Now get the velocity and energy:
markers['v'] = np.sqrt(markers['vR']**2 + markers['vz']**2 + markers['vt']**2)
markers['K'] = 0.5 * ss.par.amu2kg * markers['mass'] * markers['v']**2

fpoints['v'] = np.sqrt(fpoints['vr']**2 + fpoints['vz']**2 + fpoints['vphi']**2)
fpoints['K'] = 0.5 * ss.par.amu2kg * fpoints['mass'] * fpoints['v']**2
E_edges = np.linspace(0.9*E, 1.1*E, 25)
E_distro_ini, E_edges = np.histogram(markers['K'] / ss.par.ec)
E_distro_fin, E_edges = np.histogram(fpoints['K'] / ss.par.ec)
E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
# -- Procced to plot:
if p1:
    fig1, ax1 = plt.suplbots()
    ax1.bar(E_centers, E_distro_ini, color='r', alpha=0.5, label='Initial')
    ax1.bar(E_centers, E_distro_fin, color='k', alpha=0.5, label='Final')
    plt.legend()
    fig1.show()
