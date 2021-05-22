"""
Example of the use of the tracker under an AUG field

Jose Rueda: jrrueda@us.es

Done for version 0.4.10
"""
import Lib as ss
import os
import numpy as np

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
shot = 28061   # shot number
time = 4.495     # time point [s]
diag = 'EQH'   # diagnostic for the magnetic field
A = 2.0   # mass of the particle in amu
Ze = 1.0   # charge of the particle in |e| units
# --- custom nml and paths
directory = '/afs/ipp/home/r/ruejo/ihibpsim/jrr'
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
        'dt': 1.0e-10,
        'max_step': 10000000,
        'file_out': files['Output'],
    },
    'ORBITS_CONF': {
        'save_orbits': True,
        'num_orbits': 1.0,
        'file_orbits': files['Orbits'],
        'dt_orbit': 1.0e-8,
    },
    'DEPOSITION': {
        'markerNumber': 1,
        'depos_file': files['Deposition'],
    },
    'SCINTILLATOR': {
        'triangle_file': '/afs/ipp/home/r/ruejo/iHIBPsim/bin/plate.pod'
    }
}


# -----------------------------------------------------------------------------
# --- Field preparation
# -----------------------------------------------------------------------------
field = ss.iHIBP.fields.ihibpEMfields()                 # open the field object
field.readBfromDB(time=time, shot=shot, nR=1000, nZ=2000, diag=diag)   # read B
# -----------------------------------------------------------------------------
# --- Markers preparation
# -----------------------------------------------------------------------------
E = 90000  # energy in eV
v = np.sqrt(2. * E / A / ss.par.mp) * ss.par.c
marker = {
    'R': np.array([1.9], dtype=np.float64),
    'z': np.array([0.0], dtype=np.float64),
    'phi': np.array([0.9], dtype=np.float64),
    'vR': np.array([0.5*v], dtype=np.float64),
    'vz': np.array([0], dtype=np.float64),
    'vt': np.array([0.87*v], dtype=np.float64),
    'm': np.array([A], dtype=np.float64),
    'q': np.array([Ze], dtype=np.float64),
    'logw': np.array([0], dtype=np.float64),
    't': np.array([-0], dtype=np.float64),
}
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
ss.tracker.write_markers(marker, files['Deposition'])
# --- Launch tracker
os.system(ss.paths.tracker + ' ' + files['Namelist'])
# --- Read the orbits output
orbit_file = ss.iHIBP.orbs.orbitFile(files['Orbits'])
[orbit] = orbit_file.loadOrbit()
orbit.setMagnetics(field, magMomentumGyrocenter=True, magToroidalUseVpar=True)
orbit.plotTimeTraces(plot_coords=True)
orbit.plot()
