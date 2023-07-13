"""
Example of the use of the tracker under a dummy magnetic field

Jose Rueda: jrrueda@us.es

Done for version 0.4.10
"""
import ScintSuite.as ss
import os
import numpy as np
from scipy.interpolate import interpn

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
B0 = 1.5  # [T] Magnetic field in the Z direction
A = 2.0   # mass of the particle in amu
Ze = 1.0   # charge of the particle in |e| units
# --- grid:
R = np.linspace(0.1, 1.0, 180).astype(np.float64)
z = np.linspace(-1.0, 1.0, 400).astype(np.float64)
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
        'dt': 1.0e-9,
        'max_step': 200000,
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
# note for a full documentation of the order in which the field matrix should
# be written, see the really nice documentation of iHIBPsim. Here as the field
# will be taken as constan in the z direction, I will just include a dummy 1D
# array

field = ss.iHIBP.fields.ihibpEMfields()   # open the field object
field.bdims = 2
nR = R.size
nZ = z.size
# fill it with our dummy field
field.Bfield['R'] = R
field.Bfield['z'] = z
field.Bfield['Rmin'] = np.array((R.min()), dtype=np.float64)
field.Bfield['Rmax'] = np.array((R.max()), dtype=np.float64)
field.Bfield['Zmin'] = np.array((z.min()), dtype=np.float64)
field.Bfield['Zmax'] = np.array((z.max()), dtype=np.float64)
field.Bfield['nR'] = np.array([nR], dtype=np.int32)
field.Bfield['nZ'] = np.array([nZ], dtype=np.int32)
field.Bfield['fr'] = np.zeros((nR, nZ)).astype(dtype=np.float64)
field.Bfield['fz'] = B0 * np.ones((nR, nZ)).astype(dtype=np.float64)
field.Bfield['ft'] = np.zeros((nR, nZ)).astype(dtype=np.float64)

field.Brinterp = lambda r, z, phi, time: \
                interpn((field.Bfield['R'], field.Bfield['z']),
                        field.Bfield['fr'],
                        (r.flatten(), z.flatten()))
field.Bzinterp = lambda r, z, phi, time: \
                interpn((field.Bfield['R'], field.Bfield['z']),
                        field.Bfield['fz'],
                        (r.flatten(), z.flatten()))

field.Bphiinterp = lambda r, z, phi, time: \
                interpn((field.Bfield['R'], field.Bfield['z']),
                        field.Bfield['ft'],
                        (r.flatten(), z.flatten()))

# for the poloidal flux:
field.psipol_on = True
RR, ZZ = np.meshgrid(R, z)
Psi = np.zeros(RR.shape)
# sorry Pablo I am lazy do not kill me...
for iz in range(nZ):
    for ir in range(nR):
        Psi[iz, ir] = RR[iz, ir] ** 2
Psi *= B0 * np.pi
field.psipol['R'] = R
field.psipol['z'] = z
field.psipol['nR'] = np.array([nR], dtype=np.int32)
field.psipol['nZ'] = np.array([nZ], dtype=np.int32)
field.psipol['f'] = Psi.T.astype(dtype=np.float64)
field.psipol_interp = lambda r, z, phi, time: \
                     interpn((field.psipol['R'], field.psipol['z']),
                             field.psipol['f'],
                             (r.flatten(), z.flatten()))
# -----------------------------------------------------------------------------
# --- Markers preparation
# -----------------------------------------------------------------------------
E = 90000  # energy in eV
v = np.sqrt(2. * E / A / ss.par.mp) * ss.par.c
marker = {
    'R': np.array([0.5], dtype=np.float64),
    'z': np.array([-0.9], dtype=np.float64),
    'phi': np.array([0.9], dtype=np.float64),
    'vR': np.array([0], dtype=np.float64),
    'vz': np.array([1], dtype=np.float64),
    'vt': np.array([v], dtype=np.float64),
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
orbit.setMagnetics(field)
orbit.plotTimeTraces(plot_coords=True)
