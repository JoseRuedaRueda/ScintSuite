"""Contain physical constants and camera information

Basically here are hard-cored almost all the parameters of the suite
"""
# Physics constants
ec = 1.602176487e-19  # Electron charge, in C [from NIST]
mp = 938.272e6  # Mass of the proton, in eV/c^2
mp_kg = 1.67262192369e-27  # Mass of the proton in kg
c = 2.99792458e8       # Speed of light in m/s
amu2kg = 1.660538782e-27  # Scaling factor to go from AMU to SI units (NIST)
h_planck = 4.135667e-15         # [eV/s]
eps0 = 5.52635e7  # Vaccum permitivity in e/Vm

iHIBP = {'port_center': [0.687, -3.454, 0.03], 'sector': 13,
         'beta_std': 4.0, 'theta_std': 0.0, 'source_radius': 7.0e-3}


# -----------------------------------------------------------------------------
# --- File parameters
# -----------------------------------------------------------------------------
filetypes = [('netCDF files', '*.nc'),
             ('ASCII files', '*.txt'),
             ('cine files', ('*.cin', '*.cine')),
             ('Python files', '*.py'),
             ('Pickle4 files', '*.pk4')]

# Access to files via seek routine.
SEEK_BOF = 0
SEEK_CUR = 1
SEEK_EOF = 2
SEEK_END = 2
