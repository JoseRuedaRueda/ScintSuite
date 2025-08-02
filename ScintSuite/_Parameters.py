"""Contain physical constants and camera information

Basically here are hard-cored almost all the parameters of the suite
"""
from scipy.constants import elementary_charge as ec
from scipy.constants import electron_mass

# Physics constants
mp = 938.272e6  # Mass of the proton, in eV/c^2
mp_kg = 1.67262192369e-27  # Mass of the proton in kg
c = 2.99792458e8       # Speed of light in m/s
amu2kg = 1.660538782e-27  # Scaling factor to go from AMU to SI units (NIST)
h_planck = 4.135667e-15         # [eV/s]
eps0 = 5.52635e7  # Vaccum permitivity in e/Vm
mass_electron_amu = electron_mass/amu2kg # Electron mass in Atomic Mass Units.

# -----------------------------------------------------------------------------
# --- File parameters
# -----------------------------------------------------------------------------
filetypes = [('netCDF files', '*.nc'),
             ('ASCII files', '*.txt'),
             ('cine files', ('*.cin', '*.cine')),
             ('Python files', '*.py'),
             ('Pickle4 files', '*.pk4'),
             ('Strikemap files', '*.map')]

# Access to files via seek routine.
SEEK_BOF = 0
SEEK_CUR = 1
SEEK_EOF = 2
SEEK_END = 2

# -----------------------------------------------------------------------------
# --- Atomic parameters
# -----------------------------------------------------------------------------
species_info = {
    'H': {
        'A': 1.00784,
        'Z': 1.0,
    },
    'D': {
        'A': 2.01410177811,
        'Z': 1.0,
    },
    'T': {
        'A': 3.0160492,
        'Z': 1.0,
    },
    'He': {
        'A': 4.002602,
        'Z': 2.0,
    },
}
