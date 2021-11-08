"""
Lecture 1 of the examples to run the SINPA code: Mapping

Jose Rueda Rueda: jrrueda@us.es

Done to explain how to run a SINPA simulation

Created for version 6.0.0 of the Suite and version 0.0 of SINPA
"""
import os
# import numpy as np
import Lib as ss
from Lib.LibMachine import machine
from Lib.LibPaths import Path
import numpy as np
paths = Path(machine)

# -----------------------------------------------------------------------------
# --- Settings block
# -----------------------------------------------------------------------------
nml_options = {
    'config':  {            # General parameters
        'runid': 'hope_no_field',
        'geomID': 'Test0',
        'FILDSIMmode': False,
        'nGeomElements': 3,
        'nxi': 1,
        'nGyroradius': 1,
        'nMap': 500,
        'mapping': True,
        'signal': False,
        'resampling': False,
        'nResampling': 4,
        'saveOrbits': True,
        'saveRatio': 0.1,
        'SINPA_dir': paths.SINPA,
        'FIDASIMfolder': 'kiwi',
        'verbose': True,
        'M': 2.0,         # Mass of the particle (in uma)
        'Zin': 0.0,         # Charge before the ionization in the foil
        'Zout': 1.0,        # Charge after the ionization in the foil
        'IpBt': 1,        # Sign of toroidal current vs field (for pitch)
        'flag_efield_on': True,  # Add or not electric field

    },
    'markerinteractions': {    # Particles and foil modelling
        'energyLoss': True,
        'a_SRIM': 8.202948e-02,
        'b_SRIM': -9.984000e-05,
        'weightChange': True,
        'a_ionization': 7.711000e-01,
        'b_ionization': 1.107193e-05,
        'c_ionization': 8.254000e-01,
        'geometricTrans': 6.100000e-01,
        'scattering': False
    },
    'inputParams': {
         'nGyro': 40,
         'minAngle': -0.1,
         'dAngle': 0.2,
         'XI': [3.3],
         # 'alphas': [3.141592],
         'rL': [3.0],
         'maxT': 0.0000001
    },
    'nbi_namelist': {            # NBI geometry
        'p0': [220.78, -137.32, -2.1],  # xyz of first point in the NBI
        'u': [-0.6013878,  0.79444944,  0.08475143],  # vector of the NBI
        'd': 0.5,                # Distance between points
        'npoints': 400,          # Number of points
    },
}

# Magnetic field
B = np.array([0.0575, -2.1532, -0.1291])
E = 0.0 * np.array([0.5792, -0.0603, 0.8129])

# -----------------------------------------------------------------------------
# --- Section 0: Create the directories
# -----------------------------------------------------------------------------
runDir = os.path.join(paths.SINPA, 'runs', nml_options['config']['runid'])
inputsDir = os.path.join(runDir, 'inputs')
resultsDir = os.path.join(runDir, 'results')
os.makedirs(runDir, exist_ok=True)
os.makedirs(inputsDir, exist_ok=True)
os.makedirs(resultsDir, exist_ok=True)

# -----------------------------------------------------------------------------
# --- Section 1: Prepare the namelist
# -----------------------------------------------------------------------------
ss.sinpa.execution.write_namelist(nml_options)
# -----------------------------------------------------------------------------
# --- Section 2: Prepare the magnetic field
# -----------------------------------------------------------------------------
# Get the field
field = ss.sinpa.fieldObject()
field.createHomogeneousField(B, field='B')
field.createHomogeneousField(E, field='E')

# Write the field
fieldFileName = os.path.join(inputsDir, 'field.bin')
fid = open(fieldFileName, 'wb')
field.tofile(fid)
fid.close()
fieldFileName = os.path.join(inputsDir, 'Efield.bin')
fid = open(fieldFileName, 'wb')
field.tofile(fid, bflag=False, eflag=True)
fid.close()


# -----------------------------------------------------------------------------
# --- Section 2: Run the code
# -----------------------------------------------------------------------------
# Check the files
ss.sinpa.execution.check_files(nml_options['config']['runid'])
# Launch the simulations
ss.sinpa.execution.executeRun(nml_options['config']['runid'])
