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
        'runid': 'test34',
        'geomID': 'Jesus',
        'FILDSIMmode': True,
        'nGeomElements': 9,
        'nxi': 7,
        'nGyroradius': 2,
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
        'Zin': 1.0,         # Charge before the ionization in the foil
        'Zout': 1.0,        # Charge after the ionization in the foil
        'IpBt': 1,        # Sign of toroidal current vs field (for pitch)
        'flag_efield_on': True,  # Add or not electric field

    },
    'inputParams': {
         'nGyro': 50,
         'minAngle': -3.14,
         'dAngle': 6.24,
         'XI': [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
         'rL': [2.5, 3.0],
         'maxT': 0.0000001
    },
}

# Magnetic field
B = np.array([0.0, -1.0, 0.0])
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
