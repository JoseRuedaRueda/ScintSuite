"""
Lecture 1 of the examples to run the SINPA code: Mapping

Jose Rueda Rueda: jrrueda@us.es

Done to explain how to run a SINPA simulation

Created for version 6.0.0 of the Suite and version 0.0 of SINPA
"""
import os
import numpy as np
import Lib as ss
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)

# -----------------------------------------------------------------------------
# --- Settings block
# -----------------------------------------------------------------------------
nml_options = {
    'config':  {            # General parameters
        'runid': 'FILD1',
        'geomID': 'FILD1',
        'FILDSIMmode': True,
        'nGeomElements': 2,
        'nxi': 7,
        'nGyroradius': 2,
        'nMap': 1000,
        'mapping': True,
        'signal': False,
        'resampling': False,
        'nResampling': 4,
        'saveOrbits': True,
        'saveRatio': 0.1,
        'SINPA_dir': paths.SINPA,
        'FIDASIMfolder': '',
        'verbose': True,
        'M': 2.0,         # Mass of the particle (in uma)
        'Zin': 1.0,         # Charge before the ionization in the foil
        'Zout': 1.0,        # Charge after the ionization in the foil
        'IpBt': -1,        # Sign of toroidal current vs field (for pitch)
    },
    'inputParams': {
         'nGyro': 20,
         'minAngle': -3.14,
         'dAngle': 2*3.14,
         'alphas': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5,3.0],
         # 'alphas': [3.141592],
         'rL': [2.5, 3.0],
         'maxT': 0.0000005
    },
}

# Magnetic field
zita = 65.90115304686239
ipsilon = 36.08595339878833

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
# Get the direction of the field
direction = \
    ss.sinpa.field.constructDirection(zita, ipsilon,
                                      nml_options['config']['geomID'])
direction = [0., 0.0, -1.8]
# Get the field
field = ss.sinpa.fieldObject()
field.createFromSingleB(direction, Rmin=0.01, Rmax=25.0, zmin=-10.0, zmax=10.0)
# Write the field
fieldFileName = os.path.join(inputsDir, 'field.bin')
fid = open(fieldFileName, 'wb')
field.tofile(fid)
fid.close()


# -----------------------------------------------------------------------------
# --- Section 2: Run the code
# -----------------------------------------------------------------------------
# Check the files
ss.sinpa.execution.check_files(nml_options['config']['runid'])
# Launch the simulations
ss.sinpa.execution.executeRun(nml_options['config']['runid'])
