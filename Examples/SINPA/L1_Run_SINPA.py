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
        'runid': 'test',
        'geomID': 'Test',
        'nAlpha': 3,
        'nGyroradius': 5,
        'nMap': 1000,
        'mapping': True,
        'signal': False,
        'resampling': False,
        'nResampling': 4,
        'saveOrbits': False,
        'saveRatio': 0.1,
        'SINPA_dir': paths.SINPA,
        'verbose': True
    },
    'particlesFoil': {    # Particles and foil modelling
        'M': 2.0,         # Mass of the particle (in uma)
        'Zin': 0.0,         # Charge before the ionization in the foil
        'Zout': 1.0,        # Charge after the ionization in the foil
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
         'nGyro': 20,
         'beta0': 0.05,
         'alphas': [2.83, 3.14, 3.46],
         # 'alphas': [3.141592],
         'rL': [0.5, 0.7, 1.5, 2.0, 2.5],
         'maxT': 0.000005
    },
    'nbi_namelist': {            # NBI geometry
        'p0': [220.7590, -137.35, -2.1],  # xyz of first point in the NBI
        'u': [-0.58993824, 0.8016366,  0.0967039],  # vector of the NBI
        'd': 0.5,                # Distance between points
        'npoints': 400,          # Number of points
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
direction = [0., 0.0, -1.0]
# Get the field
field = ss.sinpa.field.sinpaField()
field.createFromSingleB(direction)
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
