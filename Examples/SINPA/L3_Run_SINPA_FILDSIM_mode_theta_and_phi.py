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
        'runid': 'theta9_phi9',
        'geomID': 'FILD1_FILDSIM',
        'FILDSIMmode': True,
        'nGeomElements': 2,
        'nxi': 7,
        'nGyroradius': 7,
        'nMap': 50000,
        'mapping': True,
        'signal': False,
        'resampling': False,
        'nResampling': 0,
        'saveOrbits': False,
        'saveRatio': 0.1,
        'saveOrbitLongMode': False,
        'SINPA_dir': paths.SINPA,
        'FIDASIMfolder': '',
        'verbose': True,
        'M': 2.0,         # Mass of the particle (in uma)
        'Zin': 1.0,         # Charge before the ionization in the foil
        'Zout': 1.0,        # Charge after the ionization in the foil
        'IpBt': -1,        # Sign of toroidal current vs field (for pitch)
        'flag_efield_on': False,  # Add or not electric field
        'save_collimator_strike_points': False,  # Save collimator points,
        'backtrace': False  # Flag to backtrace the orbits
    },
    'inputParams': {
         'nGyro': 500,
         'minAngle': -1.6,
         'dAngle': 0.4,
         'XI': [80., 70., 60., 50., 40., 30., 20.],
         'rL': [2., 3., 4., 5., 6., 7., 8.],
         'maxT': 0.00000025
    },
}
theta = 9.0
phi = 9.0
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
# Load the geometry
Geometry = ss.simcom.Geometry(nml_options['config']['geomID'], code='SINPA')
u1 = np.array(Geometry.ExtraGeometryParams['u1'])
u2 = np.array(Geometry.ExtraGeometryParams['u2'])
u3 = np.array(Geometry.ExtraGeometryParams['u3'])
# Get the field
field = ss.sinpa.fieldObject()
field.createHomogeneousFieldThetaPhi(theta, phi, field='B',
                                     u1=u1, u2=u2, u3=u3)

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
