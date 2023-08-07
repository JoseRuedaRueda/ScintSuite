"""
Lecture 2 of the examples to run the SINPA code: FILDSIM simulation

Jose Rueda Rueda: jrrueda@us.es

Done to explain how to run a SINPA simulation

Created for version 0.7.3 of the Suite and version 0.1 of SINPA

Note: All saving options, orbits, collimator impacts etc are on, so the code
run will be slow, don't be afraid, disconnect the saving of the orbits and
collimator strike points and you will see the nice speed

last revision:
    ScintSuite: version 1.0.4
    SINPA (uFILDSIM): version 2.3
"""
import os
import numpy as np
import ScintSuite.as ss
paths = ss.paths
# -----------------------------------------------------------------------------
# --- Settings block
# -----------------------------------------------------------------------------
runid = 'Example_test'
geomID = 'FILD1'
nml_options = {   # Se the PDF documentation for a complete desription of these
    'config':  {  # parameters
        'runid': runid,
        'geomfolder': os.path.join(paths.SINPA, 'Geometry', geomID),
        'FILDSIMmode': True,
        'nxi': 8,
        'nGyroradius': 10,
        'nMap': 50000,
        'n1': 1.0,
        'r1': 1.2,
        'restrict_mode': True,
        'mapping': True,
        'saveOrbits': True,
        'saveRatio': 0.01,
        'saveOrbitLongMode': False,
        'runfolder': os.path.join(paths.SINPA, 'runs', runid),
        'verbose': True,
        'IpBt': -1,              # Sign of toroidal current vs field (for pitch)
        'flag_efield_on': False, # Add or not electric field
        'save_collimator_strike_points': False,  # Save collimator points
        'backtrace': False  # Flag to backtrace the orbits
    },
    'inputParams': {
         'nGyro': 350,
         'minAngle': -2.2,
         'dAngle': 1.0,
         'XI': [85.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0],
         'rL': [1.75, 2.0, 3., 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
         'maxT': 0.00000008
    },
}
# --- magnetic field definition:
# Option 1: as FILDSIM, given by the 2 orientation angles
use_opt1 = True
theta = 8.2
phi = -0.5
# Option 2: directly a field in cartesian coordinates
direction = np.array([-0.15643447,  -0.97552826, 0.1545085])  # Arbitrary field
# There are more options such as full3D, axisymetric field, uniform
# cylindrical... just explore the field object
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
filename = ss.sinpa.execution.write_namelist(nml_options)
# -----------------------------------------------------------------------------
# --- Section 2: Prepare the magnetic field
# -----------------------------------------------------------------------------
# Get the field
field = ss.simcom.Fields()
if use_opt1:
    # Load the geometry
    Geometry = ss.simcom.Geometry(geomID, code='SINPA')
    u1 = np.array(Geometry.ExtraGeometryParams['u1'])
    u2 = np.array(Geometry.ExtraGeometryParams['u2'])
    u3 = np.array(Geometry.ExtraGeometryParams['u3'])
    # Get the field
    field.createHomogeneousFieldThetaPhi(theta, phi, field='B',
                                         u1=u1, u2=u2, u3=u3)
else:
    field.createHomogeneousField(direction, field='B')
# Write the field
fieldFileName = os.path.join(inputsDir, 'field.bin')
field.tofile(fieldFileName)

# -----------------------------------------------------------------------------
# --- Section 2: Run the code
# -----------------------------------------------------------------------------
# Check the files
ss.sinpa.execution.check_files(nml_options['config']['runid'])
# Launch the simulations
ss.sinpa.execution.executeRun(nml_options['config']['runid'])
