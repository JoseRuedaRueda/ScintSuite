"""
Script to calculate and export FILD weight function


Done for Suite Commit: 7677c5397a125262d56ade69b681aa61defd3de4
"""
import os
import numpy as np
import ScintSuite as ss

# ----------------------------------------------------------------------------
# %% Settigns
# ----------------------------------------------------------------------------
shot = 39573
time = 2.000
# Namelist for the SINPA run
runid = 'dummy'  # Will be overwritten latter
geomID = 'AUG02'
nml_options = {   # Se the PDF documentation for a complete desription of these
    'config':  {  # parameters
        'runid': runid,
        'geomfolder': os.path.join(ss.paths.SINPA, 'Geometry', geomID),
        'FILDSIMmode': True,
        'nxi': 8,
        'nGyroradius': 13,
        'nMap': 800000,
        'n1': 1.0,
        'r1': 1.2,
        'restrict_mode': False,
        'mapping': True,
        'saveOrbits': False,
        'saveRatio': 0.01,
        'saveOrbitLongMode': False,
        'runfolder': os.path.join(ss.paths.SINPA, 'runs', runid),
        'verbose': True,
        'IpBt': -1,             # Sign of toroidal current vs field (for pitch)
        'flag_efield_on': False,  # Add or not electric field
        'save_collimator_strike_points': False,  # Save collimator points
        'backtrace': False  # Flag to backtrace the orbits
    },
    'inputParams': {
         'nGyro': 350,
         'minAngle': -1.7,
         'dAngle': 0.55,
         'XI': [85.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0],
         'rL': [1.75, 2.0, 2.50, 3.0, 4.5, 5.0, 5.5, 6.0, 7.50, 8.0, 8.50, 9.0,
                9.50],
         'maxT': 0.00000008
    },
}
overwriteSINPAsim = False
# magnetic field definition:
# Option 1: as FILDSIM, given by the 2 orientation angles
use_opt1 = True
# --- Grid for the weight function
pin_options = {
    'xmin': 20,
    'xmax': 85,
    'dx': 1.0,
    'ymin': 1.2,
    'ymax': 8.0,
    'dy': 0.1,
}
scint_options = {
    'xmin': 20,
    'xmax': 85,
    'dx': 1.0,
    'ymin': 1.2,
    'ymax': 8.0,
    'dy': 0.1,
}

# ----------------------------------------------------------------------------
# %% Load the video
# ---------------------------------------------------------------------------
# Load the video and some frames as a quick way to load the angles
vid = ss.vid.FILDVideo(shot=shot, diag_ID=1)
vid.read_frame(t1=time-0.1, t2=time+0.1)
vid._getB()
# ----------------------------------------------------------------------------
# %% Launch a SINPA run
# ----------------------------------------------------------------------------
# Get the angles
phi, theta = vid.calculateBangles(t=time)
phi = np.array(phi).squeeze()      # To get rid of the SOBJ
theta = np.array(theta).squeeze()
# Get the runID
runID = 'W_%4.2f_%4.2f_%s' % (phi, theta, geomID)
# Save it in the namelist
runfolder = os.path.join(ss.paths.SINPA, 'runs', runID)
inputsDir = os.path.join(runfolder, 'inputs')
resultsDir = os.path.join(runfolder, 'results')
nml_options['config']['runfolder'] = runfolder
nml_options['config']['runid'] = runID
# see if the simulation is present
if not os.path.isdir(runfolder) or overwriteSINPAsim:
    # Create the directories
    os.makedirs(runfolder, exist_ok=True)
    os.makedirs(inputsDir, exist_ok=True)
    os.makedirs(resultsDir, exist_ok=True)
    # Create the namelist
    filename = ss.sinpa.execution.write_namelist(nml_options)
    field = ss.simcom.Fields()
    if use_opt1:
        # Create the magnetic field
        Geometry = ss.simcom.Geometry(geomID, code='SINPA')
        u1 = np.array(Geometry.ExtraGeometryParams['u1'])
        u2 = np.array(Geometry.ExtraGeometryParams['u2'])
        u3 = np.array(Geometry.ExtraGeometryParams['u3'])
        # Get the field
        field.createHomogeneousFieldThetaPhi(theta, phi, field='B',
                                             u1=u1, u2=u2, u3=u3)
    else:
        raise Exception('Not implemented in this script')
    # Write the field
    fieldFileName = os.path.join(inputsDir, 'field.bin')
    field.tofile(fieldFileName)
    # Check the files
    ss.sinpa.execution.check_files(nml_options['config']['runid'])
    # Launch the simulations
    ss.sinpa.execution.executeRun(nml_options['config']['runid'])
# ------------------------------------------------------------------------------
# %% Load the map and the strike points
# ------------------------------------------------------------------------------
mapName = os.path.join(resultsDir, '%s.map' % runID)
smap = ss.smap.Fsmap(file=mapName)
smap.load_strike_points()
# Load the scintillator efficiency
scintillator = ss.scint.Scintillator()
# ----------------------------------------------------------------------------
# %% Create the weight function
# ----------------------------------------------------------------------------
B = vid.BField.B.sel(t=time, method='nearest').values
smap.build_weight_matrix(scint_options, pin_options,
                         efficiency=scintillator.efficiency,
                         B=B)

# %% Save the W
print('Saving files')
smap.instrument_function.to_netcdf('W_%i_%4.3f.nc' % (shot, time))
