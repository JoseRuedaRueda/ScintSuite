"""
Example to calculate the energy resolution from INPA

Jose Rueda

Actually, this generate the data for one of the plots of my thesis :)

This does not include optics effect, just pure geometry of collimator
"""
import os
import numpy as np
import xarray as xr
import ScintSuite as ss
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
# ----------------------------------------------------------------------------
# %% Settings
# ----------------------------------------------------------------------------
BasicRunID = 'ThesisEResolution'
thetaAngle = np.linspace(85, 95, 11)
phiAngle = np.linspace(175, 185, 11)
Bmod = np.linspace(1.5, 3.0, 16)
runSINPA = False
overwriteSINPA = False
nml_options = {   # See the PDF documentation for a complete desription of
    'config':  {  # these parameters
        'runid': 'Kiwi',
        'geomFolder': '/tokp/work/ruejo/Programs/SINPA/Geometry/iAUG01',
        'FILDSIMmode': False,
        'nxi': 15,
        'nGyroradius': 12,
        'nMap': 7500,
        'saveOrbits': False,
        'saveRatio': 0.01,
        'saveOrbitLongMode': False,
        'runfolder': 'kiwi',
        'verbose': True,
        'M': 2.01410178,
        'Zin': 0.0,
        'Zout': 1.0,
        'IpBt': -1,        # Sign of toroidal current vs field (for pitch)
        'flag_efield_on': False,  # Add or not electric field
        'save_collimator_strike_points': False,  # Save collimator points
        'save_wrong_markers_position': False,
        'backtrace': False,  # Flag to backtrace the orbits
        'FoilElossModel': 0,
        'ScintillatorYieldModel': 2,
        'FoilYieldModel': 2,
        'restrict_mode': False,
        'resampling': True,
        'nResampling': 20,
        'mapping': True,
        'signal': False,
        'FIDASIMfolder': 'kiwi',
    },
    'inputParams': {
         'nGyro': 100,
         'minAngle': -0.15,
         'dAngle': 0.3,
         'XI': [2.90755772, 2.97122936, 3.03318827, 3.0937353, 3.15114963,
                3.20757287, 3.26024673, 3.31162787, 3.35903163, 3.40568768,
                3.44758373, 3.48947979, 3.53137584, 3.5732719, 3.61516795],
         'rL': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75],
         'maxT': 0.0000001
    },
    'markerinteractions': {
         # 'FoilInteractionParameters': [1.0]
         # 'FoilInteractionParameters': [1.0, 0.044, 20]
         # 'FoilElossParameters': [1.0, 0.1778, -5.146],
         'ScintillatorYieldParameters': [92.98, 0.9115],
         'FoilYieldParameters': [1.0, 0.825, 0.771, 0.0202],
    },
    'nbi_namelist': {            # NBI geometry
        'p0': [2.2078, -1.3732, -0.021],  # xyz of first point in the NBI
        'u': [-0.6013878,  0.79444944,  0.08475143],  # vector of the NBI
    },
}
rootSINPAfolder = '/tokp/work/ruejo/SINPAresults/ThesisScan'
executable = '/tokp/work/ruejo/Programs/SINPA/SINPA.go'
saveFile = '/tokp/work/ruejo/SINPAresults/ThesisScan/ResolutionsRScan.nc'
# Reference point
Eref = 60.0
Rref = [1.60, 1.70, 1.80, 1.90, 2.00]
# ----------------------------------------------------------------------------
# %% run SINPA
# ----------------------------------------------------------------------------
if runSINPA:
    for itheta, theta in enumerate(thetaAngle):
        for iphi, phi in enumerate(phiAngle):
            runid = BasicRunID + 't%.1f_p%.1f'%(theta, phi)
            # -----------------------------------------------------------------
            # ---- Section 0: Create the directories
            # -------------------------------------------------------------------------
            nml_options['config']['runid'] = runid
            runDir = os.path.join(rootSINPAfolder, nml_options['config']['runid'])
            strikeFile = os.path.join(runDir, 'results', nml_options['config']['runid']
                                    + '.spsignal')
            if os.path.isfile(strikeFile) and not overwriteSINPA:
                print('%s runid already saved, skipping' % runid)
                continue

            # fidasimPath = os.path.join(rootFIDASIMfolder, runid[:5], runid)
            nml_options['config']['runfolder'] = runDir
            inputsDir = os.path.join(runDir, 'inputs')
            resultsDir = os.path.join(runDir, 'results')
            os.makedirs(runDir, exist_ok=True)
            os.makedirs(inputsDir, exist_ok=True)
            os.makedirs(resultsDir, exist_ok=True)
            # -------------------------------------------------------------------------
            # %% Section 1: Prepare the namelist
            # -------------------------------------------------------------------------
            namelist_file = ss.sinpa.execution.write_namelist(nml_options)
            # -------------------------------------------------------------------------
            # %% Section 2: Prepare the magnetic field
            # -------------------------------------------------------------------------
            # Get the field
            # Geometry = ss.simcom.Geometry(geomID, code='SINPA')
            u1 = np.array([0.027857,  0.999567, -0.009725])
            u2 = np.array([-0.816590,  0.028300,  0.576519])
            u3 = np.array([0.576545, -0.008118,  0.817025])
            # rPinXYZ = Geometry.ExtraGeometryParams['rPin']
            # Rpin = Geometry.ExtraGeometryParams['r_scintillator'] #np.sqrt(rPinXYZ[0]**2 +
            # #                                                              rPinXYZ[1]**2)
            field = ss.simcom.Fields()
            field.createHomogeneousFieldThetaPhi(theta, phi, field='B',
                                                u1=u1, u2=u2, u3=u3,
                                                verbose=False,
                                                diagnostic='INPA',
                                                field_mod=1.8)
            # Write the field
            fieldFileName = os.path.join(inputsDir, 'field.bin')
            field.tofile(fieldFileName)
            # -------------------------------------------------------------------------
            # %% Section 2: Run the code
            # -------------------------------------------------------------------------
            # Check the files
            # ss.sinpa.execution.check_files(nml_options['config']['runid'])
            # # Launch the simulations
            # ss.sinpa.execution.executeRun(nml_options['config']['runid'])
            os.system('%s %s' % (executable, namelist_file))
# ----------------------------------------------------------------------------
# %% Load SINPA
# ----------------------------------------------------------------------------

sigmaE = xr.DataArray(
    np.zeros((Bmod.size, thetaAngle.size, phiAngle.size, len(Rref))),
    dims=('B', 'theta', 'phi', 'R'), 
    coords={'B':Bmod, 'theta': thetaAngle, 'phi':phiAngle, 'R':Rref})
for itheta, theta in enumerate(thetaAngle):
    print('------- We are on theta: %i'%itheta)
    for iphi, phi in enumerate(phiAngle):
        runid = BasicRunID + 't%.1f_p%.1f'%(theta, phi)
        # -----------------------------------------------------------------
        # ---- Section 0: Create the directories
        # -----------------------------------------------------------------
        runDir = os.path.join(rootSINPAfolder, runid)
        smapFile = os.path.join(runDir, 'results', runid + '.map')

        for iB, B in enumerate(Bmod):
            # Load the strike map
            smap = ss.smap.Ismap(smapFile)

            # Set the energy
            smap.calculate_energy(B)
            smap.setRemapVariables(('R0', 'e0'))

            smap.load_strike_points()

            smap.calculate_phase_space_resolution(
                diag_params={'x_method': 'Gauss', 'y_method': 'Gauss'})
            # Get the gyroradius
            gyrorad = ss.fildsim.get_gyroradius(Eref*1000, B)
            for iR, R in enumerate(Rref):
                sigmaE[iB, itheta, iphi, iR] = \
                    smap._interpolators_instrument_function['e0']['sigma'](R,
                                                                        gyrorad)
# Set the atributes
sigmaE.attrs['Eref'] = Eref
sigmaE.attrs['Rref'] = Rref
sigmaE.to_netcdf(saveFile)
# %% Small plot
fig, ax = plt.subplots(1, len(Rref), sharey=True, sharex=True)
for iR, R in enumerate(Rref):
    sigmaE.sel(R=R, B=2.3, method='nearest').plot.imshow(ax=ax[iR],interpolation='bicubic')
