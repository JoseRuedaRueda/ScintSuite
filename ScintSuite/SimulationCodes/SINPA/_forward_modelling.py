"""
Alex Reyner: alereyvinn@alum.us.es

Workflow:
    0. Obtain the distribution and define the inputs: smap, scint, WF...
    1. Run synthsig_xy to map the signal in the scintillator space
    2. Insert noises, optic system and the camera with noise_optics_camera

    - You can plot at any step given the frame and even plot each noise
    - You can also directly compute the WF and remapped synthetic signals

Functions. What can be done:
    - read_distribution: Read the ion distribution that will be used as input
    - obtain_WF: Obtain the weight function of the smap
    - synthsig_pr: Compute remapped synthetic signal in pitch-gyroradius 
    - pr_space_to_pe_space: transform the the signal phase space
    - original_synthsig_xy: compute synthetic signal in real scintillator space
    - noise_optics_camera: add noise and optic effects to the synthetic signal
    - plot_the_frame: plot the signal at a given point
    - plot_noise_contributions: plot the different noise contributions
    - synthsig_xy: compute synthetic signal in real scintillator space
    - synthsig_xy_2coll: compute synthetic signal of two pinholes
    - plot_the_frame_2coll: plot the signal
"""

import ScintSuite._Mapping as ssmapplting
from ScintSuite.SimulationCodes.FILDSIM.execution import get_energy
from ScintSuite.SimulationCodes.FILDSIM.execution import get_gyroradius
import ScintSuite.SimulationCodes.FILDSIM.forwardModelling as ssfM
import ScintSuite.SimulationCodes.Common.geometry as geometry
import ScintSuite._Plotting as ssplt
from ScintSuite._Plotting._ColorMaps import default_cmap
import ScintSuite._Plotting as ssplt
import ScintSuite._Mapping._Common as common
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

import numpy as np
import xarray as xr
import scipy.ndimage as spnd
import math
import copy
import sys

import logging
logger = logging.getLogger('ScintSuite.FwdMod')
logging.basicConfig(level=logging.INFO)
import time


# -----------------------------------------------------------------------------
# --- Inputs distributions
# -----------------------------------------------------------------------------

def read_distribution(filename, pinhole_area = None, wetted_area = None,
                    B = 4, A = 2, Z = 2, version='5.5'):
    """
    Read a distribution coming from ASCOT

    Alex Reyner: alereyvinn@alum.us.es

    Each version has a different type of input:
        - 5.5 with pitch in [VII/V] units
        - Custom read for matlab antique files (matlab)
        - Manual file for delta tipe ion flux

    :param  filename: full path to the file
    :param  pinhole_area: pinhole area in mm²
    :param  wetted_area: wetted area in mm²
    :param  B: magnetic field
    :param  A: A value of the ions (in case this does not come with the input)
    :param  Z: Z value of the ions (in case this does not come with the input)
    :param  version: ASCOT output custom, default 5.5

    :return out dictionary containing:
            'gyroradius': Array of gyroradius where the signal is evaluated
            'pitch': Array of pitches where the signal is evaluated
            'weight': Array of weights where the signal is evaluated
            other interesting parameters from the ascot files
    """
    logger.info("----- READING DISTRIBUTION ----- ")
    logger.info('Reading file: %s', filename)

    if pinhole_area == None or wetted_area == None:
        logger.error('Missing pinhole_area and/or wetted_area in input') 
        sys.exit()   
    else:
        logger.info("Wetted area: %.2f (mm²)", wetted_area)
        logger.info("Pinhole area: %.2f (mm²)", pinhole_area)
    
    out={}

    if version == '5.5':
        names = ['R', 'phi', 'Z', 'energy', 'pitch', 
                 'Anum', 'Znum', 'weight', 'time']

        # FILE PREPARATION
        with open(filename, 'r') as file:
                lines = file.readlines()[2:]
        modified_lines = []       
        for line in lines:
            if line.startswith('#'): #skips headers
                continue
            else:
                c = line.split()
                # get pitch in degree
                c[4] = math.acos(float(c[4]))*180.0/math.pi
                
                modified_line = f"{c[0]} {c[1]} {c[2]} {c[3]} \
                    {c[4]} {c[5]} {c[6]} {c[7]} {c[8]} "
                modified_lines.append(modified_line)    

    
    if version == 'matlab':
        names = ['pitch', 'energy', 'weight']
        
        with open(filename, 'r') as file:
                lines = file.readlines()
        modified_lines = []       
        for line in lines:
            if line.startswith('#'): #skips headers
                continue
            else:
                c = line.split()
                c[0] = math.acos(float(c[0]))*180.0/math.pi
                
                modified_line = \
                    f"{c[0]} {c[1]} {c[2]}"
                modified_lines.append(modified_line)    


    if version == 'manual':
        names = ['energy', 'pitch', 'Anum', 'Znum', 'weight']
        
        # FILE PREPARATION
        with open(filename, 'r') as file:
                lines = file.readlines()
        modified_lines = []        
        for line in lines:
            if line.startswith('#'): #skips headers
                continue
            else:                             
                c = line.split()
                modified_line = f"{c[0]} {c[1]} {c[2]} {c[3]} {c[4]}"
                modified_lines.append(modified_line)    

    if version == 'locust':
        names = ['R', 'z', 'phi', 'vR', 'vZ', 'vphi', 'pitch', 'energy', 
                 'rho_Larmor', 'weight', 'gyrophase', 'ID_FILD']
        
        if A==None or Z==None:
            logger.error('No A and/or B as input. STOPING')      
            sys.exit()   

        # FILE PREPARATION
        with open(filename, 'r') as file:
                lines = file.readlines()
        modified_lines = []        
        for line in lines:
            if line.startswith('#'): #skips headers
                continue
            else:
                c = line.split()
                # get pitch angle
                c[6] = math.acos(float(c[6]))*180.0/math.pi
                # get energy in eV
                c[7] = float(c[7])*1e6
                                
                modified_line = f"{c[0]} {c[1]} {c[2]} {c[3]} {c[4]} {c[5]} \
                      {c[6]} {c[7]} {c[8]} {c[9]} {c[10]} {c[11]}"
                modified_lines.append(modified_line)    

    # BUILD OUTPUT
    # -----------------------------------------------------------------------
    filename2=filename[:-4]+'_procesed.dat'
    with open(filename2, 'w') as file2:
        file2.write('\n'.join(modified_lines))

    # Load the data of this second file
    data = np.loadtxt(filename2)
    for i in range(len(names)):
        out[names[i]] = data[:, i]
    out['n'] = len(data[:, 0])

    # Fill missing arguments if necessary
    if 'Anum' not in out:
        out['Anum'] = np.full(out['n'], A)
    if 'Znum' not in out:
        out['Znum'] = np.full(out['n'], Z)
    if 'B' not in out:
        out['B'] = np.full(out['n'], B)
    # Compute gyroradius
    vect_get_gyroradius = np.vectorize(get_gyroradius)
    out['gyroradius'] = vect_get_gyroradius(out['energy'], 
                                            out['B'], out['Anum'], out['Znum'], 
                                            relativistic=True)
    # Adjust marker weight to strict pinhole area
    out['weight'] = out['weight'] / wetted_area * pinhole_area
    # Calculate the power by each marker in the pinhole
    out['power'] = out['weight'] * out['energy'] * 1.602176634e-19 # eV to J

    ion_flux = np.sum(out['weight']) / pinhole_area # per unit of area
    ion_power = np.sum(out['power']) / pinhole_area # per unit of area
    ions_head = ion_flux * wetted_area
    ions_pinhole = ion_flux * pinhole_area

    logger.info("Ion den flux   -> %e (ions/s/m²)", ion_flux*1e6) # go to /m²
    logger.info("Power den flux -> %e (W/m²)", ion_power*1e6) # go to /m²
    logger.info("Wetted flux    -> %e (ions/s)", ions_head)
    logger.info("Pinhole flux   -> %e (ions/s)", ions_pinhole)

    return out
    

# -----------------------------------------------------------------------------
# --- Synthetic signals using the weight matrix
# -----------------------------------------------------------------------------

def obtain_WF(smap, pin_params: dict = {}, scint_params: dict = {},
              efficiency_flag = False, scintillator = None, B=4, A=4, Z=2):
    '''
    Just a wrap of things to make it easier
    Efficency will be applied when generating the images

    Alex Reyner: alereyvinn@alum.us.es
    '''
    # Load the strike points
    smap.load_strike_points()
    # --- Grid for the weight function
    pin_options = {
        'xmin': 20, 'xmax': 90, 'dx': 1,
        'ymin': 1.5, 'ymax': 12, 'dy': 0.2,
        }
    scint_options = {
        'xmin': 20, 'xmax': 90, 'dx': 0.25,
        'ymin': 1, 'ymax': 12, 'dy': 0.1,
        }
    # update the matrix options
    pin_options.update(pin_params)
    scint_options.update(scint_params)

    # Build the weight function 
    if efficiency_flag == True and scintillator is not None:
        logger.info('Efficency considered in the computation of the WF')
        smap.build_weight_matrix(scint_options, pin_options,
                                efficiency=scintillator.efficiency,
                                B=B,A=A,Z=Z)
    else:
        smap.build_weight_matrix(scint_options, pin_options,
                                B=B,A=A,Z=Z)
    WF = smap.instrument_function

    smap.build_weight_matrix(scint_options, pin_options,)
    WF = smap.instrument_function
    WF.x.attrs = {'units': 'º', 'long_name': 'Pitch'}
    WF.xs.attrs = {'units': 'º', 'long_name': 'Pitch'}
    WF.y.attrs = {'units': 'cm', 'long_name': 'Gyroradius'}
    WF.ys.attrs = {'units': 'cm', 'long_name': 'Gyroradius'}

    return WF


def synthsig_pr(distro, scint, WF, 
                        gyrophases = np.pi, mode = 'photons', 
                        plot=False, cmap=default_cmap()):
    """
    Synthetic signal for pinhole and scintillator in pitch-gyroradius space

    Alex Reyner: alereyvinn@alum.us.es

    :param  distro: pinhole distribution, created by read_distribution()
    :param  WF: weight function xarray generated by obtain_WF()
    :param  gyrophases: used to renormalize the collimator factor. 
                Range of gyrophases that we consider. Default pi 
                -> range of gyrophases pointing into the pinhole
    :param  mode: select what quantity you want
                - photons: includes scintillator response (default)
                - ions: ion flux in the scintillator
                - power: power flux deposited in the scintillator
    :param  plot: plot the synthetic signals and histograms
    :param  cmap: colormap for plots

    :return out dictionary containing remapped signals:
            PH: synthetic signal at the pinhole (PH)
            SC: synthetic signal at the scintillator (SC)
    """
    logger.info('----- COMPUTING REMAPED SYNTHETIC SIGNAL USING WF -----')
    start = time.perf_counter()

    # INPUT VERIFICATION
    # -----------------------------------------------------------------------
    pitch = distro['pitch']
    gyro = distro['gyroradius']
    energy = distro['energy']
    B = distro['B']
    Anum = distro['Anum']
    Znum = distro['Znum']

    if mode == 'photons':
        if 'energy' in distro.keys() or 'e0' in distro.keys():
            eff = scint.efficiency(energy/1e3).values
        elif 'gyroradius' in distro.keys():
            energy = get_energy(gyro, B, Anum, Znum)
            eff = scint.efficiency(energy/1e3).values
        else:
            logger.error('NOT POSSIBLE TO EXTRACT EFFICENCY') 
            sys.exit() 
        weight = distro['weight']
        cbar_units = 'photons / s cm º'

    elif mode == 'ions': # to see the flux of ions in the scintillator
        eff = np.ones(distro['n']) # eficency is 1
        weight = distro['weight']
        cbar_units = 'ions / s cm º'

    elif mode == 'power': # to compute the deposited power (beta feature)
        eff = np.ones(distro['n']) # eficency is 1
        weight = distro['power']
        cbar_units = 'W / m º'

    else:
        logger.error('Wrong mode, select either: photons, ions or power')

    # VECTORIZED MAPPING
    # -----------------------------------------------------------------------
    x_val = WF.coords['x'].values
    y_val = WF.coords['y'].values
    nx, ny = len(x_val), len(y_val)
    # Remove markers outside of WF
    mask = ((pitch >= x_val.min()) & (pitch <= x_val.max()) &
            (gyro >= y_val.min()) & (gyro <= y_val.max()))
    pitch, gyro, weight, eff = pitch[mask], gyro[mask], weight[mask], eff[mask]
    # Calculate steps
    p_step = np.abs(x_val[1]-x_val[0])
    r_step = np.abs(y_val[1]-y_val[0])
    # Look for the correct indices place for each point
    p_idx = np.searchsorted(x_val, pitch)
    p_idx = np.clip(p_idx, 0, nx-1)
    r_idx = np.searchsorted(y_val, gyro)
    r_idx = np.clip(r_idx, 0, ny-1)
    # Build the weight matrix
    w_matrix = np.zeros((ny, nx))
    np.add.at(w_matrix, (r_idx, p_idx), weight*eff) # fill the weights
    w_xrarray = xr.DataArray(w_matrix,
                    coords={'y': y_val, 'x': x_val},
                    dims=('y', 'x'))
    # Compute the matrices in the pinhole and scintillator vel.-spaces
    ssPH = w_xrarray /p_step /r_step
    ssSC = ((WF*w_xrarray) * (2*np.pi/gyrophases)).sum({'x','y'})
    # Put data into Dataset and assign atributes
    synthetic_signal = xr.Dataset()
    ssPH.attrs = {'long_name': cbar_units,}
    ssSC.attrs = {'long_name': cbar_units,}
    synthetic_signal['PH'] = ssPH
    synthetic_signal['SC'] = ssSC


    end = time.perf_counter()    # end timer
    elapsed = end - start
    logger.info("   Mapping time = %.4f s", elapsed)

    # PLOTTING
    # -----------------------------------------------------------------------
    if plot == True:
        fig, ax = plt.subplots(2,2, figsize=(8, 6),
                                    facecolor='w', edgecolor='k') 
        # Plot of the synthetic signals, pinhole and scintillator
        ax_param = {'xlabel': 'Pitch [º]', 'ylabel': 'Gyroradius [cm]'}         
        ssPH.transpose().plot.imshow(ax=ax[0,0],cmap=cmap,
                                     vmax=0.5*ssPH.max().item(),
                                     cbar_kwargs={"label": 'ions/(s cm deg)'})
        ax[0,0] = ssplt.axis_beauty(ax[0,0], ax_param)
        ax[0,0].set_title("Pinhole")    
        ssSC.transpose().plot.imshow(ax=ax[0,1], cmap=cmap,
                                     vmax=0.5*ssSC.max().item(),
                                     cbar_kwargs={"label": 'ions/(s cm deg)'})
        ax[0,1] = ssplt.axis_beauty(ax[0,1], ax_param)
        ax[0,1].set_title("Scintillator")

        # Plot of the distributions of pitch and gyroradius
        ax_options_profiles = {'ylabel': 'Signal [a.u.]'}
        (ssPH.sum(dim='y')/ssPH.sum(dim='y').integrate('x')).plot.\
            line(ax=ax[1,0], color='black', label='Pinhole')
        (ssSC.sum(dim='ys')/ssSC.sum(dim='ys').integrate('xs')).plot.\
            line(ax=ax[1,0], color='red', label='Scintillator')
        ax_options_profiles['xlabel'] = 'Pitch [$\\degree$]'  
        ax[1,0] = ssplt.axis_beauty(ax[1,0], ax_options_profiles)
        ax[1,0].legend()        
        (ssPH.sum(dim='x')/ssPH.sum(dim='x').integrate('y')).plot.\
            line(ax=ax[1,1], color='black', label='Pinhole')
        (ssSC.sum(dim='xs')/ssSC.sum(dim='xs').integrate('ys')).plot.\
            line(ax=ax[1,1], color='red', label='Scintillator')      
        ax_options_profiles['xlabel'] = 'Gyroradius [cm]'
        ax[1,1] = ssplt.axis_beauty(ax[1,1], ax_options_profiles)
        ax[1,1].legend()

        fig.tight_layout()
        plt.show()

    return synthetic_signal


def pr_space_to_pe_space(synthetic_signal, B=4, A=2, Z=2, 
                         plot=False, cmap = default_cmap()):
    """
    Transfors the pitch-gyroradius signal to pitch-energy signal

    Alex Reyner: alereyvinn@alum.us.es

    :param  synthetic_signal: xarrays with the signal in the pr space. 
        This must be one of the synthetic signals xarrays produced by this 
        suite, with:
                x-> pitch in the pinhole
                y-> gyroradius in the pinhole
                xs-> pitch in the scintillator
                ys-> gyroradius in the scintillator
    :param  B: Magnetic field (to translate from r to Energy)
    :param  A: Mass, in amu, (to translate from r to Energy)
    :param  Z: Charge in e units (to translate from r to Energy)

    :return out dictionary containing remapped signals in the energy space:
            PH: synthetic signal at the pinhole (PH)
            SC: synthetic signal at the scintillator (SC)
    """
    logger.info('----- GOING FROM p-r TO p-e SPACE ----- ')
    # Synthetic signal input
    ssPH_pr = synthetic_signal['PH']
    ssSC_pr = synthetic_signal['SC']
    # Replicate the xarray.
    # Necessary to multiply by one, to "break" the relation between matrices.
    ssPH_pe = copy.deepcopy(ssPH_pr)
    ssSC_pe = copy.deepcopy(ssSC_pr)
    # Get the coordinates of the gyroradius and transform them to energy.
    gyroradius = ssPH_pe.coords['y'].values
    energy = get_energy(gyroradius,B=B,A=A,Z=Z)
    ssPH_pe['y'] = energy #change coordinates from gyroradius to energy
    # Multiply each point in the distribution by the Jacobian, to mantain the
    # integral of the signal with the same value.
    for j in range(len(energy)):
        ssPH_pe[:,j] = ssPH_pe[:,j] *gyroradius[j]/(2*energy[j])
    # Last step: interpolate the energy matrix so the indices are equally 
    # spaced.        
    e_interp=np.linspace(energy.min(),energy.max(),len(energy))
    ssPH_pe = ssPH_pe.interp(y=e_interp, method='cubic')
    # Repeat for the scintillator image
    gyroradius = ssSC_pe.coords['ys'].values
    energy = get_energy(gyroradius,B=B,A=A,Z=Z)
    ssSC_pe['ys'] = energy 
    for j in range(len(energy)):
        ssSC_pe[:,j] = ssSC_pe[:,j] *gyroradius[j]/(2*energy[j])   
    e_interp=np.linspace(energy.min(),energy.max(),len(energy))
    ssSC_pe = ssSC_pe.interp(ys=e_interp, method='cubic')
    # We don't want <0 values
    ssPH_pe = ssPH_pe.where(ssPH_pe >=0.0, 0)
    ssSC_pe = ssSC_pe.where(ssSC_pe >=0.0, 0)
    # Just a little adjustment
    integral_s = ssSC_pr.integrate('xs').integrate('ys').item()
    integral_s_e = ssSC_pe.integrate('xs').integrate('ys').item()
    out = {}
    out['PH'] = ssPH_pe
    out['SC'] = ssSC_pe/integral_s_e*integral_s

    if plot == True:
        fig, ax = plt.subplots(2,2, figsize=(8, 6),
                                    facecolor='w', edgecolor='k')
        # Plot of the synthetic signals, pinhole and scintillator
        ax_param = {'xlabel': 'Pitch [º]', 'ylabel': 'Energy [eV]'}         
        ssPH_pe.transpose().plot.imshow(ax=ax[0,0], cmap=cmap,
                                     vmax=0.5*ssPH_pe.max().item(),
                                     cbar_kwargs={"label": 'ions/(s cm deg)'})
        ax[0,0] = ssplt.axis_beauty(ax[0,0], ax_param)
        ax[0,0].set_title("Pinhole")    
        ssSC_pe.transpose().plot.imshow(ax=ax[0,1], cmap=cmap,
                                     vmax=0.5*ssPH_pe.max().item(),
                                     cbar_kwargs={"label": 'ions/(s cm deg)'})
        ax[0,1] = ssplt.axis_beauty(ax[0,1], ax_param)
        ax[0,1].set_title("Scintillator")

        # Plot of the distributions of pitch and gyroradius
        ax_options_profiles = {'ylabel': 'Signal [a.u.]'}
        (ssPH_pe.sum(dim='y')/ssPH_pe.sum(dim='y').integrate('x'))\
            .plot.line(ax=ax[1,0], color='black', label='Pinhole')
        (ssSC_pe.sum(dim='ys')/ssSC_pe.sum(dim='ys').integrate('xs'))\
            .plot.line(ax=ax[1,0], color='red', label='Scintillator')
        ax_options_profiles['xlabel'] = 'Pitch [$\\degree$]'  
        ax[1,0] = ssplt.axis_beauty(ax[1,0], ax_options_profiles)
        ax[1,0].legend()        
        (ssPH_pe.sum(dim='x')/ssPH_pe.sum(dim='x').integrate('y'))\
            .plot.line(ax=ax[1,1], color='black', label='Pinhole')
        (ssSC_pe.sum(dim='xs')/ssSC_pe.sum(dim='xs').integrate('ys'))\
            .plot.line(ax=ax[1,1], color='red', label='Scintillator')      
        ax_options_profiles['xlabel'] = 'Energy [eV]'
        ax[1,1] = ssplt.axis_beauty(ax[1,1], ax_options_profiles)
        ax[1,1].legend()

        fig.tight_layout()
        plt.show()

    return out


# -----------------------------------------------------------------------------
# --- Synthetic signal in the scintillator space and camera frame
# -----------------------------------------------------------------------------

def original_synthsig_xy(distro, smap, scint, collimator=None,
                     cam_params = {}, optic_params = {},
                     smapplt = None, 
                     gyrophases = np.pi,
                     smoother = None,
                     scint_params: dict = {}, centering = False,
                     px_shift: int = 0, py_shift: int = 0,
                     mode = 'photons',
                     **kwargs):
    """
    Maps a signal in the scintillator

    Alex Reyner: alereyvinn@alum.us.es

    Based on the origianl function by Jose Rueda    

    :param  distro: distribution in the pinhole
    :param  smap: smap to map the signal in the xyspace
    :param  smapplt: extra smap to do nice plots
    :param  scint: scintillator shape we want in the plots
    :param  collimator: add the collimator geometry to the plot (experimental)
    :param  alex: flagg to use the experimental collimator plotting
    :param  cam_params: parameters of the camera
    :param  optic_params:  parameters of the optics
    :param  gyrophases: range of gyrophases considered entering the pinhole
            (to scale the collimator factor). pi (half sr) is the default.
            Usually the collimator factor is defined over a 2pi range
    :param  smoother: adds a gaussian filter to the signal with that sigma
    :param  scint_synthetic_signal_params: grid to remap the frame

    :kwarg  eff: deactivate the scintillator efficency with None

    :return out dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
    """
    logger.info("----- BUILDING THE ORIGINAL FRAME -----")

    # Check inputs and initialise the things
    dsmap = copy.deepcopy(smap)
    if smapplt != None:
        dsmapplt = copy.deepcopy(smapplt)
    else:
        dsmapplt = copy.deepcopy(dsmap)
    dscint = copy.deepcopy(scint)
    if mode == 'photons': # the usual, photons after scintillator response
        efficiency = dscint.efficiency
        cbar_units = 'photons / s m²'
    elif mode == 'ions': # to see the flux of ions in the scintillator
        efficiency = None
        cbar_units = 'ions / s m²'
    elif mode == 'power': # to compute the deposited power (beta featura)
        efficiency = None
        cbar_units = 'W / m²'
        distro['weight'] = distro['power']
    else:
        logger.error('Wrong mode, select either: photons, ions or power')

    scint_options = {
        'rmin': 1,
        'rmax': 10.0,
        'dr': 0.1,
        'pmin': 5.0,
        'pmax': 90.0,
        'dp': 1,
    }
    scint_options.update(scint_params)    

    # SYNTHETIC SIGNAL
    # -----------------------------------------------------------------------
    logger.info('- Computing synthetic signal...')
    # Calculate the synthetic signal at the scintillator
    scint_signal = ssfM.synthetic_signal_remap(distro, dsmap,
                                          efficiency=efficiency,
                                          **scint_options)

    # LOCATE AND CENTER THE SCINTILLATOR AND SMAP
    # -----------------------------------------------------------------------
    logger.info('- Locating the smap and scintillator...')
    # Find the center of the camera frame 
    px_center = int(cam_params['nx'] / 2)
    py_center = int(cam_params['ny'] / 2)
    if 'beta' in optic_params:
        beta = optic_params['beta']
        logger.info("   Computed beta: %e", beta)
    else:
        xsize = cam_params['px_x_size'] * cam_params['nx']
        ysize = cam_params['px_y_size'] * cam_params['ny']
        chip_min_length = np.minimum(xsize, ysize)
        xscint_size = scint._coord_real['x1'].max() \
            - scint._coord_real['x1'].min()
        yscint_size = scint._coord_real['x2'].max() \
            - scint._coord_real['x2'].min()
        scintillator_max_length = np.maximum(xscint_size, yscint_size)
        beta = chip_min_length / scintillator_max_length
        logger.info('   Optics magnification, beta: %e', beta)
        optic_params['beta'] = beta
    
    if centering:
        # Center image to FoV
        xsc_percent = optic_params['FoV'][0]
        ysc_percent = optic_params['FoV'][1]
        xsc_min = scint._coord_real['x1'].min()
        xsc_max = scint._coord_real['x1'].max()
        ysc_min = scint._coord_real['x2'].min()
        ysc_max = scint._coord_real['x2'].max()
        x_scint_center = (xsc_max - xsc_min) * xsc_percent + xsc_min
        y_scint_center = (ysc_max - ysc_min) * ysc_percent + ysc_min
        dscint._coord_real['x2'] -= y_scint_center
        dscint._coord_real['x1'] -= x_scint_center        
    else:
        # Center the scintillator at the coordinate origin
        y_scint_center = 0.5 * (scint._coord_real['x2'].max()
                        + scint._coord_real['x2'].min())
        x_scint_center = 0.5 * (scint._coord_real['x1'].max()
                        + scint._coord_real['x1'].min())
        dscint._coord_real['x2'] -= y_scint_center
        dscint._coord_real['x1'] -= x_scint_center

    # Shift the imatge, if wanted
    px_0 = px_center + px_shift
    py_0 = py_center + py_shift
    # Scale to relate scintillator to camera
    xscale = beta / cam_params['px_x_size']
    yscale = beta / cam_params['px_y_size']
    # Calculate the pixel position of the scintillator vertices
    transformation_params = ssmapplting.CalParams()
    transformation_params.xscale = xscale
    transformation_params.yscale = yscale
    transformation_params.xshift = px_0
    transformation_params.yshift = py_0
    dscint.calculate_pixel_coordinates(transformation_params)
    # Shift the strike map by the same quantity:
    dsmap._data['x2'].data -= y_scint_center
    dsmap._data['x1'].data -= x_scint_center
    # Align the strike map:
    dsmap.calculate_pixel_coordinates(transformation_params)
    dsmap.interp_grid((cam_params['ny'], cam_params['nx']),
                     MC_number=0)
    # If there is an specific smap to plot, pass that smap as the plot argument
    # for strikemap. If not, the one used for the synthetic signal. We work
    # with a dumy smap, again
    dsmapplt._data['x2'].data -= y_scint_center
    dsmapplt._data['x1'].data -= x_scint_center
    dsmapplt.calculate_pixel_coordinates(transformation_params)
    dsmapplt.interp_grid((cam_params['ny'], cam_params['nx']),
                         MC_number=0)
    
    # LOCATE THE COLLIMATOR (experimental)
    # Don't use yet. Right now we take advantage of the scintillator libraries
    # to plot the collimator.
    # -----------------------------------------------------------------------
    try:
        dcoll=copy.deepcopy(collimator)
        dcoll._coord_real['x2'] -= y_scint_center
        dcoll._coord_real['x1'] -= x_scint_center
        # Calculate the pixel position of the scintillator vertices
        transformation_params = ssmapplting.CalParams()
        transformation_params.xscale = xscale
        transformation_params.yscale = yscale
        transformation_params.xshift = px_0
        transformation_params.yshift = py_0
        dcoll.calculate_pixel_coordinates(transformation_params)
        # Build the scintillator perimeter and find the area in the pixel space
        coll_perim = geometry.scint_ConvexHull(dcoll, coords='pix')
        coll_geom = True
        logger.info('- Collimator located and ready to plot')
    except:
        coll_geom = False
        logger.info('- No collimator geometry given')

    # MAP THE SIGNAL (new ridiculously fast mapping method)
    # -----------------------------------------------------------------------
    logger.info("- Mapping the signal in the scintillator space...")
    start = time.perf_counter()

    # Create a grid
    g_grid = dsmap._grid_interp['gyroradius']
    p_grid = dsmap._grid_interp['pitch']
    g_flat, p_flat = g_grid.flatten(), p_grid.flatten()
    # Bin the edges
    g_edges = scint_signal['gyroradius'] - scint_signal['dgyr']/2
    g_edges = np.append(g_edges, scint_signal['gyroradius'][-1] + scint_signal['dgyr']/2)
    p_edges = scint_signal['pitch'] - scint_signal['dp']/2
    p_edges = np.append(p_edges, scint_signal['pitch'][-1] + scint_signal['dp']/2)
    # Assign pixels to bins (to what bin does each pixel correspond)
    g_idx = np.digitize(g_flat, g_edges) - 1
    p_idx = np.digitize(p_flat, p_edges) - 1
    # Only keep valid pixels (exclude negative (in case) and only inside smap)
    valid = (g_idx >= 0) & (g_idx < scint_signal['gyroradius'].size) & \
            (p_idx >= 0) & (p_idx < scint_signal['pitch'].size)
    g_idx, p_idx = g_idx[valid], p_idx[valid]
    pixels_flat = np.zeros_like(g_flat, dtype=float)
    # Each pixel gets the signal level of the bin, it's not distributed per pix
    # Count number of pixels per bin to c
    from collections import defaultdict
    # Create a 2D index to count pixels per bin
    shape = (scint_signal['pitch'].size, scint_signal['gyroradius'].size)
    counts = np.zeros(shape, dtype=int)
    np.add.at(counts, (p_idx, g_idx), 1)  # number of pixels in each bin
    nonzero = counts > 0 # flag to skip 0 counts
    # Assign weighted values to each pixel
    # Divide the signal by th enumber of pix it is distributed
    values = scint_signal['signal'] * scint_signal['dgyr'] * scint_signal['dp']
    pixel_values = np.zeros(shape, dtype=float)
    pixel_values[nonzero] = values[nonzero] / counts[nonzero]
    # Map back to flattened array
    pixels_flat[valid] = pixel_values[p_idx, g_idx]
    # Reshape to grid
    synthetic_frame = pixels_flat.reshape(g_grid.shape)

    end = time.perf_counter()    # end timer
    elapsed = end - start
    logger.info("   Mapping time = %.4f s", elapsed)
                
    # CORRECTIONS
    # -----------------------------------------------------------------------
    logger.info('- Apllying corrections if needed...')
    # Build the original frame in the pixel space, and smooth it if wanted
    if smoother != None:
        dummy = copy.deepcopy(synthetic_frame)
        synthetic_frame = spnd.gaussian_filter(dummy,sigma=smoother)
    # Gyrophases corresponds to the range of gyrophases we consider that enter 
    # the pinhole. If we only consider the ions that are aiming to the head (pi)
    # we must have double the collimator factor, and double the particles.
    synthetic_frame *= 2*np.pi/gyrophases    

    # BUILD THE OUTPUT
    # -----------------------------------------------------------------------
    # Transform to xarray
    logger.info('- Building the signal xarray...')
    signal_frame = xr.DataArray(synthetic_frame, dims=('y', 'x'),
            coords={'y':(np.linspace(1,cam_params['ny'],cam_params['ny'])),
                    'x':(np.linspace(1,cam_params['nx'],cam_params['nx']))
                    })
    
    # Build the scintillator perimeter and find the area in the pixel space
    logger.info('- Building the scintillator perimeter and xarray...')
    scint_perim = geometry.scint_ConvexHull(dscint, coords='pix')
    scint_path = Path(scint_perim, closed=True)
    nx, ny = cam_params['nx'], cam_params['ny']
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))  # shape (ny, nx)
    points = np.vstack((x.ravel(), y.ravel())).T
    mask = scint_path.contains_points(points)
    dummy = copy.deepcopy(signal_frame)*0
    dummy_vals = dummy.values.reshape(-1)
    dummy_vals[mask] = 1
    dummy.values = dummy_vals.reshape(ny,nx)

    scint_area = copy.deepcopy(dummy)
    
    # Define the output
    signal_frame = signal_frame.where(signal_frame>=0,0)
    # Transform the output from pix units to m²
    pix_osize = ((cam_params['px_x_size']*cam_params['px_y_size'])/\
                 (optic_params['beta']**2)) # pix real size in scintillator
    signal_frame /= pix_osize

    integral_s = signal_frame.sum().item()
    integral_s *= pix_osize
    logger.info("   Total signal = %e", integral_s)

    signal_frame.attrs['long_name'] = mode
    signal_frame.attrs['units'] = cbar_units
    signal_frame.attrs['integrated_total'] = integral_s

    output = {
        'smap': dsmap,
        'smapplt': dsmapplt,
        'frame': signal_frame,        
        'scintillator': dscint,
        'scint_perim': scint_perim,
        'scint_area': scint_area
    }

    if coll_geom == True: #Flagg to add the collimator
        output['collimator'] = coll_perim

    return output


def noise_optics_camera(frame, eliminate_saturation = False,
                          cam_params: dict={},
                          optic_params: dict={},
                          noise_params: dict={},
                          radiometry = None,
                          distortion = None):
    """
    Gets the frame at the scintillator and transforms it to camera frame 
    
    Alex Reyner: alereyvinn@alum.us.es

    Feel free to update with more noises

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
    :param  optic_params:  parameters of the optics
    :param  cam_params: parameters of the camera
    :param  noise_params: types of noise
    :param  eliminate_saturation: if we want to get rid of saturated pixels.
            But we don't need to, this will be done with the plot vmax
    :param  radiometry: This adds the FoV and the transmission of the optics
            (the difference between regions assuming center equal to 1. The NA
            is a different thing). Needs to be a matrix with r (radius of FOV) 
            in x and theta (angle) in y. This matrix can't have any NaN value.
            Worry not about the number of data, it will be interpolated

    :return out dictionaryinput with the different noises added to signal_frame
            also the noise contribution of everything itself
    """
    logger.info("----- NOISE, OPTICS AND CAMERA ----- ")

    # Copy the input dictionary in the output
    out = copy.deepcopy(frame)
    # Create a copy of the signal frame where we will operate,
    # and the area covered by the scintillator
    signal_frame = copy.deepcopy(frame['frame'])
    
    scint_area = copy.deepcopy(frame['scint_area'])
    scint = copy.deepcopy(frame['scintillator'])
    radiometry = copy.deepcopy(radiometry)
    distortion = copy.deepcopy(distortion)
    # Initialise the noise dictionary. All the noises are not implemented yet.
    # This needs to be done.
    noise_opt = {
        'camera_neutrons': 0.001,
        'broken': 0.001
    }
    noise_opt.update(noise_params)
    out['noises']={}
    out['layers'] = xr.Dataset()

    # NOISES IN THE SCINTILLATOR
    # -----------------------------------------------------------------------
    # Neutron and gamma noise (constant) (just in the scintillator area)
    """
    Homogenous noise through the scintillator due to neutrons and gamma 
    reaching it and producing charged particles that will give signal. Total
    noise must be given.
    """
    if noise_opt['neutrons'] > 0:
        logger.info('- Neutron and gamma noise')
        num_pix = scint_area.sum().item() # how many pixel we the scint cover?
        # multiply by 4pi since we will consider the isotropic emision forward
        # in this model. The noise should be given per sr unit.
        dummy = copy.deepcopy(scint_area) 
        dummy *= noise_opt['neutrons']/num_pix # divide the noises equally in all pixels
        dummy *= 4*np.pi # this factor is applied since we will divide later
        # to transform from photons/pixel to photons/cm2 
        dummy /= ((cam_params['px_x_size']*cam_params['px_y_size'])\
                  /(optic_params['beta']**2))
        out['noises']['neutrons'] = dummy #storeed in photons/cm2

    # OPTICS
    # -----------------------------------------------------------------------
    # Compute the maximum counts for the camera
    max_count = 2 ** cam_params['range'] - 1
    # Now apply all factors
    # Adjust pixel sizes and beta (photons/cm2 to photons/pix)
    signal_frame *= ((cam_params['px_x_size']*cam_params['px_y_size'])\
                     /(optic_params['beta']**2))
    # Divide by 4\pi, ie, assume isotropic emission of the scintillator
    signal_frame *= 1 / 4 / np.pi
    # Consider the solid angle covered by the optics and the transmission of
    # the beam line through the lenses and mirrors:
    signal_frame *= optic_params['T'] * optic_params['omega']
    # Photon to electrons in the camera sensor (QE)
    signal_frame *= cam_params['qe']
    # Electrons to counts in the camera sensor,
    signal_frame /= cam_params['ad_gain']
    # Consider the exposure time
    signal_frame *= cam_params['exposure']
    # Apply distortion to the signal before the rest of optics and noises
    try:
        signal_frame =\
              signal_frame.interp(x=distortion.x_new, y=distortion.y_new)
    except:
        pass

    # Do the same for the noises added before the optics
    for key in out['noises']:
        out['noises'][key] *= ((cam_params['px_x_size']*cam_params['px_y_size'])\
               /(optic_params['beta']**2))
        out['noises'][key] *= 1 / 4 / np.pi
        out['noises'][key] *= optic_params['T'] * optic_params['omega']
        out['noises'][key] *= cam_params['qe']
        out['noises'][key] /= cam_params['ad_gain']
        out['noises'][key] *= cam_params['exposure']    
        try:
            out['noises'][key] =\
                  out['noises'][key].interp(x=distortion.x_new, y=distortion.y_new)
        except:
            pass
    # FINAL OF THE OPTICS
    # -----------------------------------------------------------------------

    final_frame = copy.deepcopy(signal_frame)
    # Apply the noises in the scintillator after the optics:
    for key in out['noises']:
        final_frame += out['noises'][key]

    # -----------------------------------------------------------------------
    # Add the optic FoV and the radiometry filter (stored as a noise)
    try:
        # Find the FoV in the scintillator using the % respect SW corner
        xsc_percent = optic_params['FoV'][0]
        ysc_percent = optic_params['FoV'][1]
        xsc_min = frame['scint_perim'][:,0].min()
        xsc_max = frame['scint_perim'][:,0].max()
        ysc_min = frame['scint_perim'][:,1].min()
        ysc_max = frame['scint_perim'][:,1].max()
        x_FoV = (xsc_max - xsc_min) * xsc_percent + xsc_min
        y_FoV = (ysc_max - ysc_min) * ysc_percent + ysc_min
        radiometry.coords['r'] = radiometry.coords['r'] * optic_params['beta']\
            / (cam_params['px_x_size']*1000)
        r_FoV = radiometry.coords['r'].values.max()

        #To create te filter we need to remap the R-theta matrix
        xpix = np.linspace(1,cam_params['nx'],cam_params['nx'])
        ypix = np.linspace(1,cam_params['ny'],cam_params['ny'])
        r = xr.DataArray((np.sqrt((xpix-x_FoV)**2+(ypix[:, np.newaxis]-y_FoV)**2)),
                                    dims=['y', 'x'],
                                    coords={'y':ypix, 'x':xpix})
        t = xr.DataArray(np.arctan2((ypix[:, np.newaxis]-y_FoV),(xpix-x_FoV)),
                                    dims=['y', 'x'],
                                    coords={'y':ypix, 'x':xpix}) 
        # Here we interpolate to the pixel space
        dummy = copy.deepcopy(radiometry.interp(r=r,t=t))
        dummy = dummy.drop_vars('t')
        # Now we want 0 for the filter, not NaN
        dummy = dummy.where(dummy >= 0, 0)
        # Add to the filter itself to the noise output 
        out['noises']['radiometry'] = dummy
        final_frame *= dummy
        logger.info('- Radiometry filter applied')

        # Add FoV to the output 
        out['FoV_vect'] = [x_FoV,y_FoV,r_FoV]
        logger.info('- FoV determined: (x,y,r) = (%4.1f,%4.1f,%4.1f) pix',\
                     x_FoV,y_FoV,r_FoV)
    except: # Try to at least define the FoV
        try:
            # Find the FoV in the scintillator using the % respect SW corner
            xsc_percent = optic_params['FoV'][0]
            ysc_percent = optic_params['FoV'][1]
            xsc_min = frame['scint_perim'][:,0].min()
            xsc_max = frame['scint_perim'][:,0].max()
            ysc_min = frame['scint_perim'][:,1].min()
            ysc_max = frame['scint_perim'][:,1].max()
            x_FoV = (xsc_max - xsc_min) * xsc_percent + xsc_min
            y_FoV = (ysc_max - ysc_min) * ysc_percent + ysc_min
            r_FoV = optic_params['FoV'][2] * optic_params['beta'] \
                / (cam_params['px_x_size']*1000)
            out['FoV_vect'] = [x_FoV,y_FoV,r_FoV]
            logger.info('- FoV determined: (x,y,r) = (%4.1f,%4.1f,%4.1f) pix',\
                         x_FoV,y_FoV,r_FoV)
        except:
            logger.info('- No FoV, or not good format of the input')
    

    # NOISES IN THE CAMERA  
    # -----------------------------------------------------------------------
    # Neutron impact noise
    """
    Add noise due to neutron impact on the sensor
    """
    if noise_opt['camera_neutrons'] > 0:
        logger.info('- Neutrons hitting the sensor')    
        rand = np.random.default_rng()
        hit = rand.uniform(size = final_frame.shape)
        intensity = rand.uniform(size=final_frame.shape)
        # noise frame, select only the pixels with the noise
        dummy = copy.deepcopy(final_frame)*0
        dummy += (2**cam_params['range'] - 1)
        dummy *= intensity 
        dummy = dummy.where(hit <= noise_opt['camera_neutrons'], 0)
        # eliminate those same frames from the noise frame and add noise
        final_frame = final_frame\
            .where(hit > noise_opt['camera_neutrons'], 0)
        final_frame += dummy
        out['noises']['camera_neutrons'] = dummy
        
    # Broken pixels
    """
    Simulate broken pixels
    """
    if noise_opt['broken'] > 0:
        logger.info('- Some pixel are broken')
        rand = np.random.default_rng()
        broken = rand.uniform(size = final_frame.shape)
        # eliminate the pixels
        dummy = copy.deepcopy(final_frame)*0 + broken
        dummy = dummy.where(broken > noise_opt['broken'], 0) #the broken
        dummy = dummy.where(broken <= noise_opt['broken'], 1) #the okay
        final_frame = final_frame.where(broken > noise_opt['broken'], 0)
        out['noises']['broken'] = dummy

    # Add the camera noise if both needed parameters are included
    """
    Notice: dark current and readout noise are effects always present. It is
    imposible to measure them independently, so they will be modelled as a
    single gaussian noise with centroid 'dark_centroid' and sigma
    'sigma_readout'. Both parameters to be measured for the used camera
    """
    if 'readout_noise_med' in cam_params and 'readout_noise_rmd' in cam_params:
        logger.info('- Gaussian readout noise custom')
        rand = np.random.default_rng()
        gauss = rand.standard_normal
        dummy = cam_params['readout_noise_med'] + \
            cam_params['readout_noise_rmd'] * gauss(final_frame.shape) #readjust the sigma
        dummy /= cam_params['ad_gain']
        readout_noise = copy.deepcopy(final_frame)*0 + dummy
        readout_noise = readout_noise.where(readout_noise > 0 ,0) # needed
        final_frame += readout_noise
        out['noises']['readout_noise'] = readout_noise        
    elif 'readout_noise' in cam_params and 'dark_noise' in cam_params:
        logger.info('- Gaussian readout noise standard')
        rand = np.random.default_rng()
        gauss = rand.standard_normal
        dummy = cam_params['dark_noise'] + \
            cam_params['readout_noise'] * gauss(final_frame.shape)
        dummy /= cam_params['ad_gain']
        readout_noise = copy.deepcopy(final_frame)*0 + np.round(dummy).astype(int)
        readout_noise = readout_noise.where(readout_noise > 0 ,0) # needed
        final_frame += readout_noise
        out['noises']['readout_noise'] = readout_noise

    if 'dark_noise' in cam_params and 'readout_noise' not in cam_params:
        logger.info('- Dark current noise')
        rand = np.random.default_rng()
        poiss = rand.poisson
        dummy = poiss(lam=cam_params['dark_noise'], size=final_frame.shape)
        dark_current = copy.deepcopy(final_frame)*0 + dummy
        dark_current = dark_current.where(dark_current > 0 ,0) # needed
        final_frame += dark_current
        out['noises']['dark_current'] = dark_current

    # ADJUST THE CAMERA FRAME AND OUTPUT  
    # -----------------------------------------------------------------------
    logger.info('- Buildind the output...')    
    # Cap the counts to the maximum counts
    if eliminate_saturation == True:
        final_frame = final_frame.where(final_frame < max_count, max_count) 

    # Transform the counts to integers    
    final_frame.data = final_frame.data.astype(int, copy=False)

    # Substitute the signal_frame
    out['frame'] = final_frame
    out['frame'].attrs['long_name'] = 'Pixel counts'
    out['frame'].coords['x'].attrs['longname'] = 'X'
    out['frame'].coords['x'].attrs['units'] = 'pix.'
    out['frame'].coords['y'].attrs['longname'] = 'Y'
    out['frame'].coords['y'].attrs['units'] = 'pix.'

    # Put all in the dataset
    out['layers']['FIL'] = signal_frame
    for key in out['noises']:
        out['layers'][key] = out['noises'][key]
    out['layers'].attrs['long_name'] = 'Pixel counts'
    out['layers'].coords['x'].attrs['longname'] = 'X'
    out['layers'].coords['x'].attrs['units'] = 'pix.'
    out['layers'].coords['y'].attrs['longname'] = 'Y'
    out['layers'].coords['y'].attrs['units'] = 'pix.'        


    return out


def plot_the_frame(frame, plot_smap = True, plot_scint = True, plot_FoV = False,
                cam_params: dict={}, maxval = None,
                figtitle = None, cmap = default_cmap(),
                **kwargs):
    """
    Plot one frame, the scintillator and the strikemap
    
    Alex Reyner: alereyvinn@alum.us.es

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  maxval: adjust the maximum of the colorbar to the signal
    :param  figtitle: plot a title with the optics parameters, for example

    :return fig, ax:
    """
    logger.info("----- SIGNAL PLOT ----- ")

    frame_to_plot = frame['frame']
    smapplt = frame['smapplt']
    scint_perim = frame['scint_perim']

    # Establish vmax as: 1) max camera counts, 2) to maxval of the signal
    if maxval == None:
        max_count = 2 ** cam_params['range'] - 1
        logger.info('- Maximum set to camera range')
    else:
        max_count = maxval*frame_to_plot.max().item()
        logger.info('- Maximum set to %4.2f max signal', maxval)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8,5))
    
    # frame_to_plot.plot.imshow(ax=ax, cmap=cmap, norm=LogNorm(vmin=1e14,vmax=1e22),
    #                           **kwargs)
    im = frame_to_plot.plot.imshow(ax=ax, cmap=cmap, vmin=0, vmax=max_count,
                              **kwargs)

    if plot_smap == True:
        smapplt.plot_pix(ax, labels=False, marker_params={'marker':None},
                         line_params={'color':'w', 'linewidth':1.2, 'alpha':0.8})

    if plot_scint == True:
        ax.plot(scint_perim[:,0],scint_perim[:,1], color ='w', linewidth=2)

    if plot_FoV == True:
        try:
            ax.scatter(frame['FoV_vect'][0], frame['FoV_vect'][1],
                    marker='+',s=100,c='lime')
            FoV = Circle((frame['FoV_vect'][0], frame['FoV_vect'][1]), 
                         radius=frame['FoV_vect'][2], 
                         color='lime', fill=False, linewidth=2)
            ax.add_patch(FoV)
        except:
            logger.info('- No FoV plotted beacuse whatever')
    

    if figtitle != None:
        fig.suptitle(figtitle,size=12)
        
    ax.set_xlim([1,cam_params['nx']])
    ax.set_ylim([1,cam_params['ny']])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    return fig, ax


def plot_noise_contributions(frame, cam_params: dict={}, maxval = False,
                             cmap=default_cmap()):
    """
    Plot all the noise contributions
    
    Alex Reyner: alereyvinn@alum.us.es

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  maxval: want to limit the colorbar to camera range?
    :param  figtitle: plot a title with the optics parameters, for example

    :return fig, ax:
    """
    logger.info("----- NOISE CONTRIBUTIONS PLOT ----- ")

    for i in frame['noises']:
        frame_to_plot = frame['noises'][i]
        fig, ax = plt.subplots(figsize=(8,5))
        if i == 'broken':
            bw_cmap =  LinearSegmentedColormap.from_list(
                'mycmap', ['black', 'white'], N=2)
            frame_to_plot.plot.imshow(ax=ax, cmap=bw_cmap,
                    vmin=0, vmax=1,
                    cbar_kwargs={"label": '  Broken pixel             Functioning pixel',
                                 'spacing': 'proportional'})
        elif i == 'radiometry':
            frame_to_plot.plot.imshow(ax=ax, center=1,
                    cbar_kwargs={"label": 'Relative Illumination', 'spacing': 'proportional'})
        else:
            max_count = 2 ** cam_params['range'] - 1
            frame_to_plot.plot.imshow(ax=ax, cmap=cmap,
                    vmin=0, vmax=max_count,
                    cbar_kwargs={"label": 'Pixel counts','spacing': 'proportional'})
                    
        fig.suptitle(i)
        ax.set_xlim([1,cam_params['nx']])
        ax.set_ylim([1,cam_params['ny']])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        logger.info('- %s', i)
    plt.show()
 
    return


def remap_the_frame(frame, WF = None, 
                       B=4, A=4, Z=2, efficiency_flag = True,
                       pin_params: dict = {}, scint_params: dict = {}):
    """
    Remaps the obtained camera synthetic signal.

    Alex Reyner: alereyvinn@alum.us.es

    :param  frame: dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  WF: weight function corresponfing to this case
    :param  B: magnetic field (in case WF not in the input)
    :param  A: A value of the ions (in case WF not in the input)
    :param  Z: Z value of the ions (in case WF not in the input)

    :return frame dictionary adding:
            'remapped_signal': remapped camera synthetic signal
    """    


    # In case WF is not specified, compute it (alpha is default)
    if WF is None:
        pin_options = {'xmin': 30, 'xmax': 90, 'dx': 1, 
                     'ymin': 1, 'ymax': 10, 'dy': 0.2,}
        scint_options = {'xmin': 30, 'xmax': 90, 'dx': 0.5, 
                     'ymin': 1, 'ymax': 10, 'dy': 0.1,}
        # update the matrix options
        pin_options.update(pin_params)
        scint_options.update(scint_params)
        WF = obtain_WF(smap = frame['smap'], scintillator = frame['scintillator'], 
               efficiency_flag = efficiency_flag, B=B, A=A, Z=Z,
               pin_params=pin_options, scint_params=scint_options)

    # Get WF parameters for remapping
    xstep = (WF.xs[1]-WF.xs[0]).values
    xedges=WF.xs.values-xstep
    xedges=np.append(xedges,WF.xs[-1].values+xstep)
    ystep = (WF.ys[1]-WF.ys[0]).values
    yedges=WF.ys.values-ystep
    yedges=np.append(yedges,WF.ys[-1].values+ystep)

    # Remap
    dummy = common.remap(smap=frame['smap'], frame=frame['frame'].values, x_edges=xedges, y_edges=yedges, method='centers')
    remaped_signal = xr.DataArray(dummy, dims=('x', 'y'),
                         coords={'x':WF.xs.values, 'y':WF.ys.values})

    frame['remapped_signal'] = remaped_signal

    return frame


def synthsig_xy(distro, smap, WF, scint, smapplt = None, 
                     gyrophases = np.pi, mode = 'photons',
                     cam_params = {}, optic_params = {},
                     centering = False,
                     smoother = None,
                     **kwargs):
    """
    Maps a signal in the scintillator

    Alex Reyner: alereyvinn@alum.us.es

    Based on the origianl function by Jose Rueda    

    :param  distro: distribution in the pinhole
    :param  smap: smap to map the signal in the xyspace
    :param  WF: weight function obtained with obtain WF
    :param  scint: scintillator shape we want in the plots
    :param  smapplt: extra smap to do nice plots
    :param  gyrophases: used to renormalize the collimator factor. 
                Range of gyrophases that we consider. Default pi 
                -> range of gyrophases pointing into the pinhole
    :param  mode: select what quantity you want
                - photons: includes scintillator response (default)
                - ions: ion flux in the scintillator
                - power: power flux deposited in the scintillator
    :param  cam_params: parameters of the camera
    :param  optic_params: parameters of the optics
    :param  smoother: adds a gaussian filter to the signal with that sigma

    :kwarg  eff: deactivate the scintillator efficency with None

    :return out dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'velspace': velocity space signals
    """
    logger.info("----- STARTING X-Y MAPPING -----")

    # INPUT CHECK
    # -----------------------------------------------------------------------
    logger.info('- Input checking...')
    dsmap = copy.deepcopy(smap)
    if smapplt != None:
        dsmapplt = copy.deepcopy(smapplt)
    else:
        dsmapplt = copy.deepcopy(dsmap)
    dscint = copy.deepcopy(scint)

    if mode == 'photons': # the usual, photons after scintillator response
        cbar_units = 'photons / s m²'
    elif mode == 'ions': # to see the flux of ions in the scintillator
        cbar_units = 'ions / s m²'
    elif mode == 'power': # to compute the deposited power (beta featura)
        cbar_units = 'W / m²'
        distro['weight'] = distro['power']
    else:
        logger.error('Wrong mode, select either: photons, ions or power')

    # SYNTHETIC SIGNAL
    # -----------------------------------------------------------------------
    # Calculate the synthetic signal at the scintillator
    remap_sig = synthsig_pr(distro, dscint, WF, mode=mode)
    scint_signal = remap_sig.SC
    dp = (WF.xs[1]-WF.xs[0]).values
    dr = (WF.ys[1]-WF.ys[0]).values

    # LOCATE AND CENTER THE SCINTILLATOR AND SMAP
    # -----------------------------------------------------------------------
    logger.info('- Locating the smap and scintillator...')
    # Find the center of the camera frame 
    px_center = int(cam_params['nx'] / 2)
    py_center = int(cam_params['ny'] / 2)
    if 'beta' in optic_params:
        beta = optic_params['beta']
        logger.info('   Optics magnification, beta: %e', beta)
    else:
        xsize = cam_params['px_x_size'] * cam_params['nx']
        ysize = cam_params['px_y_size'] * cam_params['ny']
        chip_min_length = np.minimum(xsize, ysize)
        xscint_size = scint._coord_real['x1'].max() \
            - scint._coord_real['x1'].min()
        yscint_size = scint._coord_real['x2'].max() \
            - scint._coord_real['x2'].min()
        scintillator_max_length = np.maximum(xscint_size, yscint_size)
        beta = chip_min_length / scintillator_max_length
        logger.info('   Optics magnification, beta: %e', beta)
        optic_params['beta'] = beta
    
    if centering:
        # Center image to FoV
        xsc_percent = optic_params['FoV'][0]
        ysc_percent = optic_params['FoV'][1]
        xsc_min = scint._coord_real['x1'].min()
        xsc_max = scint._coord_real['x1'].max()
        ysc_min = scint._coord_real['x2'].min()
        ysc_max = scint._coord_real['x2'].max()
        x_scint_center = (xsc_max - xsc_min) * xsc_percent + xsc_min
        y_scint_center = (ysc_max - ysc_min) * ysc_percent + ysc_min
        dscint._coord_real['x2'] -= y_scint_center
        dscint._coord_real['x1'] -= x_scint_center        
    else:
        # Center the scintillator at the coordinate origin
        y_scint_center = 0.5 * (scint._coord_real['x2'].max()
                        + scint._coord_real['x2'].min())
        x_scint_center = 0.5 * (scint._coord_real['x1'].max()
                        + scint._coord_real['x1'].min())
        dscint._coord_real['x2'] -= y_scint_center
        dscint._coord_real['x1'] -= x_scint_center

    # Scale to relate scintillator to camera
    xscale = beta / cam_params['px_x_size']
    yscale = beta / cam_params['px_y_size']
    # Calculate the pixel position of the scintillator vertices
    transformation_params = ssmapplting.CalParams()
    transformation_params.xscale = xscale
    transformation_params.yscale = yscale
    transformation_params.xshift = px_center
    transformation_params.yshift = py_center
    dscint.calculate_pixel_coordinates(transformation_params)
    # Shift the strike map by the same quantity:
    dsmap._data['x2'].data -= y_scint_center
    dsmap._data['x1'].data -= x_scint_center
    # Align the strike map:
    dsmap.calculate_pixel_coordinates(transformation_params)
    dsmap.interp_grid((cam_params['ny'], cam_params['nx']),
                     MC_number=0)
    # If there is an specific smap to plot, pass that smap as the plot argument
    # for strikemap. If not, the one used for the synthetic signal. We work
    # with a dumy smap, again
    dsmapplt._data['x2'].data -= y_scint_center
    dsmapplt._data['x1'].data -= x_scint_center
    dsmapplt.calculate_pixel_coordinates(transformation_params)
    dsmapplt.interp_grid((cam_params['ny'], cam_params['nx']),
                         MC_number=0)

    # MAP THE SIGNAL (new ridiculously fast mapping method)
    # -----------------------------------------------------------------------
    logger.info("- Mapping the signal in the scintillator space...")
    start = time.perf_counter()

    # Create a grid
    g_grid = dsmap._grid_interp['gyroradius']
    p_grid = dsmap._grid_interp['pitch']
    g_flat, p_flat = g_grid.flatten(), p_grid.flatten()
    # Bin the edges
    g_edges = scint_signal.ys - dr/2
    g_edges = np.append(g_edges, scint_signal.ys[-1] + dr/2)
    p_edges = scint_signal.xs - dp/2
    p_edges = np.append(p_edges, scint_signal.xs[-1] + dp/2)
    # Assign pixels to bins (to what bin does each pixel correspond)
    g_idx = np.digitize(g_flat, g_edges) - 1
    p_idx = np.digitize(p_flat, p_edges) - 1
    # Only keep valid pixels (exclude negative (in case) and only inside smap)
    valid = (g_idx >= 0) & (g_idx < scint_signal.ys.size) & \
            (p_idx >= 0) & (p_idx < scint_signal.xs.size)
    g_idx, p_idx = g_idx[valid], p_idx[valid]
    pixels_flat = np.zeros_like(g_flat, dtype=float)
    # Each pixel gets the signal level of the bin, it's not distributed per pix
    # Count number of pixels per bin to c
    from collections import defaultdict
    # Create a 2D index to count pixels per bin
    shape = (scint_signal.xs.size, scint_signal.ys.size)
    counts = np.zeros(shape, dtype=int)
    np.add.at(counts, (p_idx, g_idx), 1)  # number of pixels in each bin
    nonzero = counts > 0 # flag to skip 0 counts
    # Assign weighted values to each pixel
    # Divide the signal by th enumber of pix it is distributed
    values = scint_signal.values * dr * dp
    pixel_values = np.zeros(shape, dtype=float)
    pixel_values[nonzero] = values[nonzero] / counts[nonzero]
    # Map back to flattened array
    pixels_flat[valid] = pixel_values[p_idx, g_idx]
    # Reshape to grid
    synthetic_frame = pixels_flat.reshape(g_grid.shape)

    end = time.perf_counter()    # end timer
    elapsed = end - start
    logger.info("   Mapping time = %.4f s", elapsed)
                
    # CORRECTIONS
    # -----------------------------------------------------------------------
    logger.info('- Apllying corrections if needed...')
    # Build the original frame in the pixel space, and smooth it if wanted
    if smoother != None:
        dummy = copy.deepcopy(synthetic_frame)
        synthetic_frame = spnd.gaussian_filter(dummy,sigma=smoother)

    # BUILD THE OUTPUT
    # -----------------------------------------------------------------------
    # Transform to xarray
    logger.info('- Building the signal xarray...')
    signal_frame = xr.DataArray(synthetic_frame, dims=('y', 'x'),
            coords={'y':(np.linspace(1,cam_params['ny'],cam_params['ny'])),
                    'x':(np.linspace(1,cam_params['nx'],cam_params['nx']))
                    })
    
    # Build the scintillator perimeter and find the area in the pixel space
    logger.info('- Building the scintillator perimeter and xarray...')
    scint_perim = geometry.scint_ConvexHull(dscint, coords='pix')
    scint_path = Path(scint_perim, closed=True)
    nx, ny = cam_params['nx'], cam_params['ny']
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))  # shape (ny, nx)
    points = np.vstack((x.ravel(), y.ravel())).T
    mask = scint_path.contains_points(points)
    dummy = copy.deepcopy(signal_frame)*0
    dummy_vals = dummy.values.reshape(-1)
    dummy_vals[mask] = 1
    dummy.values = dummy_vals.reshape(ny,nx)

    scint_area = copy.deepcopy(dummy)
    
    # Define the output
    signal_frame = signal_frame.where(signal_frame>=0,0)
    # Transform the output from pix units to m²
    pix_osize = ((cam_params['px_x_size']*cam_params['px_y_size'])/\
                 (optic_params['beta']**2)) # pix real size in scintillator
    signal_frame /= pix_osize
    integral_s = signal_frame.sum().item()
    integral_s *= pix_osize
    logger.info("   Total signal = %e", integral_s)

    signal_frame.attrs['mode'] = mode
    signal_frame.attrs['long_name'] = cbar_units
    signal_frame.attrs['integrated_total'] = integral_s

    output = {
        'smap': dsmap,
        'smapplt': dsmapplt,
        'frame': signal_frame,        
        'scintillator': dscint,
        'scint_perim': scint_perim,
        'scint_area': scint_area,
        'velspace': remap_sig
    }

    return output


# -----------------------------------------------------------------------------
# --- Special routines for double pinhole FILD scintilator space
# -----------------------------------------------------------------------------

def synthsig_xy_2coll(distros, scint,
                     smaps: dict = {},
                     smapsplt: dict = {},
                     cam_params = {}, optic_params = {}, 
                     gyrophases = np.pi,
                     smoother = None,
                     scint_params: dict = {},
                     px_shift: int = 0, py_shift: int = 0):
    """
    Maps a signal in the scintillator

    Alex Reyner: alereyvinn@alum.us.es

    :param  distro: distribution in the pinhole
    :param  scint: scintillator shape we want in the plots
    :param  smaps: dictionary with the two smap objects (left, right)
    :param  smapsplt: extra smaps to plot nice
    :param  cam_params: parameters of the camera
    :param  optic_params:  parameters of the optics
    :param  gyrophases: range of gyrophases considered entering the pinhole
            (to scale the collimator factor). pi (half sr) is the default.
            Usually the collimator factor is defined over a 2pi range
    :param  smoother: adds a gaussian filter to the signal with that sigma
    :param  scint_synthetic_signal_params: grid to remap the frames

    :return out dictionary containing:
            'smap'['left','right']: the two smaps for the mapping
            'smapplt'['left','right']: the two smaps to plot
            'side_signal'['left','right']: signal of each side of the total
            'frame': total signal in the scintillator space
            'scintillator': scintillator calibrated to the signal
            'scint_area': region covered by the scintillator
    """

    output = {
        'smap': {},
        'smapplt': {},
        'side_signal': {}
    }

    dscint = copy.deepcopy(scint)
    efficiency = scint.efficiency

    # LOCATE AND CENTER THE SCINTILLATOR
    # -----------------------------------------------------------------------
    logger.info('- Locating the smap and scintillator...')
    # Find the center of the camera frame 
    px_center = int(cam_params['nx'] / 2)
    py_center = int(cam_params['ny'] / 2)
    if 'beta' in optic_params:
        beta = optic_params['beta']
    else:
        xsize = cam_params['px_x_size'] * cam_params['nx']
        ysize = cam_params['px_y_size'] * cam_params['ny']
        chip_min_length = np.minimum(xsize, ysize)
        xscint_size = scint._coord_real['x1'].max() \
            - scint._coord_real['x1'].min()
        yscint_size = scint._coord_real['x2'].max() \
            - scint._coord_real['x2'].min()
        scintillator_max_length = np.maximum(xscint_size, yscint_size)
        beta = chip_min_length / scintillator_max_length
        optic_params['beta'] = beta
        logger.info('   Optics magnification, beta: %e', beta)
    # Center the scintillator at the coordinate origin
    y_scint_center = 0.5 * (scint._coord_real['x2'].max()
                            + scint._coord_real['x2'].min())
    x_scint_center = 0.5 * (scint._coord_real['x1'].max()
                            + scint._coord_real['x1'].min())
    dscint._coord_real['x2'] -= y_scint_center
    dscint._coord_real['x1'] -= x_scint_center
    # Center of the scintillator in pixel space
    px_0 = px_center + px_shift
    py_0 = py_center + py_shift
    # Scale to relate scintillator to camera
    xscale = beta / cam_params['px_x_size']
    yscale = beta / cam_params['px_y_size']
    # Calculate the pixel position of the scintillator vertices
    transformation_params = ssmapplting.CalParams()
    transformation_params.xscale = xscale
    transformation_params.yscale = yscale
    transformation_params.xshift = px_0
    transformation_params.yshift = py_0
    dscint.calculate_pixel_coordinates(transformation_params)

    for item in smaps:
        # Check inputs and initialise the things
        dsmap = copy.deepcopy(smaps[item])
        try:
            dsmapplt = copy.deepcopy(smapsplt[item])
        except:
            dsmapplt = copy.deepcopy(dsmap)

        # SYNTHETIC SIGNAL
        # -------------------------------------------------------------------
        distro = distros[item]
        scint_options = {
            'rmin': 1,
            'rmax': 10.0,
            'dr': 0.1,
            'pmin': 5.0,
            'pmax': 90.0,
            'dp': 1,
        }
        scint_options.update(scint_params)    
        logger.info('- Computing synthetic %s signal...', item)
        # Calculate the synthetic signal at the scintillator
        scint_signal = ssfM.synthetic_signal_remap(distro, dsmap,
                                            efficiency=efficiency,
                                            **scint_options)
 
        # LOCATE AND CENTER THE SCINTILLATOR
        # -------------------------------------------------------------------
        # Shift the strike map by the same quantity:
        dsmap._data['x2'].data -= y_scint_center
        dsmap._data['x1'].data -= x_scint_center
        # Align the strike map:
        dsmap.calculate_pixel_coordinates(transformation_params)
        dsmap.interp_grid((cam_params['ny'], cam_params['nx']),
                        MC_number=0)
        # If there is an specific smap to plot, pass that smap as the plot argument
        # for strikemap. If not, the one used for the synthetic signal. We work
        # with a dumy smap, again
        dsmapplt._data['x2'].data -= y_scint_center
        dsmapplt._data['x1'].data -= x_scint_center
        dsmapplt.calculate_pixel_coordinates(transformation_params)
        dsmapplt.interp_grid((cam_params['ny'], cam_params['nx']),
                            MC_number=0)
        
        # MAP SCINTILLATOR AND GRID TO FRAME
        # -----------------------------------------------------------------------
        logger.info('- Mapping the %s signal in the scintillator space...', item)
        n_gyr = scint_signal['gyroradius'].size
        n_pitch = scint_signal['pitch'].size
        synthetic_frame = np.zeros(dsmap._grid_interp['gyroradius'].shape)
        for ir in range(n_gyr):
            # Gyroradius limits to integrate
            gmin = scint_signal['gyroradius'][ir] - scint_signal['dgyr'] / 2.
            gmax = scint_signal['gyroradius'][ir] + scint_signal['dgyr'] / 2.
            for ip in range(n_pitch):
                # Pitch limits to integrates
                pmin = scint_signal['pitch'][ip] - scint_signal['dp'] / 2.
                pmax = scint_signal['pitch'][ip] + scint_signal['dp'] / 2.
                # Look for the pixels which cover this region:
                flags = (dsmap._grid_interp['gyroradius'] >= gmin) \
                    * (dsmap._grid_interp['gyroradius'] < gmax) \
                    * (dsmap._grid_interp['pitch'] >= pmin) \
                    * (dsmap._grid_interp['pitch'] < pmax)
                flags = flags.astype(bool)
                # If there are some pixels, just divide the value weight among them
                n = np.sum(flags)
                if n > 0:
                    synthetic_frame[flags] = scint_signal['signal'][ip, ir] / n \
                        * scint_signal['dgyr'] * scint_signal['dp']
        
        # CORRECTIONS
        # -------------------------------------------------------------------
        logger.info('- Apllying corrections if needed...')
        # Build the original frame in the pixel space, and smooth it if wanted
        if smoother != None:
            dummy = copy.deepcopy(synthetic_frame)
            synthetic_frame = spnd.gaussian_filter(dummy,sigma=smoother)
        # Gyrophases corresponds to the range of gyrophases we consider that enter 
        # the pinhole. If we only consider the ions that are aiming to the head (pi)
        # we must have double the collimator factor, and double the particles.
        synthetic_frame *= 2*np.pi/gyrophases    

        # BUILD THE OUTPUT OF ONE OF THE SIDES
        # -------------------------------------------------------------------
        # Transform to xarray
        logger.info('- Building the %s signal xarray...', item)
        side_frame = xr.DataArray(synthetic_frame, dims=('y', 'x',),
                coords={'y':np.linspace(1,cam_params['ny'],cam_params['ny']),
                        'x':np.linspace(1,cam_params['nx'],cam_params['nx'])})
        side_frame = side_frame.where(side_frame>=0,0)
        
        output['smap']['left'] = dsmap
        output['smapplt'][item] = dsmapplt
        output['side_signal'][item] = side_frame
 
    signal_frame = output['side_signal']['left'] +\
                   output['side_signal']['right']
    signal_frame = signal_frame.where(signal_frame>0,0)
    integral=signal_frame.integrate('x').integrate('y').item()
    logger.info("   Total signal = %e photons/s", integral)

    # BUILD THE FINAL THINGS, AND MOST IMPORTANT, OF THE OUTPUT
    # -------------------------------------------------------------------
    # Define the scintillator perimeter and find the area in the pixel space
    logger.info('- Building the scintillator perimeter and xarray...')
    
    scint_perim = geometry.scint_ConvexHull(dscint, coords='pix')

    scint_path = Path(scint_perim, closed=True)
    dummy = copy.deepcopy(signal_frame)*0
    for i in range(cam_params['ny']):
        for j in range(cam_params['nx']):
            # Check if the (i, j) coordinates are inside the scintillator
            if scint_path.contains_point((j, i)):
                dummy[i, j] = 1 # 1 if it's inside the scintillator
    scint_area = copy.deepcopy(dummy)
    
    # Define the output
    output['frame'] = signal_frame
    output['scintillator'] = scint_perim
    output['scint_area'] = scint_area

    return output


def plot_the_frame_2coll(frame, plot_smap = True, plot_scint = True, plot_FoV = True,
                   cam_params: dict={}, maxval = None,
                   figtitle = None, cmap=default_cmap()):
    """
    Plot one frame, the scintillator and the strikemap
    
    Alex Reyner: alereyvinn@alum.us.es

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  maxval: adjust the maximum of the colorbar to the signal
    :param  figtitle: plot a title with the optics parameters, for example

    :return fig, ax:
    """
    logger.info("----- 2 COLLIMATOR SIGNAL PLOT ----- ")
    
    frame_to_plot = frame['frame']
    smapplt = frame['smapplt']
    scint_perim = frame['scintillator']

    # Establish vmax as: 1) max camera counts, 2) to maxval of the signal
    if maxval == None:
        max_count = 2 ** cam_params['range'] - 1
        logger.info('- Maximum set to camera range')
    else:
        max_count = maxval*frame_to_plot.max().item()
        logger.info('- Maximum set to %4.2f max signal', maxval)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(7,4))
    ax.set_aspect(1)      
    
    frame_to_plot.plot.imshow(ax=ax, cmap=cmap, vmin=0, vmax=max_count,
                    cbar_kwargs={"label": 'Pixel counts','spacing': 'proportional'})

    if plot_smap == True:
        for item in smapplt:
            smapplt[item].plot_pix(ax, labels=False, marker_params={'marker':None},
                         line_params={'color':'w', 'linewidth':1.2, 'alpha':0.8})

    if plot_scint == True:
        ax.plot(scint_perim[:,0],scint_perim[:,1], color ='w', linewidth=3)

    if plot_FoV == True:
        try:
            ax.scatter(frame['FoV_vect'][0], frame['FoV_vect'][1],
                    marker='+',s=100,c='lime')
            FoV = Circle((frame['FoV_vect'][0], frame['FoV_vect'][1]), 
                         radius=frame['FoV_vect'][2], 
                         color='lime', fill=False, linewidth=2)
            ax.add_patch(FoV)
        except:
            logger.info('- No FoV plotted beacuse whatever')

    ax.set_xlim((0,cam_params['nx']))
    ax.set_ylim((0,cam_params['ny']))
    ax_param = {'xlabel': 'x pix.', 'ylabel': 'y pix.'}
    ax = ssplt.axis_beauty(ax, ax_param)
    ax.set_aspect(1)

    if figtitle != None:
        fig.suptitle(figtitle, fontsize=11)
    
    plt.tight_layout()
    plt.show()

    return fig, ax
