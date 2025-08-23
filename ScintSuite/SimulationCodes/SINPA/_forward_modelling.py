"""
Alex Reyner: alereyvinn@alum.us.es

Workflow:
    0. Obtain the distribution and define the inputs 
    1. Run original_synthsig_xy to map the signal in the scintillator space
    2. Insert the different noises, optic system and the camera
       - Option to do it separately (not recomended) or at the same time (recomended)
            with noise_optics_camera

    - You can plot at any step given the frame and even plot each  noises
    - You can also directly compute the WF and remapped synthetic signals

Functions. What can be done:
    - read_ASCOT_dist: Read an ASCOT distribution with ion distribution that 
        will be used as input
    - obtain_WF: Obtain the weight function of the smap (scintillator efficency 
        or not)
    - synthetic_signal_pr: Compute remapped synthetic signal in pitch-gyroradius 
        phase space
    - pr_space_to_pe_space: transform the the signal phase space
    - original_synthsig_xy: compute synthetic signal in real scintillator space
    - noise_optics_camera: add noise and optic effects to the synthetic signal
    - plot_the_frame: plot the signal at a given point
    - plot_noise_contributions: plot the different noise contributions
    - synthsig_xy_2coll: compute synthetic signal of two pinholes
    - plot_the_frame_2coll: plot the signal
"""

import ScintSuite._Mapping as ssmapplting
from ScintSuite.SimulationCodes.FILDSIM.execution import get_energy
from ScintSuite.SimulationCodes.FILDSIM.execution import get_gyroradius
import ScintSuite.SimulationCodes.FILDSIM.forwardModelling as ssfM
import ScintSuite.SimulationCodes.Common.geometry as geometry
from ScintSuite._Plotting._ColorMaps import Gamma_II
import ScintSuite._Plotting as ssplt

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

# -----------------------------------------------------------------------------
# --- Inputs distributions
# -----------------------------------------------------------------------------

def read_ASCOT_dist(filename, pinhole_area = None, B=4, A=None, Z=None, 
                    version='5.5'):
    """
    Read a distribution coming from ASCOT

    Alex Reyner: alereyvinn@alum.us.es

    Each version has a different type of input:
        - 5.5 with pitch in [VII/V] units
        - Custom read for matlab antique files (matlab)
        - Manual file for delta tipe ion flux

    :param  filename: full path to the file
    :param  version: ASCOT output custom, default 5.5
    :param  pinhole_area: relation between pinhole and head-95% area
    :param  B: magnetic field
    :param  A: A value of the ions (in case this does not come with the input)
    :param  Z: Z value of the ions (in case this does not come with the input)

    :return out dictionary containing:
            'gyroradius': Array of gyroradius where the signal is evaluated
            'pitch': Array of pitches where the signal is evaluated
            'weight': Array of weights where the signal is evaluated
            other interesting parameters from the ascot files
    """
    logger.info("----- READING DISTRIBUTION ----- ")
    logger.info('Reading file: %s', filename)

    out={}
    ions_head = 0
    ions_pinhole = 0

    if version == '5.5':
        names = ['R', 'phi', 'z', 'energy', 'pitch', 
                        'Anum', 'Znum', 'weight', 'time']
        
        with open(filename, 'r') as file:
                lines = file.readlines()[2:]
        modified_lines = []       
        for line in lines:
            if line.startswith('#'): #skips headers
                continue
            else:
                c = line.split()
                c[4] = math.acos(float(c[4]))*180.0/math.pi
                ions_head += float(c[7])

                if pinhole_area != None: #multiply weight by pinhole_area
                    c[7] = float(c[7])*pinhole_area
                    ions_pinhole += float(c[7])
                
                modified_line = f"{c[0]} {c[1]} {c[2]} {c[3]} \
                    {c[4]} {c[5]} {c[6]} {c[7]} {c[8]} "
                modified_lines.append(modified_line)    

        # Write a second file with the pitch in degreees
        filename2=filename[:-4]+'_procesed.dat'
        with open(filename2, 'w') as file2:
            file2.write('\n'.join(modified_lines))

        # Load the data of this second file
        data = np.loadtxt(filename2)
        for i in range(len(names)):
            out[names[i]] = data[:, i]
        out['n'] = len(data[:, 0])  

        # Calculate the gyroradius
        if A==None:
            A = out['Anum']
        if Z==None:
            Z = out['Znum']
        r = get_gyroradius(out['energy'], B, A, Z)
        out['gyroradius'] = r


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
                ions_head += float(c[2])

                if pinhole_area != None: #multiply weight by pinhole_area
                    c[2] = float(c[2])*pinhole_area
                    ions_pinhole += float(c[2])
                
                modified_line = \
                    f"{c[0]} {c[1]} {c[2]}"
                modified_lines.append(modified_line)    

        # Write a second file with the pitch in degreees
        filename2=filename[:-4]+'_procesed.dat'
        with open(filename2, 'w') as file2:
            file2.write('\n'.join(modified_lines))

        # Load the data of this second file
        data = np.loadtxt(filename2)
        for i in range(len(names)):
            out[names[i]] = data[:, i]
        out['n'] = len(data[:, 0])  

        # Calculate the gyroradius
        if A==None:
            A = out['Anum']
        if Z==None:
            Z = out['Znum']
        r = get_gyroradius(out['energy'], B, A, Z)
        out['gyroradius'] = r


    if version == 'manual':
        names = ['energy', 'pitch', 'Anum', 'Znum', 'weight']
        
        # Load the data
        data = np.loadtxt(filename)
        for i in range(len(names)):
            out[names[i]] = data[:, i]
        out['n'] = len(data[:, 0])

        ions_head = np.sum(out['weight'])
        if pinhole_area != None: #multiply weight by pinhole_area
            out['weight'] *= pinhole_area
            ions_pinhole = np.sum(out['weight'])      

        # Calculate the gyroradius
        if A==None:
            A = out['Anum']
        if Z==None:
            Z = out['Znum']
        r = get_gyroradius(out['energy'], B, A, Z)
        out['gyroradius'] = r


    if version == 'locust':
        names = ['pitch', 'energy', 'rho_L', 'weight', 'gyrophase', 'ID_FILD']
        
        if A==None or Z==None:
            logger.error('No A and/or B as input. STOPING')      
            sys.exit()

        with open(filename, 'r') as file:
                lines = file.readlines()
        modified_lines = []       
        for line in lines:
            if line.startswith('#'): #skips headers
                continue
            else:
                c = line.split()
                c[0] = math.acos(float(c[0]))*180.0/math.pi

                c[1] = float(c[1])*1e6

                ions_head += float(c[3])
                if pinhole_area != None: #multiply weight by pinhole_area
                    c[3] = float(c[3])*pinhole_area
                    ions_pinhole += float(c[3])
                
                modified_line = f"{c[0]} {c[1]} {c[2]} {c[3]} {c[4]} {c[5]} "
                modified_lines.append(modified_line)    

        # Write a second file with the pitch in degreees
        filename2=filename[:-4]+'_procesed.dat'
        with open(filename2, 'w') as file2:
            file2.write('\n'.join(modified_lines))

        # Load the data of this second file
        data = np.loadtxt(filename2)
        for i in range(len(names)):
            out[names[i]] = data[:, i]
        out['n'] = len(data[:, 0])  

        # Calculate the gyroradius
        r = get_gyroradius(out['energy'], B, A, Z)
        out['gyroradius'] = r


    logger.info("ions/s arriving to the head = %e", ions_head)
    if pinhole_area != None: #multiply weight by pinhole_area
        logger.info("Area covered by pinhole = %e of the head", pinhole_area)
        logger.info("ions/s arriving to the pinhole = %e", ions_pinhole)

    return out
    

# -----------------------------------------------------------------------------
# --- Synthetic signals using the weight matrix
# -----------------------------------------------------------------------------

def obtain_WF(smap, scintillator, efficiency_flag = False, B=4, A=2, Z=2,
              pin_params: dict = {},
              scint_params: dict = {}):
    '''
    Just a wrap of things to make it easier

    Alex Reyner: alereyvinn@alum.us.es
    '''
    # Load the strike points
    smap.load_strike_points()
    # --- Grid for the weight function
    pin_options = {
        'xmin': 10,
        'xmax': 90,
        'dx': 1,
        'ymin': 1,
        'ymax': 10,
        'dy': 0.2,
    }
    scint_options = {
        'xmin': 10,
        'xmax': 90,
        'dx': 0.5,
        'ymin': 1,
        'ymax': 10,
        'dy': 0.1,
    }
    # update the matrix options
    pin_options.update(pin_params)
    scint_options.update(scint_params)

    # Build the weight function 
    if efficiency_flag == True:
        smap.build_weight_matrix(scint_options, pin_options,
                                efficiency=scintillator.efficiency,
                                B=B,A=A,Z=Z)
    else:
        smap.build_weight_matrix(scint_options, pin_options,
                                B=B,A=A,Z=Z)
    WF = smap.instrument_function

    return WF


def synthetic_signal_pr(distro, WF = None, gyrophases = np.pi, 
                        plot=False, cmap=Gamma_II()):
    """
    Synthetic signal for pinhole and scintillator in pitch-gyroradius space

    Alex Reyner: alereyvinn@alum.us.es

    :param  distro: pinhole distribution, created by one of the routines of 
        this library.
    :param  WF: weight function xarray generated in this suite. Please be sure
        that corresponds to the case you want to study.
    :param  gyrophases: used to renormalize the collimator factor. Range of 
        gyrophases that we consider. Default pi -> range of gyrophases that go 
        inside the pinhole
    :param  plot: flag to plot the synthetic signals, and the histograms in
        pitch and gyroradius.

    :return out dictionary containing remapped signals:
            PH: synthetic signal at the pinhole (PH)
            SC: synthetic signal at the scintillator (SC)
    """
    logger.info('----- COMPUTING REMAPED SYNTHETIC SIGNAL USING WF -----')
    # Get data values
    x_val = WF.coords['x'].values
    y_val = WF.coords['y'].values
    ssPH = (WF.isel(xs=1, ys=1, drop=True))*0
    xs_val = WF.coords['xs'].values
    ys_val = WF.coords['ys'].values    
    ssSC = (WF.isel(x=1, y=1, drop=True))*0
    # Define quantities
    synthetic_signal={}
    ions_pinhole = 0
    ions_pinhole_lost = 0
    ions_pinhole_found = 0
    # Number of ions that arrive to our pinhole
    input = len(distro['weight'])
    for i in range(input):
        pitch = distro['pitch'][i]
        gyro = distro['gyroradius'][i]
        weight = distro['weight'][i]
        # Control of the signal lost
        ions_pinhole += weight              
        ions_pinhole_lost += weight
        # Check if the ion is within our phase space
        p_step = np.abs(x_val[1]-x_val[0])
        r_step = np.abs(y_val[1]-y_val[0])
        if y_val.min() <= gyro <= y_val.max():
            if x_val.min() <= pitch <= x_val.max():
                # Find the nearest p-r coordinates in the pinhole
                p = ssPH.sel(x=pitch, y=gyro, method = 'nearest').x.item()
                r = ssPH.sel(x=pitch, y=gyro, method = 'nearest').y.item()
                # Fill the matrices for pinhole (PH) and scintillator (SC)
                ssPH.loc[p,r] += weight/p_step/r_step
                ssSC += (WF.sel(x=p, y=r, drop=True))\
                                *weight*(2*np.pi/gyrophases)
                # Control of the signal lost
                ions_pinhole_lost -= weight
                ions_pinhole_found += weight                

    synthetic_signal['PH'] = ssPH
    synthetic_signal['SC'] = ssSC


    logger.info("ions/s considered for the synth. sig. = %e", ions_pinhole)
    logger.info("ions/s analysed = %e", ions_pinhole_found)    
    logger.info("ions/s lost = %e", ions_pinhole_lost)

    # Default plot to control results
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
                         plot=False, cmap = Gamma_II()):
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
                     px_shift: int = 0, py_shift: int = 0):
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

    :return out dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
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
    efficiency = scint.efficiency
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

    # MAP THE SIGNAL
    # -----------------------------------------------------------------------
    logger.info("- Mapping the signal in the scintillator space...")
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
    # signal_frame.coords['x'].attrs['longname'] = 'X'
    # signal_frame.coords['x'].attrs['units'] = '[cm]'
    # signal_frame.coords['x'] = np.linspace(-1,1,cam_params['nx'])\
    #     *cam_params['nx']*cam_params['px_x_size']/2
    # signal_frame.coords['y'].attrs['longname'] = 'Y'
    # signal_frame.coords['y'].attrs['units'] = '[cm]'
    # signal_frame.coords['y'] = np.linspace(-1,1,cam_params['ny'])\
    #     *cam_params['ny']*cam_params['px_y_size']/2

    signal_frame = signal_frame.where(signal_frame>=0,0)
    integral_s=signal_frame.integrate('x').integrate('y').item()
    logger.info("   Total signal = %e photons/s", integral_s)

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
    signal_frame /=   \
       ((cam_params['px_x_size']*cam_params['px_y_size'])/(optic_params['beta']**2))
    signal_frame.attrs['long_name'] = 'Photons/cm²'
    output = {
        'smap': dsmap,
        'smapplt': dsmapplt,
        'signal_frame': signal_frame,        
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
            'signal_frame': signal in the scintillator space
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
    signal_frame = copy.deepcopy(frame['signal_frame'])
    
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
        print('a')
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
    # Cap the counts to the maximum counts
    if eliminate_saturation == True:
        final_frame = final_frame.where(final_frame < max_count, max_count) 

    # Transform the counts to integers    
    final_frame = final_frame.astype(int)

    # Substitute the signal_frame
    out['signal_frame'] = final_frame
    out['signal_frame'].attrs['long_name'] = 'Pixel counts'
    out['signal_frame'].coords['x'].attrs['longname'] = 'X'
    out['signal_frame'].coords['x'].attrs['units'] = 'pix.'
    out['signal_frame'].coords['y'].attrs['longname'] = 'Y'
    out['signal_frame'].coords['y'].attrs['units'] = 'pix.'

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


def plot_the_frame(frame, plot_smap = True, plot_scint = True, plot_FoV = True,
                   cam_params: dict={}, maxval = None,
                   figtitle = None, cmap = Gamma_II(), interpolation = 'none'):
    """
    Plot one frame, the scintillator and the strikemap
    
    Alex Reyner: alereyvinn@alum.us.es

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  maxval: adjust the maximum of the colorbar to the signal
    :param  figtitle: plot a title with the optics parameters, for example

    :return fig, ax:
    """
    logger.info("----- SIGNAL PLOT ----- ")

    frame_to_plot = frame['signal_frame']
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
    fig, ax = plt.subplots(figsize=(20/2.54,12/2.54))
    
    # frame_to_plot.plot.imshow(ax=ax, cmap=cmap, norm=LogNorm(vmin=1e14,vmax=1e22),
    #                           interpolation = interpolation, 
    #                           cbar_kwargs={'spacing': 'proportional'})
    frame_to_plot.plot.imshow(ax=ax, cmap=cmap, vmin=0, vmax=max_count,
                              interpolation = interpolation,
                              cbar_kwargs={'spacing': 'proportional'})
    
    if plot_smap == True:
        smapplt.plot_pix(ax, labels=False, marker_params={'marker':None},
                         line_params={'color':'w', 'linewidth':1.2, 'alpha':0.8})

    if plot_scint == True:
        ax.plot(scint_perim[:,0],scint_perim[:,1], color ='w', linewidth=2)

    try: #This is experimental right now, but won't mess up your code
        coll_perim=frame['collimator']
        ax.plot(coll_perim[:,0],coll_perim[:,1], color ='lime', linewidth=1)
    except:
        print('- No collimator to plot')

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
    
    # ax_param = {'xlabel': 'xpix', 'ylabel': 'ypix'}
    # ax = ssplt.axis_beauty(ax, ax_param)
    

    if figtitle != None:
        fig.suptitle(figtitle,size=12)
        
    ax.set_aspect(1)
    plt.tight_layout()
    ax.set_xlim([1,cam_params['nx']])
    ax.set_ylim([1,cam_params['ny']])
    plt.show(block=False)

    return fig, ax


def plot_noise_contributions(frame, cam_params: dict={}, maxval = False,
                             cmap=Gamma_II()):
    """
    Plot all the noise contributions
    
    Alex Reyner: alereyvinn@alum.us.es

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  maxval: want to limit the colorbar to camera range?
    :param  figtitle: plot a title with the optics parameters, for example

    :return fig, ax:
    """
    logger.info("----- NOISE CONTRIBUTIONS PLOT ----- ")

    for i in frame['noises']:
        frame_to_plot = frame['noises'][i]
        fig, ax = plt.subplots(figsize=(20/2.54,15/2.54))
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
                    
        ax.set_xlim((0,cam_params['nx']))
        ax.set_ylim((0,cam_params['ny']))
        ax_param = {'xlabel': 'x pix.', 'ylabel': 'y pix.'}
        ax = ssplt.axis_beauty(ax, ax_param)
        fig.suptitle(i)
        ax.set_aspect(1)      
        plt.tight_layout()
        print(i)    

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
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  WF: weight function corresponfing to this case
    :param  B: magnetic field (in case WF not in the input)
    :param  A: A value of the ions (in case WF not in the input)
    :param  Z: Z value of the ions (in case WF not in the input)

    :return frame dictionary adding:
            'remapped_signal': remapped camera synthetic signal
    """    
    import ScintSuite._Mapping._Common as common

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
    dummy = common.remap(smap=frame['smap'], frame=frame['signal_frame'].values, x_edges=xedges, y_edges=yedges, method='centers')
    remaped_signal = xr.DataArray(dummy, dims=('x', 'y'),
                         coords={'x':WF.xs.values, 'y':WF.ys.values})

    frame['remapped_signal'] = remaped_signal

    return frame



# These need to be worked on them a bit. Don't use it. Use the ones above.
# In a future the WF will be used for the computation of the syntt. signals.
def cretae_mapping_matrix(smap):
    gyro = smap._data['gyroradius'].data
    pitch = smap._data['pitch'].data
    x = smap._coord_pix['x']
    y = smap._coord_pix['y']


def new_synthsig_xy(distro, smap, scint, WF, collimator=None,
                     cam_params = {}, optic_params = {},
                     smapplt = None, 
                     gyrophases = 2*np.pi,
                     smoother = None, centering = False,
                     px_shift: int = 0, py_shift: int = 0):
    

                     
    """
    Maps a signal in the scintillator

    Alex Reyner: alereyvinn@alum.us.es

    :param  distro: distribution in the pinhole
    :param  smap: smap to map the signal in the xyspace
    :param  WF: weight function
    :param  smapplt: extra smap to do nice plots
    :param  scint: scintillator shape we want in the plots
    :param  cam_params: parameters of the camera
    :param  optic_params:  parameters of the optics
    :param  gyrophases: range of gyrophases considered entering the pinhole
            (to scale the collimator factor). pi (half sr) is the default.
            Usually the collimator factor is defined over a 2pi range.
    :param  smoother: adds a gaussian filter to the signal with that sigma

    :return out dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
    """
    # Check inputs and initialise the things
    dsmap = copy.deepcopy(smap)
    if smapplt != None:
        dsmapplt = copy.deepcopy(smapplt)
    else:
        dsmapplt = copy.deepcopy(dsmap)
    dscint = copy.deepcopy(scint)

    # SYNTHETIC SIGNAL
    # -----------------------------------------------------------------------
    # Calculate the synthetic signal at the scintillator
    scint_signal = synthetic_signal_pr(distro=distro, WF=WF)
    integral_s=scint_signal['SC'].integrate('xs').integrate('ys').item()
    logger.info("SS: signal at the scintillator pr %e photons/s", integral_s)

    # LOCATE AND CENTER THE SCINTILLATOR AND SMAP
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

    # MAP THE SIGNAL
    # -----------------------------------------------------------------------
    logger.info("- Mapping the signal in the scintillator space...")
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
    signal_frame = signal_frame.where(signal_frame>=0,0)
    integral_s=signal_frame.integrate('x').integrate('y').item()
    logger.info("   Total signal = %e photons/s", integral_s)

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

    # dummy = copy.deepcopy(signal_frame)*0
    # for i in range(cam_params['ny']):
    #     for j in range(cam_params['nx']):
    #         # Check if the (i, j) coordinates are inside the scintillator
    #         if scint_path.contains_point((j, i)):
    #             dummy[i, j] = 1 # 1 if it's inside the scintillator

    scint_area = copy.deepcopy(dummy)
    
    # Define the output
    signal_frame /=   \
       ((cam_params['px_x_size']*cam_params['px_y_size'])/(optic_params['beta']**2))
    signal_frame.attrs['long_name'] = 'Photons/cm²'
    output = {
        'smap': dsmap,
        'smapplt': dsmapplt,
        'signal_frame': signal_frame,        
        'scintillator': scint_perim,
        'scint_area': scint_area
    }

    if coll_geom == True: #Flagg to add the collimator
        output['collimator'] = coll_perim

    return output


def add_noise_to_frame(frame, noise_params: dict = {}, cam_params: dict = {}):
    """
    Feel free to update with more noises.
    
    Alex Reyner: alereyvinn@alum.us.es

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
    :param  optic_params:  parameters of the optics

    :return input dictionary with the noises added to signal_frame,
            and extra noises elements 
    """    

    # Copy the input dictionary in the output
    out = copy.deepcopy(frame)

    # Create a copy of the signal frame where we will add the different nosies,
    # and the area covered by the scintillator
    noise_frame = copy.deepcopy(frame['signal_frame'])
    scint_area = copy.deepcopy(frame['scint_area'])

    # Initialise the noise dictionary. All the noises are not implemented yet.
    # This needs to be done.
    noise_opt = {
        'neutrons': {},
        'camera_neutrons': {
            'percent': 1,
            'vmin': 0,
            'vmax': 1,
            'bits': cam_params['range']
        },
        'broken': {}
    }
    noise_opt.update(noise_params)
    out['noises']={}

    # Neutron and gamma noise (constant) (just in the scintillator area)
    """
    Homogenous noise through the scintillator due to neutrons and gamma 
    reaching it and producing charged particles that will give signal. Total
    noise must be given
    """
    if noise_opt['neutrons'] > 0:
        logger.info('Including neutron and gamma noise')
        num_pix = scint_area.sum().item() # how many pixel we the scint cover?
        # multiply by 4pi since we will consider the isotropic emision forward
        # in this model. The noise should be given per sr unit.
        dummy = copy.deepcopy(scint_area) 
        dummy *= noise_opt['neutrons']*4*np.pi/num_pix
        noise_frame += dummy
        out['noises']['neutrons'] = dummy

    # Put the noise frame where we had the original frame,
    # and add the nosies to the dictionary
    out['signal_frame'] = noise_frame

    return out


def add_optics_and_camera(frame, exp_time: float, eliminate_saturation = False,
                          cam_params: dict={},
                          optic_params: dict={},
                          noise_params: dict={}):
    """
    Gets the frame at the scintillator and transforms it to camera frame. 
    Adds the electronic noise also.
    Based on Jose Rueda origianl function. 
    
    Alex Reyner: alereyvinn@alum.us.es

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  cam_params: parameters of the camera
    :param  optic_params:  parameters of the optics
    :param  eliminate_saturation: if we want to get rid of saturated pixels.
            But we don't need to, this will be done with the plot vmax.

    :return input with the noises added to signal_frame
    """

    # Copy the input dictionary in the output
    out = copy.deepcopy(frame)

    # Create a copy of the signal frame to operate   
    final_frame = copy.deepcopy(out['signal_frame'])

    # Compute the maximum counts for the camera
    max_count = 2 ** cam_params['range'] - 1

    # --- Now apply all factors
    # Divide by 4\pi, ie, assume isotropic emission of the scintillator
    final_frame *= 1 / 4 / np.pi
    # Consider the solid angle covered by the optics and the transmission of
    # the beam line through the lenses and mirrors:
    final_frame *= optic_params['T'] * optic_params['omega']
    # Photon to electrons in the camera sensor (QE)
    final_frame *= cam_params['qe']
    # Electrons to counts in the camera sensor,
    final_frame /= cam_params['ad_gain']
    # Consider the exposure time
    final_frame *= exp_time

    # Do the same for the noises added before the optics
    for i in out['noises']:
        out['noises'][i] *= 1 / 4 / np.pi
        out['noises'][i] *= optic_params['T'] * optic_params['omega']
        out['noises'][i] *= cam_params['qe']
        out['noises'][i] /= cam_params['ad_gain']
        out['noises'][i] *= exp_time          

    noise_opt = {
        'neutrons': 0,
        'camera_neutrons': 0,
        'broken': 0
    }
    noise_opt.update(noise_params)        

    # Neutron impact noise
    """
    Add noise due to neutron impact on the sensor
    """
    if noise_opt['camera_neutrons'] > 0:
        logger.info('Including neutrons hitting the sensor')    
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

    # NOISES IN THE CAMERA

    # Broken pixels
    """
    Simulate broken pixels
    """
    if noise_opt['broken'] > 0:
        logger.info('Some pixel are broken')
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
    flag = ('dark_noise' in cam_params) and \
        ('readout_noise' in cam_params)
    if flag:
        logger.info('Including dark and reading noise')
        rand = np.random.default_rng()
        gauss = rand.standard_normal
        dummy = cam_params['dark_noise'] + \
            cam_params['readout_noise'] * gauss(final_frame.shape)
        readout_noise = copy.deepcopy(final_frame)*0 + dummy
        readout_noise = readout_noise.where(readout_noise > 0 ,0) # needed
        final_frame += readout_noise
        out['noises']['readout_noise'] = readout_noise

    # Cap the counts to the maximum counts
    if eliminate_saturation == True:
        final_frame = final_frame.where(final_frame < max_count, max_count) 

    # Transform the counts to integers    
    final_frame = final_frame.astype(int)

    # Change the previous frame for the camera frame
    out['signal_frame'] = final_frame
    

    return out



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
            'signal_frame': total signal in the scintillator space
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
    output['signal_frame'] = signal_frame
    output['scintillator'] = scint_perim
    output['scint_area'] = scint_area

    return output


def plot_the_frame_2coll(frame, plot_smap = True, plot_scint = True, plot_FoV = True,
                   cam_params: dict={}, maxval = None,
                   figtitle = None, cmap=Gamma_II()):
    """
    Plot one frame, the scintillator and the strikemap
    
    Alex Reyner: alereyvinn@alum.us.es

    :param  frame dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
            'noises': all the different noises as matrices
    :param  maxval: adjust the maximum of the colorbar to the signal
    :param  figtitle: plot a title with the optics parameters, for example

    :return fig, ax:
    """
    logger.info("----- 2 COLLIMATOR SIGNAL PLOT ----- ")
    
    frame_to_plot = frame['signal_frame']
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
