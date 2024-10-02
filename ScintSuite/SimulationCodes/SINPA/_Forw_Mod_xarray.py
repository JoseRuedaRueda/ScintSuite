"""
SINPA functions to compute synthetic signals

Alex Reyner: alereyvinn@alum.us.es
"""

import os
import numpy as np
import ScintSuite as ss
import ScintSuite._Mapping as ssmapplting
import ScintSuite._IO as ssio
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import xarray as xr
import math
import scipy.spatial as spsp
import scipy.ndimage as spnd
import copy
from matplotlib.colors import LinearSegmentedColormap

from ScintSuite.SimulationCodes.FILDSIM.execution import get_energy
from ScintSuite.SimulationCodes.FILDSIM.execution import get_gyroradius
import ScintSuite.SimulationCodes.FILDSIM.forwardModelling as ssfM
from skimage import measure
from matplotlib.patches import Circle

# -----------------------------------------------------------------------------
# --- Inputs distributions
# -----------------------------------------------------------------------------

def read_ASCOT_dist(filename, pinhole_area = None, B=4, A=None, Z=None, 
                    version='5.5'):
    """
    Read a distribution coming from ASCOT, with pitch in [VII/V] units (5.5)
    Custom read for matlab antique files (matlab)
    Manual file for delta tipe ion flux

    Alex Reyner: alereyvinn@alum.us.es

    :param  file: full path to the file
    :param  version: ASCOT output custom, default 5.5
    :param  pinhole_area: relation between pinhole and head-95% area

    :return out: distribution ready to be used,
    :return out dictionary containing:
            'gyroradius': Array of gyroradius where the signal is evaluated
            'pitch': Array of pitches where the signal is evaluated
            'weight': Array of weights where the signal is evaluated
            other interesting parameters from the ascot files
    """
    print('Reading file: ', filename)

    out={}
    ions_head = 0
    ions_pinhole = 0

    if version == '5.5':
        names = ['R', 'phi', 'z', 'energy', 'pitch', 
                        'Anum', 'Znum', 'weight', 'time']
        
        with open(filename, 'r') as file:
                lines = file.readlines()
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

    print("ions/s arriving to the head = "+str(f"{ions_head:e}"))
    if pinhole_area != None: #multiply weight by pinhole_area
        print("Considered the relation of areas between pinhole and head.")
        print("Area covered by pinhole: "+str(f"{pinhole_area:.2e}"+" of the head"))
        print("ions/s arriving to the pinhole = "+str(f"{ions_pinhole:e}"))

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


def synthetic_signal_pr(distro, WF = None, gyrophases = np.pi, plot=False,
                        smooth = False,
                        xmin = None, xmax = None, ymin = None, ymax = None):
    """
    Generates synthetic signal in both pinhole and scintillator in pr space
    using the WF. It's quite fast.

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

    :return ssPH: synthetic signal at the pinhole (PH)
    :return ssSC: synthetic signal at the scintillator (SC)
    """
    print(' ----- COMPUTING REMAPED SYNTHETIC SIGNAL USING WF -----')
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

    # Smooth the pinhole signal to match scintillator
    if smooth == True :   
        dummy = ssPH.interp(y=ys_val, x=xs_val, method='nearest')
        ssPH = dummy.where(dummy >=0.0, 0)

    print("ions/s considered for the synth. sig. = "+str(f"{ions_pinhole:e}"))
    print("ions/s analysed = "+str(f"{ions_pinhole_found:e}"))    
    print("ions/s lost = "+str(f"{ions_pinhole_lost:e}"))

    # Plotting block
    if plot == True:
        fig, ax = plt.subplots(2,2, figsize=(8, 6),
                                    facecolor='w', edgecolor='k') 
        cmap = ss.plt.Gamma_II()
        # Plot of the synthetic signals, pinhole and scintillator
        ax_param = {'fontsize': 10, \
                    'xlabel': 'Pitch [º]', 'ylabel': '$r_l [cm]$'}         
        ssPH.transpose().plot.imshow(ax=ax[0,0], xlim=[xmin,xmax], 
                                     ylim=[ymin,ymax], cmap=cmap,
                                     vmax=0.5*ssPH.max().item(),
                                     cbar_kwargs={"label": 'ions/(s cm deg)'})
        ax[0,0] = ss.plt.axis_beauty(ax[0,0], ax_param)
        ax[0,0].set_title("Pinhole")    
        ssSC.transpose().plot.imshow(ax=ax[0,1], xlim=[xmin,xmax], 
                                     ylim=[ymin,ymax], cmap=cmap,
                                     vmax=0.5*ssSC.max().item(),
                                     cbar_kwargs={"label": 'ions/(s cm deg)'})
        ax[0,1] = ss.plt.axis_beauty(ax[0,1], ax_param)
        ax[0,1].set_title("Scintillator")

        # Plot of the distributions of pitch and gyroradius
        ax_options_profiles = {'fontsize': 10, 'ylabel': 'Signal [a.u.]'}
        (ssPH.sum(dim='y')/ssPH.sum(dim='y').integrate('x')).plot.\
            line(ax=ax[1,0], color='black', label='Pinhole')
        (ssSC.sum(dim='ys')/ssSC.sum(dim='ys').integrate('xs')).plot.\
            line(ax=ax[1,0], color='red', label='Scintillator')
        ax_options_profiles['xlabel'] = 'Pitch [$\\degree$]'  
        ax[1,0] = ss.plt.axis_beauty(ax[1,0], ax_options_profiles)
        ax[1,0].legend()        
        (ssPH.sum(dim='x')/ssPH.sum(dim='x').integrate('y')).plot.\
            line(ax=ax[1,1], color='black', label='Pinhole')
        (ssSC.sum(dim='xs')/ssSC.sum(dim='xs').integrate('ys')).plot.\
            line(ax=ax[1,1], color='red', label='Scintillator')      
        ax_options_profiles['xlabel'] = 'Gyroradius [cm]'
        ax[1,1] = ss.plt.axis_beauty(ax[1,1], ax_options_profiles)
        ax[1,1].legend()

        fig.tight_layout()
        plt.show(block = False)

    return synthetic_signal


def pr_space_to_pe_space(synthetic_signal, B=4, A=2, Z=2, plot=False,
                        xmin = None, xmax = None, ymin = None, ymax = None):
    """
    Transfors the pitch-gyroradius space to pitch-energy.

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

    :return xarray with the signals in the pe space.
    """
    print(' ----- GOING FROM p-r TO p-e SPACE ----- ')
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
        cmap = ss.plt.Gamma_II()
        # Plot of the synthetic signals, pinhole and scintillator
        ax_param = {'fontsize': 10, \
            'xlabel': 'Pitch [º]', 'ylabel': 'Energy [eV]'}         
        ssPH_pe.transpose().plot.imshow(ax=ax[0,0], xlim=[xmin,xmax], 
                                     ylim=[ymin,ymax], cmap=cmap,
                                     cbar_kwargs={"label": 'ions/(s cm deg)'})
        ax[0,0] = ss.plt.axis_beauty(ax[0,0], ax_param)
        ax[0,0].set_title("Pinhole")    
        ssSC_pe.transpose().plot.imshow(ax=ax[0,1], xlim=[xmin,xmax], 
                                     ylim=[ymin,ymax], cmap=cmap,
                                     cbar_kwargs={"label": 'ions/(s cm deg)'})
        ax[0,1] = ss.plt.axis_beauty(ax[0,1], ax_param)
        ax[0,1].set_title("Scintillator")

        # Plot of the distributions of pitch and gyroradius
        ax_options_profiles = {'fontsize': 10, 'ylabel': 'Signal [a.u.]'}
        (ssPH_pe.sum(dim='y')/ssPH_pe.sum(dim='y').integrate('x'))\
            .plot.line(ax=ax[1,0], color='black', label='Pinhole')
        (ssSC_pe.sum(dim='ys')/ssSC_pe.sum(dim='ys').integrate('xs'))\
            .plot.line(ax=ax[1,0], color='red', label='Scintillator')
        ax_options_profiles['xlabel'] = 'Pitch [$\\degree$]'  
        ax[1,0] = ss.plt.axis_beauty(ax[1,0], ax_options_profiles)
        ax[1,0].legend()        
        (ssPH_pe.sum(dim='x')/ssPH_pe.sum(dim='x').integrate('y'))\
            .plot.line(ax=ax[1,1], color='black', label='Pinhole')
        (ssSC_pe.sum(dim='xs')/ssSC_pe.sum(dim='xs').integrate('ys'))\
            .plot.line(ax=ax[1,1], color='red', label='Scintillator')      
        ax_options_profiles['xlabel'] = 'Energy [eV]'
        ax[1,1] = ss.plt.axis_beauty(ax[1,1], ax_options_profiles)
        ax[1,1].legend()

        fig.tight_layout()
        plt.show(block = False)

    return out



# -----------------------------------------------------------------------------
# --- Synthetic signal in the scintillator space and camera frame
# -----------------------------------------------------------------------------
"""
Workflow:
    0. Obtain the distribution and define the inputs 
    1. Run original_synthsig_xy to map the signal in the scintillator space
    2. Insert the different noises, optic system and the camera
       - Option to do it separately or at the same time (recomended)
            with noise_optics_camera

    - You can plot at any step given the frame and even plot the noises
"""

def original_synthsig_xy(distro, smap, scint,
                     cam_params = {}, optic_params = {},
                     smapplt = None, 
                     gyrophases = np.pi,
                     smoother = None,
                     scint_params: dict = {},
                     px_shift: int = 0, py_shift: int = 0):
    """
    Maps a signal in the scintillator. 
    Based on the origianl function by Jose Rueda

    Alex Reyner: alereyvinn@alum.us.es

    :param  distro: distribution in the pinhole
    :param  smap: smap to map the signal in the xyspace
    :param  smapplt: extra smap to do nice plots
    :param  scint: scintillator shape we want in the plots
    :param  cam_params: parameters of the camera
    :param  optic_params:  parameters of the optics
    :param  gyrophases: range of gyrophases considered entering the pinhole
            (to scale the collimator factor). pi (half sr) is the default.
            Usually the collimator factor is defined over a 2pi range.
    :param  smoother: adds a gaussian filter to the signal with that sigma
    :param  scint_synthetic_signal_params: grid to remap the frame

    :return out dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
    """
    print(" ----- BUILDING THE ORIGINAL FRAME ----- ")

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
    print('- Computing synthetic signal...')
    # Calculate the synthetic signal at the scintillator
    scint_signal = ssfM.synthetic_signal_remap(distro, dsmap,
                                          efficiency=efficiency,
                                          **scint_options)
    
    # LOCATE AND CENTER THE SCINTILLATOR AND SMAP
    # -----------------------------------------------------------------------
    print('- Locating the smap and scintillator...')
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
        print('      Optics magnification, beta: ', beta)
        optic_params['beta'] = beta
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
    
    # MAP THE SIGNAL
    # -----------------------------------------------------------------------
    print("- Mapping the signal in the scintillator space...")
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
    print('- Apllying corrections if needed...')
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
    print('- Building the signal xarray...')
    signal_frame = xr.DataArray(synthetic_frame, dims=('y', 'x'),
            coords={'y':np.linspace(1,cam_params['ny'],cam_params['ny']),
                    'x':np.linspace(1,cam_params['nx'],cam_params['nx'])})
    signal_frame = signal_frame.where(signal_frame>=0,0)
    integral_s=signal_frame.integrate('x').integrate('y').item()
    print("      Total signal =", f"{integral_s:e}","photons/s")

    # Build the scintillator perimeter and find the area in the pixel space
    print('- Building the scintillator perimeter and xarray...')
    
    scint_perim = scint_ConvexHull(dscint)

    scint_path = Path(scint_perim, closed=True)
    dummy = copy.deepcopy(signal_frame)*0
    for i in range(cam_params['ny']):
        for j in range(cam_params['nx']):
            # Check if the (i, j) coordinates are inside the scintillator
            if scint_path.contains_point((j, i)):
                dummy[i, j] = 1 # 1 if it's inside the scintillator
    scint_area = copy.deepcopy(dummy)
    
    # Define the output
    output = {
        'smap': dsmap,
        'smapplt': dsmapplt,
        'signal_frame': signal_frame,        
        'scintillator': scint_perim,
        'scint_area': scint_area
    }

    return output


def noise_optics_camera(frame, exp_time: float, eliminate_saturation = False,
                          cam_params: dict={},
                          optic_params: dict={},
                          noise_params: dict={},
                          radiometry = None):
    """
    Gets the frame at the scintillator and transforms it to camera frame. 
    Feel free to update with more nbeoises.
    
    Alex Reyner: alereyvinn@alum.us.es

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
            But we don't need to, this will be done with the plot vmax. 
    :param  radiometry: This adds the FoV and the transmission of the optics
            (the difference between regions assuming center equal to 1. The NA
            is a different thing). Needs to be a matrix with r (radius of FOV) 
            in x and theta (angle) in y. This matrix can't have any NaN value.
            Worry not about the number of data, it will be interpolated.

    :return input with the noises added to signal_frame
            also the noise contribution of everything itself
    """
    print(" ----- NOISE, OPTICS AND CAMERA ----- ")

    # Copy the input dictionary in the output
    out = copy.deepcopy(frame)
    # Create a copy of the signal frame where we will operate,
    # and the area covered by the scintillator
    final_frame = copy.deepcopy(frame['signal_frame'])
    scint_area = copy.deepcopy(frame['scint_area'])
    scint = copy.deepcopy(frame['scintillator'])
    radiometry = copy.deepcopy(radiometry)
    # Initialise the noise dictionary. All the noises are not implemented yet.
    # This needs to be done.
    noise_opt = {
        'neutrons': {},
        'camera_neutrons': {
            'percent': 0,
            'vmin': 0,
            'vmax': 1,
            'bits': cam_params['range']
        },
        'broken': {}
    }
    noise_opt.update(noise_params)
    out['noises']={}

    # NOISES IN THE SCINTILLATOR
    # -----------------------------------------------------------------------
    # Neutron and gamma noise (constant) (just in the scintillator area)
    """
    Homogenous noise through the scintillator due to neutrons and gamma 
    reaching it and producing charged particles that will give signal. Total
    noise must be given
    """
    if noise_opt['neutrons'] > 0:
        print('- Neutron and gamma noise')
        num_pix = scint_area.sum().item() # how many pixel we the scint cover?
        # multiply by 4pi since we will consider the isotropic emision forward
        # in this model. The noise should be given per sr unit.
        dummy = copy.deepcopy(scint_area) 
        dummy *= noise_opt['neutrons']*4*np.pi/num_pix
        final_frame += dummy
        out['noises']['neutrons'] = dummy

    # OPTICS
    # -----------------------------------------------------------------------
    # Compute the maximum counts for the camera
    max_count = 2 ** cam_params['range'] - 1
    # Now apply all factors
    # Divide by 4\pi, ie, assume isotropic emission of the scintillator
    final_frame *= 1 / 4 / np.pi
    # Consider the solid angle covered by the optics and the transmission of
    # the beam line through the lenses and mirrors:
    final_frame *= optic_params['T'] * optic_params['Omega']
    # Photon to electrons in the camera sensor (QE)
    final_frame *= cam_params['qe']
    # Electrons to counts in the camera sensor,
    final_frame /= cam_params['ad_gain']
    # Consider the exposure time
    final_frame *= exp_time
    
    # Do the same for the noises added before the optics
    for i in out['noises']:
        out['noises'][i] *= 1 / 4 / np.pi
        out['noises'][i] *= optic_params['T'] * optic_params['Omega']
        out['noises'][i] *= cam_params['qe']
        out['noises'][i] /= cam_params['ad_gain']
        out['noises'][i] *= exp_time    

    # -----------------------------------------------------------------------
    # Add the optic FoV and the radiometry filter (stored as a noise)
    try:
        # Find the FoV in the scintillator using the % respect SW corner
        xsc_percent = optic_params['FoV'][0]
        ysc_percent = optic_params['FoV'][1]
        xsc_min = frame['scintillator'][:,0].min()
        xsc_max = frame['scintillator'][:,0].max()
        ysc_min = frame['scintillator'][:,1].min()
        ysc_max = frame['scintillator'][:,1].max()
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
        # dummy = dummy.drop_dims('r','t')
        # Add to the filter itself to the noise output 
        out['noises']['radiometry'] = dummy
        # Now we want 0 for the filter, not NaN
        dummy = dummy.where(dummy >= 0, 0)
        final_frame *= dummy
        print('- Radiometry filter applied')

        # Add FoV to the output 
        out['FoV_vect'] = [x_FoV,y_FoV,r_FoV]
        print('- FoV determined, in (x,y,r) pix = (',f"{x_FoV:.0f}"+',', 
                f"{y_FoV:.0f}"+',', f"{r_FoV:.0f}"+')')

    except: # Try to at least define the FoV
        try:
            # Find the FoV in the scintillator using the % respect SW corner
            xsc_percent = optic_params['FoV'][0]
            ysc_percent = optic_params['FoV'][1]
            xsc_min = frame['scintillator'][:,0].min()
            xsc_max = frame['scintillator'][:,0].max()
            ysc_min = frame['scintillator'][:,1].min()
            ysc_max = frame['scintillator'][:,1].max()
            x_FoV = (xsc_max - xsc_min) * xsc_percent + xsc_min
            y_FoV = (ysc_max - ysc_min) * ysc_percent + ysc_min
            r_FoV = optic_params['FoV'][2] * optic_params['beta'] \
                / (cam_params['px_x_size']*1000)
            out['FoV_vect'] = [x_FoV,y_FoV,r_FoV]
            print('- FoV determined, in (x,y,r) pix = (',f"{x_FoV:.0f}"+',', 
                  f"{y_FoV:.0f}"+',', f"{r_FoV:.0f}"+')')
        except:
            print('- No FoV, or not good format of the input')
    

    # NOISES IN THE CAMERA  
    # -----------------------------------------------------------------------
    # Neutron impact noise
    """
    Add noise due to neutron impact on the sensor
    """
    if noise_opt['camera_neutrons'] > 0:
        print('- Neutrons hitting the sensor')    
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
        print('- Some pixel are broken')
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
        print('- Dark and reading noise')
        rand = np.random.default_rng()
        gauss = rand.standard_normal
        dummy = cam_params['dark_noise'] + \
            cam_params['readout_noise'] * gauss(final_frame.shape)
        readout_noise = copy.deepcopy(final_frame)*0 + dummy
        readout_noise = readout_noise.where(readout_noise > 0 ,0) # needed
        final_frame += readout_noise
        out['noises']['readout_noise'] = readout_noise

    # ADJUST THE CAMERA FRAME AND OUTPUT  
    # -----------------------------------------------------------------------
    # Cap the counts to the maximum counts
    if eliminate_saturation == True:
        final_frame = final_frame.where(final_frame < max_count, max_count) 
    # Transform the counts to integers    
    final_frame = final_frame.astype(int)
    # Change the previous frame for the camera frame
    out['signal_frame'] = final_frame

    return out


def plot_the_frame(frame, plot_smap = True, plot_scint = True, plot_FoV = True,
                   cam_params: dict={},
                   maxval = True, figtitle = None):
    """
    Plot one frame, the scintillator and the strikemap. 
    
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
    print(" ----- SIGNAL PLOT ----- ")

    frame_to_plot = frame['signal_frame']
    smapplt = frame['smapplt']
    scint_perim = frame['scintillator']

    # Establish vmax as: 1) max counts, 2) 50% of the signal
    if maxval == True:
        max_count = 2 ** cam_params['range'] - 1
        print('- Maximum set to camera range')
    else:
        max_count = frame_to_plot.max().item()
        print('- Maximum set to 0.5 of the signal')

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(7,4))
    ax.set_aspect(1)      
    
    frame_to_plot.plot.imshow(ax=ax, cmap=ss.plt.Gamma_II(),
                    vmin=0, vmax=max_count,
                    cbar_kwargs={"label": 'Pixel counts','spacing': 'proportional'})

    if plot_smap == True:
        smapplt.plot_pix(ax, labels=False, marker_params={'marker':None},
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
            print('- No FoV plotted beacuse whatever')

    ax.set_xlim((0,cam_params['nx']))
    ax.set_ylim((0,cam_params['ny']))
    plt.xticks(fontsize=9)   
    plt.yticks(fontsize=9)
    ax_param = {'fontsize': 10, \
                    'xlabel': 'x pix.', 'ylabel': 'y pix.'}
    ax = ss.plt.axis_beauty(ax, ax_param)

    if figtitle != None:
        fig.suptitle(figtitle, fontsize=11)
        
    plt.show()

    return fig, ax


def plot_noise_contributions(frame, cam_params: dict={}, maxval = False):
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
    print(" ----- NOISE CONTRIBUTIONS PLOT ----- ")

    for i in frame['noises']:
        frame_to_plot = frame['noises'][i]
        if i == 'broken':
            bw_cmap =  LinearSegmentedColormap.from_list(
                'mycmap', ['black', 'white'], N=2)
            fig, ax = plt.subplots(figsize=(7,4))
            ax.set_aspect(1)   
            frame_to_plot.plot.imshow(ax=ax, cmap=bw_cmap,
                    vmin=0, vmax=1,
                    cbar_kwargs={"label": '  Broken pixel             Functioning pixel',
                                 'spacing': 'proportional'})
        elif i == 'radiometry':
            fig, ax = plt.subplots(figsize=(7,4))
            ax.set_aspect(1)
            frame_to_plot.plot.imshow(ax=ax, center=1,
                    cbar_kwargs={"label": 'Relative Illumination', 'spacing': 'proportional'})
        else:
            max_count = 2 ** cam_params['range'] - 1       
            fig, ax = plt.subplots(figsize=(7,4))
            ax.set_aspect(1)   
            frame_to_plot.plot.imshow(ax=ax, cmap=ss.plt.Gamma_II(),
                    vmin=0, vmax=max_count,
                    cbar_kwargs={"label": 'Pixel counts','spacing': 'proportional'})
                    
        ax.set_xlim((0,cam_params['nx']))
        ax.set_ylim((0,cam_params['ny']))
        plt.xticks(fontsize=9)   
        plt.yticks(fontsize=9)
        ax_param = {'fontsize': 10, \
                        'xlabel': 'x pix.', 'ylabel': 'y pix.'}
        ax = ss.plt.axis_beauty(ax, ax_param)
        fig.suptitle(i, fontsize=11)
        print(i)    

    plt.show()
 
    return


# These need to be worked on them a bit. Use the ones above. Those are the baseline.

def new_synthsig_xy(distro, smap, scint, WF,
                     cam_params = {}, optic_params = {},
                     smapplt = None, 
                     gyrophases = 2*np.pi,
                     smoother = None,
                     SC_params : dict = {},
                     px_shift: int = 0, py_shift: int = 0):
    """
    Maps a signal in the scintillator. 
    Based on the origianl function by Jose Rueda

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
    :param  SC_params: grid of the scintillator phase space

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
    scint_signal = synthetic_signal_pr(distro, WF=WF, gyrophases=gyrophases,
                    smooth=False, plot=True)
    integral_s=scint_signal['SC'].integrate('xs').integrate('ys').item()
    print("SS: signal at the scintillator pr", f"{integral_s:e}"+" photons/s")
    # LOCATE AND CENTER THE SCINTILLATOR AND SMAP
    # -----------------------------------------------------------------------
    print('Locating the smap and scintillator...')
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
        print('Optics magnification, beta: ', beta)
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
    print(" ----- MAPPING IN THE XY SPACE ----- ")
    n_gyr = scint_signal['SC']['ys'].size
    n_pitch = scint_signal['SC']['xs'].size
    synthetic_frame = np.zeros(dsmap._grid_interp['gyroradius'].shape)
    for ir in range(n_gyr):
        # Gyroradius limits to integrate
        gmin = scint_signal['SC']['ys'].values[ir] - SC_params['dy'] / 2.
        gmax = scint_signal['SC']['ys'].values[ir] + SC_params['dy'] / 2.
        for ip in range(n_pitch):
            # Pitch limits to integrates
            pmin = scint_signal['SC']['xs'].values[ip] - SC_params['dx'] / 2.
            pmax = scint_signal['SC']['xs'].values[ip] + SC_params['dx'] / 2.
            # Look for the pixels which cover this region:
            flags = (dsmap._grid_interp['gyroradius'] >= gmin) \
                * (dsmap._grid_interp['gyroradius'] < gmax) \
                * (dsmap._grid_interp['pitch'] >= pmin) \
                * (dsmap._grid_interp['pitch'] < pmax)
            flags = flags.astype(bool)
            # If there are some pixels, just divide the value weight among them
            n = np.sum(flags)
            if n > 0:
                synthetic_frame[flags] = scint_signal['SC'][ip, ir] / n \
                    * SC_params['dy'] * SC_params['dx']
                
    # CORRECTIONS
    # -----------------------------------------------------------------------
    print('Apllying corrections if needed...')
    # Build the original frame in the pixel space, and smooth it if wanted
    if smoother != None:
        dummy = copy.deepcopy(synthetic_frame)
        synthetic_frame = spnd.gaussian_filter(dummy,sigma=smoother)

    # BUILD THE OUTPUT
    # -----------------------------------------------------------------------
    # Transform to xarray
    print('Building the signal xarray...')
    signal_frame = xr.DataArray(synthetic_frame, dims=('y', 'x'),
            coords={'y':np.linspace(1,cam_params['ny'],cam_params['ny']),
                    'x':np.linspace(1,cam_params['nx'],cam_params['nx'])})
    signal_frame = signal_frame.where(signal_frame>=0,0)
    integral_s=signal_frame.integrate('x').integrate('y').item()
    print("SS: signal at the scintillator xy", f"{integral_s:e}"+" photons/s")
    # Build the scintillator perimeter and find the area in the pixel space
    print('Building the scintillator xarray...')
    dummy = copy.deepcopy(signal_frame)*0
    xdum = dscint._coord_pix['x']
    ydum = dscint._coord_pix['y'] 
    allPts = np.column_stack((xdum,ydum))
    hullPts = spsp.ConvexHull(allPts)
    scint_x = np.concatenate((allPts[hullPts.vertices,0],\
                              np.array([allPts[hullPts.vertices,0][0]])))
    scint_y = np.concatenate((allPts[hullPts.vertices,1],\
                              np.array([allPts[hullPts.vertices,1][0]])))
    scint_perim = np.column_stack((scint_x, scint_y))
    scint_path = Path(scint_perim)
    for i in range(cam_params['ny']):
        for j in range(cam_params['nx']):
            # Check if the (i, j) coordinates are inside the scintillator
            if scint_path.contains_point((j, i)):
                dummy[i, j] = 1 # 1 if it's inside the scintillator
    scint_area = copy.deepcopy(dummy)
    
    # Define the output
    output = {
        'smap': dsmap,
        'smapplt': dsmapplt,
        'scintillator': scint_perim,
        'signal_frame': signal_frame,
        'scint_area': scint_area
    }

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
        print('Including neutron and gamma noise')
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
    final_frame *= optic_params['T'] * optic_params['Omega']
    # Photon to electrons in the camera sensor (QE)
    final_frame *= cam_params['qe']
    # Electrons to counts in the camera sensor,
    final_frame /= cam_params['ad_gain']
    # Consider the exposure time
    final_frame *= exp_time

    # Do the same for the noises added before the optics
    for i in out['noises']:
        out['noises'][i] *= 1 / 4 / np.pi
        out['noises'][i] *= optic_params['T'] * optic_params['Omega']
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
        print('Including neutrons hitting the sensor')    
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
        print('Some pixel are broken')
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
        print('Including dark and reading noise')
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
    Maps a signal in the scintillator. 
    Based on the origianl function by Jose Rueda

    Alex Reyner: alereyvinn@alum.us.es

    :param  distro: distribution in the pinhole
    :param  scint: scintillator shape we want in the plots
    :param  smaps: dictionary with the two smap objects (left, right)
    :param  smapsplt: extra smaps to plot nice
    :param  cam_params: parameters of the camera
    :param  optic_params:  parameters of the optics
    :param  gyrophases: range of gyrophases considered entering the pinhole
            (to scale the collimator factor). pi (half sr) is the default.
            Usually the collimator factor is defined over a 2pi range.
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
    print('- Locating the smap and scintillator...')
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
        print('      Optics magnification, beta: ', beta)
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
        print('- Computing synthetic',item,'signal...')
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
        print('- Mapping the ',item,'signal in the scintillator space...')
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
        print('- Apllying corrections if needed...')
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
        print('- Building the',item,'signal xarray...')
        side_frame = xr.DataArray(synthetic_frame, dims=('y', 'x',),
                coords={'y':np.linspace(1,cam_params['ny'],cam_params['ny']),
                        'x':np.linspace(1,cam_params['nx'],cam_params['nx'])})
        side_frame = side_frame.where(side_frame>=0,0)
        
        output['smap']['left'] = dsmap
        output['smapplt'][item] = dsmapplt
        output['side_signal'][item] = side_frame
 
    signal_frame = output['side_signal']['left'] +\
                   output['side_signal']['right']
    integral=signal_frame.integrate('x').integrate('y').item()
    print("      Total signal =", f"{integral:e}","photons/s")

    # BUILD THE FINAL THINGS, AND MOST IMPORTANT, OF THE OUTPUT
    # -------------------------------------------------------------------
    # Define the scintillator perimeter and find the area in the pixel space
    print('- Building the scintillator perimeter and xarray...')
    
    scint_perim = scint_ConvexHull(dscint)

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
                   cam_params: dict={},
                   maxval = True, figtitle = None, smap_val = None):
    """
    Plot one frame, the scintillator and the strikemap. 
    
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
    print(" ----- 2 COLLIMATOR SIGNAL PLOT ----- ")
    
    frame_to_plot = frame['signal_frame']
    smapplt = frame['smapplt']
    scint_perim = frame['scintillator']

    # Establish vmax as: 1) max counts, 2) 50% of the signal
    if maxval == True:
        max_count = 2 ** cam_params['range'] - 1
        print('- Maximum set to camera range')
    else:
        max_count = frame_to_plot.max().item()*0.5
        print('- Maximum set to 0.5 of the signal')

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(7,4))
    ax.set_aspect(1)      
    
    frame_to_plot.plot.imshow(ax=ax, cmap=ss.plt.Gamma_II(),
                    vmin=0, vmax=max_count,
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
            print('- No FoV plotted beacuse whatever')

    ax.set_xlim((0,cam_params['nx']))
    ax.set_ylim((0,cam_params['ny']))
    plt.xticks(fontsize=9)   
    plt.yticks(fontsize=9)
    ax_param = {'fontsize': 10, \
                    'xlabel': 'x pix.', 'ylabel': 'y pix.'}
    ax = ss.plt.axis_beauty(ax, ax_param)

    if figtitle != None:
        fig.suptitle(figtitle, fontsize=11)
        
    plt.show()

    return fig, ax




# -----------------------------------------------------------------------------
# --- Auxiliary functions
# -----------------------------------------------------------------------------

def scint_ConvexHull(scint):

    xdum = scint._coord_pix['x']
    ydum = scint._coord_pix['y'] 
    allPts = np.column_stack((xdum,ydum))
    hullPts = spsp.ConvexHull(allPts)
    scint_x = np.concatenate((allPts[hullPts.vertices,0],\
                              np.array([allPts[hullPts.vertices,0][0]])))
    scint_y = np.concatenate((allPts[hullPts.vertices,1],\
                              np.array([allPts[hullPts.vertices,1][0]])))
    scint_perim = np.column_stack((scint_x, scint_y))

    return scint_perim

# work on progress
def scint_alpha(scint):
    xdum = scint._coord_pix['x']
    ydum = scint._coord_pix['y'] 
    allPts = np.column_stack((xdum,ydum))
    tri = spsp.Delaunay(allPts)
    print(tri.simplices)
    print(allPts[tri.simplices])
    print(allPts[tri.simplices,0])
    fig,ax = plt.subplots()
    ax.plot(allPts[:,0],allPts[:,1], color ='r', linewidth=3)
    ax.plot(allPts[tri.simplices,0],allPts[tri.simplices,1], color ='b', linewidth=3)

    scint_perim = 1

    return scint_perim
