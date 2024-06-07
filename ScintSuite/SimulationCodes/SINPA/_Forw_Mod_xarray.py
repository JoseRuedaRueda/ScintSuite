"""
SINPA functions to compute synthetic signals

Alex Reyner: alereyvinn@alum.us.es
"""

import os
import numpy as np
import ScintSuite as ss
import ScintSuite._Mapping as ssmapplting
import ScintSuite._IO as ssio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import xarray as xr
import math
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
import copy
from ScintSuite.SimulationCodes.FILDSIM.execution import get_energy
from ScintSuite.SimulationCodes.FILDSIM.execution import get_gyroradius
from ScintSuite.SimulationCodes.FILDSIM import synthetic_signal_remap


# -----------------------------------------------------------------------------
# --- Inputs distributions
# -----------------------------------------------------------------------------

def read_ASCOT_dist(filename, pinhole_area = None, B=4, A=None, Z=None, 
                    version=5.5):
    """
    Read a distribution coming from ASCOT, old (4)
    Read a distribution coming from ASCOT, with pitch in [VII/V] units (5.5)
    Custom read for matlab antique files (matlab)

    Alex Reyner: alereyvinn@alum.us.es

    :param  file: full path to the file
    :param  version: ASCOT version, default 5.5
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
    Generates synthetic signal in both pinhole and scintillator in pr space.

    Alex Reyner: alereyvinn@alum.us.es

    :param  distro: pinhole distribution, created by one of the routines of 
        this library.
    :param  WF: weight function xarray generate inthis suite. Please be sure
        that corresponds to the case you wnat to study.
    :param  gyrophases: used to renormalize the collimator factor. Range of 
        gyrophases that we consider. Default pi -> range of gyrophases that go 
        inside the pinhole
    :param  plot: flag to plot the synthetic signals, and the histograms in
        pitch and gyroradius.

    :return ssPH: synthetic signal at the pinhole (PH)
    :return ssSC: synthetic signal at the scintillator (SC)
    """

    x_val = WF.coords['x'].values
    y_val = WF.coords['y'].values
    ssPH = (WF.isel(xs=1, ys=1, drop=True))*0
    xs_val = WF.coords['xs'].values
    ys_val = WF.coords['ys'].values    
    ssSC = (WF.isel(x=1, y=1, drop=True))*0

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
        if y_val.min() <= gyro <= y_val.max():
            if x_val.min() <= pitch <= x_val.max():
                # Find the nearest p-r coordinates in the pinhole
                p = ssPH.sel(x=pitch, y=gyro, method = 'nearest').x.item()
                r = ssPH.sel(x=pitch, y=gyro, method = 'nearest').y.item()
                # Find the steps
                p_step = (x_val.max()-x_val.min())/(len(x_val)-1)
                r_step = (y_val.max()-y_val.min())/(len(y_val)-1)
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

    # Synthetic signal input
    ssPH_pr = synthetic_signal['PH']
    ssSC_pr = synthetic_signal['SC']

    # Replicate the xarray.
    # Necessary to multiply by one, to "break" the relation between matrices.
    ssPH_pe = ssPH_pr*1
    ssSC_pe = ssSC_pr*1

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

    ssPH_pe = ssPH_pe.where(ssPH_pe >=0.0, 0)
    ssSC_pe = ssSC_pe.where(ssSC_pe >=0.0, 0)

    out = {}
    out['PH'] = ssPH_pe
    out['SC'] = ssSC_pe

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
    1. Run original_synthsug_xy to map the signal in the scintillator space
    2. Insert the different noises
    3. Take into account the  optic system and the camera

    - You can plot any step from 1 to 3
"""

def original_synthsig_xy(pinhole_distribution: dict, efficiency, 
                     smap, scintillator,
                     cam_params, optic_params,
                     smapplt = None, 
                     gyrophases = np.pi,
                     smoother = None,
                     scint_synthetic_signal_params: dict = {},
                     spoints: str = None, px_shift: int = 0, py_shift: int = 0,
                     diag_parameters: dict = {}):
    """
    Maps a signal in the scintillator. 
    Based on the origianl function by Jose Rueda

    Alex Reyner: alereyvinn@alum.us.es

    :param  pinhole_distribution: distribution in the pinhole
    :param  efficiency: efficiency of the scintillator for that particle
    :param  smap: smap to map the signal in the xyspace
    :param  smapplt: extra smap to do nice plots
    :param  scintillator: scintillator shape we want in the plots
    :param  cam_params: parameters of the camera
    :param  optic_params:  parameters of the optics
    :param  gyrophases: range of gyrophases considered entering the pinhole
            (to scale the collimator factor). pi is the default
    :param  smoother: adds a gaussian filter to the signal with that sigma
    :param  scint_synthetic_signal_params: grid to remap the frame

    :return out dictionary containing:
            'smap': smap used calibrated to the signal
            'smapplt': smap extra to plot nice figures calibrated to the signal
            'scintillator': scintillator calibrated to the signal
            'signal_frame': signal in the scintillator space
            'scint_area': region covered by the scintillator
    """
    # --- Check inputs and initialise the settings
    # check/load strike map
    if isinstance(smap, str):  # if it is string, load the file.
        print('Reading strike map: ', smap)
        smap = ssmapplting.StrikeMap(file=smap)
        smap.load_strike_points(spoints)
    if smap._resolutions is None:
        smap.calculate_phase_space_resolution(diag_params=diag_parameters)
    dsmap=copy.deepcopy(smap)
    # check/load the scintillator
    if isinstance(scintillator, str):  # if it is string, load the file.
        print('Reading scintillator: ', scintillator)
        scintillator = ssmapplting.Scintillator(scintillator)
    dscintillator=copy.deepcopy(scintillator)
    # check that the optics parameters included all necesary elements
    if 'beta' not in optic_params:
        print('Optic magnification not set, calculating proxy')
        xsize = cam_params['px_x_size'] * cam_params['nx']
        ysize = cam_params['px_y_size'] * cam_params['ny']
        chip_min_length = np.minimum(xsize, ysize)

        xscint_size = scintillator._coord_real['x1'].max() \
            - scintillator._coord_real['x1'].min()
        yscint_size = scintillator._coord_real['x2'].max() \
            - scintillator._coord_real['x2'].min()
        scintillator_max_length = np.maximum(xscint_size, yscint_size)

        beta = chip_min_length / scintillator_max_length
        print('Optics magnification, beta: ', beta)
    # Camera parameters:
    if isinstance(cam_params, str):
        cam_params = ssio.read_camera_properties(cam_params)
    # Grid for the synthetic signal at the scintillator
    scint_signal_options = {
        'rmin': 1,
        'rmax': 10.0,
        'dr': 0.1,
        'pmin': 10.0,
        'pmax': 90.0,
        'dp': 1,
    }
    scint_signal_options.update(scint_synthetic_signal_params)

    # --- Calculate the synthetic signal at the scintillator
    print(scint_signal_options)
    scint_signal = synthetic_signal_remap(pinhole_distribution, smap,
                                          efficiency=efficiency,
                                          **scint_signal_options)
    
    # find the center of the camera frame 
    px_center = int(cam_params['ny'] / 2)
    py_center = int(cam_params['nx'] / 2)

    # center the scintillator at the coordinate origin
    y_scint_center = 0.5 * (scintillator._coord_real['x2'].max()
                            + scintillator._coord_real['x2'].min())
    x_scint_center = 0.5 * (scintillator._coord_real['x1'].max()
                            + scintillator._coord_real['x1'].min())
    dscintillator._coord_real['x2'] -= y_scint_center
    dscintillator._coord_real['x1']-= x_scint_center
    # center of the scintillator in pixel space
    px_0 = px_center + px_shift
    py_0 = py_center + py_shift
    # Scale to relate scintillator to camera
    xscale = optic_params['beta'] / cam_params['px_x_size']
    print('Xscale: ', xscale)
    yscale = optic_params['beta'] / cam_params['px_y_size']
    print('Yscale: ', yscale)
    
    # Calculate the pixel position of the scintillator vertices
    transformation_params = ssmapplting.CalParams()
    transformation_params.xscale = xscale
    transformation_params.yscale = yscale
    transformation_params.xshift = px_0
    transformation_params.yshift = py_0
    dscintillator.calculate_pixel_coordinates(transformation_params)

    # shift the strike map by the same quantity:
    dsmap._data['x2'].data -= y_scint_center
    dsmap._data['x1'].data -= x_scint_center
    # Align the strike map:
    dsmap.calculate_pixel_coordinates(transformation_params)
    dsmap.interp_grid((cam_params['nx'], cam_params['ny']),
                     MC_number=0)
    
    # If there is an specific smap to plot, pass that smap as the plot argument
    # for strikemap. If not, the one used for the synthetic signal. We work
    # with a dumy smap, again
    if smapplt == None:
        dsmapplt = copy.deepcopy(dsmap)
        dsmapplt._data['x2'].data -= y_scint_center
        dsmapplt._data['x1'].data -= x_scint_center
        dsmapplt.calculate_pixel_coordinates(transformation_params)
        dsmapplt.interp_grid((cam_params['nx'], cam_params['ny']),
                        MC_number=0)        
    else:
        dsmapplt = copy.deepcopy(smapplt)
        dsmapplt._data['x2'].data -= y_scint_center
        dsmapplt._data['x1'].data -= x_scint_center
        dsmapplt.calculate_pixel_coordinates(transformation_params)
        dsmapplt.interp_grid((cam_params['nx'], cam_params['ny']),
                        MC_number=0)
        
    # --- Map scintillator and grid to frame
    print("starting the mapping")
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

    # --- Build the original frame in the pixel space, and smooth it if wanted
    if smoother != None:
        dummy = synthetic_frame.copy()
        synthetic_frame = gaussian_filter(dummy,sigma=smoother)

    # This is the correction related to the collimater factor. Gyrophases 
    # gyrophases corresponds to the range of gyrophases we consider enter the
    # pinhole.
    synthetic_frame *= 2*np.pi/gyrophases    


    signal_frame = xr.DataArray(synthetic_frame, dims=('x', 'y'),
            coords={'x':np.linspace(0,cam_params['nx'],cam_params['nx']),
                    'y':np.linspace(0,cam_params['ny'],cam_params['ny'])})
    
    # --- Build the scintillator perimeter and find the area in the pixel space
    dummy = signal_frame.copy()*0
    xdum = dscintillator._coord_pix['x']
    ydum = dscintillator._coord_pix['y'] 
    allPts = np.column_stack((xdum,ydum))
    hullPts = ConvexHull(allPts)
    scint_x = np.concatenate((allPts[hullPts.vertices,0],\
                              np.array([allPts[hullPts.vertices,0][0]])))
    scint_y = np.concatenate((allPts[hullPts.vertices,1],\
                              np.array([allPts[hullPts.vertices,1][0]])))
    scint_perim = np.column_stack((scint_x, scint_y))
    scint_path = Path(scint_perim)
    for i in range(cam_params['nx']):
        for j in range(cam_params['ny']):
            # Check if the (i, j) coordinates are inside the scintillator
            if scint_path.contains_point((j, i)):
                dummy[i, j] = 1 # 1 if it's inside the scintillator
    scint_area = dummy.copy()
    
    # --- Define the output
    output = {
        'smap': dsmap,
        'smapplt': dsmapplt,
        'scintillator': scint_perim,
        'signal_frame': signal_frame,
        'scint_area': scint_area
    }

    return output

def add_noise_to_frame(frame,
                     noise_params: dict = {}):
    """
    Right now only includes the neutron noise only.
    Feel free to update with more noises. 
    Oh, and random briken pixels
    
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
    noise_frame = out['signal_frame'].copy()
    scint_area = out['scint_area'].copy()
    dummy = out['signal_frame'].copy()*0 # for the total noise

    # Initialise the noise dictionary. All the noises are not implemented yet.
    # This needs to be done.
    noise = {
        'camera_neutrons': None,
        'broken': None,
        'photon': None,
        'ions': None,
        'betas': None,
        'neutrons': None,
        'gamma': None,
        'camera_gamma': None
    }

    # Neutron and gamma noise (constant) (just in the scintillator area)
    """
    Homogenous noise through the scintillator due to neutrons and gamma 
    reaching it and producing charged particles that will give signal. Total
    noise must be given
    """
    if noise_params['neutrons'] > 0:
        print('Including neutron and gamma noise')
        dummy = scint_area.copy()
        num_pix = scint_area.sum().item() # how many pixel we the scint cover?
        # multiply by 4pi since we will consider the isotropic emision forward
        # in this model. The noise should be given per sr unit.
        dummy *= noise_params['neutrons']*4*np.pi/num_pix
        noise_frame += dummy
        noise['neutrons'] = dummy

    # Broken pixels
    """
    Simulate broken pixels
    This will randomly cancel the signal in a number of pixels given by percent
    """
    if noise_params['broken'] > 0:
        print('Some pixel are broken')
        rand = np.random.default_rng()
        uniform = rand.uniform(size = noise_frame.shape)
        # eliminate the pixels
        noise_frame = noise_frame.where(uniform > noise_params['broken'], 0)
        noise['broken'] = uniform

    # Put the noise frame where we had the original frame,
    # and add the nosies to the dictionary
    out['signal_frame'] = noise_frame
    out['noises'] = noise

    return out

def add_optics_and_camera(frame, exp_time: float,
                          cam_params: dict={},
                          optic_params: dict={},
                          eliminate_saturation = False):
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
    final_frame = out['signal_frame'].copy()

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


    # Add the camera noise if both needed parameters are included
    """
    Notice: dark current and readout noise are effects always present, it is
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
        readout_noise = frame['signal_frame'].copy()*0 + dummy
        out['noises']['readout_noise'] = readout_noise
        final_frame += readout_noise
        final_frame = final_frame.where(final_frame > 0 ,0) # needed
        

    # Cap the counts to the maximum counts
    if eliminate_saturation == True:
        final_frame = final_frame.where(final_frame < max_count, max_count) 

    # Transform the counts to integers    
    final_frame = final_frame.astype(int)

    # Change the previous frame for the camera frame
    out['signal_frame'] = final_frame
    

    return out

def plot_the_frame(frame, plot_smap = True, plot_scint = True,
                   cam_params: dict={},
                   maxval = True,
                   figtitle = None):
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

    frame_to_plot = frame['signal_frame']
    smapplt = frame['smapplt']
    scint_perim = frame['scintillator']

    # Establish vmax as: 1) max counts, 2) 50% of the signal
    if maxval == True:
        max_count = 2 ** cam_params['range'] - 1
    else:
        max_count = frame_to_plot.max().item()*0.5

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6,4))

    frame_to_plot.plot.imshow(ax=ax, cmap=ss.plt.Gamma_II(),
                    vmin=0, vmax=max_count)
    
    # You could use this if you are not using xarray, for example
    # img = ax.imshow(frame_to_plot, cmap=ss.plt.Gamma_II(), origin='lower',
    #                 vmin=0, vmax=max_count)
    # plt.colorbar(img, fraction=0.05, pad=0.02)

    if plot_smap == True:
        smapplt.plot_pix(ax, labels=False, marker_params={'marker':None},
                         line_params={'color':'w', 'linewidth':1.2, 'alpha':0.8})

    if plot_scint == True:
        ax.plot(scint_perim[:,0],scint_perim[:,1], color ='w', linewidth=3)
         
    if figtitle != None:
        plt.title(figtitle, fontsize=10)
        
    fig.show()
    fig.tight_layout()   

    return fig, ax
