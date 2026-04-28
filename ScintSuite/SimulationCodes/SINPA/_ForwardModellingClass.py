'''
Alex Reyner: areyner@us.es

Functions:
    - read_distribution: Read the ion distribution that will be used as input
    - obtain_WF: Obtain the weight function of the smap
FMC class:
    - synthsig_pr: Compute remapped synthetic signal in pitch-gyroradius 
    - pr_space_to_pe_space: transform the remapped signal phase space
    - synthsig_xy: (wrap) compute synthetic signal in real scintillator space
    - synthsig_camera: (wrap) compute synthetic signal in the camera
    - apply_optics_camera_noise: apply the optic effects and camera noises
    - _locate_smap_and_scint: locates the geometry elements in the frame
    - _map_signal: maps the remapped signal to the strikemap
    - _scint_perim_area: computes the scintilaltor convexhull and area covered
    - _new_synthsig_xy: compute synthetic signal in real scintillator space
    - _update_params: update dictionaries
    - plot_distribution: plot the FI distributions in pinhole and scintillator
    - plot_frame_scintillator: plot the frame in the scintillator (labels in cm)
    - plot_frame_camera: plot the camera frame
'''

import ScintSuite as ss
import ScintSuite._Mapping as ssmapplting
from ScintSuite.SimulationCodes.FILDSIM.execution import get_energy
from ScintSuite.SimulationCodes.FILDSIM.execution import get_gyroradius
import ScintSuite.SimulationCodes.Common.geometry as geometry
import ScintSuite._Plotting as ssplt
from ScintSuite._Plotting._ColorMaps import default_cmap
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle

import numpy as np
import xarray as xr
import scipy.ndimage as spnd
import math
import copy
import sys

import logging
logger = logging.getLogger('ScintSuite.FModC')
logging.basicConfig(level=logging.INFO)
import time

# -----------------------------------------------------------------------------
## --- Inputs distributions
# -----------------------------------------------------------------------------
def read_distribution(filename, pinhole_area = None, wetted_area = None,
                    B = 4, A = 2, Z = 2, version='5.5'):
    '''
    Read a distribution coming from ASCOT

    Alex Reyner: areyner@us.es
    Based on the origianl function by Jose Rueda and Anton J. van Vuuren

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
    '''
    logger.info('----- READING DISTRIBUTION ----- ')
    logger.info('Reading file: %s', filename)

    if pinhole_area == None or wetted_area == None:
        logger.error('Missing pinhole_area and/or wetted_area in input') 
        sys.exit()   
    else:
        logger.info('- Wetted area: %.2f (mm²)', wetted_area)
        logger.info('- Pinhole area: %.2f (mm²)', pinhole_area)
    
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
                # get pitch in degree, account for co (+) and counter (-)
                c[4] = math.acos(float(c[4]))*180.0/math.pi
                
                modified_line = f'{c[0]} {c[1]} {c[2]} {c[3]} \
                    {c[4]} {c[5]} {c[6]} {c[7]} {c[8]} '
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
                    f'{c[0]} {c[1]} {c[2]}'
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
                modified_line = f'{c[0]} {c[1]} {c[2]} {c[3]} {c[4]}'
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
                                
                modified_line = f'{c[0]} {c[1]} {c[2]} {c[3]} {c[4]} {c[5]} \
                      {c[6]} {c[7]} {c[8]} {c[9]} {c[10]} {c[11]}'
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

    logger.info('- Ion den flux   -> %e (ions/s/m²)', ion_flux*1e6) # go to /m²
    logger.info('- Power den flux -> %e (W/m²)', ion_power*1e6) # go to /m²
    logger.info('- Wetted flux    -> %e (ions/s)', ions_head)
    logger.info('- Pinhole flux   -> %e (ions/s)', ions_pinhole)

    return out

# -----------------------------------------------------------------------------
## --- Weight function
# -----------------------------------------------------------------------------
def obtain_WF(smap, pin_params: dict = {}, sci_params: dict = {},
              efficiency_flag: bool = False, scintillator = None, 
              B = 4, A = 4, Z = 2):
    '''
    Just a wrap of things to make it easier
    In forward modelling, do not enable the efficency, as it will be applied 
    when remapping and generating the frames.

    Alex Reyner: areyner@us.es

    :param smap: strikemap object
    :param pin_params: mesh for pinhole velocity space
    :param sci_params: mesh for scintillator velocity space
    :param efficency: flag to include the efficency
    :param scintillator: scintillator object
    :param  B: Magnetic field
    :param  A: Mass, in amu
    :param  Z: Charge in e units

    :return WF: weight function    
    '''

    # Load the strike points
    smap.load_strike_points()
    # --- Grid for the weight function
    pin_options = {
        'xmin': 20, 'xmax': 90, 'dx': 1,
        'ymin': 1.5, 'ymax': 12, 'dy': 0.2,
        }
    sci_options = {
        'xmin': 20, 'xmax': 90, 'dx': 0.25,
        'ymin': 1, 'ymax': 12, 'dy': 0.1,
        }
    # update the matrix options
    pin_options.update(pin_params)
    sci_options.update(sci_params)
    # Build the weight function 
    if efficiency_flag == True and scintillator is not None:
        logger.info('Efficency considered in the computation of the WF')
        smap.build_weight_matrix(sci_options, pin_options,
                                efficiency=scintillator.efficiency,
                                B=B,A=A,Z=Z)
    else:
        logger.info('Efficency not considered in the computation of the WF')
        smap.build_weight_matrix(sci_options, pin_options,
                                B=B,A=A,Z=Z)
    WF = smap.instrument_function
    # # Stablish units
    WF.x.attrs = {'units': 'º', 'long_name': 'Pitch'}
    WF.xs.attrs = {'units': 'º', 'long_name': 'Pitch'}
    WF.y.attrs = {'units': 'cm', 'long_name': 'Gyroradius'}
    WF.ys.attrs = {'units': 'cm', 'long_name': 'Gyroradius'}
    return WF

# -----------------------------------------------------------------------------
## --- Forward modelling class
# -----------------------------------------------------------------------------

class FMC:
    '''
    Class for forward modelling.
    Can generate dummy synthetic signals easy and fast without the necessity of
    introducing all system parameters.

    Alex Reyner: areyner@us.es
    '''
    def __init__(self, smap, scint, WF = None,
                 smapplt = None,):
        '''
        Docstring for __init__
        
        :param smap: strikemap object
        :param scint: scintillator object
        :param WF: weight function of the strikemap
        :param smapplt: strikemap to plot (if not given, smap assumed)
        '''

        # Basic needed data
        self.data = {}
        self.data['strikemap'] = smap
        self.data['scintillator'] = scint
        self.WF = WF
        if smapplt is None:
            self.data['strikemap_plot'] = smap
        else:
            self.data['strikemap_plot'] = smapplt

        # --- Basic configuration for synthetic signal production
        # Parameters of the remapped signals
        self._def_pin_params = {'xmin': 20, 'xmax': 90, 'dx': 1,
                           'ymin': 1, 'ymax': 10, 'dy': 0.1,}
        self._def_sci_params = {'xmin': 20, 'xmax': 90, 'dx': 1,
                             'ymin': 1, 'ymax': 10, 'dy': 0.1,}
        self.pin_params = self._def_pin_params.copy()
        self.sci_params = self._def_pin_params.copy()
        self.data['pin_grid'] = self.pin_params
        self.data['scint_grid'] = self.sci_params
        # Species
        self.B = 4
        self.A = 4
        self.Z = 2
        # Camera parameters (PCO.edge 5.5)
        self._def_cam_params = {'px_x_size':6.5e-6, # pixel size
                           'px_y_size':6.5e-6,
                           'nx':2560, # number of pixels
                           'ny':2160, 
                           'range':16, # pixel range
                           'qe':0.6, # quantum effiency
                           'ad_gain':0.46, # gain
                           'dark_noise':1, # dark noise
                           'readout_noise_med':2.2, # 
                           'readout_noise_rmd':2.5, # 
                           'exposure': 0.01,
                           }
        # Optic path
        self._def_opt_params = {'T': 1, # transmision
                        #    'beta': 0.2, # magnification, automatic
                           'NA': 1,
                           'omega': np.pi,
                           'FoV': [0.5, 0.5, 33.5], # position in the scintillator and FoV radius [cm]
                           }
        # Noise parameters
        self._def_noi_params = {'neutrons': 0, # background neutronic noise in the scintillator (total photons)
                           'broken':0.01, # ratio of broken pixels
                           'camera_neutrons':0.001, # ratio of pixels afected by neutrons
                           }
        self.cam_params = self._def_cam_params.copy()
        self.opt_params = self._def_opt_params.copy()
        self.noi_params = self._def_noi_params.copy()
        self.data['camera'] = self.cam_params
        self.data['optics'] = self.opt_params
        self.data['noises'] = self.noi_params

    # --- Routines for signal computations

    def synthsig_pr(self, distro, mode: str = 'ions',
                    pin_params: dict | None = None,
                    sci_params: dict | None = None,
                    gyrophases: float = np.pi,
                    ):
        '''
        Synthetic signal for pinhole and scintillator in pitch-gyroradius space

        Alex Reyner: areyner@us.es
        Based on the origianl function by Jose Rueda and Anton J. van Vuuren

        :param  distro: distribution obtained with read_distribution()
        :param  mode: select what quantity you want
                    - photons: includes scintillator response
                    - ions: ion flux in the scintillator (default)
                    - power: power flux deposited in the scintillator
        :param  pin_params: pinhole grid for the synthetic signal 
        :param  sci_params: scintillator grid for the synthetic signal

        :return pr_space atribute:
        '''
        logger.info('----- COMPUTING REMAPED SYNTHETIC SIGNAL USING WF -----')

        self.mode = mode
        logger.info(f'- Mode: {self.mode}')
        self.data['distribution'] = distro
        self.gyrophases = gyrophases
        WFrecomputation = False
        if pin_params \
        and any(self.pin_params.get(k) != v for k, v in pin_params.items()):
            logger.warning('Pinhole grid updated')
            self.pin_params.update(pin_params)
            WFrecomputation = True
        if sci_params \
        and any(self.sci_params.get(k) != v for k, v in sci_params.items()):
            logger.warning('Scintillator grid updated')
            self.sci_params.update(sci_params)
            WFrecomputation = True
        if self.WF is None or WFrecomputation:
            logger.warning('No weight function, computing...')
            start = time.perf_counter()
            self.WF = obtain_WF(smap=self.data['strikemap'], 
                                 pin_params=self.pin_params,
                                 sci_params=self.sci_params,)
            end = time.perf_counter()
            WFrecomputation = False
            logger.warning('    %.4f s', end-start)
            
        # INPUT VERIFICATION
        # -----------------------------------------------------------------------
        pitch = self.data['distribution']['pitch']
        gyro = self.data['distribution']['gyroradius']
        energy = self.data['distribution']['energy']
        B = self.data['distribution']['B']
        Anum = self.data['distribution']['Anum']
        Znum = self.data['distribution']['Znum']

        if mode == 'photons':
            if 'energy' in self.data['distribution'].keys() or 'e0' in self.data['distribution'].keys():
                eff = self.data['scintillator'].efficiency(energy/1e3).values
            elif 'gyroradius' in self.data['distribution'].keys():
                energy = get_energy(gyro, B, Anum, Znum)
                eff = self.data['scintillator'].efficiency(energy/1e3).values
            else:
                logger.error('NOT POSSIBLE TO EXTRACT EFFICENCY') 
                sys.exit() 
            weight = self.data['distribution']['weight']
            cbar_units = 'photons / s cm º'
        elif mode == 'ions': # to see the flux of ions in the scintillator
            eff = np.ones(self.data['distribution']['n']) # eficency is 1
            weight = self.data['distribution']['weight']
            cbar_units = 'ions / s cm º'
        elif mode == 'power': # to compute the deposited power (beta feature)
            eff = np.ones(self.data['distribution']['n']) # eficency is 1
            weight = self.data['distribution']['power']
            cbar_units = 'W / cm º'
        else:
            logger.error('Wrong mode, select either: photons, ions or power')

        # VECTORIZED MAPPING
        # -----------------------------------------------------------------------
        logger.info('- Mapping of the signal...')
        start = time.perf_counter()
        x_val = self.WF.coords['x'].values
        y_val = self.WF.coords['y'].values
        nx, ny = len(x_val), len(y_val)
        # Remove markers outside of WF
        total_w = weight.sum()
        mask = ((pitch >= x_val.min()) & (pitch <= x_val.max()) &
                (gyro >= y_val.min()) & (gyro <= y_val.max()))
        pitch, gyro, weight, eff = pitch[mask], gyro[mask], weight[mask], eff[mask]
        masked_w = weight.sum()
        logger.info('    Pinhole (ions/s): %e', total_w)
        logger.info('    Lost ions to smap: %e (%.2f%%)', total_w-masked_w, 
                    (total_w-masked_w)/total_w*100)
        # Calculate steps
        p_step = np.abs(x_val[1]-x_val[0])
        r_step = np.abs(y_val[1]-y_val[0])
        # Look for the correct indices place for each point
        # Nearest center index
        p_idx = np.round((pitch - x_val[0]) / p_step).astype(int)
        r_idx = np.round((gyro  - y_val[0]) / r_step).astype(int)
        # Clip to bounds
        p_idx = np.clip(p_idx, 0, nx - 1)
        r_idx = np.clip(r_idx, 0, ny - 1)
        # Build the weight matrix
        w_matrix = np.zeros((nx, ny))
        np.add.at(w_matrix, (p_idx, r_idx), (weight)) # fill the weights
        w_xrarray = xr.DataArray(w_matrix,
                        coords={'y': y_val, 'x': x_val},
                        dims=('x', 'y'))
        weff_matrix = np.zeros((nx, ny))
        np.add.at(weff_matrix, (p_idx, r_idx), (weight*eff)) # fill the weights
        weff_xrarray = xr.DataArray(weff_matrix,
                        coords={'y': y_val, 'x': x_val},
                        dims=('x', 'y'))
        # Compute the matrices in the pinhole and scintillator vel.-spaces
        ssPH = w_xrarray /p_step /r_step
        ssSC = ((self.WF*weff_xrarray) * (2*np.pi/self.gyrophases)).sum({'x','y'})
        # Put data into Dataset and assign atributes
        synthetic_signal = xr.Dataset()
        synthetic_signal['ph'] = ssPH
        synthetic_signal['ph'].attrs = {'long_name': cbar_units,}
        synthetic_signal['sc'] = ssSC
        synthetic_signal['sc'].attrs = {'long_name': cbar_units,}
        synthetic_signal.x.attrs = {'units': 'º', 'long_name': 'Pitch'}
        synthetic_signal.xs.attrs = {'units': 'º', 'long_name': 'Pitch'}
        synthetic_signal.y.attrs = {'units': 'cm', 'long_name': 'Gyroradius'}
        synthetic_signal.ys.attrs = {'units': 'cm', 'long_name': 'Gyroradius'}
        logger.info('    Scintillator (%s/s): %e', self.mode, 
                    synthetic_signal.sc.integrate(['xs', 'ys']))
        end = time.perf_counter()
        logger.info('    %.4f s', end-start)

        self.pr_space = synthetic_signal

    def pr_space_to_pe_space(self, B = None, A = None, Z = None):
        '''
        Transfors the pitch-gyroradius signal to pitch-energy signal

        Alex Reyner: areyner@us.es
        Based on the origianl function by Jose Rueda and Anton J. van Vuuren

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
        '''

        if not hasattr(self, 'pr_space'):
            logger.warning('No pr_space attribute. Call synthsig_pr first.')
            return

        B_val = self.B if B is None else B
        A_val = self.A if A is None else A
        Z_val = self.Z if Z is None else Z

        logger.info('----- GOING FROM p-r TO p-e SPACE ----- ')
        # Synthetic signal input
        ssPH_pr = self.pr_space['ph']
        ssSC_pr = self.pr_space['sc']
        # Replicate the xarray.
        # Necessary to multiply by one, to 'break' the relation between matrices.
        ssPH_pe = copy.deepcopy(ssPH_pr)
        ssSC_pe = copy.deepcopy(ssSC_pr)
        # Get the coordinates of the gyroradius and transform them to energy.
        gyroradius = ssPH_pe.coords['y'].values
        energy = get_energy(gyroradius, B=B_val, A=A_val, Z=Z_val)
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
        energy = get_energy(gyroradius, B=B_val, A=A_val, Z=Z_val)
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
        synthetic_signal = xr.Dataset()
        synthetic_signal['ph'] = ssPH_pe
        synthetic_signal['ph'].attrs = {'long_name': self.pr_space.ph.long_name,}
        synthetic_signal['sc'] = ssSC_pe/integral_s_e*integral_s
        synthetic_signal['sc'].attrs = {'long_name': self.pr_space.sc.long_name,}
        synthetic_signal.x.attrs = {'units': 'º', 'long_name': 'Pitch'}
        synthetic_signal.xs.attrs = {'units': 'º', 'long_name': 'Pitch'}
        synthetic_signal.y.attrs = {'units': 'eV', 'long_name': 'Energy'}
        synthetic_signal.ys.attrs = {'units': 'eV', 'long_name': 'Energy'}

        self.pe_space = synthetic_signal

    def synthsig_xy(self, distro, mode: str = 'photons',
                    pin_params: dict | None = None,
                    sci_params: dict | None = None,
                    cam_params: dict | None = None,
                    opt_params: dict | None = None,
                    noi_params: dict | None = None, 
                    gyrophases: float = np.pi,
                    centering: bool = False, smoother: int = 0,
                    ):
        '''
        Wrap to compute synthetic signals in the scintillator real space.

        :param  distro: distribution obtained with read_distribution()
        :param  mode: select what quantity you want
                    - photons: includes scintillator response (default)
                    - ions: ion flux in the scintillator
                    - power: power flux deposited in the scintillator
        :param  pin_params: pinhole grid for the synthetic signal 
        :param  sci_params: scintillator grid for the synthetic signal
        :param  cam_params: parameters of the camera
        :param  opt_params: parameters of the optics
        :param  noi_params: noise values
        :param  centering: center the image to the FoV
        :param  smoother: adds a gaussian filter to the signal with that sigma

        :return frame_scintillator atribute:
        '''

        self._update_params(self.cam_params, cam_params, 'camera')
        self._update_params(self.opt_params, opt_params, 'optic')
        self._update_params(self.noi_params, noi_params, 'noise')

        self.centering = centering
        self.smoother = smoother

        self.synthsig_pr(distro, mode = mode, 
                         pin_params = pin_params, sci_params = sci_params, 
                         gyrophases = gyrophases)
        self._new_synthsig_xy()
        # Delete camera frame to avoid inconsistencies with parameters
        if hasattr(self,'frame_camera'):
            del self.frame_camera

    def synthsig_camera(self, distro, mode: str = 'photons',
                        pin_params: dict | None = None,
                        sci_params: dict | None = None,
                        cam_params: dict | None = None,
                        opt_params: dict | None = None,
                        noi_params: dict | None = None,
                        centering: bool = False, smoother: int = 0,
                        rm_saturation = False,
                        radiometry = None, distortion = None,
                        scint_degree = 0,
                    ):
        '''
        Wrap to compute synthetic signals in the camera space.

        :param  distro: distribution obtained with read_distribution()
        :param  mode: select what quantity you want
                    - photons: includes scintillator response (default)
                    - ions: ion flux in the scintillator
                    - power: power flux deposited in the scintillator
        :param  pin_params: pinhole grid for the synthetic signal 
        :param  sci_params: scintillator grid for the synthetic signal
        :param  cam_params: parameters of the camera
        :param  opt_params: parameters of the optics
        :param  noi_params: noise values
        :param  centering: center the image to the FoV
        :param  smoother: adds a gaussian filter to the signal with that sigma
        :param  rm_satuation: remove saturated pixels
        :param  radiometry: relative transmission (from ZEMAX, experimental)
        :param  distortion: (from ZEMAX, experimental)

        :return frame_scintillator and frame_camera atributes:
        '''
        start = time.perf_counter()
        self.synthsig_xy(distro = distro, mode = mode, 
                         pin_params = pin_params, sci_params = sci_params, 
                         cam_params = cam_params, 
                         opt_params = opt_params,
                         noi_params = noi_params,
                         centering = centering, 
                         smoother = smoother)
        self.apply_optics_camera_noise(rm_saturation = rm_saturation,
                                       radiometry = radiometry,
                                       distortion = distortion,
                                       scint_degree = scint_degree)
        end = time.perf_counter()
        logger.info('TOTAL CAMERA SS COMPUTING TIME %.4f s', end-start)
           
    def apply_optics_camera_noise(self, 
                        cam_params: dict | None = None,
                        opt_params: dict | None = None,
                        noi_params: dict | None = None,
                        rm_saturation = False,
                        radiometry = None, distortion = None,
                        scint_degree = 0,
                        ):
        '''
        Apply the optics and camera to the frame_scintillator.

        Alex Reyner: areyner@us.es
        Based on the origianl function by Jose Rueda and Anton J. van Vuuren

        :param  cam_params: parameters of the camera
        :param  opt_params: parameters of the optics
        :param  noi_params: noise values
        :param  rm_satuation: remove saturated pixels
        :param  radiometry: relative transmission (from ZEMAX, experimental)
        :param  distortion: (from ZEMAX, experimental)

        :return frame_camera atribute:
        '''
        logger.info('----- OBTAINING CAMERA IMAGE ----- ')
        # Update parameter dictionaries, in case scans in some parameters want 
        # to be done. Carefull with this. Routines is fast enough to not need 
        # this.     
        self._update_params(self.cam_params, cam_params, 'camera')
        self._update_params(self.opt_params, opt_params, 'optic')
        self._update_params(self.noi_params, noi_params, 'noise')

        # Copy the data coming from the xy mapping
        self.frame_camera = copy.deepcopy(self.frame_scintillator)
        # The total signal will be updated during the function
        # Store special filters too
        self.radiometry = copy.deepcopy(radiometry)
        self.data['relative_T'] = self.radiometry
        self.distortion = copy.deepcopy(distortion)
        self.data['distortion_map'] = self.distortion
        
        # OPTICS
        # -----------------------------------------------------------------------
        # Compute the maximum counts for the camera
        max_count = 2 ** self.cam_params['range'] - 1

        # Now apply all variables in the result
        for key in self.frame_camera:
            # Adjust pixel sizes and beta (photons/m² to photons/pix)
            self.frame_camera[key] *= self.data['pix_scint_area']
            # Divide by 4\pi, ie, assume isotropic emission of the scintillator
            # self.frame_camera[key] *= 1/(4*np.pi) * self.opt_params['omega']
            # Advanced emission with angular dependece
            self.frame_camera[key] *= 1/(4*np.pi)/np.cos(np.deg2rad(scint_degree))*np.pi*self.opt_params['NA']**2
            # Consider the transmission of the beam line
            self.frame_camera[key] *= self.opt_params['T'] 
            # Photon to electrons in the camera sensor (QE)
            self.frame_camera[key] *= self.cam_params['qe']
            # Electrons to counts in the camera sensor,
            self.frame_camera[key] /= self.cam_params['ad_gain']
            # Consider the exposure time
            self.frame_camera[key] *= self.cam_params['exposure']
            # Apply distortion to the signal before the rest of optics and noises
            try:
                self.frame_camera[key] =\
                    self.frame_camera[key].interp(x=distortion.x_new, 
                                                y=distortion.y_new)
            except:
                pass

        # -----------------------------------------------------------------------
        # Add the optic FoV and the radiometry filter (stored as a noise)
        start = time.perf_counter()
        try:
            # Find the FoV in the scintillator using the % respect SW corner
            xsc_percent = self.opt_params['FoV'][0]
            ysc_percent = self.opt_params['FoV'][1]
            xsc_min = self.scint_perim[:,0].min()
            xsc_max = self.scint_perim[:,0].max()
            ysc_min = self.scint_perim[:,1].min()
            ysc_max = self.scint_perim[:,1].max()
            x_FoV = (xsc_max - xsc_min) * xsc_percent + xsc_min
            y_FoV = (ysc_max - ysc_min) * ysc_percent + ysc_min
            radiometry.coords['r'] = (radiometry.coords['r'] 
                                      * self.opt_params['beta'] 
                                      / (self.cam_params['px_x_size'] * 1000))
            r_FoV = radiometry.coords['r'].values.max()
            #To create te filter we need to remap the R-theta matrix
            xpix = np.linspace(1,self.cam_params['nx'],self.cam_params['nx'])
            ypix = np.linspace(1,self.cam_params['ny'],self.cam_params['ny'])
            r = xr.DataArray((np.sqrt((xpix-x_FoV)**2
                                      +(ypix[:, np.newaxis]-y_FoV)**2)),
                                      dims=['y', 'x'],
                                      coords={'y':ypix, 'x':xpix})
            t = xr.DataArray(np.arctan2((ypix[:, np.newaxis]-y_FoV),
                                        (xpix-x_FoV)),
                                        dims=['y', 'x'],
                                        coords={'y':ypix, 'x':xpix}) 
            # Here we interpolate to the pixel space
            dummy = copy.deepcopy(radiometry.interp(r=r,t=t))
            dummy = dummy.drop_vars('t')
            # Now we want 0 for the filter, not NaN
            dummy = dummy.where(dummy >= 0, 0)
            # Add the filter itself to the output 
            self.frame_camera['radiometry'] = dummy
            # Add FoV to the output 
            self.FoV_vector = [x_FoV,y_FoV,r_FoV]
            logger.info('- FoV determined: (x,y,r) = (%4.1f,%4.1f,%4.1f) pix',
                        x_FoV,y_FoV,r_FoV)
        except: # Try to at least define the FoV
            try:
                # Find the FoV in the scintillator using the % respect SW corner
                xsc_percent = self.opt_params['FoV'][0]
                ysc_percent = self.opt_params['FoV'][1]
                xsc_min = self.scint_perim[:,0].min()
                xsc_max = self.scint_perim[:,0].max()
                ysc_min = self.scint_perim[:,1].min()
                ysc_max = self.scint_perim[:,1].max()
                x_FoV = (xsc_max - xsc_min) * xsc_percent + xsc_min
                y_FoV = (ysc_max - ysc_min) * ysc_percent + ysc_min
                r_FoV = self.opt_params['FoV'][2] * self.opt_params['beta'] \
                    / (self.cam_params['px_x_size']*1000)
                self.FoV_vector = [x_FoV,y_FoV,r_FoV]
                logger.info('- FoV determined: (x,y,r) = (%4.1f,%4.1f,%4.1f) pix',
                            x_FoV,y_FoV,r_FoV)
            except:
                logger.info('- No FoV, or not good format of the input')
        end = time.perf_counter()
        logger.info('    %.4f s', end-start)   

        # NOISES IN THE CAMERA  
        # -----------------------------------------------------------------------
        final_frame = copy.deepcopy(self.frame_camera['tot'])
        # Neutron impact noise
        '''
        Add noise due to neutron impact on the sensor
        '''
        if self.noi_params['camera_neutrons'] > 0:
            start = time.perf_counter()
            rand = np.random.default_rng()
            hit = rand.uniform(size = final_frame.shape)
            intensity = rand.uniform(size=final_frame.shape)
            # noise frame, select only the pixels with the noise
            dummy = copy.deepcopy(final_frame)*0
            dummy += (2**self.cam_params['range'] - 1)
            dummy *= intensity 
            dummy = dummy.where(hit <= self.noi_params['camera_neutrons'], 0)
            # eliminate those same frames from the noise frame and add noise
            final_frame = final_frame\
                .where(hit > self.noi_params['camera_neutrons'], 0)
            final_frame += dummy
            self.frame_camera['camera_neutrons'] = dummy
            end = time.perf_counter()
            logger.info('- Neutrons hitting the sensor (%.4f s)', end-start)    
        # Broken pixels
        '''
        Simulate broken pixels
        '''
        if self.noi_params['broken'] > 0:
            start = time.perf_counter()
            rand = np.random.default_rng()
            broken = rand.uniform(size = final_frame.shape)
            # eliminate the pixels
            dummy = copy.deepcopy(final_frame)*0 + broken
            dummy = dummy.where(broken > self.noi_params['broken'], 0) #the broken
            dummy = dummy.where(broken <= self.noi_params['broken'], 1) #the okay
            final_frame = final_frame.where(broken > self.noi_params['broken'], 0)
            self.frame_camera['broken'] = dummy
            end = time.perf_counter()
            logger.info('- Some pixel are broken (%.4f s)', end-start)    

        # Add the camera noise if both needed parameters are included
        '''
        Notice: dark current and readout noise are effects always present. It is
        imposible to measure them independently, so they will be modelled as a
        single gaussian noise with centroid 'dark_centroid' and sigma
        'sigma_readout'. Both parameters to be measured for the used camera
        '''
        if 'readout_noise_med' in self.cam_params and 'readout_noise_rmd' in self.cam_params:
            start = time.perf_counter()
            rand = np.random.default_rng()
            gauss = rand.standard_normal
            dummy = self.cam_params['readout_noise_med'] + \
                self.cam_params['readout_noise_rmd'] * gauss(final_frame.shape) #readjust the sigma
            dummy /= self.cam_params['ad_gain']
            readout_noise = copy.deepcopy(final_frame)*0 + dummy
            readout_noise = readout_noise.where(readout_noise > 0 ,0) # needed
            final_frame += readout_noise
            self.frame_camera['readout_noise'] = readout_noise        
            end = time.perf_counter()
            logger.info('- Gaussian readout noise custom (%.4f s)', end-start)
        elif 'readout_noise' in self.cam_params and 'dark_noise' in self.cam_params:
            start = time.perf_counter()
            rand = np.random.default_rng()
            gauss = rand.standard_normal
            dummy = self.cam_params['dark_noise'] + \
                self.cam_params['readout_noise'] * gauss(final_frame.shape)
            dummy /= self.cam_params['ad_gain']
            readout_noise = copy.deepcopy(final_frame)*0 + np.round(dummy).astype(int)
            readout_noise = readout_noise.where(readout_noise > 0 ,0) # needed
            final_frame += readout_noise
            self.frame_camera['readout_noise'] = readout_noise
            end = time.perf_counter()
            logger.info('- Gaussian readout noise standard (%.4f s)', end-start)

        if 'dark_noise' in self.cam_params and 'readout_noise' not in self.cam_params:
            start = time.perf_counter()
            rand = np.random.default_rng()
            poiss = rand.poisson
            dummy = poiss(lam = self.cam_params['dark_noise'], 
                          size=final_frame.shape)
            dark_current = copy.deepcopy(final_frame)*0 + dummy
            dark_current = dark_current.where(dark_current > 0 ,0) # needed
            final_frame += dark_current
            self.frame_camera['dark_current'] = dark_current
            end = time.perf_counter()
            logger.info('- Dark current noise (%.4f s)', end-start)

        # PREPARE THE OUTPUT
        # -----------------------------------------------------------------------
        logger.info('- Buildind the output...')    
        # Transform the counts to integers    
        final_frame.data = final_frame.data.astype(int, copy=False)
        # Substitute the total frame
        self.frame_camera['tot'] = final_frame
        # Cap the counts to the maximum counts
        if rm_saturation == True:
            self.frame_camera =  self.frame_camera\
                .where(self.frame_camera <= max_count, max_count) 
        # Set variables in the full dataset
        for key in self.frame_camera:
            self.frame_camera[key].attrs['long_name'] = 'Pixel counts'
            self.frame_camera[key].coords['x'].attrs['longname'] = 'X'
            self.frame_camera[key].coords['x'].attrs['units'] = 'pix.'
            self.frame_camera[key].coords['y'].attrs['longname'] = 'Y'
            self.frame_camera[key].coords['y'].attrs['units'] = 'pix.'

    # --- Private routines

    def _locate_smap_and_scint(self):
        '''
        This ubicates the scintillator and strikemap in the frame

        Alex Reyner: areyner@us.es
        Based on the origianl function by Jose Rueda and Anton J. van Vuuren

        '''
        logger.info('- Locating the smap and scintillator...')
        start = time.perf_counter()

        # Find the center of the camera frame 
        px_center = int(self.cam_params['nx'] / 2)
        py_center = int(self.cam_params['ny'] / 2)
        if 'beta' not in self.opt_params:
            xsize = self.cam_params['px_x_size'] * self.cam_params['nx']
            ysize = self.cam_params['px_y_size'] * self.cam_params['ny']
            chip_min_length = np.minimum(xsize, ysize)
            xscint_size = self.scint._coord_real['x1'].max() \
                - self.scint._coord_real['x1'].min()
            yscint_size = self.scint._coord_real['x2'].max() \
                - self.scint._coord_real['x2'].min()
            scintillator_max_length = np.maximum(xscint_size, yscint_size)
            beta = chip_min_length / scintillator_max_length
            logger.info('   Optics magnification -> beta = %e', beta)
            self.opt_params['beta'] = beta
        
        if self.centering:
            # Center image to FoV
            xsc_percent = self.opt_params['FoV'][0]
            ysc_percent = self.opt_params['FoV'][1]
            xsc_min = self.scint._coord_real['x1'].min()
            xsc_max = self.scint._coord_real['x1'].max()
            ysc_min = self.scint._coord_real['x2'].min()
            ysc_max = self.scint._coord_real['x2'].max()
            x_scint_center = (xsc_max - xsc_min) * xsc_percent + xsc_min
            y_scint_center = (ysc_max - ysc_min) * ysc_percent + ysc_min
            self.scint._coord_real['x2'] -= y_scint_center
            self.scint._coord_real['x1'] -= x_scint_center        
        else:
            # Center the scintillator at the coordinate origin
            y_scint_center = 0.5 * (self.scint._coord_real['x2'].max()
                            + self.scint._coord_real['x2'].min())
            x_scint_center = 0.5 * (self.scint._coord_real['x1'].max()
                            + self.scint._coord_real['x1'].min())
            self.scint._coord_real['x2'] -= y_scint_center
            self.scint._coord_real['x1'] -= x_scint_center

        # Scale to relate scintillator to camera
        xscale = self.opt_params['beta'] / self.cam_params['px_x_size']
        yscale = self.opt_params['beta'] / self.cam_params['px_y_size']
        # Calculate the pixel position of the scintillator vertices
        transformation_params = ssmapplting.CalParams()
        transformation_params.xscale = xscale
        transformation_params.yscale = yscale
        transformation_params.xshift = px_center
        transformation_params.yshift = py_center
        self.scint.calculate_pixel_coordinates(transformation_params)
        # Shift the strike map by the same quantity:
        self.smap._data['x2'].data -= y_scint_center
        self.smap._data['x1'].data -= x_scint_center
        # Align the strike map:
        self.smap.calculate_pixel_coordinates(transformation_params)
        self.smap.interp_grid((self.cam_params['ny'], self.cam_params['nx']),
                        MC_number=0)
        # If there is an specific smap to plot, pass that smap as the plot argument
        # for strikemap. If not, the one used for the synthetic signal. We work
        # with a dumy smap, again
        self.smapplt._data['x2'].data -= y_scint_center
        self.smapplt._data['x1'].data -= x_scint_center
        self.smapplt.calculate_pixel_coordinates(transformation_params)
        self.smapplt.interp_grid((self.cam_params['ny'], self.cam_params['nx']),
                            MC_number=0)

        end = time.perf_counter()
        logger.info('   %.4f s', end-start)

    def _map_signal(self):
        '''

        Alex Reyner: areyner@us.es
        Based on the origianl function by Jose Rueda and Anton J. van Vuuren

        '''
        if not hasattr(self.pr_space, 'sc'):
           logger.warning('No remapped signal to map. Computing...')
           self.synthsig_pr(mode = self.mode)

        logger.info('- Mapping the signal in the scintillator space...')
        start = time.perf_counter()

        scint_signal = self.pr_space.sc
        dp = (self.WF.xs[1]-self.WF.xs[0]).values
        dr = (self.WF.ys[1]-self.WF.ys[0]).values
        # Create a grid
        g_grid = self.smap._grid_interp['gyroradius']
        p_grid = self.smap._grid_interp['pitch']
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

        end = time.perf_counter()
        logger.info('   %.4f s', end-start)
        
        return synthetic_frame

    def _scint_perim_area(self):
        '''
        Compute the scintillator perimeter and the area

        Alex Reyner: areyner@us.es
        Based on the origianl function by Jose Rueda and Anton J. van Vuuren

        '''
        
        logger.info('- Building the scintillator perimeter and area...')
        start = time.perf_counter()
        self.scint_perim = geometry.scint_ConvexHull(self.scint, coords='pix')

        scint_path = Path(self.scint_perim, closed=True)
        nx, ny = self.cam_params['nx'], self.cam_params['ny']
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))  # shape (ny, nx)
        points = np.vstack((x.ravel(), y.ravel())).T
        mask = scint_path.contains_points(points)
        dummy = copy.deepcopy(self.frame_scintillator['fil'])*0
        dummy_vals = dummy.values.reshape(-1)
        dummy_vals[mask] = 1
        dummy.values = dummy_vals.reshape(ny,nx)
        self.scint_area = copy.deepcopy(dummy)
        end = time.perf_counter()
        logger.info('   %.4f s', end-start)

    def _new_synthsig_xy(self):
        '''
        Maps a signal in the scintillator

        Alex Reyner: areyner@us.es
        Based on the origianl function by Jose Rueda and Anton J. van Vuuren

        :kwarg  eff: deactivate the scintillator efficency with None

        :return out dictionary containing:
                'smap': smap used calibrated to the signal
                'smapplt': smap extra to plot nice figures calibrated to the signal
                'scintillator': scintillator calibrated to the signal
                'frame': signal in the scintillator space
                'scint_area': region covered by the scintillator
                'velspace': velocity space signals
        '''
        logger.info('----- STARTING X-Y MAPPING with WF -----')

        # INPUT CHECK
        # -----------------------------------------------------------------------
        self.smap = copy.deepcopy(self.data['strikemap'])
        self.smapplt = copy.deepcopy(self.data['strikemap_plot'])
        self.scint = copy.deepcopy(self.data['scintillator'])
        if self.mode == 'photons': # the usual, photons after scintillator response
            cbar_units = 'photons / s m²'
        elif self.mode == 'ions': # to see the flux of ions in the scintillator
            cbar_units = 'ions / s m²'
        elif self.mode == 'power': # to compute the deposited power (beta featura)
            cbar_units = 'W / m²'
            self.data['distribution']['weight'] = self.data['distribution']['power']
        else:
            logger.error('Wrong mode, select either: photons, ions or power')

        # LOCATE AND CENTER THE SCINTILLATOR AND SMAP
        # -----------------------------------------------------------------------
        self._locate_smap_and_scint()

        # MAP THE SIGNAL (new ridiculously fast mapping method)
        # -----------------------------------------------------------------------
        # Build the original frame in the pixel space, and smooth it if wanted
        synthetic_frame = self._map_signal()

        # BUILD THE OUTPUT
        # -----------------------------------------------------------------------
        logger.info('- Apllying corrections...') 
        if self.smoother != None:
            dummy = copy.deepcopy(synthetic_frame)
            synthetic_frame = spnd.gaussian_filter(dummy, sigma=self.smoother)

        # Signal frame
        logger.info('- Building the frame_scintillator xarray...')
        self.frame_scintillator = xr.Dataset()
        signal_frame = xr.DataArray(synthetic_frame, dims=('y', 'x'),
                coords={'y': (np.linspace(1, self.cam_params['ny'], 
                                          self.cam_params['ny'])-1),
                        'x': (np.linspace(1, self.cam_params['nx'], 
                                          self.cam_params['nx'])-1)
                        }
                        )
        signal_frame = signal_frame.where(signal_frame >= 0, 0)
        self.frame_scintillator['fil'] = signal_frame
        # Add scintillator to output
        self._scint_perim_area()
        
        # NOISES IN THE SCINTILLATOR
        # -----------------------------------------------------------------------
        # Neutron and gamma noise (constant) (just in the scintillator area)
        '''
        Homogenous noise through the scintillator due to neutrons and gamma 
        reaching it and producing charged particles that will give signal. Total
        noise must be given.
        '''
        if self.noi_params['neutrons'] > 0:
            start = time.perf_counter()
            num_pix = self.scint_area.sum().item() # how many pixel we the scint cover?
            # multiply by 4pi since we will consider the isotropic emision forward
            # in this model. The noise should be given per sr unit.
            dummy = copy.deepcopy(self.scint_area) 
            dummy *= self.noi_params['neutrons']/num_pix # divide the noises equally in all pixels
            dummy *= 4*np.pi # this factor is applied since we will divide later
            # to transform from photons/pixel to photons/m2 
            self.frame_scintillator['neutrons'] = dummy #storeed in photons/m²
            end = time.perf_counter()
            logger.info('- Neutron and gamma noise (%.4f s)', end-start)


        # BUILD THE OUTPUT
        # -----------------------------------------------------------------------
        # Compute the total frame
        self.frame_scintillator['tot'] = (
            self.frame_scintillator.to_array().sum('variable'))
        # Transform the output from pix units to m²
        pix_osize = ((self.cam_params['px_x_size']*self.cam_params['px_y_size'])
                    /(self.opt_params['beta']**2)) # pix projection size in scintillator
        self.data['pix_scint_area'] = pix_osize
        for key in self.frame_scintillator:
            self.frame_scintillator[key] /= pix_osize # convert each to m²
            self.frame_scintillator[key].attrs['mode'] = self.mode
            self.frame_scintillator[key].attrs['long_name'] = cbar_units
            integral_s = self.frame_scintillator[key].sum().item() * pix_osize
            self.frame_scintillator[key].attrs['rate'] = integral_s

        logger.info('    Scintillator FIL (%s/s): %e', self.mode, 
                    self.frame_scintillator.fil.rate)

    def _update_params(self, cur_params: dict, new_params: dict, name: str):
        if not new_params:
            return

        changed = False

        for k, v in new_params.items():
            # If parameter is None, remove from dictionary
            if v is None:
                if k in cur_params:
                    del cur_params[k]
                    changed = True
            # Else, update parameter
            else:
                if cur_params.get(k) != v:
                    cur_params[k] = v
                    changed = True

        # if either NA or omega is in the new dict, update the other. (NA prio)
        if name == 'optic': 
            if new_params.get('NA') is not None:
                cur_params['NA'] = new_params['NA']
                theta_max = np.arcsin(new_params['NA'])
                cur_params['omega'] = 2*np.pi*(1-np.cos(theta_max))
            elif new_params.get('omega') is not None:
                cur_params['omega'] = new_params['omega']
                theta_max = np.arccos(1-new_params['omega']/(2*np.pi))
                cur_params['NA'] = np.sin(theta_max)

        if changed:
            logger.warning(f'{name} parameters updated')

    # --- Routines for plotting

    def plot_distribution(self, cmap = default_cmap(), **kwargs):
        logger.info('---- VELOCITY SPACE PLOT -----')
        fig, ax = plt.subplots(2, 2, figsize=(12, 8),
                               facecolor='w', edgecolor='k') 
        # Plot of the synthetic signals, pinhole and scintillator
        ax_param = {'xlabel': 'Pitch [º]', 'ylabel': 'Gyroradius [cm]'}         
        self.pr_space.ph.T.plot.imshow(ax=ax[0,0], cmap=cmap,
                                       vmax=0.5*self.pr_space.ph.max().item(),
                                       cbar_kwargs={'label': 'ions / s cm º'})
        ax[0,0] = ssplt.axis_beauty(ax[0,0], ax_param)
        ax[0,0].set_title('Pinhole')    
        self.pr_space.sc.T.plot.imshow(ax=ax[0,1], cmap=cmap,
                                       vmax=0.5*self.pr_space.sc.max().item())
        ax[0,1] = ssplt.axis_beauty(ax[0,1], ax_param)
        ax[0,1].set_title('Scintillator')

        # Plot of the distributions of pitch and gyroradius
        ax_options_profiles = {'ylabel': 'Signal [a.u.]'}
        (self.pr_space.ph.sum(dim='y')
         /self.pr_space.ph.sum(dim='y').integrate('x'))\
            .plot.line(ax=ax[1,0], color='black', label='Pinhole')
        (self.pr_space.sc.sum(dim='ys')
         /self.pr_space.sc.sum(dim='ys').integrate('xs'))\
            .plot.line(ax=ax[1,0], color='red', label='Scintillator')
        ax_options_profiles['xlabel'] = 'Pitch [$\\degree$]'  
        ax[1,0] = ssplt.axis_beauty(ax[1,0], ax_options_profiles)
        ax[1,0].legend()        
        (self.pr_space.ph.sum(dim='x')
         /self.pr_space.ph.sum(dim='x').integrate('y'))\
            .plot.line(ax=ax[1,1], color='black', label='Pinhole')
        (self.pr_space.sc.sum(dim='xs')
         /self.pr_space.sc.sum(dim='xs').integrate('ys'))\
            .plot.line(ax=ax[1,1], color='red', label='Scintillator')      
        ax_options_profiles['xlabel'] = 'Gyroradius [cm]'
        ax[1,1] = ssplt.axis_beauty(ax[1,1], ax_options_profiles)
        ax[1,1].legend()

        fig.tight_layout()
        plt.show()

    def plot_frame_scintillator(self, cmap = default_cmap(),
                          plot_smap = True, plot_scint = True,
                          **kwargs):
        logger.info('---- SCINTILLATOR PLOT -----')
        plot_frame = self.frame_scintillator.tot
        scint_perim = self.scint_perim
        fig, ax = plt.subplots(figsize=(8,5))
        im = plot_frame.plot.imshow(ax=ax, cmap=cmap, **kwargs)
        if plot_smap:
            self.smapplt.plot_pix(ax, labels=False, 
                                  marker_params={'marker':None},
                                  line_params={'color':'w', 
                                               'linewidth':1.2, 
                                               'alpha':0.8})
        if plot_scint:
            ax.plot(scint_perim[:,0],scint_perim[:,1], color ='w', linewidth=2,
                    alpha = 0.8)

        x_cm_pix = self.cam_params['px_x_size'] / self.opt_params['beta'] *100
        x_cm_max = len(plot_frame.x) * x_cm_pix
        xticks_cm = np.arange(0, x_cm_max, 1)
        xticks_pixels = xticks_cm / x_cm_pix
        ax.set_xticks(xticks_pixels)
        ax.set_xticklabels([str(int(x)) for x in xticks_cm])
        ax.set_xlabel('x [cm]')

        y_cm_pix = self.cam_params['px_y_size'] / self.opt_params['beta'] *100
        y_cm_max = len(plot_frame.y) * y_cm_pix
        yticks_cm = np.arange(0, y_cm_max, 1)
        yticks_pixels = yticks_cm / y_cm_pix
        ax.set_yticks(yticks_pixels)
        ax.set_yticklabels([str(int(y)) for y in yticks_cm])
        ax.set_ylabel('y [cm]')

        ax.set_xlim([plot_frame.x[0],plot_frame.x[-1]])
        ax.set_ylim([plot_frame.y[0],plot_frame.y[-1]])
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        return fig, ax

    def plot_frame_camera(self, cmap = default_cmap(), 
                          norm = None,
                          plot_smap = True, plot_scint = True, 
                          plot_FoV = False,
                          **kwargs):
        logger.info('---- CAMERA PLOT -----')
        plot_frame = self.frame_camera.tot
        scint_perim = self.scint_perim
        if norm is not None:
            if getattr(norm, 'vmin', None) is None:
                kwargs.setdefault('vmin', 0)
            if getattr(norm, 'vmax', None) is None:
                kwargs.setdefault('vmax', 2**self.cam_params['range']-1)
            kwargs['norm'] = norm
        else:
            kwargs.setdefault('vmin', 0)
            kwargs.setdefault('vmax', 2**self.cam_params['range']-1)

        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        im = plot_frame.plot.imshow(ax=ax, cmap=cmap, **kwargs)
        if plot_smap:
            self.smapplt.plot_pix(ax, labels=False, 
                                  marker_params={'marker':None},
                                  line_params={'color': 'w', 
                                               'linewidth': 1.2, 
                                               'alpha': 0.8})
        if plot_scint:
            ax.plot(scint_perim[:,0],scint_perim[:,1], color ='w', linewidth=2,
                    alpha = 0.8)
        if plot_FoV:
            try:
                ax.scatter(self.FoV_vector[0], self.FoV_vector[1],
                        marker='+',s=100,c='lime')
                FoV = Circle((self.FoV_vector[0], self.FoV_vector[1]), 
                            radius=self.FoV_vector[2], 
                            color='lime', fill=False, linewidth=2, alpha=0.8)
                ax.add_patch(FoV)
            except:
                logger.info('- No FoV plotted beacuse whatever')
            
        ax.set_xlim([plot_frame.x[0],plot_frame.x[-1]])
        ax.set_ylim([plot_frame.y[0],plot_frame.y[-1]])
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        return fig, ax

# -----------------------------------------------------------------------------
## --- Routines for the relative and deformation
# -----------------------------------------------------------------------------
# This section is work in progress. Difficult to make it uniform for every user
# since it depends on the ZEMAX information.

def Lambertian(file_path, plot=False):
    '''
    For files where the transmission is given as values in the x and y axis
    '''
    data = np.loadtxt(file_path)
    angles = np.linspace(-np.pi,np.pi,5) #because values are in the x and y axis
    radius = np.linspace(0,data[:,0].max(),int(len(data)/4))
    polars = xr.DataArray(np.zeros((len(angles),len(radius))), dims=('t', 'r'),
                        coords={'t':angles, 'r':radius})
    for i in range(len(data)):
        x_c=data[i,0]
        y_c=data[i,1]
        f_c=data[i,2]   
        r_c=np.sqrt(x_c**2+y_c**2)
        t_c=np.arctan2(y_c,x_c)
        tt = polars.sel(t=t_c, r=r_c, method = 'nearest').t.item()
        rr = polars.sel(t=t_c, r=r_c, method = 'nearest').r.item()
        polars.loc[tt,rr] = f_c
        polars[0,:] = polars[-1,:]*1 # pi equal to -pi
        polars.loc[:,0] = 1 # 1 at r=0
        #interpolation of the nan
        polars = polars.where(polars != 0, np.nan)  
        polars = polars.interpolate_na(dim='r',method='pchip')
        polars = polars.interpolate_na(dim='t',method='pchip')  

        xpix = np.linspace(data[:,0].min(),data[:,0].max(),1000)
        ypix = np.linspace(data[:,0].min(),data[:,0].max(),1000)
        rr = xr.DataArray((np.sqrt((xpix[:, np.newaxis])**2+(ypix)**2)),
                                dims=['x', 'y'],
                                coords={'y':ypix, 'x':xpix})
        tt = xr.DataArray(np.arctan2((xpix[:, np.newaxis]),(ypix)),
                                dims=['x', 'y'],
                                coords={'y':ypix, 'x':xpix})
        circulars = polars.interp(r=rr,t=tt,)

    if plot == True:
        fig, ax = plt.subplots(1,1,figsize=(8,13/2.54)) 
        im = circulars.plot.imshow(ax=ax,cmap='Blues_r',vmax=1,
                                   add_colorbar=True,
                                   cbar_kwargs={'label':'Relative light transmition',
                                                'ticks':[0.8,0.9,1]})
        ax.set_aspect(1) 
        ax.set_xticks([-30,-20,-10,0,10,20,30])
        ax.set_yticks([-30,-20,-10,0,10,20,30])
        # ax3.tick_params(axis='both',labelsize=12)
        ax_param = {'xlabel':'x (cm)', 'ylabel':'y (cm)', }
        ax = ss.plt.axis_beauty(ax, ax_param)
        plt.tight_layout()

    return polars, circulars

def Vignete(file_path, plot=False):
    '''
    transmision is given in terms of the radius
    '''
    data = np.loadtxt(file_path,encoding='utf-16')
    angles = np.linspace(-np.pi,np.pi,10) #because values are in the x and y axis
    radius = data[:,0]
    dummy = xr.DataArray(np.zeros((len(angles),len(radius))), dims=('t', 'r'),
                        coords={'t':angles, 'r':radius})
    for i in range(len(data)):
        r_c=data[i,0]
        f_c=data[i,1]   
        # tt = polars.sel(t=0, r=r_c, method = 'nearest').t.item()
        rr = dummy.sel(r=r_c, method = 'nearest').r.item()
        dummy.loc[:,rr] = f_c
        polars = (dummy/dummy.max().values)

    if plot == True:
        xpix = np.linspace(-data[:,0].max(),data[:,0].max(),1000)
        ypix = np.linspace(-data[:,0].max(),data[:,0].max(),1000)
        rr = xr.DataArray((np.sqrt((xpix[:, np.newaxis])**2+(ypix)**2)),
                                dims=['x', 'y'],
                                coords={'y':ypix, 'x':xpix})
        tt = xr.DataArray(np.arctan2((xpix[:, np.newaxis]),(ypix)),
                                dims=['x', 'y'],
                                coords={'y':ypix, 'x':xpix})

        # plt.close('all')
        circulars = polars.interp(r=rr,t=tt)
        fig, ax = plt.subplots(1,1,figsize=(8,13/2.54)) 
        im = circulars.plot.imshow(ax=ax,cmap='jet',
                                   interpolation='gaussian',vmin=0.8,vmax=1,
                                   add_colorbar=True,
                                   cbar_kwargs={'label':'Relative light transmition',
                                                'ticks':[0.8,0.9,1]})
        ax.set_aspect(1) 
        ax.set_xticks([-30,-20,-10,0,10,20,30])
        ax.set_yticks([-30,-20,-10,0,10,20,30])
        # ax3.tick_params(axis='both',labelsize=12)
        ax_param = {'xlabel':'x (cm)', 'ylabel':'y (cm)', }
        ax = ss.plt.axis_beauty(ax, ax_param)
        plt.tight_layout()

    return polars

def Deformation(file_path, cam_params, opt_params, plot=False):
    '''
    Implement the deformation of the signal computed with 
    '''
    data = np.loadtxt(file_path,encoding='utf-16')
    # i   j   X-Field     Y-Field     R-Field    Predicted X  Predicted Y   Real X        Real Y        Distortion
    x_p = np.rint(data[:,2] /1000/cam_params['px_x_size']*opt_params['beta'] + cam_params['nx']/2)[::-1]
    y_p = np.rint(data[:,3] /1000/cam_params['px_x_size']*opt_params['beta'] + cam_params['ny']/2)
    x_r = (data[:,7] /1000 /cam_params['px_x_size'] + cam_params['nx']/2) #invertion in X
    y_r = (data[:,8] /1000 /cam_params['px_x_size'] + cam_params['ny']/2)
    u = x_p-x_r
    v = y_p-y_r
    # get only the non repeated values anf build an xarray dataset with them
    x_pi = np.unique(x_p.astype(int))
    y_pi = np.unique(y_p.astype(int))
    deformation = xr.Dataset()
    dummx = xr.DataArray(dims={'x': x_pi,'y': y_pi,}, coords={'x': x_pi,'y': y_pi,})
    dummy = xr.DataArray(dims={'x': x_pi,'y': y_pi,}, coords={'x': x_pi,'y': y_pi,})
    dummt = xr.DataArray(dims={'x': x_pi,'y': y_pi,}, coords={'x': x_pi,'y': y_pi,})
    for i in np.arange(len(x_p)):
        xx = dummx.sel(x=x_p[i], y=y_p[i], method='nearest').x.item()
        yy = dummx.sel(x=x_p[i], y=y_p[i], method='nearest').y.item()
        # its necessary to do it this way, to 'invert' the transforation, and make it easier later
        dummx.loc[xx,yy] = x_p[i]+u[i]
        dummy.loc[xx,yy] = y_p[i]+v[i]
        dummt.loc[xx,yy] = np.sqrt(u[i]**2 + v[i]**2)
    deformation['x_new'] = dummx
    deformation['y_new'] = dummy
    deformation['total'] = dummt
    # interpolate and extrapolate the matrix to the full chip
    xpix = np.linspace(1,cam_params['nx'],cam_params['nx'])
    ypix = np.linspace(1,cam_params['ny'],cam_params['ny'])
    deformation_pix = deformation.interp(x=xpix,y=ypix)
    deformation_pix = deformation_pix.interpolate_na(dim='x',method='linear',fill_value='extrapolate')
    deformation_pix = deformation_pix.interpolate_na(dim='y',method='linear',fill_value='extrapolate')

    if plot == True:
        fig, ax = plt.subplots(figsize=(8,5))
        plt.scatter(x_p,y_p)
        plt.scatter(x_r,y_r,c='cyan')
        # ax.quiver(x_p,y_p,u,v)
        ax.set_xlim([1,cam_params['nx']])
        ax.set_ylim([1,cam_params['ny']])
        # ax_param = {'xlabel': 'xpix', 'ylabel': 'ypix'}
        # ax = ssplt.axis_beauty(ax, ax_param)
        ax.set_aspect(1)
        plt.tight_layout()

    return deformation_pix

