"""
Created on Wed May  5 09:19:50 2021

@author: ajvv

Run FILDSIM and calculate resolutions

"""

import Lib as ss
import numpy as np
import os
import IPython
import matplotlib.pylab as plt
import time
import pickle

def write_plate_files(root_dir = '/afs/ipp/home/a/ajvv/FILDSIM/geometry/W7X/Scans/'
                      , string_mod = ''
                      , pinhole_lenght  = 0.2
                      , pinhole_width   = 0.1
                      , pinhole_scint_dist = 0.5
                      , slit_height     = 1.0
                      , slit_length     = 5.
                      , scint_width     = 10.
                      , scint_height    = 10.
                      , pinhole_padding = 3.
                      , pinhole_depth = 10.):
    '''
    Parameters
    ----------
    pinhole_lenght : TYPE, optional
        DESCRIPTION. The default is 0.2.
    pinhole_width : TYPE, optional
        DESCRIPTION. The default is 0.1.
    pinhole_scint_dist : TYPE, optional
        DESCRIPTION. The default is 0.5.
    slit_height : TYPE, optional
        DESCRIPTION. The default is 1.0.
    slit_length : TYPE, optional
        DESCRIPTION. The default is 5..
    scint_width : TYPE, optional
        DESCRIPTION. The default is 10..
    scint_height : TYPE, optional
        DESCRIPTION. The default is 10..

    Returns
    -------
    None.
    '''
    plates = []
    #scintilator_plate
    N_vertices = 4
    xyz_scint = np.zeros((N_vertices, 3))
    xyz_scint[:, 0] = - pinhole_scint_dist * np.ones(N_vertices) # scintilor x points all the same
    xyz_scint[0, 1], xyz_scint[0, 2] = -0.5 * pinhole_lenght , 0
    xyz_scint[1, 1], xyz_scint[1, 2] = -0.5 * pinhole_lenght ,  -scint_height
    xyz_scint[2, 1], xyz_scint[2, 2] = -0.5 * pinhole_lenght + scint_width ,  -scint_height
    xyz_scint[3, 1], xyz_scint[3, 2] = -0.5 * pinhole_lenght + scint_width ,  0
    scint_normal = np.array([1.,0.,0.])
    plates.append({'name': 'W7X_scint_scan', 'N_vertices':N_vertices,'vertices':xyz_scint, 'normal':scint_normal })
    #slit_plates
    #pinhole 1
    N_vertices = 8
    xyz_pinhole_1 = np.zeros((N_vertices, 3))
    xyz_pinhole_1[0, 0], xyz_pinhole_1[0, 1] = - pinhole_scint_dist -pinhole_padding ,  -0.5 * pinhole_lenght - pinhole_padding
    xyz_pinhole_1[1, 0], xyz_pinhole_1[1, 1] = - pinhole_scint_dist + pinhole_padding + pinhole_depth ,  -0.5 * pinhole_lenght - pinhole_padding
    xyz_pinhole_1[2, 0], xyz_pinhole_1[2, 1] = - pinhole_scint_dist + pinhole_padding + pinhole_depth , -0.5 * pinhole_lenght  + pinhole_padding +  scint_width
    xyz_pinhole_1[3, 0], xyz_pinhole_1[3, 1] =   0.5 * pinhole_width,  -0.5 * pinhole_lenght  + pinhole_padding +  scint_width
    xyz_pinhole_1[4, 0], xyz_pinhole_1[4, 1] =   0.5 * pinhole_width,  -0.5 * pinhole_lenght
    xyz_pinhole_1[5, 0], xyz_pinhole_1[5, 1] = -0.5 * pinhole_width,  -0.5 * pinhole_lenght
    xyz_pinhole_1[6, 0], xyz_pinhole_1[6, 1] = -0.5 * pinhole_width,  -0.5 * pinhole_lenght  + pinhole_padding +  scint_width
    xyz_pinhole_1[7, 0], xyz_pinhole_1[7, 1] = - pinhole_scint_dist -pinhole_padding, -0.5 * pinhole_lenght  + pinhole_padding +  scint_width
    pinhole_1_normal = np.array([0.,   0,   1.])
    plates.append({'name': 'W7X_pinhole_1_scan', 'N_vertices':N_vertices,'vertices':xyz_pinhole_1, 'normal':pinhole_1_normal })
    #pinhole 2
    N_vertices = 4
    xyz_pinhole_2 = np.zeros((N_vertices, 3))
    xyz_pinhole_2[0, 0], xyz_pinhole_2[0, 1] =   0.5 * pinhole_width,  0.5 * pinhole_lenght
    xyz_pinhole_2[1, 0], xyz_pinhole_2[1, 1] =   0.5 * pinhole_width,  -0.5 * pinhole_lenght  + pinhole_padding +  scint_width
    xyz_pinhole_2[2, 0], xyz_pinhole_2[2, 1] = -0.5 * pinhole_width,  -0.5 * pinhole_lenght  + pinhole_padding +  scint_width
    xyz_pinhole_2[3, 0], xyz_pinhole_2[3, 1] = -0.5 * pinhole_width,  0.5 * pinhole_lenght
    pinhole_2_normal = np.array([0.,   0,   1.])
    plates.append({'name': 'W7X_pinhole_2_scan', 'N_vertices':N_vertices,'vertices':xyz_pinhole_2, 'normal':pinhole_2_normal })
    #slit 1
    N_vertices = 8
    xyz_slit_1 = np.zeros((N_vertices, 3))
    xyz_slit_1[:, 2] = np.ones(N_vertices) *(-slit_height) 
    xyz_slit_1[0, 0], xyz_slit_1[0, 1] = - pinhole_scint_dist -pinhole_padding ,  -0.5 * pinhole_lenght - pinhole_padding
    xyz_slit_1[1, 0], xyz_slit_1[1, 1] = - pinhole_scint_dist +pinhole_padding + pinhole_depth ,  -0.5 * pinhole_lenght - pinhole_padding
    xyz_slit_1[2, 0], xyz_slit_1[2, 1] = - pinhole_scint_dist +pinhole_padding + pinhole_depth , 0.5 * pinhole_lenght  
    xyz_slit_1[3, 0], xyz_slit_1[3, 1] =   0.5 * pinhole_width,  0.5 * pinhole_lenght  
    xyz_slit_1[4, 0], xyz_slit_1[4, 1] =   0.5 * pinhole_width,  -0.5 * pinhole_lenght
    xyz_slit_1[5, 0], xyz_slit_1[5, 1] = -0.5 * pinhole_width,  -0.5 * pinhole_lenght
    xyz_slit_1[6, 0], xyz_slit_1[6, 1] = -0.5 * pinhole_width,  0.5 * pinhole_lenght
    xyz_slit_1[7, 0], xyz_slit_1[7, 1] = - pinhole_scint_dist -pinhole_padding, 0.5 * pinhole_lenght
    slit_1_normal = np.array([0.,   0,   1.])
    plates.append({'name': 'W7X_slit_1_scan', 'N_vertices':N_vertices,'vertices':xyz_slit_1, 'normal':slit_1_normal })
    #slit back plate
    N_vertices = 4
    xyz_slit_back = np.zeros((N_vertices, 3))
    xyz_slit_back[:, 1] = np.ones(N_vertices) *(-0.5 * pinhole_lenght) 
    xyz_slit_back[0, 0], xyz_slit_back[0, 2] = -0.5 * pinhole_width,  -slit_height
    xyz_slit_back[1, 0], xyz_slit_back[1, 2] =  0.5 * pinhole_width,  -slit_height
    xyz_slit_back[2, 0], xyz_slit_back[2, 2] =  0.5 * pinhole_width, 0
    xyz_slit_back[3, 0], xyz_slit_back[3, 2] = -0.5 * pinhole_width, 0
    slit_back_normal = np.array([0.,   1.,   0.])
    plates.append({'name': 'W7X_slit_back_scan', 'N_vertices':N_vertices,'vertices':xyz_slit_back, 'normal':slit_back_normal })
    #slit lateral 1
    N_vertices = 4
    xyz_slit_lateral_1 = np.zeros((N_vertices, 3))
    xyz_slit_lateral_1[:, 0] = np.ones(N_vertices) *(-0.5 * pinhole_width) 
    xyz_slit_lateral_1[0, 1], xyz_slit_lateral_1[0, 2] =   -0.5 * pinhole_lenght, -slit_height
    xyz_slit_lateral_1[1, 1], xyz_slit_lateral_1[1, 2] = 0.5 * pinhole_lenght, -slit_height
    xyz_slit_lateral_1[2, 1], xyz_slit_lateral_1[2, 2] = slit_length, 0
    xyz_slit_lateral_1[3, 1], xyz_slit_lateral_1[3, 2] = -0.5 * pinhole_lenght,  0.
    slit_lateral_1_normal = np.array([1.,   0.,   0.])
    plates.append({'name': 'W7X_slit_lateral_1_scan', 'N_vertices':N_vertices,'vertices':xyz_slit_lateral_1, 'normal':slit_lateral_1_normal })
    #slit lateral 2
    N_vertices = 4
    xyz_slit_lateral_2 = np.zeros((N_vertices, 3))
    xyz_slit_lateral_2[:, 0] = np.ones(N_vertices) *(0.5 * pinhole_width)
    xyz_slit_lateral_2[0, 1], xyz_slit_lateral_2[0, 2] =  -0.5 * pinhole_lenght, -slit_height
    xyz_slit_lateral_2[1, 1], xyz_slit_lateral_2[1, 2] =  0.5 * pinhole_lenght, -slit_height
    xyz_slit_lateral_2[2, 1], xyz_slit_lateral_2[2, 2] =  slit_length, 0
    xyz_slit_lateral_2[3, 1], xyz_slit_lateral_2[3, 2] =  -0.5 * pinhole_lenght,  0.
    slit_lateral_2_normal = np.array([1.,   0.,   0.])
    plates.append({'name': 'W7X_slit_lateral_2_scan', 'N_vertices':N_vertices,'vertices':xyz_slit_lateral_2, 'normal':slit_lateral_2_normal })
    ##write geometry files

    for plate in plates:
        plate_filename = root_dir + plate['name']+ string_mod+'.pl'
        f = open(plate_filename, 'w')
        f.write('# Plate file for FILDSIM.f90\n')
        f.write('Name='+plate['name']+'\n')
        f.write('N_vertices='+str(plate['N_vertices'])+'\n')
        
        for i in range(plate['N_vertices']):
            f.write(str(plate['vertices'][i, 0]) + ',' + str(plate['vertices'][i, 1]) + ','+  str(plate['vertices'][i, 2]) + '\n')
        
        f.write('Normal_vector\n')
        f.write(str(plate['normal'][0]) + ',' + str(plate['normal'][1]) + ','+  str(plate['normal'][2]) + '\n')
        
        f.close()
 
        
if __name__ == '__main__':
    #set plotting settings
    ss.LibPlotting.plotSettings(plot_mode='Presentation')
    # -----------------------------------------------------------------------------
    namelist_path = ss.paths.FILDSIM + 'results/W7X_scans/'   # Paths to save namelist
    # -----------------------------------------------------------------------------
    # Section 0: FILDSIM Settings
    # -----------------------------------------------------------------------------
    run_scan = False
    plot_plate_geometry = True
    
    read_scan = True
    plot_strike_points = True
    plot_strikemap = True

    #for best case    
    plot_gyro_res= True
    plot_pitch_res= True
    plot_collimating_factor = True
    plot_synthetic_signal= True
    
    marker_params = {'markersize':6, 'marker':'o','color':'b'}
    line_params =  {'ls':'solid','color':'k'}
    
    #[cm]
    pinhole_lenght  = 0.1 #0.2 
    pinhole_width   = 0.08 #0.1
    pinhole_scint_dist = 0.8 #0.6 #0.5
    slit_height     = 0.5 #1.0
    slit_length     = 2.0#6. #5.
    scint_width     = 10 #large scintilator
    scint_height    = 10
    
    Br, Bz, Bt = -0.3263641727027391, -0.3266729522875698 , 2.3419531967879372#0.515, -0.091, 2.621  #[T]
    alpha = 0.16  #From J. Hidalgo
    beta = 215. #180
    
    n_markers = int(1e5)
    
    scan_Parameters = {}
    scan_Parameters['Pinhole width'] = {'scan_str': 'pw',
                                        'scan_values':[0.08],#np.arange(0.03, 0.15, 0.01), #
                                        'scan': True,
                                        'value': pinhole_width}
    scan_Parameters['Pinhole lenght'] = {'scan_str': 'pl',
                                    'scan_values':np.arange(0.05, 0.25, 0.02), #[0.05, 0.07, 0.13, 0.19],#
                                    'scan': False,
                                    'value': pinhole_lenght}
    scan_Parameters['Pinhole area'] = {'scan_str': 'pa',
                                    'scan_values': [0],
                                    'scan': False,
                                    'value': 0}
    scan_Parameters['Pinhole scint dist'] = {'scan_str': 'psd',
                                    'scan_values': np.arange(0.1, 1.5, 0.1),
                                    'scan': False ,
                                    'value': pinhole_scint_dist}
    scan_Parameters['Slit height'] = {'scan_str': 'sh',
                                    'scan_values': np.arange(0.1, 2.5, 0.2),
                                    'scan': False,
                                    'value': slit_height}
    scan_Parameters['Slit length'] = {'scan_str': 'sl',
                                    'scan_values': np.arange(0.1, 8.0, 1.),
                                    'scan': False,
                                    'value': slit_length}
    
    scan_Parameters['Beta'] = {'scan_str': 'b',
                                    'scan_values': np.arange(200, 255, 5),
                                    'scan': False,
                                    'value': beta}    
    # -----------------------------------------------------------------------------
    # --- Section 1: Run FILDSIM scan over scan parameters
    # -----------------------------------------------------------------------------    
    if run_scan:
        for scan_paramter in scan_Parameters.keys():
            if scan_Parameters[scan_paramter]['scan']:
                for value in scan_Parameters[scan_paramter]['scan_values']:
                    '''
                    Loop over scan paramater
                    '''
                    scan_Parameters[scan_paramter]['value'] = value

                    pinhole_lenght = scan_Parameters['Pinhole lenght']['value']
                    pinhole_width  = scan_Parameters['Pinhole width']['value']
                    pinhole_scint_dist = scan_Parameters['Pinhole scint dist']['value']
                    slit_height     = scan_Parameters['Slit height']['value']
                    slit_length     = scan_Parameters['Slit length']['value']
                    
                    beta = scan_Parameters['Beta']['value']
                    phi, theta = ss.LibFILDSIM.FILDSIMexecution.calculate_fild_orientation(Br, Bz, Bt, alpha, beta, verbose=False)
                    
                    string_mod = '_%s_%s' %(scan_Parameters[scan_paramter]['scan_str'], f'{float(value):g}')
                    write_plate_files(root_dir = ss.paths.FILDSIM + '/geometry/W7X/Scans/',
                                    string_mod = string_mod,
                                    pinhole_lenght = pinhole_lenght,
                                    pinhole_width  = pinhole_width ,
                                    pinhole_scint_dist = pinhole_scint_dist,
                                    slit_height     = slit_height,
                                    slit_length     = slit_length,
                                    scint_width     = scint_width, 
                                    scint_height    = scint_height)
                    
                    runid = 'scan'+string_mod
                    start_x = [-0.25*pinhole_width, 0.25*pinhole_width]
                    start_y = [-0.5*pinhole_lenght, 0.5*pinhole_lenght]
        
                    FILDSIM_namelist = {
                        'config': {
                            'runid': runid,
                            'result_dir': ss.paths.FILDSIM + '/results/W7X_scans/',
                            'backtrace': False,
                            'n_gyroradius': 10,
                            'n_pitch': 8,
                            'save_orbits': 0,
                            'verbose': True,
                        },
                        'input_parameters': {
                            'n_ions': n_markers,
                            'step': 0.01,
                            'helix_length': 11.0,
                            'gyroradius': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2., 3., 4.],
                            'pitch_angle': [115., 105., 95., 85., 75., 65., 55., 45.],#[85., 80., 70., 60., 50., 40., 30., 20., 10, 0.],
                            'gyrophase_range': [0.0, 3.14],
                            'start_x': start_x,
                            'start_y': start_y,
                            'start_z': [0.0, 0.0],
                            'theta': theta,
                            'phi': phi
                        },
                        'plate_setup_cfg': {
                            'geometry_dir': ss.paths.FILDSIM + './geometry/W7X/Scans/',
                            'n_scintillator': 1,
                            'n_slits': 6
                        },
                        'plate_files': {
                            'scintillator_files': ['W7X_scint_%s.pl' %(runid)],
                            'slit_files': ['W7X_pinhole_1_%s.pl' %(runid),
                                        'W7X_pinhole_2_%s.pl' %(runid),
                                        'W7X_slit_1_%s.pl' %(runid),
                                        'W7X_slit_back_%s.pl' %(runid),
                                        'W7X_slit_lateral_1_%s.pl' %(runid),
                                        'W7X_slit_lateral_2_%s.pl' %(runid)]
                        }
                    }
                    # Write namelist
                    ss.fildsim.write_namelist(FILDSIM_namelist, p=namelist_path)
                    namelist_name = os.path.join(namelist_path,
                                                FILDSIM_namelist['config']['runid'] + '.cfg')
                    
                    if plot_plate_geometry:
                        ss.LibFILDSIM.FILDSIMplot.plot_geometry(namelist_name)
                        plt.show()
                    # Run FILDSIM
                    ss.fildsim.run_FILDSIM(namelist_name, queue = True)
                    time.sleep(10.0)
    # -----------------------------------------------------------------------------
    # --- Section 2: Analyse the results
    # -----------------------------------------------------------------------------
    SMAPS = []
    
    if read_scan:
        for scan_paramter in scan_Parameters.keys():
            if scan_Parameters[scan_paramter]['scan']:
                min_gyro, max_gyro = [],[]
                gyro_1_res, gyro_2_res, gyro_3_res = [],[],[]
                min_pitch, max_pitch = [], []
                min_gyro_res, max_gyro_res = [], []
                min_pitch_res, max_pitch_res, pitch_85_res = [], [], []
                
                avg_collimating_factor, pitch_85_gyro_1_collimating_factor = [], []
                
                for value in scan_Parameters[scan_paramter]['scan_values']:
                    ## Loop over scan variables
                    
                    scan_Parameters[scan_paramter]['value'] = value
                    
                    pinhole_lenght = scan_Parameters['Pinhole lenght']['value']
                    pinhole_width  = scan_Parameters['Pinhole width']['value']
                    pinhole_scint_dist = scan_Parameters['Pinhole scint dist']['value']
                    slit_height     = scan_Parameters['Slit height']['value']
                    slit_length     = scan_Parameters['Slit length']['value']
                    
                    string_mod = '_%s_%s' %(scan_Parameters[scan_paramter]['scan_str'], f'{float(value):g}')
                    # Load the result of the simulation
                    runid = 'scan'+string_mod
                    namelist_name = os.path.join(namelist_path,runid + '.cfg')
                    result_dir =  ss.paths.FILDSIM + '/results/W7X_scans/'
                    base_name = result_dir + runid
                    strike_map_file = base_name + '_strike_map.dat'
                    strike_points_file = base_name + '_strike_points.dat'
                    # Load the strike map
                    Smap = ss.mapping.StrikeMap('FILD', strike_map_file)
                    # Load the strike points used to calculate the map
                    try:
                        Smap.load_strike_points(strike_points_file)
                    except:
                        min_gyro.append(np.nan)
                        max_gyro.append(np.nan)
                        min_pitch.append(np.nan)
                        max_pitch.append(np.nan)
                        min_gyro_res.append(np.nan)
                        max_gyro_res.append(np.nan)
                        pitch_85_res.append(np.nan)
                        gyro_1_res.append(np.nan)
                        gyro_2_res.append(np.nan)
                        gyro_3_res.append(np.nan)
                        min_pitch_res.append(np.nan)
                        max_pitch_res.append(np.nan)
                        avg_collimating_factor.append(np.nan)
                        pitch_85_gyro_1_collimating_factor.append(np.nan)
                        print('Strike map could not be loaded')
                        continue
                    # Calculate the resolutions
                    
                    if plot_plate_geometry:
                        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(18, 10),
                                           facecolor='w', edgecolor='k', dpi=100)
                        ax2D_xy, ax2D_yz, ax2D_xz = axarr[0], axarr[1], axarr[2]
                        
                        ax2D_xy.set_xlabel('X [cm]')
                        ax2D_xy.set_ylabel('Y [cm]')
                        ax2D_xy.set_title('Top down view (X-Y plane)')
                        ax2D_yz.set_xlabel('Y [cm]')
                        ax2D_yz.set_ylabel('Z [cm]')
                        ax2D_yz.set_title('Camera view (Y-Z plane)')
                        ax2D_xz.set_xlabel('X [cm]')
                        ax2D_xz.set_ylabel('Z [cm]')
                        ax2D_xz.set_title('Side view (Y-Z plane)')

                        ss.LibFILDSIM.FILDSIMplot.plot_geometry(namelist_name, axarr=axarr)
                        if plot_strike_points:
                            Smap.plot_strike_points(ax=ax2D_yz)#, plt_param=marker_params)
                        if plot_strikemap:
                            Smap.plot_real(ax=ax2D_yz, marker_params=marker_params, line_params = line_params)
                            
                        fig.show()

                    
                    # Smap.calculate_resolutions(min_statistics = 1000,
                    #                            adaptative=False,
                    #                            diag_params = {
                    #                             'dpitch': 0.02,
                    #                             'dgyr': 0.05,
                    #                             'p_method': 'Gauss',
                    #                             'g_method': 'sGauss'
                    #                         })  # Default call,
                    try:
                        Smap.calculate_resolutions(min_statistics = 100)
                    except:
                        min_gyro.append(np.nan)
                        max_gyro.append(np.nan)
                        min_pitch.append(np.nan)
                        max_pitch.append(np.nan)
                        min_gyro_res.append(np.nan)
                        max_gyro_res.append(np.nan)
                        pitch_85_res.append(np.nan)
                        gyro_1_res.append(np.nan)
                        gyro_2_res.append(np.nan)
                        gyro_3_res.append(np.nan)
                        min_pitch_res.append(np.nan)
                        max_pitch_res.append(np.nan)
                        avg_collimating_factor.append(np.nan)
                        pitch_85_gyro_1_collimating_factor.append(np.nan)
                        print('Could not calcualte resolutions')
                        continue

                    # Save the result in the SMAP list
                    
                    if len(Smap.gyroradius)==0:
                        min_gyro.append(np.nan)
                        max_gyro.append(np.nan)
                    else:
                        min_gyro.append(np.nanmin(Smap.gyroradius))
                        max_gyro.append(np.nanmax(Smap.gyroradius))
                    
                    if len(Smap.pitch)==0:
                        min_pitch.append(np.nan)
                        max_pitch.append(np.nan)
                    else:
                        min_pitch.append(np.nanmin(Smap.pitch))
                        max_pitch.append(np.nanmax(Smap.pitch)) 
                        

                    min_gyro_res.append(np.nanmin(Smap.resolution['Gyroradius']['sigma']))
                    max_gyro_res.append(np.nanmax(Smap.resolution['Gyroradius']['sigma']))
                    idx_85 = np.argmin( abs(Smap.strike_points['pitch'] - 85))
                    try:
                        pitch_85_res.append(np.nanmax(Smap.resolution['Gyroradius']['sigma'][idx_85]))
                    except:
                        pitch_85_res.append(np.nan)
                    
                    idx1 = np.argmin( abs(Smap.strike_points['gyroradius'] -0.5))
                    idx2 = np.argmin( abs(Smap.strike_points['gyroradius'] -1.0))
                    idx3 = np.argmin( abs(Smap.strike_points['gyroradius'] -1.5))
                    gyro_1_res.append(np.nanmin(Smap.resolution['Gyroradius']['sigma'][idx1,:]))
                    gyro_2_res.append(np.nanmax(Smap.resolution['Gyroradius']['sigma'][idx2,:]))
                    gyro_3_res.append(np.nanmax(Smap.resolution['Gyroradius']['sigma'][idx3,:]))

                    
                    min_pitch_res.append(np.nanmin(Smap.resolution['Pitch']['sigma']))
                    max_pitch_res.append(np.nanmax(Smap.resolution['Pitch']['sigma']))
                    
                    avg_collimating_factor.append(np.nanmean(Smap.collimator_factor_matrix))
                    
                    try:
                        pitch_85_gyro_1_collimating_factor.append(Smap.collimator_factor_matrix[idx2, idx_85])
                    except:
                        pitch_85_gyro_1_collimating_factor.append(0)
                    
                    SMAPS.append(Smap)
                    
                    #IPython.embed()
                    del Smap
                
                ##plot metrics
                x = scan_Parameters[scan_paramter]['scan_values']
                
                fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12, 12),
                                           facecolor='w', edgecolor='k', dpi=100)
                
                ax_gyro, ax_pitch, ax_gyro_res, ax_pitch_res = axarr[0,0], axarr[0,1], axarr[1,0], axarr[1,1]
                ax_gyro.set_xlabel(scan_paramter + ' [cm]')
                ax_gyro.set_ylabel('Gyroradius [cm]')
                
                #ax_pitch.set_xlabel(scan_paramter + ' [cm]')
                #ax_pitch.set_ylabel('Pitch angle [$\\degree$]')
                
                ax_gyro_res.set_xlabel(scan_paramter + ' [cm]')
                ax_gyro_res.set_ylabel('Gyroradius resolution [cm]')
                
                ax_pitch_res.set_xlabel(scan_paramter + ' [cm]')
                ax_pitch_res.set_ylabel('Pitch angle resolution [$\\degree$]')
                
                ax_gyro.plot(x, min_gyro, marker = 'o', label = 'min')
                ax_gyro.plot(x, max_gyro, marker = 'o', label = 'max')
                ax_gyro.legend(loc='upper right')
                #ax_pitch.plot(x, min_pitch, marker = 'o', label = 'min')
                #ax_pitch.plot(x, max_pitch, marker = 'o', label = 'max')
                #ax_pitch.legend(loc='upper right')
                
                ax_gyro_res.plot(x, gyro_1_res, marker = 'o', label = '0.5 cm')
                ax_gyro_res.plot(x, gyro_2_res, marker = 'o', label = '1 cm')
                ax_gyro_res.plot(x, gyro_3_res, marker = 'o', label = '1.5 cm')
                ax_gyro_res.legend(loc='upper right')
                
                ax_pitch_res.plot(x, min_pitch_res, marker = 'o', label = 'min')
                ax_pitch_res.plot(x, max_pitch_res, marker = 'o', label = 'max')
                ax_pitch_res.plot(x, pitch_85_res, marker = 'o', label = '85$\\degree$')
                ax_pitch_res.legend(loc='upper right')

                
                #fig, ax_coll = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                #                           facecolor='w', edgecolor='k', dpi=100)
                ax_coll = axarr[0,1]
                ax_coll.plot(x, avg_collimating_factor, marker = 'o', label='avg')
                ax_coll.plot(x, pitch_85_gyro_1_collimating_factor, marker = 'o', label='1 cm, 85$\\degree$')
                ax_coll.legend(loc='upper right')
                ax_coll.set_xlabel(scan_paramter + ' [cm]')
                ax_coll.set_ylabel('Average collimator factor %')
                
                ax_gyro_res.set_ylim([0, 1.0])
                ax_pitch_res.set_ylim([0, 5.0])
                
                fig.tight_layout()
                fig.show()
                
    # -----------------------------------------------------------------------------
    # --- Section 3: Analyse the synthetic signal
    # -----------------------------------------------------------------------------
    smap = SMAPS[0]


    if plot_gyro_res:
        smap.plot_gyroradius_histograms(pitch = 85)
    
    if plot_pitch_res:
        smap.plot_pitch_histograms(gyroradius = 1.)

    if plot_collimating_factor:
        smap.plot_collimator_factor()
        plt.gcf().show()
    
    if plot_synthetic_signal:
        dist_file = '/afs/ipp/home/a/ajvv/ascot5/RUNS/W7X_distributions/pos_02_FILD_distro.pck'
        distro = pickle.load( open( dist_file, "rb" ) )
        g_array, p_array, signal = ss.LibFILDSIM.FILDSIMforward.synthetic_signal(distro, smap, 
                                                                                 gmin=0.75, 
                                                                                 gmax=3, 
                                                                                 pmin=55,
                                                                                 pmax=105)
        
        fig, ax_syn = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                   facecolor='w', edgecolor='k', dpi=100)   
        ax_syn.set_xlabel('Pitch [$\\degree$]')
        ax_syn.set_ylabel('Gyroradius [cm]')
        ax_syn.set_ylim([0, 2.5])
        ss.LibFILDSIM.FILDSIMforward.plot_synthetic_signal(g_array, p_array
                                                           , signal, 
                                                           ax=ax_syn, fig=fig)
        fig.tight_layout()
        fig.show()
        