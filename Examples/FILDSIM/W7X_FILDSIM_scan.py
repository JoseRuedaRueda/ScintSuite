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


def write_plate_files(root_dir = '/afs/ipp/home/a/ajvv/FILDSIM/geometry/W7X/Scans/', string_mod = '', pinhole_lenght  = 0.2, pinhole_width   = 0.1, pinhole_scint_dist = 0.5, slit_height     = 1.0, slit_length     = 5.,scint_width     = 10., scint_height    = 10.):
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
    pinhole_padding = 3.
    pinhole_depth = 10.
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
    #slit_plates
    #pinhole 1
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
    #slit back
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
    #plt.ion()
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
    marker_params, line_params = {'markersize':6, 'marker':'o','color':'b'}, {'ls':'solid','color':'k'}
    
    pinhole_lenght  = 0.2
    pinhole_width   = 0.1
    pinhole_scint_dist = 0.5
    slit_height     = 1.0
    slit_length     = 5.
    scint_width     = 10 #large scintilator
    scint_height    = 10

    n_markers = 1e4
    pinhole_widths = np.linspace( 0.02, 0.8, 1)
    scan_str = 'pw'  #pw = pinhole_width, pl = pinhole_lenght, pa = pinhole_area, psd = pinhole_scint_dist, sh = slit_height, sl =slit_length
    
    # -----------------------------------------------------------------------------
    # --- Section 1: Run FILDSIM scan over scan parameters
    # -----------------------------------------------------------------------------    
    if run_scan:
        for pinhole_width in pinhole_widths:
        #for pinhole_lenght in pinhole_lenghts:
        #for pinhole_scint_dist in pinhole_scint_dists:
        #for slit_height in slit_heights:
        #for slit_length in slit_lengths:
            '''
            Loop over scan paramater
            '''
            string_mod = '_%s_%s' %(scan_str, f'{float(pinhole_width):g}')
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
                    'n_gyroradius': 11,
                    'n_pitch': 10,
                    'save_orbits': 0,
                    'verbose': True,
                },
                'input_parameters': {
                    'n_ions': n_markers,
                    'step': 0.01,
                    'helix_length': 10.0,
                    'gyroradius': [1.5, 1.75, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                    'pitch_angle': [85., 80., 70., 60., 50., 40., 30., 20., 10, 0.],
                    'gyrophase_range': [0.0, 3.14],
                    'start_x': start_x,
                    'start_y': start_y,
                    'start_z': [0.0, 0.0],
                    'theta': 0.0,
                    'phi': 0.0
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

            # Run FILDSIM
            ss.fildsim.run_FILDSIM(namelist_name, queue = True)

    # -----------------------------------------------------------------------------
    # --- Section 2: Analyse the results
    # -----------------------------------------------------------------------------
    SMAPS = []
    
    if read_scan:
        for pinhole_width in pinhole_widths:
            string_mod = '_%s_%s' %(scan_str, f'{float(pinhole_width):g}')
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
            Smap.load_strike_points(strike_points_file)
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
            else:
                fig, ax2D_yz = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                   facecolor='w', edgecolor='k', dpi=100)
                ax2D_yz.set_xlabel('Y [cm]')
                ax2D_yz.set_ylabel('Z [cm]')
                ax2D_yz.set_title('Front view (Y-Z plane)')
                
            if plot_strike_points:
                Smap.plot_strike_points(ax=ax2D_yz)#, plt_param=marker_params)
            if plot_strikemap:
                Smap.plot_real(ax=ax2D_yz, marker_params=marker_params, line_params = line_params)
                
            fig.show()
            
            
            Smap.calculate_resolutions()  # Default call,
            # Save the result in the SMAP list
            SMAPS.append(Smap)
            
            #IPython.embed()
            del Smap
