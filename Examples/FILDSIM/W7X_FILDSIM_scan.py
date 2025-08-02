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

def get_normal_vector(p1, p2, p3):
    '''
    '''
    #https://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    
    cp = cp/(cp**2).sum()**0.5
    
    return cp

def get_xx_points_on_circle(xy1, xy2, r, y, right=True):
    '''
    '''
    center_x, center_y = circle_center(xy1, xy2, r, right=right)
    #IPython.embed()
    return np.sqrt(r**2 - (y - center_x)**2 ) + center_y

def get_x_points_on_circle(xy1, xy2, r, y, right=True):
    '''
    '''
    center_x, center_y = circle_center(xy1, xy2, r, right=right)
    #IPython.embed()
    return np.sqrt(r**2 - (y - center_y)**2 ) + center_x

def circle_center(xy1, xy2, r, right=True):
    #https://stackoverflow.com/questions/4914098/centre-of-a-circle-that-intersects-two-points
    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]
    q = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    if r*2 < q:
        
        IPython.embed()
        r = q/2.
    
    y3 = (y1+y2)/2
    x3 = (x1+x2)/2
    # first circle
    #IPython.embed()
    center_x = x3 + np.sqrt(r**2-(q/2)**2)*(y1-y2)/q 
    center_y = y3 + np.sqrt(r**2-(q/2)**2)*(x2-x1)/q 
    if not right:
        # second circle
        center_x = x3 - np.sqrt(r**2-(q/2)**2)*(y1-y2)/q
        center_y = y3 - np.sqrt(r**2-(q/2)**2)*(x2-x1)/q  
    
    return center_x, center_y
  
def get_curved_collimating_plate(head_params, col_slit_dist):
    '''
    Parameters
    ----------
    Returns
    -------
    None.
    '''
    pinhole_lenght = head_params["pinhole_lenght"]

    slit_height = head_params["slit_height"]
    slit_length = head_params["slit_length"]


    n_curve_points = head_params["n_curve_points"]
    opening_curve_radius = head_params["opening_curve_radius"]
    inward_curve_radius = head_params["inward_curve_radius"]
    inward_distance = head_params["inward_distance"]


    
    plates = []
    
    yz1 = np.array([-0.5 * pinhole_lenght, -slit_height])
    yz2 = np.array([slit_length,  0.]) #Directly below slit opening
    
    
    N_vertices =  n_curve_points + 1
    xyz_slit_lateral = np.zeros((N_vertices, 3))
    
    xyz_slit_lateral[:, 0] = np.ones(N_vertices) * col_slit_dist #(0.5 * pinhole_width)
    slit_lateral_normal = np.array([1.,   0.,   0.])
    
    xyz_slit_lateral[0, 1], xyz_slit_lateral[0, 2] =  -0.5 * pinhole_lenght, 0
    
    z_points = np.linspace(-slit_height, 0, n_curve_points)
    y_points = get_x_points_on_circle(yz1, yz2, opening_curve_radius, z_points)

    xyz_slit_lateral[1:, 1] = y_points
    xyz_slit_lateral[1:, 2] = z_points
    
    #IPython.embed()
    if not inward_curve_radius:
        if col_slit_dist<0:
            name = 'slit_lateral_1_'
        else:
            name = 'slit_lateral_2_' 
            
        plates.append({'name': name,
            'N_vertices':N_vertices,
            'vertices':xyz_slit_lateral, 
            'normal':slit_lateral_normal })
    else:
        N_vertices =  4 #* n_curve_points 
        
        
        xz2 = np.array([col_slit_dist,0]) 
        xz1 = np.array([col_slit_dist - inward_distance,  -slit_height,])
        
        x_points = get_x_points_on_circle(xz1, xz2, inward_curve_radius, z_points)  # -  col_slit_dist    
        plates = []
        #IPython.embed()
        for i in range(n_curve_points-1):
            xyz_slit_lateral = np.zeros((N_vertices, 3))
            xyz_slit_lateral[0, 0] = x_points[i]
            xyz_slit_lateral[0, 1] = -0.5 * pinhole_lenght
            xyz_slit_lateral[0, 2] = z_points[i]

            xyz_slit_lateral[1, 0] = x_points[i+1]
            xyz_slit_lateral[1, 1] = -0.5 * pinhole_lenght
            xyz_slit_lateral[1, 2] = z_points[i+1]
            
            xyz_slit_lateral[2, 0] = x_points[i+1]
            xyz_slit_lateral[2, 1] = y_points[i+1]
            xyz_slit_lateral[2, 2] = z_points[i+1]
            
            xyz_slit_lateral[3, 0] = x_points[i]
            xyz_slit_lateral[3, 1] = y_points[i]
            xyz_slit_lateral[3, 2] = z_points[i]           
            
            slit_lateral_normal =  get_normal_vector(xyz_slit_lateral[0, :],
                                     xyz_slit_lateral[1, :], 
                                     xyz_slit_lateral[2, :])
            
            if col_slit_dist<0:
                name = 'slit_lateral_1_' + str(i)
            else:
                name = 'slit_lateral_2_' + str(i)
                
            plates.append({'name': name,
            'N_vertices':4,
            'vertices':xyz_slit_lateral, 
            'normal':slit_lateral_normal }
                          )
            
        pass
    

    return plates

def get_tilted_collimating_plate(head_params, col_slit_dist):
    '''
    Parameters
    ----------
    Returns
    -------
    None.
    '''
    pinhole_lenght = head_params["pinhole_lenght"]

    slit_height = head_params["slit_height"]
    slit_length = head_params["slit_length"]



    slit_theta = head_params["slit_theta"]
    
    N_vertices = 4
    xyz_slit_lateral = np.zeros((N_vertices, 3))
    xyz_slit_lateral[:, 0] = np.ones(N_vertices) * col_slit_dist 
    
 
    xyz_slit_lateral[1, 0] = col_slit_dist - np.tan(slit_theta) * slit_height
    xyz_slit_lateral[1, 1] = -0.5 * pinhole_lenght
    xyz_slit_lateral[1, 2] = -slit_height
    
    xyz_slit_lateral[2, 0] = col_slit_dist - np.tan(slit_theta) * slit_height
    xyz_slit_lateral[2, 1] = 0.5 * pinhole_lenght
    xyz_slit_lateral[2, 2] = -slit_height
    
    xyz_slit_lateral[3, 1] = slit_length
    xyz_slit_lateral[3, 2] = 0
    
    xyz_slit_lateral[0, 1] = -0.5 * pinhole_lenght
    xyz_slit_lateral[0, 2] = 0.
    
    slit_lateral_normal =  get_normal_vector(xyz_slit_lateral[0, :],
                                     xyz_slit_lateral[1, :], 
                                     xyz_slit_lateral[2, :])

    return {'name': '',
            'N_vertices':N_vertices,
            'vertices':xyz_slit_lateral, 
            'normal':slit_lateral_normal }

def get_scint_plate(head_params):
    '''

    Returns
    -------
    None.

    '''

    pinhole_lenght = head_params["pinhole_lenght"]

    pinhole_scint_dist = head_params["pinhole_scint_dist"]

    scint_width = head_params["scint_width"] 
    scint_height = head_params["scint_height"]

    #IPython.embed()

    scint_theta = head_params["scint_theta"]


    N_vertices = 4
    xyz_scint = np.zeros((N_vertices, 3))
    xyz_scint[:, 0] = - pinhole_scint_dist * np.ones(N_vertices) # scintilor x points all the same
    
    xyz_scint[0, 0] = - pinhole_scint_dist
    xyz_scint[0, 1] = -0.5 * pinhole_lenght
    xyz_scint[0, 2] =  0
    
    #IPython.embed()
    xyz_scint[1, 0] = - pinhole_scint_dist + np.tan(scint_theta) * scint_height
    xyz_scint[1, 1] = -0.5 * pinhole_lenght
    xyz_scint[1, 2] =  -scint_height
    
    xyz_scint[2, 0] = - pinhole_scint_dist + np.tan(scint_theta) * scint_height
    xyz_scint[2, 1] = -0.5 * pinhole_lenght + scint_width
    xyz_scint[2, 2] = -scint_height
    
    xyz_scint[3, 0] = - pinhole_scint_dist
    xyz_scint[3, 1] = -0.5 * pinhole_lenght + scint_width
    xyz_scint[3, 2] = 0
    
    
    scint_normal = get_normal_vector(xyz_scint[0, :],
                                     xyz_scint[1, :], 
                                     xyz_scint[2, :])

    #slit_plates

    return {'name': 'W7X_scint_scan',
            'N_vertices':N_vertices,
            'vertices':xyz_scint, 
            'normal':scint_normal }

def get_slit_backplate(head_params,
                     slit_theta):
    '''
    '''
    
    pinhole_lenght = head_params["pinhole_lenght"]
    pinhole_width = head_params["pinhole_width"]
    pinhole_scint_dist = head_params["pinhole_scint_dist"]
    slit_height = head_params["slit_height"]


    #slit back plate
    N_vertices = 4
    xyz_slit_back = np.zeros((N_vertices, 3))
    xyz_slit_back[:, 1] = np.ones(N_vertices) *(-0.5 * pinhole_lenght) 
    
    #xyz_slit_back[0, 0] = -0.5 * pinhole_width - np.tan(slit_theta) * slit_height
    xyz_slit_back[0, 0] = - pinhole_scint_dist
    xyz_slit_back[0, 2] = -slit_height
    
    #xyz_slit_back[1, 0] =  0.5 * pinhole_width - np.tan(slit_theta) * slit_height
    xyz_slit_back[1, 0] =  0.5 * pinhole_width + 2
    xyz_slit_back[1, 2] =  -slit_height
    
    #xyz_slit_back[2, 0] =  0.5 * pinhole_width
    xyz_slit_back[2, 0] =  0.5 * pinhole_width + 2
    xyz_slit_back[2, 2] =  0
    
    #xyz_slit_back[3, 0] = -0.5 * pinhole_width
    xyz_slit_back[3, 0] = - pinhole_scint_dist
    xyz_slit_back[3, 2] = 0
    
    
    slit_back_normal = np.array([0.,   1.,   0.])

    
    return {'name': 'W7X_slit_back_scan', 
            'N_vertices':N_vertices,
            'vertices':xyz_slit_back, 
            'normal':slit_back_normal }
    
def write_plate_files(root_dir = '/afs/ipp/home/a/ajvv/FILDSIM/geometry/W7X/Scans/'
                      , string_mod = ''
                      , head_params = {}):
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
    
    pinhole_padding = 0.2
    pinhole_depth = 10.
    
    
    pinhole_lenght = head_params["pinhole_lenght"]
    pinhole_width = head_params["pinhole_width"]
    pinhole_scint_dist = head_params["pinhole_scint_dist"]
    slit_height = head_params["slit_height"]
    slit_length = head_params["slit_length"]
    scint_width = head_params["scint_width"] 
    scint_height = head_params["scint_height"]
    
    n_curve_points = head_params["n_curve_points"]
    opening_curve_radius = head_params["opening_curve_radius"]
    inward_curve_radius = head_params["inward_curve_radius"]
    inward_distance = head_params["inward_distance"]
    n_curve_points = head_params["n_curve_points"]
    scint_theta = head_params["scint_theta"]
    slit_theta = head_params["slit_theta"]
    
    
    
    plates = []
    #scintilator_plate
    
    scint_plate = get_scint_plate(head_params)
    plates.append(scint_plate)
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
     
    
    if slit_theta:
        #slit lateral 1
        col_slit_dist = -0.5 * pinhole_width
        slit_lateral_1 = get_tilted_collimating_plate(head_params, col_slit_dist)
           
        
        slit_lateral_1['name'] = 'W7X_slit_lateral_1_scan'
        
        #slit lateral 2
        col_slit_dist = 0.5 * pinhole_width
        slit_lateral_2 = get_tilted_collimating_plate(head_params, col_slit_dist)       
        
        slit_lateral_2['name'] = 'W7X_slit_lateral_2_scan'
        plates.append(slit_lateral_1)
        plates.append(slit_lateral_2)
    else:
        #slit lateral 1
        col_slit_dist = -0.5 * pinhole_width
        slit_lateral_1 = get_curved_collimating_plate(head_params, col_slit_dist)        
        
        #slit_lateral_1['name'] = 'W7X_slit_lateral_1_scan'
        plates = plates + slit_lateral_1
        #slit lateral 2
        col_slit_dist = 0.5 * pinhole_width
        slit_lateral_2 = get_curved_collimating_plate(head_params, col_slit_dist)        
        #IPython.embed()
        #slit_lateral_2['name'] = 'W7X_slit_lateral_2_scan'
        plates = plates + slit_lateral_2
    
    

    back_plate = get_slit_backplate(head_params,
                     slit_theta)

    plates.append(back_plate)
    
    plate_files = []
    #IPython.embed()
    for plate in plates:
        plate_filename = root_dir + plate['name']+ string_mod+'.pl'
        if not plate['name'] == 'W7X_scint_scan':
            plate_files.append(plate['name']+ string_mod+'.pl')
            
        f = open(plate_filename, 'w')
        f.write('# Plate file for FILDSIM.f90\n')
        f.write('Name='+plate['name']+'\n')
        f.write('N_vertices='+str(plate['N_vertices'])+'\n')
        
        for i in range(plate['N_vertices']):
            f.write(str(np.round(plate['vertices'][i, 0],2)) + ',' + str(np.round(plate['vertices'][i, 1],2)) + ',' +  str(np.round(plate['vertices'][i, 2],2)) + '\n')
        
        f.write('Normal_vector\n')
        f.write(str(np.round(plate['normal'][0],2))  + ',' + str(np.round(plate['normal'][1],2)) + ','+  str(np.round(plate['normal'][2],2)) + '\n')
        
        f.close()
    
    #IPython.embed()
    
    return plate_files
        
if __name__ == '__main__':
    #set plotting settings
    ss.LibPlotting.plotSettings(plot_mode='Presentation')
    # -----------------------------------------------------------------------------
    namelist_path = ss.paths.FILDSIM + 'results/W7X_scans/'   # Paths to save namelist
    # -----------------------------------------------------------------------------
    # Section 0: FILDSIM Settings
    # -----------------------------------------------------------------------------
    Test = True  #if true don't submit run
    
    run_scan = False
    save_orbits =0#0=Don't Save orbit trajectories, 1= Save
    plot_plate_geometry = True
    plot_3D = False
    
    read_scan = not run_scan#True
    plot_strike_points = True
    plot_strikemap = True
    plot_orbits = False
    
    #for best case    
    plot_gyro_res= False
    plot_pitch_res= False
    plot_collimating_factor = False
    plot_synthetic_signal= False
    
    marker_params = {'markersize':6, 'marker':'o','color':'b'}
    line_params =  {'ls':'solid','color':'k'}
    
    #[cm]
    pinhole_lenght  = 0.1 #0.2 
    pinhole_width   = 0.08 #0.1
    pinhole_scint_dist = 0.8 #0.6 #0.5
    slit_height     = 0.5 #1.0
    slit_length     = 1.0#6. #5.
    scint_width     = 6 #large scintilator
    scint_height    = 6
    
    n_curve_points = 50#2
    opening_curve_radius = 2
    scint_theta  = 0
    slit_theta = 0
    
    inward_curve_radius = 0
    inward_distance = 0.02
    
    Br, Bz, Bt = -0.3263641727027391, -0.3266729522875698 , 2.3419531967879372#0.515, -0.091, 2.621  #[T]
    alpha = 0.16  #From J. Hidalgo
    beta = 215. #180
    
    n_markers = int(1e5)
    
    scan_Parameters = {}
    scan_Parameters['Pinhole width'] = {'scan_str': 'pw',
                                        'scan_values':[0.08],#np.arange(0.03, 0.15, 0.01), #[0.08],#
                                        'scan': False,
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
                                    'scan_values': np.arange(0.5, 3.0, 0.5),
                                    'scan': False,
                                    'value': slit_length}
    scan_Parameters['Beta'] = {'scan_str': 'b',
                                    'scan_values': np.arange(200, 255, 5),
                                    'scan': False,
                                    'value': beta}    
    scan_Parameters['opening_curve_radius'] = {'scan_str': 'oc',
                                    'scan_values': [4],#np.arange(5, 17, 2),
                                    'scan': False,
                                    'value': opening_curve_radius} 
    scan_Parameters['Scint theta'] = {'scan_str': 'st',
                                    'scan_values': np.arange(-20, 20, 5),
                                    'scan': True,
                                    'value': scint_theta} 
    scan_Parameters['Slit theta'] = {'scan_str': 'ct',
                                    'scan_values': np.arange(-20, 20, 5),
                                    'scan': False,
                                    'value': slit_theta} 
    scan_Parameters['inward_curve_radius'] = {'scan_str': 'icr',
                                    'scan_values': [0.5],#np.arange(0.5, 4, 0.5),
                                    'scan': False,
                                    'value': inward_curve_radius} 
    
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
                    
                    opening_curve_radius = scan_Parameters['opening_curve_radius']['value']
                    scint_theta = scan_Parameters['Scint theta']['value']
                    slit_theta = scan_Parameters['Slit theta']['value']
                    
                    inward_curve_radius = scan_Parameters['inward_curve_radius']['value']
                    #IPython.embed()
                    
                    phi, theta = ss.LibFILDSIM.FILDSIMexecution.calculate_fild_orientation(Br, Bz, Bt, alpha, beta, verbose=False)
                    
                    string_mod = '_%s_%s' %(scan_Parameters[scan_paramter]['scan_str'], f'{float(value):g}')
                    
                    head_params = {
                                    "pinhole_lenght": pinhole_lenght,
                                    "pinhole_width" : pinhole_width ,
                                    "pinhole_scint_dist": pinhole_scint_dist,
                                    "slit_height"   : slit_height,
                                    "slit_length"   : slit_length,
                                    "scint_width"   : scint_width, 
                                    "scint_height"  : scint_height,
                                    "beta"          : beta,
                                    "n_curve_points": n_curve_points,
                                    "opening_curve_radius": opening_curve_radius,
                                    "scint_theta"   : np.deg2rad(scint_theta),
                                    "slit_theta"    : np.deg2rad(slit_theta),
                                    "inward_curve_radius" : inward_curve_radius,
                                    "inward_distance" : inward_distance
                                    }
                    
                    
                    plate_files = write_plate_files(root_dir = ss.paths.FILDSIM + '/geometry/W7X/Scans/',
                                    string_mod = string_mod,
                                    head_params = head_params )
                        
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
                            'save_orbits': save_orbits,
                            'verbose': True,
                            'export_scint_coords': False
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
                            'n_slits': len(plate_files)
                        },
                        'plate_files': {
                            'scintillator_files': ['W7X_scint_%s.pl' %(runid)],
                            'slit_files': plate_files
                        }
                    }
                    # Write namelist
                    ss.fildsim.write_namelist(FILDSIM_namelist, p=namelist_path)
                    namelist_name = os.path.join(namelist_path,
                                                FILDSIM_namelist['config']['runid'] + '.cfg')
                    
                    if plot_plate_geometry:
                        ss.LibFILDSIM.FILDSIMplot.plot_geometry(namelist_name, plot_3D = plot_3D)
                        plt.show()
                    # Run FILDSIM
                    if not Test:
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
                            #Smap.plot_strike_points(ax=ax2D_yz)#, plt_param=marker_params)
                            
                            Strikes = ss.LibFILDSIM.FILDSIMmarkers.Strikes(file = strike_points_file)
                            
                            mar_params = {'zorder':3 }
                            
                            Strikes.plot2D(per=0.1, ax=ax2D_yz, mar_params = mar_params)
                            
                        if plot_strikemap:
                            Smap.plot_real(ax=ax2D_yz, marker_params=marker_params, line_params = line_params)
                            
                        

                        if plot_orbits:
                            # Load the result of the simulation
                        
                            base_name = namelist_path + runid
                            orbits_file = base_name + '_orbits.dat'
                            orbits_index_file = base_name + '_orbits_index.dat'                        
                        
                            # plot geometry of calculated orbits
                            Orbits = ss.LibFILDSIM.FILDSIMmarkers.Orbits(orbits_file, orbits_index_file)
                            Orbits.plot(per=0.5, ax3D=None, axarr=axarr, dpi=100)
                        
                        
                        
                        fig.show()
    
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
                    #IPython.embed()
                    idx_85 = np.argmin( abs(Smap.strike_points.header['pitch'] - 85))
                    try:
                        pitch_85_res.append(np.nanmax(Smap.resolution['Gyroradius']['sigma'][idx_85]))
                    except:
                        pitch_85_res.append(np.nan)
                    
                    idx1 = np.argmin( abs(Smap.strike_points.header['gyroradius'] -0.5))
                    idx2 = np.argmin( abs(Smap.strike_points.header['gyroradius'] -1.0))
                    idx3 = np.argmin( abs(Smap.strike_points.header['gyroradius'] -1.5))
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
    #smap = SMAPS[0]


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
        
