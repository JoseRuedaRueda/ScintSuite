#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:11:34 2022

@author: ajvv
"""

import numpy as np
import os
import matplotlib.pylab as plt
import Lib as ss
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)
import IPython
import pickle
#import open3d
from matplotlib import cm
from stl import mesh 

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

def write_collimator_plates(file_name_save= 'lateral_collimator_plates'
                            , ph1 = 1
                            , ph2 = 1
                            , pl1 = 2
                            , pl2 = 2
                            , pin_0 = 0
                            , pin_2 = 0
                            , pin_0_right = 0
                            , pin_2_right = 0):
    '''
    '''
    #dl in cm converted to mm
    ph1 *= 10
    ph2 *= 10
    
    pl1 *= 10
    pl2 *= 10
    
    ##left collimator
    p1 = np.array([-728.328, 5789.25, 370.004])
    p1_delta = np.array([-728.138, 5788.95, 370.469])
    p2=np.array([-742.387, 5784.41, 369.301])
    p2=np.array([-745.693, 5784.27, 367.113])
    #follow vector for dl length
    vi = p1_delta - p1
    vi/= np.linalg.norm(vi) # unit vector
    ic = np.argmax(np.abs(vi))
    dpos=vi/np.abs(vi[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*ph1
    
    if np.sum(pin_0) != 0:
        p1 = pin_0
        #lengt vector
        p2l = np.array([-729.753, 5788.8, 369.88])
        p1l = np.array([-728.328,5789.25, 370.004])
 
        p2l = np.array([-745.693, 5784.27, 367.113])
        p1l = pin_0   

        #lenght vector
        v_pl = p2l - p1l
        v_pl/= np.linalg.norm(v_pl)
        ic = np.argmax(np.abs(v_pl))
        d_pl=v_pl/np.abs(v_pl[ic])
        d_pl=d_pl/np.sqrt(np.sum(d_pl**2))*pl1 
        p2 = p1 + d_pl  
        
    p1_dl = p1 + dpos
    
    xx = np.array([p1[0], p1_dl[0],p2[0]])
    yy = np.array([p1[1], p1_dl[1],p2[1]])
    zz = np.array([p1[2], p1_dl[2],p2[2]])
    
    p1 = np.array([-728.482, 5789.55, 370.732])
    p1_delta = np.array([-728.292, 5789.25, 371.196])
    p2=np.array([-742.541, 5784.71, 370.028])
    p2=np.array([-745.893, 5784.65, 367.904])
    #follow vector for dl length
    vi = p1_delta - p1
    vi/= np.linalg.norm(vi) # unit vector
    ic = np.argmax(np.abs(vi))
    dpos=vi/np.abs(vi[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*ph1
    
    if np.sum(pin_2) != 0:
        p1 = pin_2
         #lengt vector
        p2l = np.array([-729.753, 5788.8, 369.88])
        p1l = np.array([-728.328,5789.25, 370.004])

        p2l = np.array([-745.693, 5784.27, 367.113])
        p1l = pin_0   
    
        #lenght vector
        v_pl = p2l - p1l
        v_pl/= np.linalg.norm(v_pl)
        ic = np.argmax(np.abs(v_pl))
        d_pl=v_pl/np.abs(v_pl[ic])
        d_pl=d_pl/np.sqrt(np.sum(d_pl**2))*pl1 
        p2 = p1 + d_pl  
        
    p1_dl = p1 + dpos
    
    xx = np.concatenate([xx, np.array([p1[0], p1_dl[0],p2[0]])])
    yy = np.concatenate([yy, np.array([p1[1], p1_dl[1],p2[1]])])
    zz = np.concatenate([zz, np.array([p1[2], p1_dl[2],p2[2]])])
    
    
    ###right collimator
    
    p1 = np.array([-786.853, 5796.68, 333.852])
    p1_delta = np.array([-786.688, 5796.38, 334.314])
    p2=np.array([-767.689, 5802.43, 335.872])
    p2=np.array([-771.133, 5803.19, 337.5])
    #follow vector for dl length
    vi = p1_delta - p1
    vi/= np.linalg.norm(vi) # unit vector
    ic = np.argmax(np.abs(vi))
    dpos=vi/np.abs(vi[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*ph2
    
    if np.sum(pin_0_right) != 0:
        p1 = pin_0_right
        #lengt vector
        p2l = np.array([-785.428, 5797.14, 333.969])
        p1l = np.array([-786.853, 5796.68, 333.852])

        p2l = np.array([-771.133, 5803.19, 337.5])
        p1l = pin_0_right     
        #scint_vector:
        v_scint = p2l-p1l
        v_scint/= np.linalg.norm(v_scint)
        ic = np.argmax(np.abs(v_scint))
        d_pl=v_scint/np.abs(v_scint[ic])
        d_pl=d_pl/np.sqrt(np.sum(d_pl**2))*pl2 
        p2 = p1 + d_pl  
        
    
    p1_dl = p1 + dpos

    xx = np.concatenate([xx, np.array([p1[0], p1_dl[0],p2[0]])])
    yy = np.concatenate([yy, np.array([p1[1], p1_dl[1],p2[1]])])
    zz = np.concatenate([zz, np.array([p1[2], p1_dl[2],p2[2]])])
    
    p1_1old = np.array([p1[0], p1[1], p1[2]])
    p1_dl_1old = np.array([p1_dl[0], p1_dl[1], p1_dl[2]])
    p2_1old = np.array([p2[0], p2[1], p2[2]])

    p1 = np.array([-787.007, 5796.98, 334.579])
    p1_delta = np.array([-786.842, 5796.67, 335.041])
    p2=np.array([-767.842, 5802.72, 336.599])
    #follow vector for dl length
    vi = p1_delta - p1
    vi/= np.linalg.norm(vi) # unit vector
    ic = np.argmax(np.abs(vi))
    dpos=vi/np.abs(vi[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*ph2
#    
    if np.sum(pin_2_right) != 0:
        p1 = pin_2_right
        #lengt vector
        p2l = np.array([-785.428, 5797.14, 333.969])
        p1l = np.array([-786.853, 5796.68, 333.852])

        p2l = np.array([-771.133, 5803.19, 337.5])
        p1l = pin_0_right   
    
        #scint_vector:
        v_scint = p2l-p1l
        v_scint/= np.linalg.norm(v_scint)
        ic = np.argmax(np.abs(v_scint))
        d_pl=v_scint/np.abs(v_scint[ic])
        d_pl=d_pl/np.sqrt(np.sum(d_pl**2))*pl2 
        p2 = p1 + d_pl  
        
    
    p1_dl = p1 + dpos

    xx = np.concatenate([xx, np.array([p1[0], p1_dl[0],p2[0]])])
    yy = np.concatenate([yy, np.array([p1[1], p1_dl[1],p2[1]])])
    zz = np.concatenate([zz, np.array([p1[2], p1_dl[2],p2[2]])])

    p1_2old = np.array([p1[0], p1[1], p1[2]])
    p1_dl_2old = np.array([p1_dl[0], p1_dl[1], p1_dl[2]])
    p2_2old = np.array([p2[0], p2[1], p2[2]])


    ###backplate:
    xx = np.concatenate([xx, np.array([p1_1old[0], p1_dl_1old[0],p1_2old[0]])])
    yy = np.concatenate([yy, np.array([p1_1old[1], p1_dl_1old[1],p1_2old[1]])])
    zz = np.concatenate([zz, np.array([p1_1old[2], p1_dl_1old[2],p1_2old[2]])])
    
    xx = np.concatenate([xx, np.array([p1_2old[0], p1_dl_2old[0],p1_1old[0]])])
    yy = np.concatenate([yy, np.array([p1_2old[1], p1_dl_2old[1],p1_1old[1]])])
    zz = np.concatenate([zz, np.array([p1_2old[2], p1_dl_2old[2],p1_1old[2]])])


    xx = np.concatenate([xx, np.array([p1_dl_1old[0], p1_dl_2old[0],p1_1old[0]])])
    yy = np.concatenate([yy, np.array([p1_dl_1old[1], p1_dl_2old[1],p1_1old[1]])])
    zz = np.concatenate([zz, np.array([p1_dl_1old[2], p1_dl_2old[2],p1_1old[2]])])
    
    xx = np.concatenate([xx, np.array([p1_dl_1old[0], p1_dl_2old[0],p1_2old[0]])])
    yy = np.concatenate([yy, np.array([p1_dl_1old[1], p1_dl_2old[1],p1_2old[1]])])
    zz = np.concatenate([zz, np.array([p1_dl_1old[2], p1_dl_2old[2],p1_2old[2]])])

   
    n_triang = 8
    data = np.zeros(n_triang, dtype=mesh.Mesh.dtype)
    mesh_object = mesh.Mesh(data, remove_empty_areas=False)
    mesh_object.x[:] = np.reshape(xx, (n_triang, 3))
    mesh_object.y[:] = np.reshape(yy, (n_triang, 3))
    mesh_object.z[:] = np.reshape(zz, (n_triang, 3))
    mesh_object.save(file_name_save+'.stl') 



def write_slit_cover_plate(pin_width, 
                           pin_length, 
                           scint_dist, 
                           slit_edge_dist, 
                           wall_thickness,
                           file_name_save= 'pinhole_cover'):
    #make stl file
    
    pin_width *=10
    pin_length*=10
    scint_dist*=10
    slit_edge_dist*=10
    wall_thickness*=10
    
    scint_dist*=1 #due to 60 degree angle ##only for right side


    ###left
    p1 = np.array([-736.628, 5777.78, 350.404])
    p2 = np.array([-746.487, 5774.16, 349.79])
    p3 = np.array([-718.967, 5802.15, 392.035])
    p4 = np.array([-732.395, 5793.62, 383.037])
    

    p5 = np.array([-718.869, 5746.71, 397.272])
    p6 = np.array([-700.622, 5771.91, 440.323])
    ## pinhole vectors
    ## lengt vector
    p2l = np.array([-729.753, 5788.8, 369.88])
    p1l = np.array([-728.328,5789.25, 370.004])
    ##width
    p2w = np.array([-728.482, 5789.55, 370.732])
    
    
    
    ###right
    p1 = np.array([-782.654, 5789.48, 315.895])
    p2 = np.array([-774.232, 5792.03, 316.302])
    p3 = np.array([-789.885, 5806.95, 357.669])
    p4 = np.array([-785.375, 5818.47, 379.646])
    

    p5 = np.array([-756.996, 5737.83, 391.419])
    p6 = np.array([-766.33, 5759.63, 435.953])
    #pinhole vectors
    #lengt vector
    p2l = np.array([-785.428, 5797.14, 333.969])
    p1l = np.array([-786.853, 5796.68, 333.852])
    #width
    p2w = np.array([-787.007, 5796.98, 334.579])


    #adjust points p1 and p3 according to wall thickness
    v = p1-p2
    v/= np.linalg.norm(v)
    ic = np.argmax(np.abs(v))
    dpos=v/np.abs(v[ic])
    dpos_wall=dpos/np.sqrt(np.sum(dpos**2))*(wall_thickness) #the 16 is the wall + scint thickness

    #pinhole vectors
    #lengt vector


    #scint_vector:
    v_scint = p1l-p1
    v_scint/= np.linalg.norm(v_scint)
    ic = np.argmax(np.abs(v_scint))
    dpos=v_scint/np.abs(v_scint[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*(scint_dist + 15.33-0.17) #the 16 is the wall + scint thickness
    pin_p1 = p1 + dpos  
    
    #lenght vector
    v_pl = p2l - p1l
    v_pl/= np.linalg.norm(v_pl)
    ic = np.argmax(np.abs(v_pl))
    dpos=v_pl/np.abs(v_pl[ic])
    dpos_edge=dpos/np.sqrt(np.sum(dpos**2))*(slit_edge_dist)
    pin_p1 = pin_p1 + dpos  
    
    dpos=v_pl/np.abs(v_pl[ic])
    dpos2=dpos/np.sqrt(np.sum(dpos**2))*pin_length
    pin_p2 = pin_p1 + dpos2   
    
    

    v_pw = p2w - p1l
    v_pw/= np.linalg.norm(v_pw)
    ic = np.argmax(np.abs(v_pw))
    dpos=v_pw/np.abs(v_pw[ic])
    dpos3=dpos/np.sqrt(np.sum(dpos**2))*pin_width
    pin_p3 = pin_p1 + dpos3   
    
    pin_p4 = pin_p1 + dpos3 + dpos2
    #8 triangles needed
    
    ####rigth
    pin_p1 += dpos_edge
    pin_p2 += dpos_edge
    pin_p3 += dpos_edge
    pin_p4 += dpos_edge


    p1 = p1 + dpos_wall   
    p3 = p3 + dpos_wall
 
    p5 = p5 + dpos_wall   
    p6 = p6 + dpos_wall   

    #make stl file
    vertices = np.array([\
    p1,
    p2,
    p3,
    p4,
    pin_p1,
    pin_p2,
    pin_p3,
    pin_p4,
    p5,
    p6])
    
    faces = np.array([\
    [0, 1, 5],
    [1, 3, 7],
    [2, 3, 6],
    [0, 2, 4],
    [0, 4, 5],
    [1, 5, 7],
    [3, 7, 6],
    [2, 4, 6],
    [0, 2, 8],
    [8, 2, 9]])
        
    surface = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            surface.vectors[i][j] = vertices[f[j],:]

    surface.save(file_name_save+'.stl')
    
    return np.array([pin_p1, pin_p2, pin_p3, pin_p4])
    


def write_stl_geometry_files(root_dir
                              , run_name = ''
                              , collimator_stl_files = {}
                              , scintillator_stl_files = {}
                              , pinhole = []
                              ):
    '''
    Parameters
    ----------


    Returns
    -------
    None.
    '''
    scan_folder = run_name + '/'
    directory = os.path.join(root_dir, scan_folder)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Making directory')

    element_nr = 1
    
    
    # Gather collimator triangles
    collimators = collimator_stl_files.keys()
    for coll in collimators:
        collimator_filename = directory + 'Element%i.txt'%element_nr
        element_nr += 1
        f = open(collimator_filename, 'w')
        f.write('Collimator file for SINPA FILDSIM\n')
        f.write('Run name is ' + run_name + '\n')
        f.write('STL collimator %s \n' %coll)
        f.write('0  ! Kind of plate\n')
        f.close()
        ss.LibCAD.write_file_for_fortran_numpymesh(collimator_stl_files[coll],
                                              collimator_filename, 
                                              convert_mm_2_m = True)      

    scint_norm = [1., 0., 0.] 
    ps = np.zeros(3)
    rot = np.identity(3)
    # Dummy scintillator normal vector,reference point and rottion vector
    #, in case we don't include a scintillator in the run
    #
    # Write scintillator to file
    scintillators = scintillator_stl_files.keys()
    for scint in scintillators:
        scint_filename = directory + 'Element%i.txt'%element_nr
        ##write geometory file "header"
        f = open(scint_filename, 'w')
        f.write('Scintillator file for SINPA FILDSIM\n')
        f.write('Scintilator stl file: ' + scint + '\n')
        f.write('File by '+ os.getenv('USER') + ' \n')
        f.write('2  ! Kind of plate\n')
        f.close()
        # Append triangle data from stl file
        ss.LibCAD.write_file_for_fortran_numpymesh(scintillator_stl_files[scint], 
                                         scint_filename, 
                                         convert_mm_2_m = True) 
        
        # --- Open and load the stil file
        mesh_obj = mesh.Mesh.from_file(scintillator_stl_files[scint])
    
        x1x2x3 = mesh_obj.x  
        y1y2y3 = mesh_obj.y  
        z1z2z3 = mesh_obj.z  
    
        itriang = 18 #for version 7 #18 for versio 6 #3, forversion 3
        p1 = np.array((x1x2x3[itriang, 0],
                       y1y2y3[itriang, 0],
                       z1z2z3[itriang, 0]) ) * 0.001 #convert mm to m
        p2 = np.array((x1x2x3[itriang, 1],
                       y1y2y3[itriang, 1],
                       z1z2z3[itriang, 1]) ) * 0.001 #convert mm to m
        p3 = np.array((x1x2x3[itriang, 2],
                       y1y2y3[itriang, 2],
                       z1z2z3[itriang, 2]) ) * 0.001 #convert mm to m
        
        scint_norm = -get_normal_vector(p1, p2, p3)        
        
      
        ps = p1 #Arbitrarily choose the first point as the reference point
        u1_scint = p3 - p1
        u1_scint /= np.linalg.norm(u1_scint) #Only needed to align the scintilattor
        
        rot = ss.sinpa.geometry.calculate_rotation_matrix(scint_norm, u1 = -u1_scint
                                                          ,verbose=False)[0]

   
    
    ## Pinhole properties
    pinhole_points =pinhole['points']
    pinhole_points = pinhole_points * 0.001 #convert Catia points to m
    rPin = np.mean(pinhole_points, axis = 0)
    #TO do: get d1 and d2
    if pinhole['pinholeKind'] == 1:
        d1 = np.sqrt(np.sum((pinhole_points[1,:] - pinhole_points[0,:])**2) )
        u1 = (pinhole_points[1] - pinhole_points[0]) / d1
        
        d2 = np.sqrt(np.sum((pinhole_points[3,:] - pinhole_points[1,:])**2) )
        u2 = (pinhole_points[3] - pinhole_points[1])  / d2    
    else:
        d1 = pinhole['pinholeRadius']* 0.001 #convert Catia points to m
        u1 = (pinhole_points[1] - pinhole_points[0]) 
        u1 /= np.linalg.norm(u1)
        
        d2 = 0 #not needed
        u2 = (pinhole_points[2] - pinhole_points[1])   
        u2 /= np.linalg.norm(u2)
        
        rPin = pinhole['pinholeCentre']* 0.001 #convert Catia points to m
        
        
    u3 = np.cross(u1, u2)
    
    extra_filename = directory + 'ExtraGeometryParams.txt'
    nGeomElements = element_nr
    # make sure to convert all to m
    f = open(extra_filename,'w')
    f.write('&ExtraGeometryParams   ! Namelist with the extra geometric parameters\n')
    f.write('  nGeomElements = ' + (str(nGeomElements)) + '\n')
    f.write('  ! Pinhole\n')
    f.write('  rPin(1) = ' + (str(np.round(rPin[0],6))) 
            + ',        ! Position of the pinhole XYZ\n')
    f.write('  rPin(2) = ' + (str(np.round(rPin[1],6))) + ',\n')
    f.write('  rPin(3) = ' + (str(np.round(rPin[2],6))) + ',\n')
    f.write('  pinholeKind = 1     ! 0 = Circular, 1 = rectangle\n')
    f.write('  d1 = ' + (str(np.round(d1,6))) + '  ! Pinhole radius, or size along u1 (in m)\n')
    f.write('  d2 = ' + (str(np.round(d2,6))) + '   ! Size along u2, not used if we have a circular pinhole\n\n')
    f.write('  ! Unitary vectors:\n')
    f.write('  u1(1) =  %f\n' %(u1[0]))
    f.write('  u1(2) =  %f\n' %(u1[1]))
    f.write('  u1(3) =  %f\n\n' %(u1[2]))
    f.write('  u2(1) =  %f\n' %(u2[0]))
    f.write('  u2(2) =  %f\n' %(u2[1]))
    f.write('  u2(3) =  %f\n\n' %(u2[2]))
    f.write('  u3(1) =  %f   ! Normal to the pinhole plane\n' %(u3[0]))
    f.write('  u3(2) =  %f\n' %(u3[1]))
    f.write('  u3(3) =   %f\n\n' %(u3[2]))
    f.write('  ! Reference system of the Scintillator:\n')
    f.write('  ps(1) =  ' + (str(np.round(ps[0] ,6))) + '\n')
    f.write('  ps(2) =  ' + (str(np.round(ps[1] ,6))) + '\n')
    f.write('  ps(3) =  ' + (str(np.round(ps[2] ,6))) + '\n\n')
    # f.write('  ScintNormal(1) =  ' + (str(np.round(scint_norm[0],4))) + '   ! Normal to the scintillator\n')
    # f.write('  ScintNormal(2) =  ' + (str(np.round(scint_norm[1],4))) + '\n')
    # f.write('  ScintNormal(3) =  ' + (str(np.round(scint_norm[2],4))) + '\n\n')
    f.write('  rotation(1,1) = ' + (str(np.round(rot[0,0],4))) + '\n')
    f.write('  rotation(1,2) = ' + (str(np.round(rot[0,1],4))) + '\n')
    f.write('  rotation(1,3) = ' + (str(np.round(rot[0,2],4))) + '\n')
    f.write('  rotation(2,1) = ' + (str(np.round(rot[1,0],4))) + '\n')
    f.write('  rotation(2,2) = ' + (str(np.round(rot[1,1],4))) + '\n')
    f.write('  rotation(2,3) = ' + (str(np.round(rot[1,2],4))) + '\n')
    f.write('  rotation(3,1) = ' + (str(np.round(rot[2,0],4))) + '\n')
    f.write('  rotation(3,2) = ' + (str(np.round(rot[2,1],4))) + '\n')
    f.write('  rotation(3,3) = ' + (str(np.round(rot[2,2],4))) + '\n\n')
    f.write('/')
    f.close()
    
    
    return

if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # --- Options
    # -----------------------------------------------------------------------------
    plt.close('all')
    Test = False  #if true don't submit run
    
    run_code = False
    
    geom_name = 'W7X_std_v14_orb1' # Only used if running a single iteration

    scan = False # Set to "False" to run a single iteration
    scan_param = ['wt2'] # , 'pw2', 'pl', 'pl2', 'psd', 'sh', 'sh2', 'sl', 'sl2', 'oc', 'oc2', 'st', 'ct', 'ct2']
    
    read_results = not run_code #True
    
    self_shadowing = True
    save_self_shadowing_collimator_strike_points = True
    save_orbits = True
    backtrace = False

    run_slit = [True, True] # Run particles starting at one slit or the other?
    read_slit = [True, True] # Read results from the left slit, the tt slit, or both?
    
    if save_orbits:
        nGyro = 36
        maxT = 0.000006 *  1
    else:
        nGyro = 360
        maxT = 0.00000006 
        
    plot_plate_geometry = True
    plot_3D = True
    
    plot_strike_points = False
    save_strike_points_txt = True
    
    plot_strikemap = False
    save_strikemap_txt = False
    
    plot_orbits = True
    orbit_kind=(3,)
    # 2 colliding with the scintillator
    # 0 colliding with the collimator
    # 9 are just orbits exceeding
    # the scintillator and nothing is wrong with the code
    save_orb_txt = True
    seperated = True
    
    plot_self_shadowing_collimator_strike_points = False
    save_self_shadowing_collimator_strike_points_txt = False
    
    plot_metrics = True
    
    if run_code:
        if not run_slit[0] and not run_slit[1]:
            print('You need to choose at least one slit!')
    
    if read_results:
        if not read_slit[0] and not read_slit[1]:
            print('You need to choose at least one slit!')
    
    #for best case: choose the scan parameter and value you want to look at!
    best_param = 'pw3'
    best_param_value = 0.08
    plot_resolutions = False
    plot_gyro_res= False
    plot_pitch_res= False
    plot_collimating_factor = False
    plot_synthetic_signal= False
    plot_camera_signal=False
                            
    
    marker_params = [{'markersize':6, 'marker':'o','color':'b'},
                     {'markersize':6, 'marker':'o','color':'m'}]
    line_params =  [{'ls':'solid','color':'k'},
                    {'ls':'solid','color':'w'}]
    mar_params = [{'zorder':3,'color':'k'},
                  {'zorder':3,'color':'w'}]
    

    n_markers = int(1e2)

    gyro_arrays = [[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.],
                   [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.]]

    gyro_arrays = [[0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.],
                   [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.]]



    pitch_arrays = [[90.,88., 85., 83., 80., 78., 75., 73., 70., 65., 55., 45., 35., 25.],
                    [90.,93., 95., 98., 100., 103., 105., 108., 110., 115., 125., 135., 145., 155.]]
    

    # gyro_arrays = [[0.5, 1.0, 1.5,  2., 3., 4.],
    #                 [0.5, 1.0, 1.5,  2., 3., 4.]]
    # pitch_arrays = [[85., 75., 65., 55., 45., 35., 25.],
    #                 [95., 105., 115., 125., 135., 145., 155., 165., 175.]]

    gyro_arrays = [[1.0, 1.5],
                    [1.0, 1.5]]
    
    pitch_arrays = [[25, 80, 85],
                    [105, 110, 155]]
    
    #gyrophase_range = np.array([np.deg2rad(250),np.deg2rad(350)])
    gyrophase_range = np.array([np.deg2rad(250),np.deg2rad(350)])
    gyrophase_range = np.array([np.deg2rad(0),np.deg2rad(360)])
    #gyrophase_range = np.array([np.deg2rad(324),np.deg2rad(326)])
    #gyrophase_range = np.array([np.deg2rad(330-180),np.deg2rad(330-180)])
    # Set n1 and r1 for the namelist, 0 and 0 are defaults, setting to 0.02 and 0.4 gives ~300,000 particles for rl = 0.25 
    # and ~400,000 for 0.5
    n1 = 0.0
    r1 = 0.0
    
    Br, Bz, Bt = -0.438929, -0.211763, 2.296651   #STD #left pinhole
    #Br, Bz, Bt =  -0.37123,	-0.381688, 1.940142    #KJM
    #Br, Bz, Bt =  -0.456466	, -0.472407, 2.094869  #FTM
    
    modB = np.sqrt(Br**2 + Bz**2 + Bt**2)
    
    alpha = 0.0  #alligned
    beta = 0.0 #180
    
    use_ascot_B = True
    use_single_B = False
    ascot_bfield_File ='Fields/std_bfield.pickle' #'Ascot_bfield_20180918045_cooker4700.pck'# 
    ascot_boozer_File = 'Fields/std_boozer.pickle'
    dist_file = '/afs/ipp/home/a/ajvv/ScintSuite/Examples/SINPA/Distros/std_fild_fine__FILD_distro.pck'
    dist_file = '/afs/ipp/home/a/ajvv/ScintSuite/Examples/SINPA/Distros/kjm_lim__FILD_distro.pck'
    
    new_b_field = False
    
    ##STL files
    collimator_stl_files = {'collimator': 'STLs/probe_head_v14.stl',
                            }
    scintillator_stl_files = {'scintillator':  'STLs/scintillator_plate_v14.stl'}
    pinholes = [{}, {}]
    pinholes[0]['pinholeKind'] =1
    pinholes[0]['pinholeCentre'] = None
    pinholes[0]['pinholeRadius'] = None
    pinholes[0]['points'] = np.array([[-730.358, 5787.98, 368.998],
                                      [-731.807, 5787.65, 368.79],
                                      [-730.557, 5788.36, 369.788],
                                      [-732.006, 5788.03, 369.581]] )#co- going slit opening

    
    pinholes[1]['pinholeKind'] =1
    pinholes[1]['pinholeCentre'] = None
    pinholes[1]['pinholeRadius'] = None
    pinholes[1]['points'] = np.array([[-785.783, 5800.21, 334.972],
                                      
                                      [-784.334, 5800.54, 335.179],
                                      [-785.983, 5800.59, 335.762],
                                      [-784.534, 5800.92, 335.97] ] )#counter going slit opening
    
    test_cover = False
    #cm
    pinhole_length  = 0.15 #0.2 
    pinhole_width   = 0.09 #0.1
    pinhole_scint_dist = 0.7 #0.6 #0.5
    slit_height     = 0.4 #1.0
    slit_length     = 1.5#6. #5.
    
    slit_edge_dist = 0.1
    wall_thickness = 0.1
    
    
    pinhole_length_right  = 0.15 #0.2 
    pinhole_width_right   = 0.09 #0.1
    pinhole_scint_dist_right = 0.7 #0.6 #0.5
    slit_height_right   = 0.4#0.5 
    slit_length_right     = 1.5#6. #5.
    
    slit_edge_dist_right = 0.1
    wall_thickness_right = 0.   
    
    
    

    head_params = {
                   "pinhole_length": pinhole_length,
                   "pinhole_width" : pinhole_width ,
                   "pinhole_scint_dist": pinhole_scint_dist,
                   "slit_height"   : slit_height,
                   "slit_length"   : slit_length,
                   "slit_edge_dist":slit_edge_dist,
                   "wall_thickness" : wall_thickness,

                   "pinhole_length_right": pinhole_length_right,
                   "pinhole_width_right" : pinhole_width_right ,
                   "pinhole_scint_dist_right": pinhole_scint_dist_right,
                   "slit_height_right"   : slit_height_right,
                   "slit_length_right"   : slit_length_right,
                    "slit_edge_dist_right":slit_edge_dist_right,
                    "wall_thickness_right" : wall_thickness_right
                   }




    if scan:
        scan_Parameters = {}

        scan_Parameters['Pinhole width'] = {'scan_str': 'pw',
                                            'scan_param': 'pinhole_width',
                                            'scan_values':np.arange(0.05, 0.15, 0.01),
                                            'scan': False,
                                            'value': pinhole_width}
        scan_Parameters['Pinhole length'] = {'scan_str': 'pl',
                                            'scan_param': 'pinhole_length',
                                        'scan_values':np.arange(0.05, 0.25, 0.02), 
                                        'scan': False,
                                        'value': pinhole_length}
        scan_Parameters['Slit height'] = {'scan_str': 'sh',
                                        'scan_param': 'slit_height',
                                        'scan_values': [0.4],#np.arange(0.1, 1., 0.1) ,
                                        'scan': False,
                                        'value': slit_height}
        scan_Parameters['Slit length'] = {'scan_str': 'sl',
                                        'scan_param': 'slit_length',
                                        'scan_values': np.arange(0.6, 2., 0.2) ,
                                        'scan': False,
                                        'value': slit_height}
        scan_Parameters['Pinhole scint dist'] = {'scan_str': 'psd',
                                                'scan_param': 'pinhole_scint_dist',
                                                'scan_values': [1.4],#np.arange(1.1, 2., 0.1),#np.arange(0.2, 1., 0.1),
                                                'scan': False ,
                                                'value': pinhole_scint_dist}
        scan_Parameters['Pinhole edge dist'] = {'scan_str': 'ped',
                                               'scan_param': 'slit_edge_dist',
                                               'scan_values': [0.2],#np.arange(0., 1., 0.1),
                                               'scan': False ,
                                               'value': slit_edge_dist}
        scan_Parameters['Wall thickness'] = {'scan_str': 'wt',
                                               'scan_param': 'wall_thickness',
                                               'scan_values': np.arange(0.1, 1.0, 0.1),
                                               'scan': False ,
                                               'value': wall_thickness}

        scan_Parameters['Pinhole width right'] = {'scan_str': 'pw2',
                                            'scan_param': 'pinhole_width_right',
                                            'scan_values':np.arange(0.05, 0.15, 0.01),
                                            'scan': False,
                                            'value': pinhole_width_right}
        scan_Parameters['Pinhole length right'] = {'scan_str': 'pl2',
                                            'scan_param': 'pinhole_length_right',
                                        'scan_values':np.arange(0.05, 0.25, 0.02), 
                                        'scan': False,
                                        'value': pinhole_length_right}
        scan_Parameters['Slit height right'] = {'scan_str': 'sh2',
                                        'scan_param': 'slit_height_right',
                                        'scan_values': [0.4],#np.arange(0.1, 1., 0.1) ,
                                        'scan': False,
                                        'value': slit_height_right}
        scan_Parameters['Slit length right'] = {'scan_str': 'sl2',
                                        'scan_param': 'slit_length_right',
                                        'scan_values': np.arange(0.6, 2., 0.2) ,
                                        'scan': False,
                                        'value': slit_length_right}
        scan_Parameters['Pinhole scint dist right'] = {'scan_str': 'psd2',
                                                'scan_param': 'pinhole_scint_dist_right',
                                                'scan_values': [0.7],#np.arange(0.3, 1., 0.1),
                                                'scan': False ,
                                                'value': pinhole_scint_dist_right}
        scan_Parameters['Pinhole edge dist right'] = {'scan_str': 'ped2',
                                                'scan_param': 'slit_edge_dist_right',
                                                'scan_values': np.arange(0., 0.5, 0.1),
                                                'scan': False ,
                                                'value': slit_edge_dist_right}
        scan_Parameters['Wall thickness right'] = {'scan_str': 'wt2',
                                                'scan_param': 'wall_thickness_right',
                                                'scan_values': [0.2],#np.arange(0.1, 1.0, 0.1),
                                                'scan': False ,
                                                'value': wall_thickness_right}



        for scan_parameter in scan_Parameters.keys():
            if scan_Parameters[scan_parameter]['scan_str'] in scan_param:
                scan_Parameters[scan_parameter]['scan'] = True
# -----------------------------------------------------------------------------
# --- Run SINPA FILDSIM
# -----------------------------------------------------------------------------
                
    if run_code:
        geom_dir = os.path.join(paths.SINPA,'Geometry/')
        nml_options = {
            'config':  {  # parameters
                'runid': '',
                'geomfolder': '',
                'FILDSIMmode': True,
                'nxi': 0,
                'nGyroradius': 0,
                'nMap': n_markers,
                'n1': n1,
                'r1': r1,
                'restrict_mode': False,
                'mapping': True,
                'saveOrbits': save_orbits,
                'saveRatio': 1,
                'saveOrbitLongMode': False,
                'runfolder': '',
                'verbose': True,
                'IpBt': 1,        # Sign of toroidal current vs field (for pitch), need to check
                'flag_efield_on': False,  # Add or not electric field
                'save_collimator_strike_points': False,  # Save collimator points
                'backtrace': backtrace,  # Flag to backtrace the orbits
                'save_self_shadowing_collimator_strike_points': 
                    save_self_shadowing_collimator_strike_points,
                'self_shadowing': self_shadowing
                },
            'inputParams': {
                'nGyro': nGyro,
                'minAngle': 0,
                'dAngle': 0,
                'XI': [],
                'rL': [],
                'maxT': maxT
                },
            }
        
        if not scan:
            geom_names = [geom_name,geom_name + '_right']
            #Create magnetic field
            field = ss.simcom.Fields()
                    
            if use_single_B:    
                field.createFromSingleB(B = np.array([Br,Bz,Bt]), Rmin = 4.5,
                                          Rmax = 6.5,
                                          zmin = -1.0, zmax = 1.0,
                                          nR = 100, nZ = 100)

            elif use_ascot_B:
                f = open(ascot_bfield_File, 'rb')
                ascot_bfield = pickle.load(f)
                f.close()
                #Field geometry saved in "boozer" structure
                f = open(ascot_boozer_File, 'rb')
                ascot_boozer = pickle.load(f)
                f.close()
                
                field.Bfield['R'] = np.asfortranarray(np.linspace(ascot_boozer['rmin'][0], 
                                       ascot_boozer['rmax'][0], 
                                       ascot_boozer['nr'][0]), dtype=np.float64 )
                field.Bfield['z'] = np.asfortranarray(np.linspace(ascot_boozer['zmin'][0], 
                                       ascot_boozer['zmax'][0], 
                                       ascot_boozer['nz'][0]), dtype=np.float64 )
                
                
                field.Bfield['nR'] = np.asfortranarray(len(field.Bfield['R']), dtype=np.int32)
                field.Bfield['nZ'] = np.asfortranarray(len(field.Bfield['z']), dtype=np.int32)
                field.Bfield['Rmin'] = np.asfortranarray(ascot_boozer['rmin'][0], dtype=np.float64)
                field.Bfield['Rmax'] = np.asfortranarray(ascot_boozer['rmax'][0], dtype=np.float64)
                field.Bfield['Zmin'] = np.asfortranarray(ascot_boozer['zmin'][0], dtype=np.float64)
                field.Bfield['Zmax'] = np.asfortranarray(ascot_boozer['zmax'][0], dtype=np.float64)            
                #Ascot stellarator fields only store data for a single period
                #bfield [idx_R, idx_phi, idx_Z], thus rrepeat along axis = 1

                nfp = int(ascot_bfield['toroidalPeriods'])
                #br = np.concatenate([ascot_bfield['br'], (ascot_bfield['br'][:,0,:])[:,None,:]], axis=1)
                br = np.tile(ascot_bfield['br'],[1,nfp,1])
                bphi = np.tile(ascot_bfield['bphi'],[1,nfp,1])
                bz = np.tile(ascot_bfield['bz'],[1,nfp,1])
                
                field.Bfield['fr'] = np.asfortranarray(br, dtype=np.float64)
                field.Bfield['fz'] = np.asfortranarray(bz, dtype=np.float64)
                field.Bfield['ft'] = np.asfortranarray(bphi, dtype=np.float64)
                
                field.Bfield['nPhi'] = np.asfortranarray(np.shape(br)[1], dtype=np.int32 )
                field.Bfield['Phimin'] = np.asfortranarray(0., dtype=np.float64)

                field.Bfield['Phimax'] = np.asfortranarray(2.*np.pi*(1-1/(np.shape(br)[1])), dtype=np.float64)#Alex
                field.bdims = 3
                
                #To do FILD is at +-97 deg, plot b-field at correct phi position
                field.plot('br', phiSlice = 0 ,plot_vessel = False)
                plt.show()
                
            #Write geometry files
            for i in range(2):
                if run_slit[i]:
                    
                    if test_cover:
                        pinhole_points = write_slit_cover_plate(  pinhole_width_right
                                                                , pinhole_length_right
                                                                , pinhole_scint_dist_right
                                                                , slit_edge_dist_right
                                                                , wall_thickness_right
                                                                , file_name_save= 'pinhole_cover')
                        pinholes[i]['pinholeKind'] =1
                        pinholes[i]['pinholeCentre'] = None
                        pinholes[i]['pinholeRadius'] = None
                        pinholes[i]['points'] = np.array(pinhole_points)
                        
                        collimator_stl_files['pinhole_cover'] = 'pinhole_cover.stl'
                        
                        write_collimator_plates(file_name_save= 'lateral_collimator_plates'
                                                , pl1 = slit_length
                                                , pl2 = slit_length_right
                                                , ph1=slit_height
                                                , ph2=slit_height_right
                                                , pin_0 = pinholes[0]['points'][0]
                                                , pin_2 = pinholes[0]['points'][2]
                                                , pin_0_right = pinholes[1]['points'][0]
                                                , pin_2_right = pinholes[1]['points'][2])
                        collimator_stl_files['collimator_lateral'] = 'lateral_collimator_plates.stl'
                    
                    
                    write_stl_geometry_files(root_dir = geom_dir,
                                          run_name = geom_names[i],
                                          collimator_stl_files = collimator_stl_files,
                                          scintillator_stl_files = scintillator_stl_files,
                                          pinhole = pinholes[i])            
                
                    if not Test:
                        # Create directories
                        runDir = os.path.join(paths.SINPA, 'runs', geom_names[i])
                        inputsDir = os.path.join(runDir, 'inputs')
                        resultsDir = os.path.join(runDir, 'results')
                        os.makedirs(runDir, exist_ok=True)
                        os.makedirs(inputsDir, exist_ok=True)
                        os.makedirs(resultsDir, exist_ok=True)
                        
                        # Set namelist parameters
                        nml_options['config']['runid'] = geom_names[i]
                        nml_options['config']['geomfolder'] = geom_dir + '/' + geom_names[i]
                        nml_options['config']['runfolder'] = runDir
                        nml_options['config']['nxi'] = len(pitch_arrays[i])
                        nml_options['config']['nGyroradius'] = len(gyro_arrays[i])
                        nml_options['inputParams']['XI'] = pitch_arrays[i]
                        nml_options['inputParams']['rL'] = gyro_arrays[i]
                        nml_options['inputParams']['minAngle'] = gyrophase_range[0]
                        nml_options['inputParams']['dAngle'] = (gyrophase_range[1]
                                                                  - gyrophase_range[0])
                                                
                        #Make field
                        if new_b_field:
                            fieldFileName = os.path.join(inputsDir, 'field.bin')
                            fid = open(fieldFileName, 'wb')
                            field.tofile(fid)
                            fid.close()

    
                        # Create namelist
                        ss.sinpa.execution.write_namelist(nml_options)
                        
                        # Missing a step: create B field!!
                        # Check the files
                        ss.sinpa.execution.check_files(nml_options['config']['runid'])
                        # Launch the simulations
                        ss.sinpa.execution.executeRun(nml_options['config']['runid'], queue = True)
                
            if plot_plate_geometry:
                for i in range(2):
                    if run_slit[i]:
                        geomID = geom_names[i]
                        Geometry = ss.simcom.Geometry(GeomID=geomID)
                        if plot_3D:
                            Geometry.plot3Dfilled(element_to_plot = [0,2])
                            
                        # Geometry.plot2Dfilled(view = 'XY', element_to_plot = [0,2], 
                        #                       plot_pinhole = False, surface_params = {'alpha': 1.0})
                        Geometry.plot2Dfilled(view = 'Scint', 
                                              referenceSystem = 'scintillator',
                                              element_to_plot = [2], plot_pinhole = True)
    
        else:
            for scan_parameter in scan_Parameters.keys():
                if scan_Parameters[scan_parameter]['scan']:
                    for value in scan_Parameters[scan_parameter]['scan_values']:
                        '''
                        Loop over scan paramater
                        '''
                        name = scan_Parameters[scan_parameter]['scan_param']
                        temp = head_params[name]
                        head_params[name] = value


                        string_mod = '_%s_%s' %(scan_Parameters[scan_parameter]['scan_str'], 
                                                f'{float(value):g}')
                        run_names = ['W7X_scan' + string_mod, 'W7X_scan' + string_mod + '_right']
                        
                        
                        field = ss.simcom.Fields()
                                
                        if use_single_B:    
                            field.createFromSingleB(B = np.array([Br,Bz,Bt]), Rmin = 4.5,
                                          Rmax = 6.5,
                                          zmin = -1.0, zmax = 1.0,
                                          nR = 100, nZ = 100)
                        
                        elif use_ascot_B:
                            f = open(ascot_bfield_File, 'rb')
                            ascot_bfield = pickle.load(f)
                            f.close()
                            #Field geometry saved in "boozer" structure
                            f = open(ascot_boozer_File, 'rb')
                            ascot_boozer = pickle.load(f)
                            f.close()
                            
                            field.Bfield['R'] = np.asfortranarray(np.linspace(ascot_boozer['rmin'][0], 
                                                   ascot_boozer['rmax'][0], 
                                                   ascot_boozer['nr'][0]), dtype=np.float64 )
                            field.Bfield['z'] = np.asfortranarray(np.linspace(ascot_boozer['zmin'][0], 
                                                   ascot_boozer['zmax'][0], 
                                                   ascot_boozer['nz'][0]), dtype=np.float64 )
                            
                            
                            field.Bfield['nR'] = np.asfortranarray(len(field.Bfield['R']), dtype=np.int32)
                            field.Bfield['nZ'] = np.asfortranarray(len(field.Bfield['z']), dtype=np.int32)
                            field.Bfield['Rmin'] = np.asfortranarray(ascot_boozer['rmin'][0], dtype=np.float64)
                            field.Bfield['Rmax'] = np.asfortranarray(ascot_boozer['rmax'][0], dtype=np.float64)
                            field.Bfield['Zmin'] = np.asfortranarray(ascot_boozer['zmin'][0], dtype=np.float64)
                            field.Bfield['Zmax'] = np.asfortranarray(ascot_boozer['zmax'][0], dtype=np.float64)            
                            #Ascot stellarator fields only store data for a single period
                            #bfield [idx_R, idx_phi, idx_Z], thus rrepeat along axis = 1
            
                            nfp = int(ascot_bfield['toroidalPeriods'])
                            #br = np.concatenate([ascot_bfield['br'], (ascot_bfield['br'][:,0,:])[:,None,:]], axis=1)
                            br = np.tile(ascot_bfield['br'],[1,nfp,1])
                            bphi = np.tile(ascot_bfield['bphi'],[1,nfp,1])
                            bz = np.tile(ascot_bfield['bz'],[1,nfp,1])
                            
                            field.Bfield['fr'] = np.asfortranarray(br, dtype=np.float64)
                            field.Bfield['fz'] = np.asfortranarray(bz, dtype=np.float64)
                            field.Bfield['ft'] = np.asfortranarray(bphi, dtype=np.float64)
                            
                            field.Bfield['nPhi'] = np.asfortranarray(np.shape(br)[1], dtype=np.int32 )
                            field.Bfield['Phimin'] = np.asfortranarray(0., dtype=np.float64)
            
                            field.Bfield['Phimax'] = np.asfortranarray(2.*np.pi*(1-1/(np.shape(br)[1])), dtype=np.float64)#Alex
                            field.bdims = 3
                            
                            #To do FILD is at +-97 deg, plot b-field at correct phi position
                            field.plot('br', phiSlice = 0 ,plot_vessel = False)
                            plt.show()
                        
 
                        #Write geometry files
                        for i in range(2):
                            if run_slit[i]:
                                if test_cover:
                                    
                                    pinhole_points = write_slit_cover_plate(  head_params['pinhole_width_right']
                                                                            , head_params['pinhole_length_right']
                                                                            , head_params['pinhole_scint_dist_right']
                                                                            , head_params['slit_edge_dist_right']
                                                                            , head_params['wall_thickness_right']
                                                                            , file_name_save= 'pinhole_cover_right')

                                    # pinhole_points = write_slit_cover_plate(  head_params['pinhole_width']
                                    #                                         , head_params['pinhole_length']
                                    #                                         , head_params['pinhole_scint_dist']
                                    #                                         , head_params['slit_edge_dist']
                                    #                                         , head_params['wall_thickness']
                                    #                                         , file_name_save= 'pinhole_cover')


                                    pinholes[i]['pinholeKind'] =1
                                    pinholes[i]['pinholeCentre'] = None
                                    pinholes[i]['pinholeRadius'] = None
                                    pinholes[i]['points'] = np.array(pinhole_points)
                                    
                                    collimator_stl_files['pinhole_cover'] = 'pinhole_cover_right.stl'
                                    
                                write_collimator_plates(file_name_save= 'lateral_collimator_plates'
                                                        , pl1 = head_params['slit_length']
                                                        , pl2 = head_params['slit_length_right']
                                                        , ph1= head_params['slit_height']
                                                        , ph2= head_params['slit_height_right']
                                                        , pin_0 = pinholes[0]['points'][0]
                                                        , pin_2 = pinholes[0]['points'][2]
                                                        , pin_0_right = pinholes[1]['points'][0]
                                                        , pin_2_right = pinholes[1]['points'][2])



                                collimator_stl_files['collimator_lateral'] = 'lateral_collimator_plates.stl'
                                write_stl_geometry_files(root_dir = geom_dir,
                                                      run_name = run_names[i],
                                                      collimator_stl_files = collimator_stl_files,
                                                      scintillator_stl_files = scintillator_stl_files,
                                                      pinhole = pinholes[i])                                    
                                    
                                if not Test:
                                    # Create directories
                                    runDir = os.path.join(paths.SINPA, 'runs', run_names[i])
                                    inputsDir = os.path.join(runDir, 'inputs')
                                    resultsDir = os.path.join(runDir, 'results')
                                    os.makedirs(runDir, exist_ok=True)
                                    os.makedirs(inputsDir, exist_ok=True)
                                    os.makedirs(resultsDir, exist_ok=True)
                                    
                                    # Set namelist parameters
                                    nml_options['config']['runid'] = run_names[i]
                                    nml_options['config']['geomfolder'] = (geom_dir + '/' + run_names[i])
                                    nml_options['config']['runfolder'] = runDir
                                    nml_options['config']['nxi'] = len(pitch_arrays[i])
                                    nml_options['config']['nGyroradius'] = len(gyro_arrays[i])
                                    nml_options['inputParams']['XI'] = pitch_arrays[i]
                                    nml_options['inputParams']['rL'] = gyro_arrays[i]
                                    nml_options['inputParams']['minAngle'] = gyrophase_range[0]
                                    nml_options['inputParams']['dAngle'] = (gyrophase_range[1]
                                                                      - gyrophase_range[0])
                         
                                    #Make field
                                    if new_b_field:
                                        fieldFileName = os.path.join(inputsDir, 'field.bin')
                                        fid = open(fieldFileName, 'wb')
                                        field.tofile(fid)
                                        fid.close()

                                    # Create namelist
                                    ss.sinpa.execution.write_namelist(nml_options)
                                
                                    # Missing a step: create B field!!
                                    # Check the files
                                    ss.sinpa.execution.check_files(nml_options['config']['runid'])
                                    # Launch the simulations
                                    #ss.sinpa.execution.executeRun(nml_options['config']['runid'], queue = False)
                                    ss.sinpa.execution.executeRun(nml_options['config']['runid'], queue = True)
                                
                        
                            if plot_plate_geometry:
                                if run_slit[i]:
                                    geomID = run_names[i]
                                    Geometry = ss.simcom.Geometry(GeomID=geomID)
                                    if plot_3D:
                                        Geometry.plot3Dfilled(element_to_plot = [0,2], plot_pinhole = False)
                                    #Geometry.plot2Dfilled(view = 'XY', element_to_plot = [0,2], plot_pinhole = False)
                                    Geometry.plot2Dfilled(view = 'Scint', element_to_plot = [0,2], plot_pinhole = False)
    
                        
# -----------------------------------------------------------------------------
# --- Section 2: Analyse the results
# -----------------------------------------------------------------------------
    if read_results:
        if not scan:
            runid = [geom_name,geom_name + '_right']
            strike_points_file = ['','']
            Smap = [[],[]]
            p0 = [75, 115]
            
            for i in range(2):
                if read_slit[i] :# or mixnmatch:
                    runDir = os.path.join(paths.SINPA, 'runs', runid[i])
                    inputsDir = os.path.join(runDir, 'inputs/')
                    resultsDir = os.path.join(runDir, 'results/')
                    base_name = resultsDir + runid[i]
                    smap_name = base_name + '.map'
            
                    # Load the strike map
                    Smap[i] = ss.mapping.StrikeMap('FILD', file=smap_name)
                    try:
                        Smap[i].load_strike_points()
                    except:
                        print('Strike map ' + str(i+1) + ' could not be loaded')
                        continue
                    
                    try:
                        selfmap_name = base_name + '.spcself'
                        Smap[i].strike_points_self = ss.simcom.strikes.Strikes(code = 'SINPA', file=selfmap_name, type = 'mapcollimator')#(runid[i],type='mapcollimator')
                    except:
                        print('Self shadowing points' + str(i+1) + ' could not be loaded')
                        continue

                    if plot_synthetic_signal:
                        # I haven't tested this yet
                        
                        distro = pickle.load( open( dist_file, "rb" ) )
    
    
                        per = 0.05
                        flags = np.random.rand(distro['n']) < per
                        
                        distro['BR'] = distro['BR'][flags]
                        distro['n'] = len(distro['BR'])
                        distro['Bphi'] = distro['Bphi'][flags]
                        distro['Bz'] = distro['Bz'][flags]
                        distro['Anum'] =distro['Anum'][flags]
                        distro['Znum'] = distro['Znum'][flags]
                        distro['vR']=distro['vR'][flags]
                        distro['vphi'] =distro['vphi'][flags]
                        distro['vz'] =distro['vz'][flags]
                        distro['pitch'] = distro['pitch'][flags]
                        distro['energy'] =distro['energy'][flags]
        
                        distro['gyroradius'] =distro['gyroradius'][flags]
                        distro['weight'] =    distro['weight'] [flags]
                        
                        Gyro_radius=distro['gyroradius']
                        pitch = distro['pitch']
                        weight = distro['weight']
                        gridlev = 100
                        fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                        facecolor='w', edgecolor='k', dpi=100)
            
                        ax4.set_ylabel('Gyro radius [cm]')
                        ax4.set_xlabel('Pitch angle [$\\degree$]')
                        
                        rl = np.linspace(0, 2., gridlev)
                        p = np.linspace(-1, 1, gridlev)
                        p = np.linspace(55,105, gridlev)
                        #IPython.embed()
                        f, rl, p = np.histogram2d(Gyro_radius, pitch, bins=[rl, p],
                                                           weights=weight)
                        cmap = cm.inferno # Colormap
                        cmap._init()                    
                        im4 = ax4.imshow(f, origin='lower',
                                        extent=(p[0],
                                                p[-1],
                                                rl[0],
                                                rl[-1]),
                                        cmap = cmap,
                                        aspect='auto', interpolation='nearest')#,
                    
                        
                        fig4.colorbar(im4, ax=ax4, label='fast-ions/s/keV')
                        fig4.tight_layout()
                        fig4.show()
    
                        output = ss.fildsim.synthetic_signal_remap(distro, Smap[i],
                                                                   rmin=0.1, rmax=4.0, dr=0.01,
                                                                   pmin=55.0, pmax=105.0,
                                                                   dp=0.01)
    
    
    
    
                        fig, ax_syn = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                                   facecolor='w', edgecolor='k', dpi=100)   
                        ax_syn.set_xlabel('Pitch [$\\degree$]')
                        ax_syn.set_ylabel('Gyroradius [cm]')
                        ax_syn.set_ylim([0, 2.0])
                        
                        cmap = cm.inferno # Colormap
                        cmap._init()
                        
                        ss.fildsim.plot_synthetic_signal(output['gyroradius'], output['pitch']
                                                   , output['signal'], 
                                                   ax=ax_syn, fig=fig, cmap = cmap)
                        if read_slit[0] and read_slit[1]:
                            fig.title('Slit #' + str(i+1))
                        fig.tight_layout()
                        fig.show()
                            
                    if False:
                        dist_file = '/afs/ipp/home/a/ajvv/ScintSuite/Examples/SINPA/Distros/std_fild_fine__FILD_distro.pck'

                        distro = pickle.load( open( dist_file, "rb" ) )
                        
                        
                        prune = 100
                        flags = np.random.choice(np.arange(len(distro['BR'])),size=(prune,),replace=False)
                        
                        #flags = (distro['gyroradius']>0.) #+ (distro['gyroradius']<1.0)
                        distro['BR'] = distro['BR'][flags]
                        distro['n'] = len(distro['BR'])
                        distro['Bphi'] = distro['Bphi'][flags]
                        distro['Bz'] = distro['Bz'][flags]
                        distro['Anum'] =distro['Anum'][flags]
                        distro['Znum'] = distro['Znum'][flags]
                        distro['vR']=distro['vR'][flags]
                        distro['vphi'] =distro['vphi'][flags]
                        distro['vz'] =distro['vz'][flags]
                        distro['pitch'] = distro['pitch'][flags]
                        distro['energy'] =distro['energy'][flags]
                        
                        distro['gyroradius'] =distro['gyroradius'][flags]
                        distro['weight'] =    distro['weight'] [flags]

                        efficiency = ss.LibScintillatorCharacterization.ScintillatorEfficiency()

                        scintillator = ss.LibMap.Scintillator('/afs/ipp/home/a/ajvv/SINPA/Geometry/W7X_std_v14/Element2.txt', format = 'SINPA')
                        
                        optics_parameters = {}
                        optics_parameters['beta'] = 1/37
                        optics_parameters['T'] = 0.67
                        optics_parameters['Omega'] = 4.7e-4
                        
                        camera_parameters = {}
                        camera_parameters['px_x_size'] = 9e-6
                        camera_parameters['nx'] = 229
                        camera_parameters['px_y_size'] = 9e-6
                        camera_parameters['ny'] = 240
                        camera_parameters['range'] = 12
                        camera_parameters['qe'] = 0.7
                        camera_parameters['ad_gain'] = 0.25
                        camera_parameters['dark_noise'] = 0.96
                        camera_parameters['readout_noise'] = 2.33
                        
                        exp_time = 1.
                        
                        scint_signal_options = {}
                        scint_signal_options['rmin']=0.0
                        scint_signal_options['rmax']=2.0
                        scint_signal_options['dr']=0.1
                        scint_signal_options['pmin']=55.0
                        scint_signal_options['pmax']=125.0
                        scint_signal_options['dp']=0.1
                        
                        
                        output_cam = ss.fildsim.synthetic_signal(distro, efficiency, optics_parameters,
                                             Smap[i], scintillator, camera_parameters,
                                             exp_time = exp_time, px_shift = 0, py_shift = 0,
                                             plot=True)

                    if plot_camera_signal:
                        """
                        Calculate FILD synthetic signal
                        """
 
                        # ------------------------------------------------------------------------------
                        # %% Load distribution
                        # ------------------------------------------------------------------------------
                        distro = pickle.load(open(dist_file, "rb" ))
                        prune = 100
                        flags = np.random.choice(np.arange(len(distro['BR'])),
                                                 size=(prune,), replace=False)
                        distro['BR'] = distro['BR'][flags]
                        distro['n'] = len(distro['BR'])
                        distro['Bphi'] = distro['Bphi'][flags]
                        distro['Bz'] = distro['Bz'][flags]
                        distro['Anum'] = distro['Anum'][flags]
                        distro['Znum'] = distro['Znum'][flags]
                        distro['vR'] = distro['vR'][flags]
                        distro['vphi'] = distro['vphi'][flags]
                        distro['vz'] = distro['vz'][flags]
                        distro['pitch'] = distro['pitch'][flags]
                        distro['energy'] = distro['energy'][flags]
                        distro['gyroradius'] = distro['gyroradius'][flags]
                        distro['weight'] = distro['weight'][flags]
                       
                        # ------------------------------------------------------------------------------
                        # %% Settings
                        # ------------------------------------------------------------------------------
                        # ---- Geometry files
                        geomFolder = \
                           '/afs/ipp-garching.mpg.de/home/a/ajvv/pub/rueda/SINPA_test/Geom/W7X_std_v14'
                        # ---- SINPA run


                        # ---- Camera options
                        optics_parameters = {}
                        optics_parameters['beta'] = 1/37
                        optics_parameters['T'] = 0.67
                        optics_parameters['Omega'] = 4.7e-4
                        
                        camera_parameters = {}
                        camera_parameters['px_x_size'] = 9e-6
                        camera_parameters['nx'] = 229
                        camera_parameters['px_y_size'] = 9e-6
                        camera_parameters['ny'] = 240
                        camera_parameters['range'] = 12
                        camera_parameters['qe'] = 0.7
                        camera_parameters['ad_gain'] = 0.25
                        camera_parameters['dark_noise'] = 0.96
                        camera_parameters['readout_noise'] = 2.33
                        
                        exp_time = 1.
                        
                        scint_signal_options = {}
                        scint_signal_options['rmin'] = 0.0
                        scint_signal_options['rmax'] = 2.0
                        scint_signal_options['dr'] = 0.1
                        scint_signal_options['pmin'] = 55.0
                        scint_signal_options['pmax'] = 125.0
                        scint_signal_options['dp'] = 0.1
                        
                        # ---- plotting options
                        p1 = True  # Plot the signal in the remap variables
                        # ------------------------------------------------------------------------------
                        # %% Load geometry ans map
                        # ------------------------------------------------------------------------------
                        Geom = ss.simcom.Geometry(GeomID=runid[i])
                        #scintillator = ss.scint.Scintillator(os.path.join(geomFolder, 'Element2.txt'),
                        #                                     format='SINPA')
                        scintillator = ss.LibMap.Scintillator('/afs/ipp/home/a/ajvv/SINPA/Geometry/W7X_std_v14/Element2.txt', format = 'SINPA')
                        Geom.apply_movement()
                        #smap = ss.smap.Fsmap(smapFile)
                        #smap.load_strike_points()
                        smap=Smap[i]
                        # apply the calibration
                        cal = ss.mapping.CalParams()
                        xscale = optics_parameters['beta'] / camera_parameters['px_x_size']
                        yscale = optics_parameters['beta'] / camera_parameters['px_y_size']
                        # calculate the pixel position of the scintillator vertices
                        cal.xscale = xscale
                        cal.yscale = yscale
                        cal.xshift = 0
                        cal.yshift = 0
                        
                        cal.deg = 75
                        
                        smap.calculate_pixel_coordinates(cal)
                        scintillator.calculate_pixel_coordinates(cal)
                        # Aling the center of the map with the camera center
                        
                        if i == 0:
                            x_shift_smap =  camera_parameters['nx']/2.0 - smap.xpixel.mean()
                            y_shift_smap = camera_parameters['ny']/2.0 - smap.ypixel.mean()
                            
                            scintillator_x_shift = camera_parameters['nx']/2.0 - scintillator.xpixel.mean()
                            scintillator_y_shift = camera_parameters['ny']/2.0 - scintillator.ypixel.mean()
                        
                        smap.xpixel += x_shift_smap+70
                        smap.ypixel += y_shift_smap+50
                        
                        scintillator.xpixel += scintillator_x_shift
                        scintillator.ypixel += scintillator_y_shift

                        # ------------------------------------------------------------------------------
                        # %% Calculate signal in remap coordinates
                        # ------------------------------------------------------------------------------
                        signalRemapCoordinates = \
                            ss.fildsim.synthetic_signal_remap(
                                distro, smap, efficiency=None,
                                **scint_signal_options)
                        if False:
                            fig, ax = plt.subplots()
                            ax.contourf(signalRemapCoordinates['pitch'],
                                        signalRemapCoordinates['gyroradius'],
                                        signalRemapCoordinates['signal'].T,
                                        cmap=ss.plt.Gamma_II())
                            ax.set_xlabel('Pitch [º]')
                            ax.set_ylabel('Gyroradius [cm]')
                        
                        # %% Calculate transformation matrix
                        # ------------------------------------------------------------------------------
                        smap.interp_grid((camera_parameters['nx'],
                                          camera_parameters['ny']), MC_number=250,
                                         grid_params={
                                             'ymin': scint_signal_options['rmin'],
                                             'ymax': scint_signal_options['rmax'],
                                             'xmin': scint_signal_options['pmin'],
                                             'xmax': scint_signal_options['pmax'],
                                             'dx': scint_signal_options['dp'],
                                             'dy': scint_signal_options['dr'],
                                             })
                        H = np.tensordot(signalRemapCoordinates['signal'],
                                         smap.grid_interp['transformation_matrix'],
                                         2)
                        # %% Plot the frame
                        if True:
                            if i ==0:
                                fig2, ax2 = plt.subplots(figsize=(10, 8),
                                facecolor='w', edgecolor='k', dpi=100)
                            #ax2.imshow(H, cmap=ss.plt.Gamma_II())

                                fig3, ax3 = plt.subplots(figsize=(8, 6),
                                facecolor='w', edgecolor='k', dpi=100)
                            else:
                                fig4, ax4 = plt.subplots(figsize=(8, 6),
                                facecolor='w', edgecolor='k', dpi=100)
                            if i ==1:
                                lowerBound = 2
                                H =np.ma.masked_where((lowerBound > H), H)
                             
                            cal = 2000 * 1 / 4 / np.pi * optics_parameters['T'] * optics_parameters['Omega'] * camera_parameters['qe'] / camera_parameters['ad_gain'] * exp_time

                            im2 = ax2.imshow(H * cal, cmap=cm.inferno)
                            smap.plot_pix(line_params={'color': 'r', 'alpha':0.45}, 
                                          marker_params = {   'markersize': 0},
                                          ax=ax2, labels=True)
                            ax2.set_xlabel('X [pixel]')
                            ax2.set_ylabel('Y [pixel]')
                            if i ==0:
                                fig2.colorbar(im2, ax=ax2, label='Counts')
                                fig2.tight_layout()

                            if i ==0:
                                im2 = ax3.imshow(H * cal, cmap=cm.inferno)
                                smap.plot_pix(line_params={'color': 'r', 'alpha':0.45}, 
                                              marker_params = {   'markersize': 0},
                                              ax=ax3, labels=True)
                                ax3.set_xlabel('X [pixel]')
                                ax3.set_ylabel('Y [pixel]')
                                ax3.set_xlim([135, 205.])
                                ax3.set_ylim([200, 130.])
                                fig3.colorbar(im2, ax=ax3, label='Counts')
                                fig3.tight_layout()

                            if i ==1:
                                #lowerBound = np.min(H)
                                #H =np.ma.masked_where((lowerBound >= H), H)
                                im2 = ax4.imshow(H.data * cal, cmap=cm.inferno)
                                smap.plot_pix(line_params={'color': 'r', 'alpha':0.45}, 
                                              marker_params = {   'markersize': 0},
                                              ax=ax4, labels=True)
                                ax4.set_xlabel('X [pixel]')
                                ax4.set_ylabel('Y [pixel]')
                                ax4.set_xlim([85, 155.])
                                ax4.set_ylim([70, 0.])                                
                                fig4.colorbar(im2, ax=ax4, label='Counts')
                                fig4.tight_layout()


                    if plot_resolutions:
                        Smap[i].calculate_resolutions(min_statistics = 1000)
                        Smap[i].plot_resolutions()
                        
                    if plot_gyro_res:
                        Smap[i].plot_resolution_fits(var='Gyroradius',
                                                      pitch=np.array([75]),
                                                      kind_of_plot='normal',
                                                      include_legend=True)
                        plt.gcf().show()
                    if plot_pitch_res:
                        Smap[i].plot_resolution_fits(var='Pitch',
                                                      gyroradius=1,
                                                      kind_of_plot='normal',
                                                      include_legend=True)
                        plt.gcf().show()
                    if plot_collimating_factor:
                        Smap[i].plot_collimator_factor()
                        plt.gcf().show()
    
    



                
            if plot_plate_geometry:
                fig, ax = plt.subplots()
                orb = []
                    
                ax.set_xlabel('Y [cm]')
                ax.set_ylabel('Z [cm]')
                ax.set_title('Camera view (YZ plane)')
                
                # If you want to mix and match, it assumes they have the same scintillator
                if read_slit[0]:
                    Geometry = ss.simcom.Geometry(GeomID=runid[0])
                else:
                    Geometry = ss.simcom.Geometry(GeomID=runid[1])
                    
                Geometry.plot2Dfilled(ax=ax, view = 'Scint', element_to_plot = [2],
                                      plot_pinhole = False)
                    
                for i in range(2):
                    if read_slit[i]:# or mixnmatch:
                        if plot_strike_points:
                            #IPython.embed()
                            Smap[i].strike_points.scatter(ax=ax, per=0.2
                                                          , xscale=100.0, yscale=100.0
                                                          ,mar_params = mar_params[i]
                                                          , varx='ys', vary='zs')
                            
                        if plot_strikemap:
                            Smap[i].plot_real(ax=ax, marker_params=marker_params[i],
                                              line_params=line_params[i], factor=100., labels=True)
                                
                        if plot_orbits:
                            orb.append(ss.sinpa.orbits(runID=runid[i]))
                            #orb[i].plot2D(ax=ax,line_params={'color': 'r'}, kind=(2,),factor=100.0)
                            #orb[i].plot2D(ax=ax,line_params={'color': 'b'}, kind=(0,),factor=100.0)
                            #orb[i].plot2D(ax=ax,line_params={'color': 'k'}, kind=(9,),factor=100.0)
                    else:
                        orb.append([])
                          
                fig.show()
                    
                if plot_3D:
                    Geometry.plot3Dfilled(element_to_plot = [0, 2], units = 'm')
                    ax3D = plt.gca()
                    
                  
                    for i in range(2):
                        if read_slit[i]:# or mixnmatch:
                            
                            if plot_strike_points:
                                Smap[i].strike_points.plot3D(ax=ax3D)
                                if save_strike_points_txt:
                                    Smap[i].strike_points.points_to_txt(file_name_save = 'Strikes_%s.txt' % (runid[i])
                                                                        , per = 0.01)
                                if plot_self_shadowing_collimator_strike_points:
                                    Smap[i].strike_points_self.plot3D(ax=ax3D)  
                                if save_self_shadowing_collimator_strike_points_txt:
                                    Smap[i].strike_points_self.points_to_txt(file_name_save = 'Strikes_shadow_%s.txt' % (runid[i])
                                                                             , per = 1)
                                    
                            # plot in red the ones colliding with the scintillator
                            if plot_orbits:
                                orb[i].plot3D(line_params={'color': 'r'}, kind=(2,), ax=ax3D, per = 1.)
                                # plot in blue the ones colliding with the collimator
                                orb[i].plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax3D, per = 1.)
                                # plot the 'wrong orbits', to be sure that the are just orbits exceeding
                                # the scintillator and nothing is wrong with the code
                                orb[i].plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax3D, per = 1.)
                                
                                if save_orb_txt:
                                    orb[i].save_orbits_to_txt( kind=orbit_kind, units = 'mm', seperated = seperated)






        elif scan:
            orb = [[],[]]
            for scan_parameter in scan_Parameters.keys():
                if scan_Parameters[scan_parameter]['scan']:
                    min_gyro, max_gyro = [[],[]], [[],[]]
                    min_gyro_p0, max_gyro_p0 = [[],[]], [[],[]]
                    gyro_1_res, gyro_2_res, gyro_3_res = [[],[]], [[],[]], [[],[]]
                    min_pitch, max_pitch = [[],[]], [[],[]]
                    min_gyro_res, max_gyro_res = [[],[]], [[],[]]
                    min_pitch_res, max_pitch_res, pitch_p0_res = [[],[]], [[],[]], [[],[]]
                    # p0 is 75 for left slit, 125 for right slit
                    p0 = [75,120]
                
                    avg_collimating_factor, pitch_p0_gyro_1_collimating_factor = [[],[]], [[],[]]
                
                    for value in scan_Parameters[scan_parameter]['scan_values']:
                        ## Loop over scan variables
                        Smap = [[],[]]
                        scan_Parameters[scan_parameter]['value'] = value
                        
                        string_mod = string_mod = '_%s_%s' %(scan_Parameters[scan_parameter]['scan_str'], 
                                                f'{float(value):g}')
                    
                        # Load the result of the simulation
                        runid = ['W7X_scan' + string_mod, 'W7X_scan' + string_mod + '_right']
                    
                        for i in range(2):
                            if read_slit[i]:
                                runDir = os.path.join(paths.SINPA, 'runs', runid[i])
                                inputsDir = os.path.join(runDir, 'inputs/')
                                resultsDir = os.path.join(runDir, 'results/')
                                base_name = resultsDir + runid[i]
                                smap_name = base_name + '.map'
                    
                                # Load the strike map
                                Smap[i] = ss.mapping.StrikeMap('FILD', file=smap_name)
                                try:
                                    Smap[i].load_strike_points()
                                except:
                                    print('Strike map ' + str(i+1) + ' could not be loaded')
                                    continue
                                
                                try:
                                    selfmap_name = base_name + '.spcself'
                                    Smap[i].strike_points_self = ss.simcom.strikes.Strikes(code = 'SINPA', file=selfmap_name, type = 'mapcollimator')#(runid[i],type='mapcollimator')
                                except:
                                    print('Self shadowing points' + str(i+1) + ' could not be loaded')
                                    continue 

                               
                                # Load the strike points used to calculate the map
                                try:
                                    Smap[i].load_strike_points()
                                except:
                                    min_gyro[i].append(np.nan)
                                    max_gyro[i].append(np.nan)
                                    min_pitch[i].append(np.nan)
                                    max_pitch[i].append(np.nan)
                                    min_gyro_res[i].append(np.nan)
                                    max_gyro_res[i].append(np.nan)
                                    pitch_p0_res[i].append(np.nan)
                                    gyro_1_res[i].append(np.nan)
                                    gyro_2_res[i].append(np.nan)
                                    gyro_3_res[i].append(np.nan)
                                    min_pitch_res[i].append(np.nan)
                                    max_pitch_res[i].append(np.nan)
                                    avg_collimating_factor[i].append(np.nan)
                                    pitch_p0_gyro_1_collimating_factor[i].append(np.nan)
                                    print('Strike map ' + str(i+1) + ' could not be loaded')
                                    continue
                                
                        if plot_plate_geometry:
                            fig, ax = plt.subplots()
                    
                            ax.set_xlabel('Y [cm]')
                            ax.set_ylabel('Z [cm]')
                            ax.set_title('Camera view (YZ plane)')
                        
                            if read_slit[0]:
                                Geometry = ss.simcom.Geometry(GeomID=runid[0])
                            else:
                                Geometry = ss.simcom.Geometry(GeomID=runid[1])
                        
                            Geometry.plot2Dfilled(ax=ax, view = 'Scint', element_to_plot = [2],
                                                  plot_pinhole = False)
    
                            # If you want to mix and match, it assumes they have the same scintillator
                            
                            for i in range(2):
                                if read_slit[i]:# or mixnmatch:
                                    if plot_strike_points:
                                        Smap[i].strike_points.scatter(ax=ax, per=1
                                                                      , xscale=100.0, yscale=100.0
                                                                      ,mar_params = mar_params[i]
                                                                      , varx='ys', vary='zs')
                                
                                    if plot_strikemap:
                                        Smap[i].plot_real(ax=ax, marker_params=marker_params[i],
                                                          line_params=line_params[i], factor=100.0, labels=True)
                                
                                    if plot_orbits:
                                        if read_slit[i]:
                                            orb[i] = ss.sinpa.orbits(runID=runid[i])
                                        #orb[i].plot2D(ax=ax,line_params={'color': 'r'}, kind=(2,),factor=100.0)
                                        #orb[i].plot2D(ax=ax,line_params={'color': 'b'}, kind=(0,),factor=100.0)
                                        #orb[i].plot2D(ax=ax,line_params={'color': 'k'}, kind=(9,),factor=100.0)
                        
                          
                            fig.show()
                            
                            if plot_3D:
                                Geometry.plot3Dfilled(element_to_plot = [0,2])
                                ax3D = plt.gca()
                                
                                if plot_orbits:
                                    for i in range(2):
                                        if read_slit[i]:# or mixnmatch:
                                            # plot in red the ones colliding with the scintillator
                                            #orb[i].plot3D(line_params={'color': 'r'}, kind=(2,), ax=ax3D, factor=100.0)
                                            # plot in blue the ones colliding with the collimator
                                            #orb[i].plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax3D, factor=100.0)
                                            # plot the 'wrong orbits', to be sure that the are just orbits exceeding
                                            # the scintillator and nothing is wrong with the code
                                            #orb[i].plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax3D, factor=100.0)

                                            orb[i].plot3D(line_params={'color': 'r'}, kind=(2,), ax=ax3D, per = 1.)
                                            # plot in blue the ones colliding with the collimator
                                            orb[i].plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax3D, per = 1.)
                                            # plot the 'wrong orbits', to be sure that the are just orbits exceeding
                                            # the scintillator and nothing is wrong with the code
                                            orb[i].plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax3D, per = 1.)
                                            
                                            if save_orb_txt:
                                                orb[i].save_orbits_to_txt( kind=orbit_kind, units = 'mm', seperated = seperated)



                            
                        for i in range(2):
                            if read_slit[i]:
                                try:
                                    Smap[i].calculate_resolutions(min_statistics = 100)
                                except:
                                    min_gyro[i].append(np.nan)
                                    max_gyro[i].append(np.nan)
                                    min_pitch[i].append(np.nan)
                                    max_pitch[i].append(np.nan)
                                    min_gyro_res[i].append(np.nan)
                                    max_gyro_res[i].append(np.nan)
                                    pitch_p0_res[i].append(np.nan)
                                    gyro_1_res[i].append(np.nan)
                                    gyro_2_res[i].append(np.nan)
                                    gyro_3_res[i].append(np.nan)
                                    min_pitch_res[i].append(np.nan)
                                    max_pitch_res[i].append(np.nan)
                                    avg_collimating_factor[i].append(np.nan)
                                    pitch_p0_gyro_1_collimating_factor[i].append(np.nan)
                                    print('Could not calculate resolutions for slit ' + str(i+1))
                                    continue
    
                                if len(Smap[i].gyroradius)==0:
                                    min_gyro[i].append(np.nan)
                                    max_gyro[i].append(np.nan)
                                else:
                                    min_gyro[i].append(np.nanmin(Smap[i].gyroradius))
                                    max_gyro[i].append(np.nanmax(Smap[i].gyroradius))
    
                                index_p0 = np.where(Smap[i].pitch == p0[i])[0]
                                if len(index_p0) == 0:
                                    min_gyro_p0[i].append(np.nan)
                                    max_gyro_p0[i].append(np.nan)
                                else:
                                    min_gyro_p0[i].append(np.nanmin(Smap[i].gyroradius[index_p0]))
                                    max_gyro_p0[i].append(np.nanmax(Smap[i].gyroradius[index_p0]))
                            
                                if len(Smap[i].pitch)==0:
                                    min_pitch[i].append(np.nan)
                                    max_pitch[i].append(np.nan)
                                else:
                                    min_pitch[i].append(np.nanmin(Smap[i].pitch))
                                    max_pitch[i].append(np.nanmax(Smap[i].pitch)) 
                        
                                min_gyro_res[i].append(np.nanmin(Smap[i].resolution['Gyroradius']['sigma']))
                                max_gyro_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma']))
                            
                                idx_p0 = np.argmin( abs(Smap[i].strike_points.header['pitch'] - p0[i]))
                                try:
                                    pitch_p0_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma'][idx_p0]))
                                except:
                                    pitch_p0_res[i].append(np.nan)
                                
                                idx1 = np.argmin( abs(Smap[i].strike_points.header['gyroradius'] -0.5))
                                idx2 = np.argmin( abs(Smap[i].strike_points.header['gyroradius'] -1.0))
                                idx3 = np.argmin( abs(Smap[i].strike_points.header['gyroradius'] -1.5))
                                gyro_1_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma'][idx1,:]))
                                gyro_2_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma'][idx2,:]))
                                gyro_3_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma'][idx3,:]))
                                
                                min_pitch_res[i].append(np.nanmin(Smap[i].resolution['Pitch']['sigma']))
                                max_pitch_res[i].append(np.nanmax(Smap[i].resolution['Pitch']['sigma']))
                    
                                avg_collimating_factor[i].append(np.nanmean(Smap[i].collimator_factor_matrix))
                    
                                try:
                                    pitch_p0_gyro_1_collimating_factor[i].append(Smap[i].collimator_factor_matrix[idx2, idx_p0])
                                except:
                                    pitch_p0_gyro_1_collimating_factor[i].append(0)                    
                        
                                # for best case: haven't added a way to title these by slit yet
                                scan_string = scan_Parameters[scan_parameter]['scan_str']
                                diff = abs(best_param_value-value)/best_param_value
                                if scan_string == best_param and diff < 0.01:
                                    if plot_resolutions:
                                        Smap[i].plot_resolutions()
    
                                    if plot_gyro_res:
                                        Smap[i].plot_resolution_fits(var='Gyroradius',pitch=p0[i])
                                        plt.gcf().show()
    
                                    if plot_pitch_res:
                                        Smap[i].plot_resolution_fits(var='Pitch',gyroradius = 1.)
                                        plt.gcf().show()
    
                                    if plot_collimating_factor:
                                        Smap[i].plot_collimator_factor()
                                        plt.gcf().show()
                                        
                                if plot_synthetic_signal:
                                    # I haven't tested this yet
                                    
                                    distro = pickle.load( open( dist_file, "rb" ) )


                                    per = 0.05
                                    flags = np.random.rand(distro['n']) < per
                                    
                                    distro['BR'] = distro['BR'][flags]
                                    distro['n'] = len(distro['BR'])
                                    distro['Bphi'] = distro['Bphi'][flags]
                                    distro['Bz'] = distro['Bz'][flags]
                                    distro['Anum'] =distro['Anum'][flags]
                                    distro['Znum'] = distro['Znum'][flags]
                                    distro['vR']=distro['vR'][flags]
                                    distro['vphi'] =distro['vphi'][flags]
                                    distro['vz'] =distro['vz'][flags]
                                    distro['pitch'] = distro['pitch'][flags]
                                    distro['energy'] =distro['energy'][flags]
                    
                                    distro['gyroradius'] =distro['gyroradius'][flags]
                                    distro['weight'] =    distro['weight'] [flags]
                                    
                                    Gyro_radius=distro['gyroradius']
                                    pitch = distro['pitch']
                                    weight = distro['weight']
                                    gridlev = 100
                                    fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                                    facecolor='w', edgecolor='k', dpi=100)
                        
                                    ax4.set_ylabel('Gyro radius [cm]')
                                    ax4.set_xlabel('Pitch angle [$\\degree$]')
                                    
                                    rl = np.linspace(0, 2., gridlev)
                                    p = np.linspace(-1, 1, gridlev)
                                    p = np.linspace(55,145, gridlev)
                                    #IPython.embed()
                                    f, rl, p = np.histogram2d(Gyro_radius, pitch, bins=[rl, p],
                                                                       weights=weight)
                                    cmap = cm.inferno # Colormap
                                    cmap._init()                    
                                    im4 = ax4.imshow(f, origin='lower',
                                                    extent=(p[0],
                                                            p[-1],
                                                            rl[0],
                                                            rl[-1]),
                                                    cmap = cmap,
                                                    aspect='auto', interpolation='nearest')#,
                                
                                    
                                    fig4.colorbar(im4, ax=ax4, label='fast-ions/s/keV')
                                    fig4.tight_layout()
                                    fig4.show()
    
                                    output = ss.fildsim.synthetic_signal_remap(distro, Smap[i],
                                                                               rmin=0.1, rmax=4.0, dr=0.01,
                                                                               pmin=55.0, pmax=145.0,
                                                                               dp=0.01)
    


    
                                    fig, ax_syn = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                                               facecolor='w', edgecolor='k', dpi=100)   
                                    ax_syn.set_xlabel('Pitch [$\\degree$]')
                                    ax_syn.set_ylabel('Gyroradius [cm]')
                                    ax_syn.set_ylim([0, 2.0])
                                    
                                    cmap = cm.inferno # Colormap
                                    cmap._init()
                                    
                                    ss.fildsim.plot_synthetic_signal(output['gyroradius'], output['pitch']
                                                               , output['signal'], 
                                                               ax=ax_syn, fig=fig, cmap = cmap)
                                    if read_slit[0] and read_slit[1]:
                                        fig.title('Slit #' + str(i+1))
                                    fig.tight_layout()
                                    fig.show()
                                        

                                if plot_resolutions:
                                    Smap[i].plot_resolutions()
                                    
                                if plot_gyro_res:
                                    Smap[i].plot_resolution_fits(var='Gyroradius',
                                                                  pitch=np.array([75]),
                                                                  kind_of_plot='normal',
                                                                  include_legend=True)
                                    plt.gcf().show()
                                if plot_pitch_res:
                                    Smap[i].plot_resolution_fits(var='Pitch',
                                                                  gyroradius=1,
                                                                  kind_of_plot='normal',
                                                                  include_legend=True)
                                    plt.gcf().show()
                                if plot_collimating_factor:
                                    Smap[i].plot_collimator_factor()
                                    plt.gcf().show()

                        
                        #del Smap
                    ##plot metrics
                    if plot_metrics:
                        for i in range(2):
                            if read_slit[i]:
                            
                                x = scan_Parameters[scan_parameter]['scan_values']
                        
                                fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12, 12),
                                                   facecolor='w', edgecolor='k', dpi=100)
                
                                ax_gyro, ax_pitch, ax_gyro_res, ax_pitch_res = axarr[0,0], axarr[0,1], axarr[1,0], axarr[1,1]
                                ax_gyro.set_xlabel(scan_parameter + ' [cm]')
                                ax_gyro.set_ylabel('Gyroradius [cm]')
                
                                ax_gyro_res.set_xlabel(scan_parameter + ' [cm]')
                                ax_gyro_res.set_ylabel('Gyroradius resolution [cm]')
                    
                                ax_pitch_res.set_xlabel(scan_parameter + ' [cm]')
                                ax_pitch_res.set_ylabel('Pitch angle resolution [$\\degree$]')
                
                                ax_gyro.plot(x, min_gyro[i], marker = 'o', label = 'min')
                                ax_gyro.plot(x, max_gyro[i], marker = 'o', label = 'max')
                                ax_gyro.plot(x, min_gyro_p0[i], marker = 'o', label = 'min at ' + str(p0[i]) + '$\\degree$')
                                ax_gyro.plot(x, max_gyro_p0[i], marker = 'o', label = 'max at ' + str(p0[i]) + '$\\degree$')
                                ax_gyro.legend(loc='upper right')
                
                                ax_gyro_res.plot(x, gyro_1_res[i], marker = 'o', label = '0.5 cm')
                                ax_gyro_res.plot(x, gyro_2_res[i], marker = 'o', label = '1 cm')
                                ax_gyro_res.plot(x, gyro_3_res[i], marker = 'o', label = '1.5 cm')
                                ax_gyro_res.legend(loc='upper right')
                
                                ax_pitch_res.plot(x, min_pitch_res[i], marker = 'o', label = 'min')
                                ax_pitch_res.plot(x, max_pitch_res[i], marker = 'o', label = 'max')
                                ax_pitch_res.plot(x, pitch_p0_res[i], marker = 'o', label = str(p0[i]) + '$\\degree$')
                                ax_pitch_res.legend(loc='upper right')
    
                
                                ax_coll = axarr[0,1]
                                ax_coll.plot(x, avg_collimating_factor[i], marker = 'o', label='avg')
                                ax_coll.plot(x, pitch_p0_gyro_1_collimating_factor[i], marker = 'o', label='1 cm, ' + str(p0[i]) + '$\\degree$')
                                ax_coll.legend(loc='upper right')
                                ax_coll.set_xlabel(scan_parameter + ' [cm]')
                                ax_coll.set_ylabel('Average collimator factor %')
                
                                ax_gyro_res.set_ylim([0, 1.0])
                                ax_pitch_res.set_ylim([0, 5.0])
                            
                                if read_slit[0] and read_slit[1]:
                                    fig.suptitle('Slit '  + str(i+1) + '           ')
                                fig.tight_layout()
                                fig.show()  
                        
                    # if plot_metrics:
                    #     for i in range(2):
                    #         if read_slit[i]:
                    #           Smap[i].calculate_resolutions(min_statistics=1)
              
                                if plot_resolutions:
                                    Smap[i].plot_resolutions()
                                    
                                if plot_gyro_res:
                                    Smap[i].plot_resolution_fits(var='Gyroradius',
                                                                  pitch=np.array(75),
                                                                  kind_of_plot='normal',
                                                                  include_legend=True)
                                    plt.gcf().show()
                                if plot_pitch_res:
                                    Smap[i].plot_resolution_fits(var='Pitch',
                                                                  gyroradius=1,
                                                                  kind_of_plot='normal',
                                                                  include_legend=True)
                                    plt.gcf().show()
                                if plot_collimating_factor:
                                    Smap[i].plot_collimator_factor()
                                    plt.gcf().show()
                            
                                    #del Smap
                                        
                        
                        

                        
                        
