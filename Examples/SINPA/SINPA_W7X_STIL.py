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
                            , dl1 = 10
                            , dl2 = 10):
    '''
    '''
    #dl in cm converted to mm
    dl1 *= 10
    dl2 *= 10
    ##left collimator
    p1 = np.array([-783.033, 5789.71, 345.661])
    p1_delta = np.array([-782.865, 5789.4, 346.13])
    p2=np.array([-763.865, 5795.45, 347.684])
    #follow vector for dl length
    vi = p1_delta - p1
    vi/= np.linalg.norm(vi) # unit vector
    ic = np.argmax(np.abs(vi))
    dpos=vi/np.abs(vi[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*dl1
    p1_dl = p1 + dpos
    
    xx = np.array([p1[0], p1_dl[0],p2[0]])
    yy = np.array([p1[1], p1_dl[1],p2[1]])
    zz = np.array([p1[2], p1_dl[2],p2[2]])
    
    p1 = np.array([-782.879, 5789.41, 344.934])
    p1_delta = np.array([-782.711, 5789.11, 345.403])
    p2=np.array([-763.711, 5795.15, 346.957])
    #follow vector for dl length
    vi = p1_delta - p1
    vi/= np.linalg.norm(vi) # unit vector
    ic = np.argmax(np.abs(vi))
    dpos=vi/np.abs(vi[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*dl1
    p1_dl = p1 + dpos
    
    xx = np.concatenate([xx, np.array([p1[0], p1_dl[0],p2[0]])])
    yy = np.concatenate([yy, np.array([p1[1], p1_dl[1],p2[1]])])
    zz = np.concatenate([zz, np.array([p1[2], p1_dl[2],p2[2]])])
    
    
    ###right collimator
    
    p1 = np.array([-724.249, 5782.36, 381.294])
    p1_delta = np.array([-724.059, 5782.07, 381.764])
    p2=np.array([-738.309, 5777.53, 380.598])
    #follow vector for dl length
    vi = p1_delta - p1
    vi/= np.linalg.norm(vi) # unit vector
    ic = np.argmax(np.abs(vi))
    dpos=vi/np.abs(vi[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*dl2
    p1_dl = p1 + dpos

    xx = np.concatenate([xx, np.array([p1[0], p1_dl[0],p2[0]])])
    yy = np.concatenate([yy, np.array([p1[1], p1_dl[1],p2[1]])])
    zz = np.concatenate([zz, np.array([p1[2], p1_dl[2],p2[2]])])

    p1 = np.array([-724.403, 5782.66, 382.021])
    p1_delta = np.array([-724.213, 5782.36, 382.491])
    p2=np.array([-738.463, 5777.83, 381.325])
    #follow vector for dl length
    vi = p1_delta - p1
    vi/= np.linalg.norm(vi) # unit vector
    ic = np.argmax(np.abs(vi))
    dpos=vi/np.abs(vi[ic])
    dpos=dpos/np.sqrt(np.sum(dpos**2))*dl2
    p1_dl = p1 + dpos

    xx = np.concatenate([xx, np.array([p1[0], p1_dl[0],p2[0]])])
    yy = np.concatenate([yy, np.array([p1[1], p1_dl[1],p2[1]])])
    zz = np.concatenate([zz, np.array([p1[2], p1_dl[2],p2[2]])])
    
    n_triang = 4
    data = np.zeros(n_triang, dtype=mesh.Mesh.dtype)
    mesh_object = mesh.Mesh(data, remove_empty_areas=False)
    mesh_object.x[:] = np.reshape(xx, (n_triang, 3))
    mesh_object.y[:] = np.reshape(yy, (n_triang, 3))
    mesh_object.z[:] = np.reshape(zz, (n_triang, 3))
    mesh_object.save(file_name_save+'.stl') 

def write_scint_triangles(file_name_save= 'SCINTILLATOR_PLATE_test'):
    #make stl file
    
    p1 = np.array([-763.522, 5746.9, 399.36])
    p2 = np.array([-738.88, 5801.67, 344.496])
    p3 = np.array([-785.658, 5790.75, 337.194])
    p4 = np.array([-716.744, 5757.82, 406.662])
    
    vertices = np.array([\
    p1,
    p2,
    p3,
    p4])
    
    faces = np.array([\
    [0, 1, 2],
    [2, 3, 0],
    [1, 3, 0],
    [1, 3, 1]])
        
    surface = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            surface.vectors[i][j] = vertices[f[j],:]

    surface.save(file_name_save+'.stl')


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
        ss.LibCAD.write_file_for_fortran_test(collimator_stl_files[coll],
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
        ss.LibCAD.write_file_for_fortran(scintillator_stl_files[scint], 
                                         scint_filename, 
                                         convert_mm_2_m = True) 
        
        # --- Open and load the stil file
        mesh_obj = mesh.Mesh.from_file(scintillator_stl_files[scint])
    
        x1x2x3 = mesh_obj.x  
        y1y2y3 = mesh_obj.y  
        z1z2z3 = mesh_obj.z  
    
        itriang = 22
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
        
        # a = open3d.io.read_triangle_mesh(stl_files['scintillator'])
        # vertices = np.asarray(a.vertices)
        # index = np.asarray(a.triangles)
        
        # # Get the points of the first triangle in the scintillator stl file
        # itriang = 22
        # j=0
        # p1 = np.array((vertices[index[itriang, j], 0],
        #                vertices[index[itriang, j], 1],
        #                vertices[index[itriang, j], 2]) ) * 0.001 #convert mm to m
        # j=1
        # p2 = np.array((vertices[index[itriang, j], 0],
        #                vertices[index[itriang, j], 1],
        #                vertices[index[itriang, j], 2]) ) * 0.001 #convert mm to m
        # j=2
        # p3 = np.array((vertices[index[itriang, j], 0],
        #                vertices[index[itriang, j], 1],
        #                vertices[index[itriang, j], 2]) ) * 0.001 #convert mm to m
        

        # scint_norm = -get_normal_vector(p1, p2, p3)
        # To do: Make sure normal vector points away from plasma
        # need a plasma point to test
        
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
        
        d2 = np.sqrt(np.sum((pinhole_points[2,:] - pinhole_points[1,:])**2) )
        u2 = (pinhole_points[2] - pinhole_points[1])  / d2    
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
    
    run_code = True
    geom_name = 'W7X_stl_QHS_test2' # Only used if running a single iteration

    scan = False # Set to "False" to run a single iteration
    scan_param = ['sh'] # , 'pw2', 'pl', 'pl2', 'psd', 'sh', 'sh2', 'sl', 'sl2', 'oc', 'oc2', 'st', 'ct', 'ct2']

    
    save_orbits = False
    plot_plate_geometry = True
    plot_3D = True
    
    read_results = not run_code #True
    plot_strike_points = True
    plot_strikemap = True
    plot_orbits = False
    
    backtrace = False
    
    if save_orbits:
        nGyro = 360
        maxT = 0.00000006  * 10
    else:
        nGyro = 360
        maxT = 0.00000006 
        
    plot_metrics = True

    double = False # Plot both slits in the geometry files
    run_slit = [True, False] # Run particles starting at one slit or the other?
    read_slit = [True, True] # Read results from the left slit, the right slit, or both?
    
    if run_code:
        if not run_slit[0] and not run_slit[1]:
            print('You need to choose at least one slit!')
    
    if read_results:
        if not read_slit[0] and not read_slit[1]:
            print('You need to choose at least one slit!')

    
    #for best case: choose the scan parameter and value you want to look at!
    best_param = 'sh'
    best_param_value = 0.50
    plot_resolutions = False
    plot_gyro_res= False
    plot_pitch_res= False
    plot_collimating_factor = False
    plot_synthetic_signal= False
                            
    
    marker_params = [{'markersize':6, 'marker':'o','color':'b'},
                     {'markersize':6, 'marker':'o','color':'m'}]
    line_params =  [{'ls':'solid','color':'k'},
                    {'ls':'solid','color':'w'}]
    mar_params = [{'zorder':3,'color':'k'},
                  {'zorder':3,'color':'w'}]
    

    n_markers = int(1e4)

    gyro_arrays = [[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2., 3., 4.],
                   [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2., 3., 4.]]
    
    pitch_arrays = [[85., 75., 65., 55., 45., 35., 25., 15., 5.],
                    [95., 105., 115., 125., 135., 145., 155., 165., 175.]]
    
    gyrophase_range = np.array([np.deg2rad(220),np.deg2rad(300)])

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
    
    use_aligned_B = False    
    use_rotated_FILD_B = False
    use_ascot_B = True
    use_single_B = False
    ascot_bfield_File = 'std_bfield.pickle'
    ascot_boozer_File = 'std_boozer.pickle'

    ##STL files
    
    
    collimator_stl_files = {'collimator': 'probe_head_with_pinholes.stl',
                            }
    scintillator_stl_files = {'scintillator':  'SCINTILLATOR_PLATE_test2.stl'}
    pinholes = [{}, {}]
    pinholes[0]['pinholeKind'] =1
    pinholes[0]['pinholeCentre'] = None
    pinholes[0]['pinholeRadius'] = None
    pinholes[0]['points'] = np.array([[-725.674, 5781.91, 381.177],
                                    [-725.828, 5782.21, 381.904],
                                    [-724.403, 5782.66, 382.021],
                                    [-724.249, 5782.36, 381.294] ] )#co- going slit opening

    
    pinholes[1]['pinholeKind'] =1
    pinholes[1]['pinholeCentre'] = None
    pinholes[1]['pinholeRadius'] = None
    pinholes[1]['points'] = np.array([[-782.879, 5789.41, 344.934],
                                   [-783.033, 5789.71, 345.661],
                                   [-781.608, 5790.16, 345.778],
                                   [-781.454, 5789.86, 345.051] ] )#counter going slit opening
    
    
    #cm
    slit_height     = 0.5 
    slit_height_2   = 0.5 
    head_params = {
                   "slit_height"   : slit_height,
                   "slit_height_2"   : slit_height_2,
                   }


    if scan:
        scan_Parameters = {}
        scan_Parameters['Slit height'] = {'scan_str': 'sh',
                                        'scan_param': 'slit_height',
                                        'scan_values': np.arange(0.1, 2.5, 0.2) ,
                                        'scan': False,
                                        'value': slit_height}
        scan_Parameters['Slit height_2'] = {'scan_str': 'sh2',
                                        'scan_param': 'slit_height_2',
                                        'scan_values': np.arange(0.1, 2.5, 0.2) *10,
                                        'scan': False,
                                        'value': slit_height_2}
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
                'backtrace': backtrace  # Flag to backtrace the orbits
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
                    
            if use_aligned_B:
                # This is negative so the particles will gyrate the way we want
                direction = np.array([0, -1, 0])
                Field = modB * direction
                field.createHomogeneousField(Field, field='B')
                #IPython.embed()
            if use_single_B:    
                field.createFromSingleB(B = np.array([Br,Bz,Bt]), Rmin = 4.5,
                                          Rmax = 6.5,
                                          zmin = -1.0, zmax = 1.0,
                                          nR = 100, nZ = 100)
            elif use_rotated_FILD_B:
                # Haven't tried this out yet, waiting on updates to the angles
                Field = np.array([Br, Bz, Bt])
                phi, theta = ss.fildsim.calculate_fild_orientation(Br, Bz, Bt, alpha, beta, verbose=False)
                u1 = np.array([1., 0., 0.])
                u2 = np.array([0., 1., 0.])
                u3 = np.array([0., 0., 1.])
                field.createHomogeneousFieldThetaPhi(theta, phi, field_mod = modB,
                                                     field='B', u1=u1, u2=u2, u3=u3,
                                                     IpBtsign = 1., verbose = False)
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
                br = np.repeat(ascot_bfield['br'], 
                               ascot_bfield['toroidalPeriods'],
                               axis = 1)
                bphi = np.repeat(ascot_bfield['bphi'], 
                                 ascot_bfield['toroidalPeriods'],
                                 axis = 1)
                bz = np.repeat(ascot_bfield['bz'], 
                               ascot_bfield['toroidalPeriods'],
                               axis = 1)
                
                field.Bfield['fr'] = np.asfortranarray(br, dtype=np.float64)
                field.Bfield['fz'] = np.asfortranarray(bz, dtype=np.float64)
                field.Bfield['ft'] = np.asfortranarray(bphi, dtype=np.float64)
                
                field.Bfield['nPhi'] = np.asfortranarray(np.shape(br)[1], dtype=np.int32 )
                field.Bfield['Phimin'] = np.asfortranarray(0., dtype=np.float64)
                field.Bfield['Phimax'] = np.asfortranarray(2.*np.pi, dtype=np.float64)
                
                field.bdims = 3
                
                #To do FILD is at +-97 deg, plot b-field at correct phi position
                field.plot('br', phiSlice = 0 ,plot_vessel = False)
                plt.show()
            #Write geometry files
            for i in range(2):
                if run_slit[i]:
                    
                    write_collimator_plates(file_name_save= 'lateral_collimator_plates'
                                            , dl1=slit_height
                                            , dl2=slit_height_2)
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
                                
                        if use_aligned_B:
                            # This is negative so the particles will gyrate the way we want
                            direction = np.array([0, -1, 0])
                            Field = modB * direction
                            field.createHomogeneousField(Field, field='B')
                        if use_single_B:    
                            field.createFromSingleB(B = np.array([Br,Bz,Bt]), Rmin = 4.5,
                                          Rmax = 6.5,
                                          zmin = -1.0, zmax = 1.0,
                                          nR = 100, nZ = 100)
                        elif use_rotated_FILD_B:
                            # Haven't tried this out yet, waiting on updates to the angles
                            Field = np.array([Br, Bz, Bt])
                            phi, theta = ss.fildsim.calculate_fild_orientation(Br, Bz, Bt, alpha, beta, verbose=False)
                            u1 = np.array([1., 0., 0.])
                            u2 = np.array([0., 1., 0.])
                            u3 = np.array([0., 0., 1.])
                            field.createHomogeneousFieldThetaPhi(theta, phi, field_mod = modB,
                                                                 field='B', u1=u1, u2=u2, u3=u3,
                                                                 IpBtsign = 1., verbose = False)
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
                            br = np.repeat(ascot_bfield['br'], 
                                           ascot_bfield['toroidalPeriods'],
                                           axis = 1)
                            bphi = np.repeat(ascot_bfield['bphi'], 
                                             ascot_bfield['toroidalPeriods'],
                                             axis = 1)
                            bz = np.repeat(ascot_bfield['bz'], 
                                           ascot_bfield['toroidalPeriods'],
                                           axis = 1)
                            
                            field.Bfield['fr'] = np.asfortranarray(br, dtype=np.float64)
                            field.Bfield['fz'] = np.asfortranarray(bz, dtype=np.float64)
                            field.Bfield['ft'] = np.asfortranarray(bphi, dtype=np.float64)
                            
                            field.Bfield['nPhi'] = np.asfortranarray(np.shape(br)[1], dtype=np.int32 )
                            field.Bfield['Phimin'] = np.asfortranarray(0., dtype=np.float64)
                            field.Bfield['Phimax'] = np.asfortranarray(2.*np.pi, dtype=np.float64)
                            
                            field.bdims = 3
                            
                            #To do FILD is at +-97 deg, plot b-field at correct phi position
                            field.plot('bphi', phiSlice = 0 ,plot_vessel = False)
                            plt.show()
                        #Write geometry files
                        for i in range(2):
                            if run_slit[i]:
                                write_collimator_plates(file_name_save= 'lateral_collimator_plates'
                                                        , dl1=head_params["slit_height"]
                                                        , dl2=head_params["slit_height_2"])
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
                        
                                head_params[name] = temp
                        
                        if plot_plate_geometry:
                            geomID = run_names[0]
                            Geometry = ss.simcom.Geometry(GeomID=geomID)
                            if plot_3D:
                                Geometry.plot3Dfilled(element_to_plot = [0,2], plot_pinhole = False)
                            #Geometry.plot2Dfilled(view = 'XY', element_to_plot = [0,2], plot_pinhole = False)
                            Geometry.plot2Dfilled(view = 'Scint', element_to_plot = [2], plot_pinhole = False)
    
                        
    # -----------------------------------------------------------------------------
    # --- Section 2: Analyse the results
    # -----------------------------------------------------------------------------
    if read_results:
        if not scan:
            runid = [geom_name,geom_name + '_right']
            strike_points_file = ['','']
            Smap = [[],[]]
            p0 = [75, 120]
            
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
                            Smap[i].strike_points.scatter(ax=ax, per=0.1
                                                          , xscale=100.0, yscale=100.0
                                                          ,mar_params = mar_params[i]
                                                          , varx='ys', vary='zs')
                                
                        if plot_strikemap:
                            Smap[i].plot_real(ax=ax, marker_params=marker_params[i],
                                              line_params=line_params[i], factor=100., labels=True)
                                
                        if plot_orbits:
                            orb.append(ss.sinpa.orbits(runID=runid[i]))
                            orb[i].plot2D(ax=ax,line_params={'color': 'r'}, kind=(2,),factor=100.0)
                            #orb[i].plot2D(ax=ax,line_params={'color': 'b'}, kind=(0,),factor=100.0)
                            #orb[i].plot2D(ax=ax,line_params={'color': 'k'}, kind=(9,),factor=100.0)
                    else:
                        orb.append([])
                          
                fig.show()
                    
                if plot_3D:
                    Geometry.plot3Dfilled(element_to_plot = [0, 2], units = 'm')
                    ax3D = plt.gca()
    
                    if plot_strike_points:
                        #IPython.embed()
                        Smap[i].strike_points.plot3D(ax=ax3D)
                    
                    for i in range(2):
                        if read_slit:# or mixnmatch:
                            # plot in red the ones colliding with the scintillator
                            if plot_orbits:
                                orb[i].plot3D(line_params={'color': 'r'}, kind=(2,), ax=ax3D, per = 1.)
                                # plot in blue the ones colliding with the collimator
                                orb[i].plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax3D, per = 1.)
                                # plot the 'wrong orbits', to be sure that the are just orbits exceeding
                                # the scintillator and nothing is wrong with the code
                                orb[i].plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax3D, per = 1.)

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
                                        Smap[i].strike_points.scatter(ax=ax, per=0.1, 
                                                                      mar_params = mar_params[i])
                                
                                    if plot_strikemap:
                                        Smap[i].plot_real(ax=ax, marker_params=marker_params[i],
                                                          line_params=line_params[i], factor=100.0, labels=True)
                                
                                    if plot_orbits:
                                        if read_slit[i]:
                                            orb[i] = ss.sinpa.orbits(runID=runid[i])
                                        orb[i].plot2D(ax=ax,line_params={'color': 'r'}, kind=(2,),factor=100.0)
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
                                            orb[i].plot3D(line_params={'color': 'r'}, kind=(2,), ax=ax3D, factor=100.0)
                                            # plot in blue the ones colliding with the collimator
                                            #orb[i].plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax3D, factor=100.0)
                                            # plot the 'wrong orbits', to be sure that the are just orbits exceeding
                                            # the scintillator and nothing is wrong with the code
                                            #orb[i].plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax3D, factor=100.0)
                            
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
                                        dist_file = '/afs/ipp/home/a/ajvv/ascot5/RUNS/W7X_distributions/250mm_FILD_distro.pck'
                                        distro = pickle.load( open( dist_file, "rb" ) )
        
                                        output = ss.fildsim.synthetic_signal_remap(distro, Smap[i],
                                                                                   rmin=0.1, rmax=4.0, dr=0.01,
                                                                                   pmin=55.0, pmax=105.0,
                                                                                   dp=1.0)
        
    
    
        
                                        fig, ax_syn = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                                                   facecolor='w', edgecolor='k', dpi=100)   
                                        ax_syn.set_xlabel('Pitch [$\\degree$]')
                                        ax_syn.set_ylabel('Gyroradius [cm]')
                                        ax_syn.set_ylim([0, 2.5])
                                        ss.fildsim.plot_synthetic_signal(output['gyroradius'], output['pitch']
                                                                   , output['signal'], 
                                                                   ax=ax_syn, fig=fig)
                                        if read_slit[0] and read_slit[1]:
                                            fig.title('Slit #' + str(i+1))
                                        fig.tight_layout()
                                        fig.show()
                                        
                        
                        del Smap


        if plot_metrics:
            #for i in range(2):
            #if read_slit[i] or mixnmatch:
            Smap.calculate_resolutions(min_statistics=100)
  
            if plot_resolutions:
                Smap.plot_resolutions()
                
            if plot_gyro_res:
                Smap.plot_resolution_fits(var='Gyroradius',
                                             pitch=p0,
                                             kind_of_plot='normal',
                                             include_legend=True)
                plt.gcf().show()
            if plot_pitch_res:
                Smap.plot_resolution_fits(var='Pitch',
                                             gyroradius=1,
                                             kind_of_plot='normal',
                                             include_legend=True)
                plt.gcf().show()
            if plot_collimating_factor:
                Smap.plot_collimator_factor()
                plt.gcf().show()

        #del Smap
                        
                        
                        
                        
                        
                        
                        
                        
