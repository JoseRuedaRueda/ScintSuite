#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:11:34 2022

@author: ajvv
"""

import numpy as np
import os
import matplotlib.pylab as plt
import ScintSuite as ss
from ScintSuite._Machine import machine
from ScintSuite._Paths import Path
paths = Path(machine)

from matplotlib import cm


import ScintSuite.LibData.TCV.Equilibrium as TCV_equilibrium

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

def write_stl_geometry_files(root_dir
                              , run_name = ''
                              , collimator_stl_files = {}
                              , scintillator_stl_files = {}
                              #, itriang_scint = 0
                              , ps = [0,0,0]
                              , u1_scint = [1,0,0]
                              , scint_norm = [0,1,0]
                              , pinhole = []
                              ):
    '''
    Parameters
    ----------
    Returns
    -------
    None.
    '''
    run_folder = run_name + '/'
    directory = os.path.join(root_dir, run_folder)

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
        ss._CAD.write_file_for_fortran_numpymesh(collimator_stl_files[coll],
                                              collimator_filename, 
                                              convert_mm_2_m = True )      


    rot = np.identity(3)
    # Dummy scintillator normal vector,reference point and rotation vector
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
        ss._CAD.write_file_for_fortran_numpymesh(scintillator_stl_files[scint], 
                                         scint_filename, 
                                         convert_mm_2_m = True )    

        rot = ss.sinpa.geometry.calculate_rotation_matrix(scint_norm, u1 = u1_scint
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

    Test = False  # if true don't do run, just check the input geometry
    # test geometry
    plot_plate_geometry = True
    plot_3D = False

    shot = 79302
    Rinsertion = -17.2 #[mm] #negative means inserted

    if shot <=77469:
        year = 2022
    else:
        year = 2023

    time = 1.2
    use_reduced_stl_models = True
    use_1mm_pinhole = False

    run_code = True  # Set flag to run FILDSIM
    run_slit = [False, False, True, False]  # Run particles starting at slits set to true, starting with ul, ur, lr, ll
    read_slit = [False, False, True, False] # Read results from diffrent slits
    slits = ['ul', 'ur', 'lr', 'll']
    idx_slit = np.where(run_slit)[0][0]
    run_name = '%i@%.3f' %(shot, time) #Choose unique run identifier, like shot number and time
    run_names = [run_name +'_ul', run_name +'_ur', run_name +'_lr', run_name +'_ll']
    read_results = not run_code # Flag to read output after run

    ###
    #Input settings
    ###
    self_shadowing = False  #Flag to remove markers that would have been shadowed, leave this on
    backtrace = False  #Do backtracing
    ###

    #Output data to save
    ###
    #Flags
    save_orbits = False # Save orbit data
    if save_orbits:
        nGyro = 36
        maxT = 0.000006 *  1
        nGyro = 360
        maxT = 0.00000006 
    else:
        nGyro = 360
        maxT = 0.00000006 
    
    save_self_shadowing_collimator_strike_points = False
    ###
    # FILD probe head
    ###    
    alpha = 0.0 # TCV FILD has no inclination or rotation angles.
    beta = 0.0  #Besides we use TCV coordinates, so for ow this is not needed
    #STL files
    geom_dir = os.path.join(paths.SINPA,'Geometry/')
    
    reduced_string = ['', 'reduced', '_reduced']
    slitSize_string = ['08', '10']
    year_string = str(year)
    use_upper_slit = slits[idx_slit][0]=='u'
    collimator_string = ['DOWN', 'UP','']

    collimator_stl_files = {'heatshield': geom_dir+'TCV_FILD/%s/%s/HeatShield_%s%s.stl'%(year_string, 
                                                                                        reduced_string[int(use_reduced_stl_models)], 
                                                                                         year_string, 
                                                                                         reduced_string[int(use_reduced_stl_models)])
                            }
    scintillator_stl_files = {'scintillator':geom_dir+'TCV_FILD/%s/%s/Scintillator_%s%s.stl'%(year_string, 
                                                                                              reduced_string[int(use_reduced_stl_models)], 
                                                                                              year_string,
                                                                                              reduced_string[int(use_reduced_stl_models)])
                                }
    collimator_stl_files['collimator'] = geom_dir+'TCV_FILD/%s/%s/Collimator%s_%s_%s%s.stl'%(year_string, 
                                                                                            reduced_string[int(use_reduced_stl_models)], 
                                                                                            collimator_string[int(use_upper_slit) + 2023%year], 
                                                                                            slitSize_string[int(use_1mm_pinhole)]+'mm', 
                                                                                            year_string, 
                                                                                            reduced_string[int(use_reduced_stl_models)])
                            
    if year == 2022:
        #choose points on scintillator for reference coordinate system
        p1 = np.array([-195.225, 1130.11, 29.8262]) * 0.001 #convert mm to m
        p2 = np.array([-210.381, 1127.1, 29.826]) * 0.001 
        p3 = np.array([-202.901, 1128.59, -14.5706]) * 0.001

        scint_norm = get_normal_vector(p1, p2, p3)        
        ps = p1 
        u1_scint = p2 - p1
        u1_scint /= np.linalg.norm(u1_scint)

    if year == 2023:
        p1 = np.array([-195.63, 1130.03, 29.4993]) * 0.001 
        p2 = np.array([-209.592, 1127.26, 29.4995] ) * 0.001
        p3 = np.array([-210.13, 1127.15, -4.66387]) * 0.001

        scint_norm = get_normal_vector(p1, p2, p3)        
        ps = p2 
        u1_scint = p2 - p1
        u1_scint /= np.linalg.norm(u1_scint)     
               
    #Pinhole coordinates in [mm]
    pinholes = [{}, {}, {}, {}]
    if use_1mm_pinhole:
        ### 1.0mm collimator
        pinholes[0]['pinholeKind'] =1
        pinholes[0]['pinholeCentre'] = None
        pinholes[0]['pinholeRadius'] = None
        pinholes[0]['points'] = np.array([[-192.96, 1137.7, 35.4414],  #Important, the vector p1 to p2 should be one dimension, and 
                                        [-193.155, 1138.68, 35.4414],  # the vector p2 to p3 the other dimension of the slit
                                        [-195.117, 1138.29, 35.4414],
                                        [-194.921, 1137.31, 35.4414]] ) #upper right, looking from plasma to FILD head

        pinholes[1]['pinholeKind'] =1
        pinholes[1]['pinholeCentre'] = None
        pinholes[1]['pinholeRadius'] = None
        pinholes[1]['points'] = np.array([[-257.103, 1124.94, 35.4414],
                                        [-257.298, 1125.92, 35.4414],
                                        [-255.337, 1126.31, 35.4414],
                                        [-255.142, 1125.33, 35.4414] ] )#upper left

        if year == 2023:
            pinholes[0]['points'] = np.array([[-192.96, 1137.7, 35.4992],  
                                            [-193.155, 1138.68, 35.4992], 
                                            [-195.117, 1138.29, 35.4992],
                                            [-194.922, 1137.31, 35.4992]] ) 


            pinholes[1]['points'] = np.array([[-257.104, 1124.94, 35.5002],
                                            [-257.299, 1125.92, 35.5002],
                                            [-255.337, 1126.31, 35.5002],
                                            [-255.142, 1125.33, 35.5002] ] )

    else:    
        ### 0.8mm collimator

        pinholes[0]['pinholeKind'] =1
        pinholes[0]['pinholeCentre'] = None
        pinholes[0]['pinholeRadius'] = None
        pinholes[0]['points'] = np.array([[-192.96, 1137.7, 35.4414],  #Important, the vector p1 to p2 should be one dimension, and 
                                        [-193.155, 1138.68, 35.4414],  # the vector p2 to p3 the other dimension of the slit
                                        [-195.117, 1138.29, 35.4414],
                                        [-194.921, 1137.31, 35.4414]] ) #upper left, looking from camera to FILD head

        pinholes[1]['pinholeKind'] =1
        pinholes[1]['pinholeCentre'] = None
        pinholes[1]['pinholeRadius'] = None
        pinholes[1]['points'] = np.array([[-257.103, 1124.94, 35.4414],
                                        [-257.259, 1125.73, 35.4414],
                                        [-255.298, 1126.12, 35.4414],
                                        [-255.142, 1125.33, 35.4414] ] )#upper right


        if year == 2023:
            pinholes[3]['pinholeKind'] =1
            pinholes[3]['pinholeCentre'] = None
            pinholes[3]['pinholeRadius'] = None
            pinholes[3]['points'] = np.array([[-192.961, 1137.7, -10.5008],
                                            [-193.117, 1138.49, -10.5008],
                                            [-195.079, 1138.1, -10.5007],
                                            [-194.922, 1137.31, -10.5007]] )#lower left, looking from plasma to FILD head

            pinholes[2]['pinholeKind'] =1
            pinholes[2]['pinholeCentre'] = None
            pinholes[2]['pinholeRadius'] = None
            pinholes[2]['points'] = np.array([[-257.104, 1124.94, -10.4998],
                                            [-257.26, 1125.73, -10.4998],
                                            [-255.299, 1126.12, -10.4998],
                                            [-255.143, 1125.33, -10.4998] ] )#lower right          

    ###Magnetic Field
    use_1D_Bfield = True  #Use the magnetic field at the slit
    use_2D_Bfield = not use_1D_Bfield
    if use_1D_Bfield:
        xyzPin = np.mean(pinholes[idx_slit]['points'], axis = 0)   #get magnetic field for specific slit
        xyzPin *=0.001 #convert to [m]
        Rpin = np.sqrt(xyzPin[0]**2 + xyzPin[1]**2)
        zPin = xyzPin[2]
        
        #Br, Bz, Bt, bp =  TCV_equilibrium.get_mag_field(shot, Rpin + Rinsertion*0.001, zPin, time)#, use_gdat = True)
        #print('LIUQE: ',Br, Bz, Bt, bp, np.sqrt(Br**2 + Bz**2 + Bt**2) )
        Br, Bz, Bt, bp =  TCV_equilibrium.get_mag_field(shot, Rpin + Rinsertion*0.001, zPin, time, use_gdat = True)
        print('gdat: ',Br, Bz, Bt, bp, np.sqrt(Br**2 + Bz**2 + Bt**2)  )

        modB = np.sqrt(Br**2 + Bz**2 + Bt**2) 
    elif  use_2D_Bfield:
        rmin = 0.614  #[m]
        rmax = 1.147
        nr = 129#28
        zmin = -0.76
        zmax = 0.76
        nz = 129#65

        r_grid = np.linspace(rmin, rmax, nr, dtype=np.float32)
        z_grid = np.linspace(zmin, zmax, nz, dtype=np.float32)
        r_mesh, z_mesh = np.meshgrid(r_grid, z_grid, indexing = 'ij')

        Br, Bz, Bt = TCV_equilibrium.get_mag_field(shot, Rpin + Rinsertion*0.001, zPin, time)#, use_gdat = True)
        
        modB = np.array([1.12817047]) #np.mean( np.sqrt(Br**2 + Bz**2 + Bt**2) ) #this has to be manual :(


    ###
    # Marker inputs
    ###
    #Number of markers per pitch-gyroradius pair
    n_markers = int(1e4)
    # Set n1 and r1 are paremeters for adjusteing # markers per gyroradius. If zero number markers is uniform
    n1 = 0.0
    r1 = 0.0
    #Grids

    energy_arrays = np.arange(3000, 65000, 20000)
    #Gyroradii grid in [cm]
    g_r = ss.SimulationCodes.FILDSIM.execution.get_gyroradius(energy_arrays, modB)
    #g_r = np.asarray([0.88,1.08, 1.2567])
    #Alternatively use cm grid
    #g_r = np.arange(0.5, 5, 0.2)

    gyro_arrays = [list(np.around(g_r, decimals = 5)), #For each indivudual slit, ul->lr 
                list(np.around(g_r, decimals = 5)),
                list(np.around(g_r, decimals = 5)),
                list(np.around(g_r, decimals = 5))]
    #pitch angle grid in [degrees]
    #JP: Added rounded on p, because otherwise not correct the arccos (too many zeros gives NaN)
    p = np.around(np.arange(0.0, 1, 0.2), decimals = 5)
    pitch_arrays = [list(np.around(np.rad2deg(np.arccos(p)), decimals = 5)), 
                    list(np.around(np.rad2deg(np.arccos(-p)), decimals = 5)),
                    list(np.around(np.rad2deg(np.arccos(-p)), decimals = 5)),
                    list(np.around(np.rad2deg(np.arccos(p)), decimals = 5))
                    ]

    #alternatively use degree grid
    #p = np.arange(0.00, 90, 4)
    #pitch_arrays = [ list(p),
    #                list(180-p),
    #                list(p),
    #                list(180-p)
    #                ]

    #Range of gyrophase to use. Smaller range can be used, but for now allow all gyrophases
    gyrophase_range = [[np.deg2rad(185),np.deg2rad(359)],  #UL  [np.deg2rad(185),np.deg2rad(359)]
                        [np.deg2rad(70),np.deg2rad(160)],  #UR [np.deg2rad(22),np.deg2rad(182)]
                        [np.deg2rad(200),np.deg2rad(350)], #LR [np.deg2rad(185),np.deg2rad(359)]
                        [np.deg2rad(1),np.deg2rad(189)]  #LL [np.deg2rad(1),np.deg2rad(189)]
                        ]        

    # -----------------------------------------------------------------------------
    # --- Run SINPA FILDSIM
    # -----------------------------------------------------------------------------
               
    if run_code:
        # prepare namelist
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
                'IpBt': -1,        # Sign of toroidal current vs field (for pitch), need to check
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
        
        # write geometry files
        for i in range(4):
            if run_slit[i]:                   
                # prepare magnetic field
                ###
                # Magnetic field input
                ###
                field = ss.simcom.Fields()
                if use_1D_Bfield:
                    field.createFromSingleB(B = np.array([Br, Bz, Bt]), 
                                    Rmin = Rpin - 0.25,
                                    Rmax = Rpin + 0.25,
                                    zmin = zPin -0.25, zmax = zPin + 0.25,
                                    nR = 10, nz = 10)

                elif use_2D_Bfield:
                    Br = Br.astype(dtype=np.float64)
                    Bphi = Bt.astype(dtype=np.float64)
                    Bz = Bz.astype(dtype=np.float64)

                    field.Bfield['R'] = np.float64(r_grid - Rinsertion*0.001)
                    field.Bfield['z'] = np.float64(z_grid)   
                    field.Bfield['Rmin'] = np.float64(np.min(field.Bfield['R']))
                    field.Bfield['Rmax'] = np.float64(np.max(field.Bfield['R']))
                    field.Bfield['zmin'] = np.float64(np.min(field.Bfield['z']))
                    field.Bfield['zmax'] = np.float64(np.max(field.Bfield['z']))
                    field.Bfield['nR'] = np.int32(len(field.Bfield['R']))
                    field.Bfield['nz'] = np.int32(len(field.Bfield['z']))

                    field.Bfield['fr'] = np.asfortranarray(Br)
                    field.Bfield['fz'] = np.asfortranarray(Bz)
                    field.Bfield['ft'] = np.asfortranarray(Bphi)
       
                    #field.Bfield['nTime'] = np.int32(0)
                    #field.Bfield['Timemin'] = np.float64(0.)
                    #field.Bfield['Timemax'] = np.float64(0.)
                    
                    field.bdims = 2
                    field.plot('br', phiSlice = 0 ,plot_vessel = False)
                    plt.show()

                write_stl_geometry_files(root_dir = geom_dir,
                                        run_name = run_names[i],
                                        collimator_stl_files = collimator_stl_files,
                                        scintillator_stl_files = scintillator_stl_files,
                                        ps = ps,
                                        u1_scint= u1_scint,
                                        scint_norm= scint_norm,
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
                    nml_options['inputParams']['minAngle'] = gyrophase_range[i][0]
                    nml_options['inputParams']['dAngle'] = (gyrophase_range[i][1]
                                                        - gyrophase_range[i][0])
            
                    #Make field

                    fieldFileName = os.path.join(inputsDir, 'field.bin')
                    print('Writing new B-field file!!!!!')
                    fid = open(fieldFileName, 'wb')
                    field.tofile(fid)
                    fid.close()

                    # Create namelist
                    ss.sinpa.execution.write_namelist(nml_options )
                
                    # Missing a step: create B field!!
                    # Check the files
                    ss.sinpa.execution.check_files(nml_options['config']['runid'])
                    # Launch the simulations
                    ss.sinpa.execution.executeRun(nml_options['config']['runid'], queue = False)
                
        
            if plot_plate_geometry:
                if run_slit[i]:
                    geomID = run_names[i]
                    Geometry = ss.simcom.Geometry(GeomID=geomID)
                    if plot_3D:
                        Geometry.plot3Dfilled(element_to_plot = [0,2], plot_pinhole = False)
                    #Geometry.plot2Dfilled(view = 'XY', element_to_plot = [0,2], plot_pinhole = False)
                    Geometry.plot2Dfilled(view = 'Scint', element_to_plot = [0,2], plot_pinhole = False)
                    plt.show()


    # -----------------------------------------------------------------------------
    # --- Section 2: Analyse the results
    # -----------------------------------------------------------------------------
    ###
    #Output plotting flags
    ###
    # plot outputs
    plot_strike_points = False
    plot_strikemap = True
    # strike plotting settings
    marker_params = [{'markersize':6, 'marker':'o','color':'b'},
                     {'markersize':6, 'marker':'o','color':'m'},
                     {'markersize':6, 'marker':'o','color':'m'},
                     {'markersize':6, 'marker':'o','color':'b'}]
    line_params =  [{'ls':'solid','color':'k'},
                    {'ls':'solid','color':'w'},
                    {'ls':'solid','color':'r'},
                    {'ls':'solid','color':'b'}]
    mar_params = [{'zorder':3,'color':'k'},
                  {'zorder':3,'color':'w'},
                  {'zorder':3,'color':'r'},
                  {'zorder':3,'color':'b'}]

    plot_orbits = True
    orbit_kind=(2,) # 2 colliding w. scint., 0 colliding w. coll., 9 missing all, 3 scint. markers traced backwards
    plot_self_shadowing_collimator_strike_points = False
    # Save data to txt files for ParaView inspection
    save_strike_points_txt = False
    save_strikemap_txt = False
    save_orb_txt = True  # Save orbit data to .txt file
    seperated = False      # If this flag is true, make separate txt file for each orbit, otherwise one file is written
    save_self_shadowing_collimator_strike_points = False #Flag to see where self shadowing happens
    save_self_shadowing_collimator_strike_points_txt = False
    # plot some metrics
    plot_metrics = True
    plot_resolutions = True
    plot_gyro_res= False
    plot_pitch_res= False
    plot_collimating_factor = True
    plot_synthetic_signal= False
    plot_camera_signal=False


    if read_results:

        runid = run_names
        strike_points_file = ['','','','']
        Smap = [[],[],[],[]]
        p0 = [75, 115]
        
        for i in range(4):
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
                
            for i in range(4):
                if read_slit[i]:# or mixnmatch:

                    Geometry = ss.simcom.Geometry(GeomID=runid[i])

                        
                    Geometry.plot2Dfilled(ax=ax, view = 'Scint', element_to_plot = [2],
                                            plot_pinhole = False)

                    print('HOLAAA!')
                    if plot_strike_points:
                        #IPython.embed()rundid=
                        Smap[i].strike_points.scatter(ax=ax, per=0.2
                                                        , xscale=100.0, yscale=100.0
                                                        ,mar_params = mar_params[i]
                                                        , varx='x1', vary='x2')
                        
                    if plot_strikemap:
                        Smap[i].plot_real(ax=ax, marker_params=marker_params[i],
                                            line_params=line_params[i], factor=100., labels=True)
                            
                    if plot_orbits:
                        orb.append(ss.sinpa.orbits(runID=runid[i]))
                        #orb[i].plot2D(ax=ax,line_params={'color': 'r'}, kind=(2,),factor=100.0)
                        #orb[i].plot2D(ax=ax,line_params={'color': 'b'}, kind=(0,),factor=100.0)
                        #orb[i].plot2D(ax=ax,line_params={'color': 'k'}, kind=(9,),factor=100.0)
                        if save_orb_txt:
                            orb[i].save_orbits_to_txt( kind=orbit_kind, units = 'mm', seperated = seperated)
                            print('I arrived here!!!!')
                else:
                    orb.append([])
                        
            fig.show()
                
            if plot_3D:
                Geometry.plot3Dfilled(element_to_plot = [0, 2], units = 'm')
                ax3D = plt.gca()
                
                
                for i in range(4):
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
                                orb[i].save_orbits_to_txt( kind=orbit_kind, units = 'mm')#, seperated = seperated)



    plt.show()

                   
                        
