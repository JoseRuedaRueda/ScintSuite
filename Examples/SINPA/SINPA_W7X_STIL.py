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
import open3d

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
                              , scan_name = ''
                              , stl_files = {}
                              , pinhole = []
                              ):
    '''
    Parameters
    ----------


    Returns
    -------
    None.
    '''
    scan_folder = scan_name + '/'
    directory = os.path.join(root_dir, scan_folder)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Making directory')

    element_nr = 1
    
    
    # Gather collimator triangles
    if 'collimator' in stl_files.keys():
        collimator_filename = directory + 'Element%i.txt'%element_nr
        element_nr += 1
        f = open(collimator_filename, 'w')
        f.write('Collimator file for SINPA FILDSIM\n')
        f.write('Scan name is ' + scan_name + '\n')
        f.write('STL collimator \n')
        f.write('0  ! Kind of plate\n')
        f.close()
        ss.LibCAD.write_file_for_fortran(stl_files['collimator'], collimator_filename, 
                                         convert_mm_2_m = True)      

    scint_norm = [1., 0., 0.] 
    ps = np.zeros(3)
    rot = np.identity(3)
    # Dummy scintillator normal vector,reference point and rottion vector
    #, in case we don't include a scintillator in the run
    #
    # Write scintillator to file
    
    if 'scintillator' in stl_files.keys():
        scint_filename = directory + 'Element%i.txt'%element_nr
        
        ##write geometory file "header"
        f = open(scint_filename, 'w')
        f.write('Scintillator file for SINPA FILDSIM\n')
        f.write('Scintilator stl file: ' + stl_files['scintillator'] + '\n')
        f.write('File by '+ os.getenv('USER') + ' \n')
        f.write('2  ! Kind of plate\n')
        f.close()
        # Append triangle data from stl file
        ss.LibCAD.write_file_for_fortran(stl_files['scintillator'], 
                                         scint_filename, 
                                         convert_mm_2_m = True) 
        
        # --- Open and load the stil file
        a = open3d.io.read_triangle_mesh(stl_files['scintillator'])
        vertices = np.asarray(a.vertices)
        index = np.asarray(a.triangles)
        
        # Get the points of the first triangle in the scintillator stl file
        itriang = 0#10
        j=0
        p1 = np.array((vertices[index[itriang, j], 0],
                       vertices[index[itriang, j], 1],
                       vertices[index[itriang, j], 2]) ) * 0.001 #convert mm to m
        j=1
        p2 = np.array((vertices[index[itriang, j], 0],
                       vertices[index[itriang, j], 1],
                       vertices[index[itriang, j], 2]) ) * 0.001 #convert mm to m
        j=2
        p3 = np.array((vertices[index[itriang, j], 0],
                       vertices[index[itriang, j], 1],
                       vertices[index[itriang, j], 2]) ) * 0.001 #convert mm to m
        

        scint_norm = -get_normal_vector(p1, p2, p3)
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
    nGeomElements = len(stl_files.keys())
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
    geom_name = 'W7X_stl_QHS' # Only used if running a single iteration
    
    save_orbits = False
    plot_plate_geometry = True
    plot_3D = True
    
    read_results = not run_code #True
    plot_strike_points = False
    plot_strikemap = True
    plot_orbits = False
    
    backtrace = False
    
    if save_orbits:
        nGyro = 36
        maxT = 0.00000006  * 10
    else:
        nGyro = 350
        maxT = 0.00000006 
        
    plot_metrics = False
    
    #for best case: choose the scan parameter and value you want to look at!
    plot_resolutions = False
    plot_gyro_res= False
    plot_pitch_res= False
    plot_collimating_factor = False
    plot_synthetic_signal= False
                            
    
    marker_params = {'markersize':6, 'marker':'o','color':'b'}
    line_params =  {'ls':'solid','color':'k'}
    mar_params = {'zorder':3,'color':'k'}
    

    n_markers = int(1e4)

    gyro_array = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2., 3., 4.]

    pitch_array = [85., 75., 65., 55., 45., 35., 25., 15., 5.]
    #pitch_array = [95., 105., 115., 125., 135., 145., 155., 165., 175.]
    #pitch_arrays = [[160., 140., 120., 100., -80., -60., -40., -20.],
    #                [160., 140., 120., 100., 80., 60., 40., 20.]]
    gyrophase_range = np.array([np.deg2rad(90),np.deg2rad(270)])

    # Set n1 and r1 for the namelist, 0 and 0 are defaults, setting to 0.02 and 0.4 gives ~300,000 particles for rl = 0.25 
    # and ~400,000 for 0.5
    n1 = 0.0
    r1 = 0.0
    
    Br, Bz, Bt = -0.430942, -0.463285,	2.201483	   #STD
    #Br, Bz, Bt =  -0.37123,	-0.381688, 1.940142    #KJM
    #Br, Bz, Bt =  -0.456466	, -0.472407, 2.094869  #FTM
    
    modB = np.sqrt(Br**2 + Bz**2 + Bt**2)
    
    alpha = 0.16  #From J. Hidalgo
    beta = 215. #180
    
    use_aligned_B = False    
    use_rotated_FILD_B = False
    use_ascot_B = True
    ascot_bfield_File = 'std_bfield.pickle'
    ascot_boozer_File = 'std_boozer.pickle'
    
    ##STL files
    #stl_files = {'collimator': 'graphite_cup_QHF.stl',
    #             'scintillator': 'sensor_QHF.stl'}
    pinhole = {}
    pinhole['pinholeKind'] = 1
    pinhole['pinholeCentre'] = None
    pinhole['points'] = np.array([[-604.214, 5737.866, 737.998],
                                   [-599.764, 5740.079, 738.542],
                                   [-612.243, 5759.764, 714.652],
                                   [-607.793, 5761.976, 715.197] ] )
    
    stl_files = {'collimator': 'BOTTON_HEAD_WITH_HOLES.stl',
                 'scintillator': 'SCINTILLATOR_PLATE.stl'}

    pinhole['pinholeCentre'] = np.array([-738.196, 5807.961, 350.292 ] ) #co- going slit opening
    pinhole['pinholeCentre'] = np.array([-780.393, 5798.132, 343.755] ) #counter going slit opening
    pinhole['pinholeKind'] = 0
    pinhole['pinholeRadius'] = 2.5
    pinhole['points'] = np.array([[-812.676, 5778.52, 315.535],
                                   [-788.217, 5812.3, 373.24],
                                   [-726.769, 5798.58, 328.944],
                                   [-702.375, 5832.35, 386.639] ] )

    ##STL files
    stl_files = {'collimator': 'probe_head_with_pinholes.stl',
                 'scintillator':  'SCINTILLATOR_PLATE_test.stl'}
    pinhole = {}
    pinhole['pinholeKind'] =1
    pinhole['pinholeCentre'] = None
    pinhole['pinholeRadius'] = None
    pinhole['points'] = np.array([[-782.879, 5789.41, 344.934],
                                   [-783.033, 5789.71, 345.661],
                                   [-781.608, 5790.16, 345.778],
                                   [-781.454, 5789.86, 345.051] ] )#counter going slit opening


    pinhole['points'] = np.array([[-725.674, 5781.91, 381.177],
                                   [-725.828, 5782.21, 381.904],
                                   [-724.403, 5782.66, 382.021],
                                   [-724.249, 5782.36, 381.294] ] )#co- going slit opening

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
        
        #Create magnetic field
        field = ss.simcom.Fields()
                
        if use_aligned_B:
            # This is negative so the particles will gyrate the way we want
            direction = np.array([0, -1, 0])
            Field = modB * direction
            field.createHomogeneousField(Field, field='B')
            IPython.embed()
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
        #for i in range(2):
        #if run_slit[i]:
        write_stl_geometry_files(root_dir = geom_dir,
                              scan_name = geom_name,
                              stl_files = stl_files,
                              pinhole = pinhole)            
            
        if not Test:
                    # Create directories
                    runDir = os.path.join(paths.SINPA, 'runs', geom_name)
                    inputsDir = os.path.join(runDir, 'inputs')
                    resultsDir = os.path.join(runDir, 'results')
                    os.makedirs(runDir, exist_ok=True)
                    os.makedirs(inputsDir, exist_ok=True)
                    os.makedirs(resultsDir, exist_ok=True)
                    
                    # Set namelist parameters
                    nml_options['config']['runid'] = geom_name
                    nml_options['config']['geomfolder'] = geom_dir + '/' + geom_name
                    nml_options['config']['runfolder'] = runDir
                    nml_options['config']['nxi'] = len(pitch_array)
                    nml_options['config']['nGyroradius'] = len(gyro_array)
                    nml_options['inputParams']['XI'] = pitch_array
                    nml_options['inputParams']['rL'] = gyro_array
                    nml_options['inputParams']['minAngle'] = gyrophase_range[0]
                    nml_options['inputParams']['dAngle'] = (gyrophase_range[1]
                                                              - gyrophase_range[0])
                                            
                    #Make field
                    # fieldFileName = os.path.join(inputsDir, 'field.bin')
                    # fid = open(fieldFileName, 'wb')
                    # field.tofile(fid)
                    # fid.close()

                    # Create namelist
                    ss.sinpa.execution.write_namelist(nml_options)
                    
                    # Missing a step: create B field!!
                    # Check the files
                    ss.sinpa.execution.check_files(nml_options['config']['runid'])
                    # Launch the simulations
                    ss.sinpa.execution.executeRun(nml_options['config']['runid'], queue = True)
                    
        if plot_plate_geometry:
            geomID = geom_name
            Geometry = ss.simcom.Geometry(GeomID=geomID)
            if plot_3D:
                Geometry.plot3Dfilled(element_to_plot = [0,2])
            # Geometry.plot2Dfilled(view = 'XY', element_to_plot = [0,2], 
            #                       plot_pinhole = False, surface_params = {'alpha': 1.0})
            Geometry.plot2Dfilled(view = 'Scint', 
                                  referenceSystem = 'scintillator',
                                  element_to_plot = [0,2], plot_pinhole = True)


    # -----------------------------------------------------------------------------
    # --- Section 2: Analyse the results
    # -----------------------------------------------------------------------------
    if read_results:
        runid = geom_name#[geom_name,geom_name + '_right']
        strike_points_file = ['','']
        Smap = [[],[]]
        p0 = [75, 120]
        
        #for i in range(2):
        #if read_slit[i] or mixnmatch:
        runDir = os.path.join(paths.SINPA, 'runs', runid)
        inputsDir = os.path.join(runDir, 'inputs/')
        resultsDir = os.path.join(runDir, 'results/')
        base_name = resultsDir + runid#[i]
        smap_name = base_name + '.map'
        
        # Load the strike map
        Smap = ss.mapping.StrikeMap('FILD', file=smap_name)
        try:
            Smap.load_strike_points()
        except:
            print('Strike map  could not be loaded')
            #continue

                
        if plot_plate_geometry:
            fig, ax = plt.subplots()
            orb = []
                
            ax.set_xlabel('Y [cm]')
            ax.set_ylabel('Z [cm]')
            ax.set_title('Camera view (YZ plane)')
            
            # If you want to mix and match, it assumes they have the same scintillator
            Geometry = ss.simcom.Geometry(GeomID=runid)

                    
            Geometry.plot2Dfilled(ax=ax, view = 'scint', element_to_plot = [2],
                                      plot_pinhole = False,referenceSystem = 'scintillator')
                
            #for i in range(2):
            #if read_slit[i] or mixnmatch:
            if plot_strike_points:
                Smap.strike_points.scatter(ax=ax, per=0.1, xscale=100.0, yscale=100.0,
                                              mar_params = mar_params,
                                              varx='ys', vary='zs')
                    
            if plot_strikemap:
                Smap.plot_real(ax=ax, marker_params=marker_params,
                                  line_params=line_params, factor=100.0, labels=True)
                    
            if plot_orbits:
                orb.append(ss.sinpa.orbits(runID=runid))
                #orb[0].plot2D(ax=ax,line_params={'color': 'r'}, kind=(2,),factor=100.0)
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
                    Smap.strike_points.plot3D(ax=ax3D)
                
                #for i in range(2):
                #if read_slit or mixnmatch:
                # plot in red the ones colliding with the scintillator
                if plot_orbits:
                    orb[0].plot3D(line_params={'color': 'r'}, kind=(2,), ax=ax3D, per = 1.)
                    # plot in blue the ones colliding with the collimator
                    orb[0].plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax3D, per = 1.)
                    # plot the 'wrong orbits', to be sure that the are just orbits exceeding
                    # the scintillator and nothing is wrong with the code
                    orb[0].plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax3D, per = 1.)

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
                        
                        
                        
                        
                        
                        
                        
                        
