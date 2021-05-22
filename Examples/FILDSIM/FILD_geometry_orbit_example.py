"""
Created on Sat May  8 21:00:25 2021

@author: ajvv

Run FILDSIM and calculate resolutions

"""

import Lib as ss
import numpy as np
import os
import IPython
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    #set plotting settings
    ss.LibPlotting.plotSettings(plot_mode='Presentation')
    
    # -----------------------------------------------------------------------------
    namelist_path = ss.paths.FILDSIM   # Paths to save namelist
    # -----------------------------------------------------------------------------
    # Section 0: FILDSIM Settings
    # -----------------------------------------------------------------------------
    
    FILDSIM_namelist = {
        'config': {
            'runid': 'orbit_geometry_exaple',
            'result_dir': ss.paths.FILDSIM + '/results/',
            'backtrace': False,
            'n_gyroradius': 3,
            'n_pitch': 4,
            'save_orbits': 1,
            'verbose': True,
        },
        'input_parameters': {
            'n_ions': 100,
            'step': 0.01,
            'helix_length': 10.0,
            'gyroradius': [2., 4., 8],
            'pitch_angle': [70., 50., 30., 10, ],
            'gyrophase_range': [1.0, 1.8],
            'start_x': [-0.025, 0.025],
            'start_y': [-0.1, 0.1],
            'start_z': [0.0, 0.0],
            'theta': 0.0,
            'phi': 0.0
        },
        'plate_setup_cfg': {
            'geometry_dir': ss.paths.FILDSIM + './geometry/AUG/',
            'n_scintillator': 1,
            'n_slits': 3#6
        },
        'plate_files': {
            'scintillator_files': ['aug_fild1_scint.pl'],
            'slit_files': [#'aug_fild1_pinhole_1_v2.pl',
                           #'aug_fild1_pinhole_2_v2.pl',
                           #'aug_fild1_slit_1_v2.pl',
                           'aug_fild1_slit_back_v2.pl',
                           'aug_fild1_slit_lateral_1_v2.pl',
                           'aug_fild1_slit_lateral_2_v2.pl']
        }
    }
    
    # Write namelist
    ss.fildsim.write_namelist(FILDSIM_namelist, namelist_path)
    namelist_name = os.path.join(namelist_path,
                                 FILDSIM_namelist['config']['runid'] + '.cfg')
    # Run FILDSIM
    ss.fildsim.run_FILDSIM(namelist_name)
    
    # Load the result of the simulation

    base_name = FILDSIM_namelist['config']['result_dir'] + FILDSIM_namelist['config']['runid']
    orbits_file = base_name + '_orbits.dat'
    orbits_index_file = base_name + '_orbits_index.dat'

    fig = plt.figure(figsize=(6, 10), facecolor='w', edgecolor='k',
                     dpi=100)
    ax3D = fig.add_subplot(111, projection='3d')
    ax3D.set_xlabel('X [cm]')
    ax3D.set_ylabel('Y [cm]')
    ax3D.set_zlabel('Z [cm]')
    

    fig2, axarr = plt.subplots(nrows=1, ncols=3, figsize=(18, 10),
                               facecolor='w', edgecolor='k', dpi=100)
    ax2D_xy = axarr[0]  # topdown view, i.e should see pinhole surface
    ax2D_xy.set_xlabel('X [cm]')
    ax2D_xy.set_ylabel('Y [cm]')
    ax2D_xy.set_title('Top down view (X-Y plane)')
    ax2D_yz = axarr[1]  # front view, i.e. should see scintilator plate
    ax2D_yz.set_xlabel('Y [cm]')
    ax2D_yz.set_ylabel('Z [cm]')
    ax2D_yz.set_title('Front view (Y-Z plane)')
    ax2D_xz = axarr[2]  # side view, i.e. should see slit plate surface(s)
    ax2D_xz.set_xlabel('X [cm]')
    ax2D_xz.set_ylabel('Z [cm]')
    ax2D_xz.set_title('Side view (Y-Z plane)')
    
    #Plot geometry of input FILD plates
    ss.LibFILDSIM.FILDSIMplot.plot_geometry(namelist_name, 
                                            ax3D =ax3D, axarr=axarr,
                                            plates_all_black=True)
    # plot geometry of calculated orbits
    ss.LibFILDSIM.FILDSIMplot.plot_orbits(orbits_file, orbits_index_file, 
                                          ax3D =ax3D , axarr=axarr)
    
    plt.show()