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
    plot_strike_points = True
    plot_strikemap = True
    
    marker_params = {'markersize':6, 'marker':'o','color':'b'}
    line_params =  {'ls':'solid','color':'k'}
    # -----------------------------------------------------------------------------
    namelist_path = ss.paths.FILDSIM   # Paths to save namelist
    # -----------------------------------------------------------------------------
    # Section 0: FILDSIM Settings
    # -----------------------------------------------------------------------------
    
    FILDSIM_namelist = {
        'config': {
            'runid': 'resolution_example',
            'result_dir': ss.paths.FILDSIM + '/results/',
            'backtrace': False,
            'n_gyroradius': 5,
            'n_pitch': 10,
            'save_orbits': 0,
            'verbose': True,
        },
        'input_parameters': {
            'n_ions': int(1e5),
            'step': 0.01,
            'helix_length': 10.0,
            'gyroradius': [2.0, 3.0, 4.0, 5.0, 6.0],
            'pitch_angle': [85.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10, 0.0],
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
    #ss.fildsim.run_FILDSIM(namelist_name)
    
    # Load the result of the simulation

    base_name = FILDSIM_namelist['config']['result_dir'] + FILDSIM_namelist['config']['runid']
    #orbits_file = base_name + '_orbits.dat'
    #orbits_index_file = base_name + '_orbits_index.dat'
    
    strike_map_file = base_name + '_strike_map.dat'
    strike_points_file = base_name + '_strike_points.dat'
    # Load the strike map
    Smap = ss.mapping.StrikeMap('FILD', strike_map_file)
    # Load the strike points used to calculate the map
    Smap.load_strike_points(strike_points_file)
    
    
    Smap.calculate_resolutions()   # Default call,
    # Example changing the binning and settings skewGaussian for boths
    # Smap.calculate_resolutions(diag_params = {'dpitch':2.0, 'dgyr':0.25,
    #                                           'p_method': 'sGauss'},
    #                            g_method='sGauss')
    # Example changing the minimum of markers needed to consider making the fit,
    # default is 100
    # Smap.calculate_resolutions(diag_params = {'dpitch':2.0, 'dgyr':0.25,
    #                                           'p_method': 'sGauss'},
    #                            g_method='sGauss', min_statistics=500)
    Smap.plot_gyroradius_histograms(pitch = 50,
                                    plot_fit = True,
                                    axarr=None, dpi=100,alpha=0.5)
    
    Smap.plot_pitch_histograms(gyroradius = 5,
                                plot_fit = True,
                                axarr=None, dpi=100,alpha=0.5)
    
    Smap.plot_resolutions(ax_param={'dpi':100})
    plt.gcf().show()
    
    
    Smap.plot_collimator_factor(ax_param={'dpi':100})
    plt.gcf().show()