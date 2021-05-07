"""Plot FILDSIM geometry and results from FILDSIM.f90"""
import Lib.LibFILDSIM.FILDSIMexecution as ssfildsimA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interpolate


def plot_geometry(filename, ax3D=None, axarr=None, dpi=100):
    '''
    Plot the input FILDSIM geometry, namely the slits and scintillator

    ajvv: avanvuuren@us.es

    ----------
    @param filename: FILDSIM configuration file, eg. /path/to/cfg_example.cfg
    @param ax3D: 3D axis where to plot the geometry, if None, new will be made
    @param axarr: array of axis to plot projections, if None, new will be made
    @param dpi: dpi to render the figures, only used if the axis are created
    by this function
    '''
    namelist = ssfildsimA.read_namelist(filename)
    geometry_dir = namelist['plate_setup_cfg']['geometry_dir']
    scintillator_files = namelist['plate_files']['scintillator_files']
    n_scintillator = namelist['plate_setup_cfg']['n_scintillator']
    slit_files = namelist['plate_files']['slit_files']
    n_slits = namelist['plate_setup_cfg']['n_slits']

    if n_scintillator == 1:
        scintillator_files = [scintillator_files]
    if n_slits == 1:
        slit_files = [slit_files]

    scintillator_plates = []
    # --- read scintillator files
    for scint_file in scintillator_files:
        file = geometry_dir + scint_file
        scintillator_plates.append(ssfildsimA.read_plate(file))

    slit_plates = []
    # --- read slit files
    for slit_file in slit_files:
        file = geometry_dir + slit_file
        slit_plates.append(ssfildsimA.read_plate(file))

    # --- Open the figure
    if ax3D is None:
        fig = plt.figure(figsize=(6, 10), facecolor='w', edgecolor='k',
                         dpi=dpi)
        ax3D = fig.add_subplot(111, projection='3d')
        ax3D.set_xlabel('X [cm]')
        ax3D.set_ylabel('Y [cm]')
        ax3D.set_zlabel('Z [cm]')
        created_3D = True
    else:
        created_3D = False
    if axarr is None:
        fig2, axarr = plt.subplots(nrows=1, ncols=3, figsize=(18, 10),
                                   facecolor='w', edgecolor='k', dpi=dpi)
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
        created_2D = True
    else:
        ax2D_xy = axarr[0]  # topdown view, i.e should see pinhole surface
        ax2D_yz = axarr[1]  # front view, i.e. should see scintilator plate
        ax2D_xz = axarr[2]  # side view, i.e. should see slit plate surface(s)
        created_2D = False
    # --- Plot scintilator plates
    for scintillator_plate in scintillator_plates:
        x = scintillator_plate['vertices'][:, 0]
        y = scintillator_plate['vertices'][:, 1]
        z = scintillator_plate['vertices'][:, 2]
        # the scintilator should be in the y-z plane, by definition of
        # FILDSIM
        grid_y, grid_z = np.meshgrid(np.linspace(min(y), max(y), num=50),
                                     np.linspace(min(z), max(z), num=50))
        grid_x = interpolate.griddata((y, z), x, (grid_y, grid_z))
        # 3D view
        ax3D.plot_surface(grid_x, grid_y, grid_z, color='green', alpha=1.,
                          label=scintillator_plate['name'])
        # 2D view
        # planer view
        ax2D_yz.plot(np.append(y, y[0]), np.append(z, z[0]), ls='solid',
                     marker='', color='green', label=scintillator_plate['name'])
        ax2D_yz.fill(np.append(y, y[0]), np.append(z, z[0]), color='green',
                     alpha=0.8)

        # line views
        ax2D_xy.plot(x, y, ls='solid', marker='', color='green',
                     label=scintillator_plate['name'])
        ax2D_xz.plot(x, z, ls='solid', marker='', color='green',
                     label=scintillator_plate['name'])

    # -- plot slit plates
    for slit_plate in slit_plates:
        x = slit_plate['vertices'][:, 0]
        y = slit_plate['vertices'][:, 1]
        z = slit_plate['vertices'][:, 2]

        slit_line = ax2D_xy.plot(np.append(x, x[0]), np.append(y, y[0]),
                                 ls='solid', marker='',
                                 label=slit_plate['name'])
        ax2D_yz.plot(np.append(y, y[0]), np.append(z, z[0]),
                     color=slit_line[0].get_color(), ls='solid', marker='',
                     label=slit_plate['name'])
        ax2D_xz.plot(np.append(x, x[0]), np.append(z, z[0]),
                     color=slit_line[0].get_color(), ls='solid', marker='',
                     label=slit_plate['name'])

        # slit plates may be in different planes, therefore check first,
        # otherwise method fails
        if min(x) == max(x):
            grid_y, grid_z = np.meshgrid(np.linspace(min(y), max(y), num=50),
                                         np.linspace(min(z), max(z), num=50))
            grid_x = interpolate.griddata((y, z), x, (grid_y, grid_z))
            ax2D_yz.fill(np.append(y, y[0]), np.append(z, z[0]),
                         color=slit_line[0].get_color(), alpha=0.8)
        elif min(y) == max(y):
            grid_x, grid_z = np.meshgrid(np.linspace(min(x), max(x), num=50),
                                         np.linspace(min(z), max(z), num=50))
            grid_y = interpolate.griddata((x, z), y, (grid_x, grid_z))
            ax2D_xz.fill(np.append(x, x[0]), np.append(z, z[0]),
                         color=slit_line[0].get_color(), alpha=0.8)

        elif min(z) == max(z):
            grid_x, grid_y = np.meshgrid(np.linspace(min(x), max(x), num=50),
                                         np.linspace(min(y), max(y), num=50))
            grid_z = interpolate.griddata((x, y), z, (grid_x, grid_y))
            ax2D_xy.fill(np.append(x, x[0]), np.append(y, y[0]),
                         color=slit_line[0].get_color(), alpha=0.8)

        ax3D.plot_surface(grid_x, grid_y, grid_z, color=slit_line[0].get_color(), alpha=.5,
                          label=scintillator_plate['name'])
    ax2D_xz.legend(loc='best')
    if created_2D:
        fig2.tight_layout()
        fig2.show()
    if created_3D:
        fig.tight_layout()
        fig.show()
    return

def plot_orbits(orbits_file, orbits_index_file, ax3D=None, axarr=None, dpi=100,
                marker_params: dict = {}, line_params: dict = {}):
    '''
    Plot the output FILDSIM orbits

    ajvv: avanvuuren@us.es

    ----------
    @param orbits_file: full path to the orbits file. 
        eg. /path/to/runid_example_orbits.dat
    @param orbits_index_file: full path to the orbits_index file. 
        eg. /path/to/runid_example_orbits_index.dat
    @param ax3D: 3D axis where to plot the orbits, if None, new will be made
    @param axarr: array of axis to plot projections, if None, new will be made
    @param dpi: dpi to render the figures, only used if the axis are created
    by this function
    @param line_params: Parameters for plot orbit lines, 
                        Default: linestyle = 'solid' or color = 'red'
    @param marker_params: Parameters for plot orbit end points, 
                        Default: maker = 'circle' or color = 'red'
    '''
    # Default plot parameters:
    marker_options = {
        'markersize': 3,
        #'fillstyle': 'none',
        'color': 'r',
        'marker': 'o',
        'linestyle': 'none'
    }
    marker_options.update(marker_params)    
    line_options = {
                    'color': 'red',
                    'marker': ''
                    }
    line_options.update(line_params)
    
    orbits = ssfildsimA.read_orbits(orbits_file, orbits_index_file)
    
    # --- Open the figure
    created_3D = False
    if ax3D is None:
        fig = plt.figure(figsize=(6, 10), facecolor='w', edgecolor='k',
                         dpi=dpi)
        ax3D = fig.add_subplot(111, projection='3d')
        ax3D.set_xlabel('X [cm]')
        ax3D.set_ylabel('Y [cm]')
        ax3D.set_zlabel('Z [cm]')
        created_3D = True
    if axarr is None:
        fig2, axarr = plt.subplots(nrows=1, ncols=3, figsize=(18, 10),
                                   facecolor='w', edgecolor='k', dpi=dpi)
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
        created_2D = True
    else:
        ax2D_xy = axarr[0]  # topdown view, i.e should see pinhole surface
        ax2D_yz = axarr[1]  # front view, i.e. should see scintilator plate
        ax2D_xz = axarr[2]  # side view, i.e. should see slit plate surface(s)
        created_2D = False
        
    for orbit in orbits:
        xline = orbit[:, 0]
        yline = orbit[:, 1]
        zline = orbit[:, 2]
        ax3D.plot3D(xline, yline, zline, **line_options)
        
        ax2D_xy.plot(xline, yline, **line_options)
        ax2D_xy.plot(xline[0], yline[0], **marker_options)
        ax2D_xy.plot(xline[-1], yline[-1], **marker_options)
        
        ax2D_yz.plot(yline, zline, **line_options)
        ax2D_yz.plot(yline[0], zline[0], **marker_options)
        ax2D_yz.plot(yline[-1], zline[-1], **marker_options)
        
        ax2D_xz.plot(xline, zline, **line_options)
        ax2D_xz.plot(xline[0], zline[0], **marker_options)
        ax2D_xz.plot(xline[-1], zline[-1], **marker_options)
        
    if created_2D:
        fig2.tight_layout()
        fig2.show()
    if created_3D:
        fig.tight_layout()
        fig.show()
    
    return