"""
Contains the routines to read and plot the plates used in the simulation codes.

Maintainer: Jose Rueda Rueda: jrrueda@us.es
Contributors:

Introduced in version 0.6.6, before each code had his own object and library

All interaction is recomended using the object Geometry, please do not use the
single routines independently.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Lib._Machine import machine
from Lib._Paths import Path
from Lib._Plotting import axis_beauty, axisEqual3D, clean3Daxis
import Lib._CAD as libcad
import f90nml
paths = Path(machine)


def read_element(file, code: str = 'SINPA'):
    """
    Read a SINPA of FILDSIM gemetric element.

    Jose Rueda: jrrueda@us.es

    @param file: file to be read
    @param FILDSIM: flag to indicate if we have the oldFILDSIM format

    @return geom: Dictionary containing:
        - 'name': Name of the element
        - 'description': Description
        - 'kind': kind of plate
        - 'n': number of triangles
        - 'triangles': vertices

    Note: oldFILDSIM does not include what kind of plot it is, so the kind of
    plate is deduced for the name. In this case, all is considered to be a
    collimator plate, except 'Scintillator' [case insensitive] is written at
    some point in the name. Examples: AUG_FILD1_SCINTILLATOR or
    AUG_scintillator_FILD1

    Note2: To be consistend between both of codes, coorditanes will be
    transformed to m when reading the files
    """
    if code.lower() == 'fildsim':
        with open(file, 'r') as f:
            dummy = f.readline().strip()   # Comment line
            dummy = f.readline().strip()   # Line with the number of vertex
            dum, name = dummy.split('=')
            dummy = f.readline().strip()   # Line with vertex number
            number = int(dummy.split('=')[-1])
            # FILDSIM old format do not include the kind of plate, so let's try
            # to get it from the name:
            possible_names_plates = name.lower().split('_')
            if 'scintillator' in possible_names_plates:
                kind = 2
            else:        # assume that it is a collimator, better than nothing
                kind = 0

            geom = {
                'name': name,
                'description': None,
                'kind': kind,
                'n': number
            }
        # read the vertex
        geom['vertex'] = np.loadtxt(
            file, skiprows=3, max_rows=number, comments='!', delimiter=',')
        geom['triangles'] = None
        geom['trianglesScint'] = None
        geom['vertexScint'] = None
        # Transform the vertex coordinates to m:
        geom['vertex'] /= 100.0
    elif code.lower() == 'sinpa':
        with open(file, 'r') as f:
            geom = {
                'name': f.readline().strip(),
                'description': [f.readline().strip(), f.readline().strip()],
            }
        geom['kind'] = int(np.loadtxt(file, max_rows=1,
                                      skiprows=3, comments='!'))
        geom['n'] = int(np.loadtxt(file, max_rows=1, skiprows=4,
                                   comments='!'))
        geom['triangles'] = np.loadtxt(file, skiprows=5, comments='!')
        geom['trianglesScint'] = None
        geom['vertex'] = None
        geom['vertexScint'] = None
    elif code.lower() == 'ihibpsim':
        with open(file, 'r') as f:
            geom = {
                'name': 'iHIBPgeom',
                'description': 'iHIBPsim plate',
                'kind': 0,  # Assume everything is collimator
                'n': int(np.loadtxt(file, max_rows=1, comments='!'))
            }
        geom['triangles'] = np.loadtxt(file, skiprows=1, comments='!')
        geom['trianglesScint'] = None
        geom['vertex'] = None
        geom['vertexScint'] = None
        if len(geom['triangles'].shape) == 2:
            # we have a 2D iHIBPsim file
            geom['wallDim'] = 2
        else:
            geom['wallDim'] = 3

    else:
        raise Exception('Sorry, code not understood')
    return geom


def plotLinesElement(geom: dict, ax=None, line_params: dict = {},
                     referenceSystem='absolute', plot2D: bool = False,
                     units: str = 'cm'):
    """
    Plot a geometry element.

    Jose Rueda Rueda: jrrueda@us.es

    @param geom: dictionary created by read_element()
    @param ax: axes where to plot
    @param line_params: parameters for the plt.plot function
    @param referenceSystem: if absolute, the absolute coordinates will be
        used, if 'scintillator', the scintillator coordinates will be used
    @param plot2D: flag, if true a 2D plot of the element will be represented,
        in the scintillator system, so the flag 'referenceSystem' is
        is overwritten if this is true

    Note: The use of this routine is not recomended if you use a fine mesh with
    several triangles, as several lines will be plotted and therefore we can
    overload memory
    """
    # --- Initialize options
    line_options = {
        'color': 'k'
    }
    line_options.update(line_params)
    # Overwrite the referenceSystem label in case there is a 2D plot:
    if plot2D:
        referenceSystem = 'scintillator'
    # Get the units:
    if units not in ['m', 'cm', 'mm']:
        raise Exception('Not understood units?')
    possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
    factor = possible_factors[units]
    # --- Plot
    # see if we have a new file (triangles grid) or old file (perimeter)
    if geom['vertex'] is not None:
        triangleFile = False
        key_base = 'vertex'
    else:
        triangleFile = True
        key_base = 'triangles'
    # See which data we need to plot
    if referenceSystem.lower() == 'absolute':
        key = key_base
    elif referenceSystem.lower() == 'scintillator':
        key = key_base + 'Scint'
    else:
        raise Exception('Not understood reference system')
    # Create the axis
    if ax is None:
        if plot2D:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.tight_layout()
    # Plot the lines
    if triangleFile:
        if plot2D:
            for i in range(geom['n']):
                x = factor \
                    * np.array([geom[key][3*i, 1], geom[key][3*i + 1, 1],
                                geom[key][3*i+2, 1], geom[key][3*i + 1, 1],
                                geom[key][3*i, 1], geom[key][3*i + 2, 1]])
                y = factor \
                    * np.array([geom[key][3*i, 2], geom[key][3*i + 1, 2],
                                geom[key][3*i+2, 2], geom[key][3*i + 1, 2],
                                geom[key][3*i, 2], geom[key][3*i + 2, 2]])

                ax.plot(x, y, **line_options)
        else:
            for i in range(geom['n']):
                z = factor \
                    * np.array([geom[key][3*i, 2], geom[key][3*i + 1, 2],
                                geom[key][3*i+2, 2], geom[key][3*i + 1, 2],
                                geom[key][3*i, 2], geom[key][3*i + 2, 2]])
                y = factor \
                    * np.array([geom[key][3*i, 1], geom[key][3*i + 1, 1],
                                geom[key][3*i+2, 1], geom[key][3*i + 1, 1],
                                geom[key][3*i, 1], geom[key][3*i + 2, 1]])
                x = factor \
                    * np.array([geom[key][3*i, 0], geom[key][3*i + 1, 0],
                                geom[key][3*i+2, 0], geom[key][3*i + 1, 0],
                                geom[key][3*i, 0], geom[key][3*i + 2, 0]])

                ax.plot(x, y, z, ** line_options)
    else:
        if plot2D:
            ax.plot(geom[key][:, 1] * factor, geom[key][:, 2] * factor,
                    **line_options)
            ax.plot([geom[key][0, 1] * factor, geom[key][-1, 1] * factor],
                    [geom[key][0, 2] * factor, geom[key][-1, 2] * factor],
                    **line_options)
        else:
            ax.plot(geom[key][:, 0] * factor, geom[key][:, 1] * factor,
                    geom[key][:, 2] * factor,
                    **line_options)
            ax.plot([geom[key][0, 0] * factor, geom[key][-1, 0] * factor],
                    [geom[key][0, 1] * factor, geom[key][-1, 1] * factor],
                    [geom[key][0, 2] * factor, geom[key][-1, 2] * factor],
                    **line_options)
    return ax


def plotShadedElement(geom: dict, ax=None, surface_params: dict = {},
                      referenceSystem='absolute', plot2D: bool = False,
                      units: str = 'cm', view: str = 'absolute'):
    """
    Plot the geometric elements with a filled contour.

    Jose Rueda Rueda: jrrueda@us.es

    2D view implemented by Alex LeViness: leviness@pppl.gov

    @param geom: dictionary created by read_element()
    @param ax: axes where to plot, if none, they will be created
    @param surface_params: parameters for the plt.plot function
    @param referenceSystem: if absolute, the absolute coordinates will be
        used, if 'scintillator', the scintillator coordinates will be used
    @param plot2D: flag, if true a 2D plot of the element will be represented,
        in the scintillator system, so the flag 'referenceSystem' is
        is overwritten if this is true [Still not implemented]
    @param units: units to plot the geometry. Acepted: m, cm, mm
    @param view: plot XY plane, XZ, YZ, or the scintillator plane. This option
        is only used for the 2D plotting. You can select the scintillator
        option writtin 'Scint', 'Scintillator' or any capitalization of those

    Note: The use of this routine is not recomended if you use a fine mesh
    with several triangles, as several lines will be plotted and therefore
    we can overload memory
    """
    # --- Initialise and check settings
    # Plotting options:
    surface_options = {
        'color': 'k',
        'alpha': 0.25,    # Transparency factor
        'linewidth': 0.0  # Width of the line, if zero, no contour will be plot
    }
    if plot2D:  # just add a bit of line width if we deal with 2D plots
        surface_options['linewidth'] = 0.1
        surface_options['alpha'] = 0.95  # And reduce the transparency
    surface_options.update(surface_params)

    # select the reference system for 2D stuff:
    if (view.lower() == 'scint') or (view.lower() == 'scintillator'):
        referenceSystem = 'scintillator'
    else:
        referenceSystem = 'absolute'

    # --- Check the scale
    if units not in ['m', 'cm', 'mm']:
        raise Exception('Not understood units?')
    possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
    factor = possible_factors[units]
    # --- Plot
    # see if we have a new file (triangles grid) or old file (perimeter)
    if geom['vertex'] is not None:
        triangleFile = False
        key_base = 'vertex'
    else:
        triangleFile = True
        key_base = 'triangles'
    # See which data we need to plot
    if referenceSystem.lower() == 'absolute':
        key = key_base
    elif referenceSystem.lower() == 'scintillator':
        key = key_base + 'Scint'
    else:
        raise Exception('Not understood reference system')
    # Create the axis
    if ax is None:
        if plot2D:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.tight_layout()
    # --- Plot
    if plot2D:
        if triangleFile:
            for i in range(geom['n']):
                x = np.append(
                    geom[key][3*i: (3*i + 3), 0], geom[key][3*i, 0]) * factor
                y = np.append(
                    geom[key][3*i: (3*i + 3), 1], geom[key][3*i, 1]) * factor
                z = np.append(
                    geom[key][3*i: (3*i + 3), 2], geom[key][3*i, 2]) * factor
                if view.lower() == 'scint' or view.lower() == 'yz':
                    ax.fill_between(y, z, **surface_options)
                elif view.lower() == 'xy':
                    ax.fill_between(x, y, **surface_options)
                elif view.lower() == 'xz':
                    ax.fill_between(x, z, **surface_options)
        else:
            raise Exception('Sorry still not implemented feature')
    else:  # 3D plot
        if not triangleFile:
            x = np.append(geom[key][:, 0], geom[key][-1, 0]) * factor
            y = np.append(geom[key][:, 1], geom[key][-1, 1]) * factor
            z = np.append(geom[key][:, 2], geom[key][-1, 2]) * factor
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(verts, **surface_options))
        else:
            for it in range(geom['n']):
                x = np.append(geom[key][3*it: (3*it + 3), 0],
                              geom[key][3*it, 0]) * factor
                y = np.append(geom[key][3*it: (3*it + 3), 1],
                              geom[key][3*it, 1]) * factor
                z = np.append(geom[key][3*it: (3*it + 3), 2],
                              geom[key][3*it, 2]) * factor
                verts = [list(zip(x, y, z))]
                ax.add_collection3d(Poly3DCollection(verts, **surface_options))
    return ax


class Geometry:
    """
    Class containing the geometry introduced in the simulation

    Introduced in version 0.6.0. Adapted to oldFILDSIM in version 0.6.6
    Adapted to iHIBPsim in version 0.6.7

    Maintainers: Jose Rueda jrrueda@us.es & Pablo Oyola pablo.oyola@ipp.mpg.de
    """

    def __init__(self, GeomID: str = 'Test0', code: str = 'SINPA', files=None):
        """
        Initialise the class.

        Jose Rueda Rueda: jrrueda@us.es

        @param GeomID: Geom ID
        @param code: Which code it is
        @param files: a list with the files to be loaded, if present, GeomID
            will be ignored.
        Notice: the GeomID would be the name of the folder inside the geometry
        folder of the FILDSIM or SINPA code. If GeomID is an absolute path,
        files inside that paths will be loaded

        @ToDo: allow to pass as input a code namelist
        """
        # initialise the object as empty if needed:
        if (GeomID is None) and (files is None):
            self.ExtraGeometryParams = None
            self.code = None
            self.elements = []
            return
        # Get the folder with the geometry data
        if code.lower() == 'fildsim':
            folder = os.path.join(paths.FILDSIM, 'geometry', GeomID)
            # For FILDSIM, there is no extra file with the pinhole position and
            # other data so we will assume the pinhole as in 0,0,0 as default
            # FILDSIM and that the rotation is none. The user can latelly
            # modify this latelly if he wants
            self.ExtraGeometryParams = {
                'rotation': np.eye(3),
                'rpin': np.array([0.0, 0.0, 0.0]),
                'ps': np.array([0.0, 0.0, 0.0]),
            }
        elif code.lower() == 'sinpa':
            folder = os.path.join(paths.SINPA, 'Geometry', GeomID)
            dummy = f90nml.read(os.path.join(folder,
                                             'ExtraGeometryParams.txt'))
            self.ExtraGeometryParams = dummy['ExtraGeometryParams']
            self.ExtraGeometryParams['rotation'] = \
                np.array(self.ExtraGeometryParams['rotation']).T
            # This last rotation is just to translate from fortran to Python
        else:
            # For iHIBsim, there is no extra file with the pinhole position and
            # other data so we will assume the pinhole as in 0,0,0 and that the
            # rotation is none. So no transformation is done when passing from
            # one system to another
            self.ExtraGeometryParams = {
                'rotation': np.eye(3),
                'rpin': np.array([0.0, 0.0, 0.0]),
                'ps': np.array([0.0, 0.0, 0.0]),
            }
            folder = ''
        if files is None:
            files = os.listdir(folder)
        self.elements = []
        # Read the files
        for f in files:
            if f.startswith('Elem') or f.endswith('.pl') or f.endswith('.3d'):
                filename = os.path.join(folder, f)
                self.elements.append(read_element(filename, code=code))
        # Just set a variable to see from where it was comming
        self.code = code

    @property
    def size(self):
        """Get the number of geometrical elements."""
        return len(self.elements)

    def __getitem__(self, idx):
        """
        Overload of the method to be able to access the geometry element.

        It returns the whole data of a geometry elements

        Copied from PabloOrbit object (see iHIBSIM library)

        @param idx: element number

        @return self.data[idx]: Element dictionary
        """
        return self.elements[idx]

    def apply_movement(self):
        """Apply the rototranslation to the geometric element"""
        rot = self.ExtraGeometryParams['rotation']
        tras = self.ExtraGeometryParams['ps']
        if self[0]['triangles'] is not None:
            key1 = 'triangles'
            key2 = 'trianglesScint'
        else:
            key1 = 'vertex'
            key2 = 'vertexScint'
        for i in range(self.size):
            shape = self[i][key1].shape
            self[i][key2] = np.zeros(shape)
            for it in range(shape[0]):
                self[i][key2][it, :] = \
                        rot @ (self[i][key1][it, :] - tras)
        # Apply this also for the pinhole point:
        self.ExtraGeometryParams['rpinScint'] = \
            rot @ (np.array(self.ExtraGeometryParams['rpin']) - tras)

    def plot3Dlines(self, line_params: dict = {}, ax=None,
                    element_to_plot=[0, 1, 2], plot_pinhole: bool = True,
                    referenceSystem='absolute', units: str = 'cm'):
        """
        Plot the geometric elements.

        Jose Rueda Rueda: jrrueda@us.es

        @param geom: dictionary created by read_element()
        @param ax: axes where to plot, if none, they will be created
        @param line_params: parameters for the plt.plot function
        @param element_to_plot: kind of plates we want to plot:
            -0: Collimator
            -1: Ionizers (INPA carbon foil)
            -2: Scintillator
        @param plot_pinhole: flag to plot a point on the pinhole or not
        @param referenceSystem: if absolute, the absolute coordinates will be
            used, if 'scintillator', the scintillator coordinates will be used

        Note: The use of this routine is not recomended if you use a fine mesh
        with several triangles
        """
        # Initialise the plotting parameters:
        line_options = {
            'color': 'k'
        }
        line_options.update(line_params)
        line_colors = ['k', 'b', 'r']  # Default color for coll, foil, scint
        # --- Check the scale
        if units not in ['m', 'cm', 'mm']:
            raise Exception('Not understood units?')
        possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
        factor = possible_factors[units]
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.tight_layout()
            created = True
        else:
            created = False
        for ele in self.elements:
            if ele['kind'] in element_to_plot:
                # If the user did not provided a custom color, generate a color
                # for each tipe of plate
                if 'color' not in line_params:
                    line_options['color'] = line_colors[ele['kind']]
                plotLinesElement(ele, ax=ax, line_params=line_options,
                                 referenceSystem=referenceSystem,
                                 plot2D=False, units=units)
        # --- Plot pinhole
        if plot_pinhole:
            ax.plot([self.ExtraGeometryParams['rpin'][0] * factor],
                    [self.ExtraGeometryParams['rpin'][1] * factor],
                    [self.ExtraGeometryParams['rpin'][2] * factor], 'og')
        if created:
            axisEqual3D(ax)
            clean3Daxis(ax)
            fig.show()
        return ax

    def plot2Dlines(self, line_params: dict = {}, ax=None,
                    ax_params: dict = {},
                    element_to_plot=[0, 1, 2], plot_pinhole: bool = True,
                    units: str = 'cm'):
        """
        Plot the geometric elements.

        Jose Rueda Rueda: jrrueda@us.es

        @param geom: dictionary created by read_element()
        @param ax: axes where to plot, if none, they will be created
        @param ax_param: parameter for the axis beauty function
        @param line_params: parameters for the plt.plot function
        @param element_to_plot: kind of plates we want to plot:
            -0: Collimator
            -1: Ionizers
            -2: Scintillator
        @param plot_pinhole: flag to plot a point on the pinhole or not
        @param referenceSystem: if absolute, the absolute coordinates will be
            used, if 'scintillator', the scintillator coordinates will be used

        Note: The use of this routine is not recomended if you use a fine mesh
        with several triangles
        """
        # Check if the rotation was done
        if self[0]['trianglesScint'] is None or self[0]['vertexScint'] is None:
            self.apply_movement()
        # Initialise the plotting parameters:
        line_options = {
            'color': 'k'
        }
        line_options.update(line_params)
        line_colors = ['k', 'b', 'r']  # Default color for coll, foil, scint
        ax_options = {
            'grid': 'both',
            'xlabel': 'x [' + units + ']',
            'ylabel': 'y [' + units + ']',
        }
        ax_options.update(ax_params)
        # --- Check the scale
        if units not in ['m', 'cm', 'mm']:
            raise Exception('Not understood units?')
        possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
        factor = possible_factors[units]
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        for ele in self.elements:
            if ele['kind'] in element_to_plot:
                # If the user did not provided a custom color, generate a color
                # for each tipe of plate
                if 'color' not in line_params:
                    line_options['color'] = line_colors[ele['kind']]
                plotLinesElement(ele, ax=ax, line_params=line_options,
                                 plot2D=True, units=units)
        # --- Plot pinhole
        if plot_pinhole:
            ax.plot([self.ExtraGeometryParams['rpinScint'][1] * factor],
                    [self.ExtraGeometryParams['rpinScint'][2] * factor], 'og')
        if created:
            axis_beauty(ax, ax_options)
            fig.show()
        return ax

    def plot3Dfilled(self, surface_params: dict = {}, ax=None,
                     element_to_plot=[0, 1, 2], plot_pinhole: bool = True,
                     referenceSystem='absolute', units: str = 'cm'):
        """
        Plot the geometric elements.

        Jose Rueda Rueda: jrrueda@us.es

        @param geom: dictionary created by read_element()
        @param ax: axes where to plot, if none, they will be created
        @param surface_params: parameters for the plt.plot function
        @param element_to_plot: kind of plates we want to plot:
            -0: Collimator
            -1: Ionizers
            -2: Scintillator
        @param plot_pinhole: flag to plot a point on the pinhole or not. Only
            work for SINPA geometry, as oldFILDSIM has not the extra namelist
        @param referenceSystem: if absolute, the absolute coordinates will be
            used, if 'scintillator', the scintillator coordinates will be used
        @param units: units to plot the geometry. Acepted: m, cm, mm

        Note: The use of this routine is not recomended if you use a fine mesh
        with several triangles
        """
        # Initialise the plotting parameters:
        surface_options = {
            'color': 'k',
            'alpha': 0.25,   # Transparency factor
        }
        surface_options.update(surface_params)
        surface_colors = ['k', 'b', 'g']  # Default color for coll, foil, scint
        # --- Check the scale
        if units not in ['m', 'cm', 'mm']:
            raise Exception('Not understood units?')
        possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
        factor = possible_factors[units]
        # see if we have a new file (triangles grid) or old file (perimeter),
        # This is just for the axis limits for the 3D plot
        if self[0]['vertex'] is not None:
            key_base = 'vertex'
        else:
            key_base = 'triangles'
        # See which data we need to plot
        if referenceSystem.lower() == 'absolute':
            key = key_base
        elif referenceSystem.lower() == 'scintillator':
            key = key_base + 'Scint'
        else:
            raise Exception('Not understood reference system')
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.tight_layout()
            created = True
            # axis limits, dummy values, they will be updated in the loop
            xmin = 1e6
            xmax = -1e6
            ymin = 1e6
            ymax = -1e6
            zmin = 1e6
            zmax = -1e6
        else:
            created = False

        for ele in self.elements:
            if ele['kind'] in element_to_plot:
                # If the user did not provided a custom color, generate a color
                # for each tipe of plate
                if 'color' not in surface_params:
                    surface_options['color'] = surface_colors[ele['kind']]
                # plot the plate
                plotShadedElement(ele, ax=ax, surface_params=surface_options,
                                  referenceSystem=referenceSystem,
                                  plot2D=False, units=units)
                # get the limit of the plate
                if created:
                    xmin = min(xmin, ele[key][:, 0].min())
                    xmax = max(xmax, ele[key][:, 0].max())
                    ymin = min(ymin, ele[key][:, 1].min())
                    ymax = max(ymax, ele[key][:, 1].max())
                    zmin = min(zmin, ele[key][:, 2].min())
                    zmax = max(zmax, ele[key][:, 2].max())

        # --- Plot pinhole
        if plot_pinhole:
            ax.plot([self.ExtraGeometryParams['rpin'][0] * factor],
                    [self.ExtraGeometryParams['rpin'][1] * factor],
                    [self.ExtraGeometryParams['rpin'][2] * factor], 'og')
        # --- Set the scale:
        if created:
            dx = xmax - xmin
            dy = ymax - ymin
            dz = zmax - zmin
            ax.set_xlim((xmin - 0.1 * dx) * factor, (xmax + 0.1 * dx) * factor)
            ax.set_ylim((ymin - 0.1 * dy) * factor, (ymax + 0.1 * dy) * factor)
            ax.set_zlim((zmin - 0.1 * dz) * factor, (zmax + 0.1 * dz) * factor)
            axisEqual3D(ax)
            clean3Daxis(ax)
            fig.show()
        return ax

    def plot2Dfilled(self, surface_params: dict = {}, ax=None,
                     ax_params: dict = {},
                     element_to_plot=[0, 1, 2], plot_pinhole: bool = True,
                     units: str = 'cm',
                     view: str = 'Scint',
                     referenceSystem: str = 'absolute'):
        """
        Plot the geometric elements in 2D.

        Added by Alex LeViness: leviness@pppl.gov

        @param geom: dictionary created by read_element()
        @param ax: axes where to plot, if none, they will be created
        @param surface_params: parameters for the plt.plot function
        @param element_to_plot: kind of plates we want to plot:
            -0: Collimator
            -1: Ionizers
            -2: Scintillator
        @param plot_pinhole: flag to plot a point on the pinhole or not. Only
            work for SINPA geometry, as oldFILDSIM has not the extra namelist
        @param referenceSystem: if absolute, the absolute coordinates will be
            used, if 'scintillator', the scintillator coordinates will be used
        @param units: units to plot the geometry. Acepted: m, cm, mm
        @param view: plot XY plane, XZ, YZ, or the scintillator plane

        Note: The use of this routine is not recomended if you use a fine mesh
        with several triangles
        """
        # Check if the rotation was done
        if self[0]['trianglesScint'] is None or self[0]['vertexScint'] is None:
            self.apply_movement()

        # Initialise the plotting parameters:
        surface_options = {
            'color': 'k'
        }
        surface_options.update(surface_params)
        surface_colors = ['k', 'b', 'g']  # Default color for coll, foil, scint
        if view.lower() == 'scint':
            labels = ['x', 'y']
        elif view.lower() == 'yz':
            labels = ['y', 'z']
        elif view.lower() == 'xy':
            labels = ['x', 'y']
        elif view.lower() == 'xz':
            labels = ['x', 'z']
        else:
            raise Exception('Viewing angle not understood')
        ax_options = {
            'grid': 'both',
            'xlabel': labels[0] + ' [' + units + ']',
            'ylabel': labels[1] + ' [' + units + ']',
        }
        # --- Check the scale
        if units not in ['m', 'cm', 'mm']:
            raise Exception('Not understood units?')
        possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
        factor = possible_factors[units]
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            created = False
        for ele in self.elements:
            if ele['kind'] in element_to_plot:
                # If the user did not provided a custom color, generate a color
                # for each tipe of plate
                if 'color' not in surface_params:
                    surface_options['color'] = surface_colors[ele['kind']]
                # plot the plate
                plotShadedElement(ele, ax=ax, surface_params=surface_options,
                                  plot2D=True, units=units, view=view,
                                  referenceSystem=referenceSystem)

        # --- Plot pinhole
        if plot_pinhole:
            if view.lower() == 'scint':
                ax.plot([self.ExtraGeometryParams['rpinScint'][1] * factor],
                        [self.ExtraGeometryParams['rpinScint'][2] * factor],
                        'og')
            elif view.lower() == 'yz':
                ax.plot([self.ExtraGeometryParams['rpin'][1] * factor],
                        [self.ExtraGeometryParams['rpin'][2] * factor], 'og')
            elif view.lower() == 'xy':
                ax.plot([self.ExtraGeometryParams['rpin'][0] * factor],
                        [self.ExtraGeometryParams['rpin'][1] * factor], 'og')
            elif view.lower() == 'xz':
                ax.plot([self.ExtraGeometryParams['rpin'][0] * factor],
                        [self.ExtraGeometryParams['rpin'][2] * factor], 'og')

        if created:
            axis_beauty(ax, ax_options)
            fig.show()

        return ax

    def writeGeometry(self, path):
        """
        Write the geometry into the folder

        Note: Only working for SINPA/iHIBPsim code
        """
        if self.code.lower() == 'sinpa':
            for i in range(self.size):
                name = os.path.join(path, 'Element' + str(i + 1) + '.txt')
                with open(name, 'w') as f:
                    f.writelines([self[i]['name'] + '\n'])
                    f.writelines([self[i]['description'][0] + '\n',
                                  self[i]['description'][1] + '\n'])
                    f.writelines([str(self[i]['kind']) + '\n',
                                  str(self[i]['n']) + '\n'])
                    for it in range(3 * self[i]['n']):
                        f.writelines([str(self[i]['triangles'][it, 0]) + ' ',
                                      str(self[i]['triangles'][it, 1]) + ' ',
                                      str(self[i]['triangles'][it, 2]) + '\n'])
            # Write the namelist
            file = os.path.join(path, 'ExtraGeometryParams.txt')
            f90nml.write({'ExtraGeometryParams': self.ExtraGeometryParams}, 
                         file, force=True)
        elif self.code.lower() == 'ihibpsim':
            if os.path.isdir(path):
                name = os.path.join(path, 'wall.txt')
            else:
                name = path

            with open(name, 'w') as f:
                f.writelines('%i  ! Number of elements\n'%self[0]['n'])
                if self['wallDim'] == 3:
                    for j in range(3 * self[0]['n']):
                        f.writelines([str(self[0]['triangles'][j, 0]) + ' ',
                                      str(self[0]['triangles'][j, 1]) + ' ',
                                      str(self[0]['triangles'][j, 2]) + '\n'])
                if self['wallDim'] == 2:
                    for j in range(2 * self[0]['n']):
                        f.writelines([str(self[0]['triangles'][j, 0]) + ' ',
                                      str(self[0]['triangles'][j, 1]) + '\n'])

                

    def elements_to_stl(self, element_to_save=[0, 1, 2], units: str = 'cm'
                           ,file_name_save: str = 'Test'):
        """
        Store the geometric elements to stl files. Useful for testing SINPA inputs
        Anton van Vuuren: avanvuuren@us.es
        @param geom: dictionary created by read_element()

        @param element_to_save: kind of plates we want to plot:
            -0: Collimator
            -1: Ionizers
            -2: Scintillator

        @param units: units to plot the geometry. Acepted: m, cm, mm
        @param file_name_save: name of stl file to be generated (don't add ".stl")
        """
        file_mod = ["Collimator", "Ionizers", "Scintillator"]
        for ele in self.elements:
            if ele['kind'] in element_to_save:
                libcad.write_triangles_to_stl(ele, units=units ,
                                                file_name_save = file_name_save
                                                + "_" + file_mod[ele['kind']])

##

