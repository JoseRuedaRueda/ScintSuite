"""
SINPA geometry

Jose Rueda: jrrueda@us.es

Introduced in version 0.6.0
"""
import os
import math
import f90nml
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)


def calculate_rotation_matrix(n, u1=None, verbose=True):
    """
    Calculate a rotation matrix to leave n as ux

    Jose Rueda: jrrueda@us.es

    It gives that rotation matrix, R, such that ux' = R @ n, where ux' is the
    unitaty vector in the x direction. If u1 is provided, the rotation will
    also fulfil uy = R @ u1. In this way the scintillator will finish properly
    aligned after the rotation

    @param n: unit vector
    @param u1: Unit vector normal to n

    @return M: Rotation matrix
    """
    # --- Check the unit vector
    modn = math.sqrt(np.sum(n * n))
    if abs(modn - 1) > 1e-5:
        print('The vector is not normalised, applying normalization')
        n /= modn
    if u1 is not None:
        perp = np.sum(u1 * n)
        if perp > 1e-3:
            raise Exception('U1 is not normal to n, revise inputs')
    # --- Check if the normal vector is already not in the x direction
    if abs(abs(n[0]) - 1) < 1e-5:
        print('The scintillator is already in a plane of constant x')
        return np.eye(3)  # Return just the identity matrix
    # --- Calculate the normal vector to perform the rotation
    ux = np.array([1.0, 0.0, 0.0])
    u_turn = np.cross(n, ux)
    u_turn /= math.sqrt(np.sum(u_turn * u_turn))
    # --- Calculate the proper angle
    alpha = math.acos(n[0])
    if verbose:
        print('The rotation angle is:', alpha)
        # print('Please write in fortran this matrix but transposed!!')
    # --- Calculate the rotation
    r = R.from_rotvec(alpha * u_turn)
    rot1 = r.as_matrix()
    # --- Now perform the second rotation to orientate the scintillator
    if u1 is not None:
        uy = np.array([0.0, 1.0, 0.0])
        # Get u1 in the second system of coordinates
        u1_new = r.apply(u1)
        u_turn = np.cross(u1_new, uy)
        u_turn /= math.sqrt(np.sum(u_turn * u_turn))
        # --- Calculate the proper angle
        alpha = math.acos(u1_new[1])
        r = R.from_rotvec(alpha * u_turn)
        rot2 = r.as_matrix()
    else:
        rot2 = np.eye(3)
    return rot2 @ rot1


def read_element(file):
    """
    Read a SINPA gemetric element.

    Jose R: jrrueda@us.es

    @param file: file to be read

    @return geom: Dictionary containing:
        - 'name': Name of the element
        - 'description': Description
        - 'kind': kind of plate
        - 'n': number of triangles
        - 'triangles': vertices
    """
    with open(file, 'r') as f:
        geom = {
            'name': f.readline().strip(),
            'description': [f.readline().strip(), f.readline().strip()],
        }
    geom['kind'] = np.loadtxt(file, max_rows=1,
                              skiprows=3, comments='!').astype(np.int)
    geom['n'] = np.loadtxt(file, max_rows=1, skiprows=4,
                           comments='!').astype(np.int)
    geom['triangles'] = np.loadtxt(file, skiprows=5, comments='!')
    return geom


def plot_element(geom: dict, ax=None, line_params: dict = {},
                 referenceSystem='absolute'):
    """
    Plot a geometry element.

    Jose Rueda Rueda: jrrueda@us.es

    @param geom: dictionary created by read_element()
    @param ax: axes where to plot
    @param line_params: parameters for the plt.plot function
    @param referenceSystem: if absolute, the absolute coordinates will be
        used, if 'scintillator', the scintillator coordinates will be used

    Note: The use of this routine is not recomended if you use a fine mesh with
    several triangles
    """
    # --- Initialize options
    line_options = {
        'color': 'k'
    }
    line_options.update(line_params)
    # --- Create the axis
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.tight_layout()
    # --- Plot
    if referenceSystem.lower() == 'absolute':
        key = 'triangles'
    elif referenceSystem.lower() == 'scintillator':
        key = 'trianglesScint'
    else:
        raise Exception('Not understood reference system')

    for i in range(geom['n']):
        ax.plot([geom[key][3*i, 0], geom[key][3*i + 1, 0]],
                [geom[key][3*i, 1], geom[key][3*i + 1, 1]],
                [geom[key][3*i, 2], geom[key][3*i + 1, 2]],
                **line_options)
        ax.plot([geom[key][3*i+2, 0], geom[key][3*i + 1, 0]],
                [geom[key][3*i+2, 1], geom[key][3*i + 1, 1]],
                [geom[key][3*i+2, 2], geom[key][3*i + 1, 2]],
                **line_options)
        ax.plot([geom[key][3*i, 0], geom[key][3*i + 2, 0]],
                [geom[key][3*i, 1], geom[key][3*i + 2, 1]],
                [geom[key][3*i, 2], geom[key][3*i + 2, 2]],
                **line_options)


def plot_element2D(geom: dict, ax=None, line_params: dict = {}):
    """
    Plot a geometry element.

    Jose Rueda Rueda: jrrueda@us.es

    This will plot the geometric elemets in 2D, in the scintillator system

    @param geom: dictionary created by read_element()
    @param ax: axes where to plot
    @param line_params: parameters for the plt.plot function

    Note: The use of this routine is not recomended if you use a fine mesh with
    several triangles
    """
    # --- Initialize options
    line_options = {
        'color': 'k'
    }
    line_options.update(line_params)
    # --- Create the axis
    if ax is None:
        fig, ax = plt.subplots()
    # --- Plot
    key = 'trianglesScint'

    for i in range(geom['n']):
        ax.plot([geom[key][3*i, 1], geom[key][3*i + 1, 1]],
                [geom[key][3*i, 2], geom[key][3*i + 1, 2]],
                **line_options)
        ax.plot([geom[key][3*i+2, 1], geom[key][3*i + 1, 1]],
                [geom[key][3*i+2, 2], geom[key][3*i + 1, 2]],
                **line_options)
        ax.plot([geom[key][3*i, 1], geom[key][3*i + 2, 1]],
                [geom[key][3*i, 2], geom[key][3*i + 2, 2]],
                **line_options)


class Geometry:
    """
    Class containing the geometry introduced in the simulation

    In the future it will contain also routines to turn and translate the
    plates. Now it is just reading and plotting

    Introduced in version 0.6.0
    """

    def __init__(self, GeomID: str = 'Test0'):
        """
        Initialise the class.

        Jose Rueda Rueda: jrrueda@us.es
        """
        folder = os.path.join(paths.SINPA, 'Geometry', GeomID)
        files = os.listdir(folder)
        self.elements = []
        for f in files:
            if f.startswith('Element'):
                filename = os.path.join(folder, f)
                self.elements.append(read_element(filename))
        dummy = f90nml.read(os.path.join(folder, 'ExtraGeometryParams.txt'))
        self.ExtraGeometryParams = dummy['ExtraGeometryParams']
        self.ExtraGeometryParams['rotation'] = \
            np.array(self.ExtraGeometryParams['rotation']).T

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
        for i in range(self.size):
            self[i]['trianglesScint'] = np.zeros(self[i]['triangles'].shape)
            for it in range(self[i]['n']*3):
                self[i]['trianglesScint'][it, :] = \
                    rot @ (self[i]['triangles'][it, :]-tras)

    def plot3D(self, line_params: dict = {}, ax=None,
               element_to_plot=[0, 1, 2], plot_pinhole: bool = True,
               referenceSystem='absolute'):
        """
        Plot the geometric elements.

        Jose Rueda Rueda: jrrueda@us.es

        @param geom: dictionary created by read_element()
        @param ax: axes where to plot
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

        @todo: add a flag to plot only a given type of plate
        """
        # Initialise the plotting parameters:
        line_options = {
            'color': 'k'
        }
        line_options.update(line_params)
        line_colors = ['k', 'b', 'r']  # Default color for coll, foil, scint
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.tight_layout()
        for ele in self.elements:
            if ele['kind'] in element_to_plot:
                # If the user did not provided a custom color, generate a color
                # for each tipe of plate
                if 'color' not in line_params:
                    line_options['color'] = line_colors[ele['kind']]
                plot_element(ele, ax=ax, line_params=line_options,
                             referenceSystem=referenceSystem)
        # --- Plot pinhole
        if plot_pinhole:
            ax.plot([self.ExtraGeometryParams['rpin'][0]],
                    [self.ExtraGeometryParams['rpin'][1]],
                    [self.ExtraGeometryParams['rpin'][2]], 'og')

    def plot2D(self, line_params: dict = {}, ax=None,
               element_to_plot=[1, 2]):
        """
        Plot the geometric elements in 2D (scintillator system)

        Jose Rueda Rueda: jrrueda@us.es

        @param geom: dictionary created by read_element()
        @param ax: axes where to plot
        @param line_params: parameters for the plt.plot function
        @param element_to_plot: kind of plates we want to plot:
            -0: Collimator
            -1: Ionizers
            -2: Scintillator

        Note: The use of this routine is not recomended if you use a fine mesh
        with several triangles

        @todo: add a flag to plot only a given type of plate
        """
        # Initialise the plotting parameters:
        line_options = {
            'color': 'k'
        }
        line_options.update(line_params)
        line_colors = ['k', 'b', 'r']  # Default color for coll, foil, scint
        if ax is None:
            fig, ax = plt.subplots()

        # Apply the rotation in case is not applied
        if 'trianglesScint' not in self[0].keys():
            print('Applying rotation')
            self.apply_movement()
        for ele in self.elements:
            if ele['kind'] in element_to_plot:
                # If the user did not provided a custom color, generate a color
                # for each tipe of plate
                if 'color' not in line_params:
                    line_options['color'] = line_colors[ele['kind']]
                plot_element2D(ele, ax=ax, line_params=line_options)
