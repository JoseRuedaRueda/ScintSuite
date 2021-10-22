"""
SINPA geometry

Jose Rueda: jrrueda@us.es

Introduced in version 0.6.0
"""
import os
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)


def calculate_rotation_matrix(n, verbose=True):
    """
    Calculate a rotation matrix to leave n as ux

    Jose Rueda: jrrueda@us.es

    It gives that rotation matrix, R, such that ux' = R @ n, where ux' is the
    unitaty vector in the x direction

    @param n: unit vector

    @return M: Rotation matrix
    """
    # --- Check the unit vector
    modn = math.sqrt(np.sum(n * n))
    if abs(modn - 1) > 1e-2:
        print('The vector is not normalised, applying normalization')
        n /= modn
    # --- Calculate the normal vector to perform the rotation
    ux = np.array([1.0, 0.0, 0.0])
    u_turn = np.cross(n, ux)
    u_turn /= math.sqrt(np.sum(u_turn * u_turn))
    # --- Calculate the proper angle
    alpha = math.acos(-n[0])
    if verbose:
        print('The rotation angle is:', alpha)
        print('Please write in fortran this matrix but transposed!!')
    # --- Calculate the rotation
    r = R.from_rotvec(alpha * u_turn)
    return r.as_matrix().T


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


def plot_element(geom: dict, ax=None, line_params: dict = {}):
    """
    Plot a geometry element.

    Jose Rueda Rueda: jrrueda@us.es

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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.tight_layout()
    # --- Plot
    for i in range(geom['n']):
        ax.plot([geom['triangles'][3*i, 0], geom['triangles'][3*i + 1, 0]],
                [geom['triangles'][3*i, 1], geom['triangles'][3*i + 1, 1]],
                [geom['triangles'][3*i, 2], geom['triangles'][3*i + 1, 2]],
                **line_options)
        ax.plot([geom['triangles'][3*i+2, 0], geom['triangles'][3*i + 1, 0]],
                [geom['triangles'][3*i+2, 1], geom['triangles'][3*i + 1, 1]],
                [geom['triangles'][3*i+2, 2], geom['triangles'][3*i + 1, 2]],
                **line_options)
        ax.plot([geom['triangles'][3*i, 0], geom['triangles'][3*i + 2, 0]],
                [geom['triangles'][3*i, 1], geom['triangles'][3*i + 2, 1]],
                [geom['triangles'][3*i, 2], geom['triangles'][3*i + 2, 2]],
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

    @property
    def size(self):
        """Get the number of geometrical elements."""
        return len(self.elements)

    def __getitem__(self, idx):
        """
        Overload of the method to be able to access the data in the orbit data.

        It returns the whole data of a geometry elements

        Copied from PabloOrbit object (see iHIBSIM library)

        @param idx: element number

        @return self.data[idx]: Element dictionary
        """
        return self.elements[idx]

    def plot3D(self, line_params: dict = {}, ax=None):
        """
        Plot the geometric elements.

        Jose Rueda Rueda: jrrueda@us.es

        @param geom: dictionary created by read_element()
        @param ax: axes where to plot
        @param line_params: parameters for the plt.plot function

        Note: The use of this routine is not recomended if you use a fine mesh
        with several triangles

        @todo: add a flag to plot only a given type of plate
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.tight_layout()
        for ele in self.elements:
            plot_element(ele, ax=ax, line_params=line_params)
