"""
Rutines to prepare geometry files for the fortran codes starting from .stl file

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 0.5.9
"""
import numpy as np
import warnings
try:
    import open3d
except ImportError:
    warnings.warn('You cannot process CAD files',
                  category=UserWarning)
try:
    from stl import mesh
except ImportError:
    warnings.warn('You cannot process CAD files',
                  category=UserWarning)


def write_file_for_fortran(stlfile, outputfile, convert_mm_2_m=False):
    """
    Transform .stl files into a format compatible with SINPA/iHIBPsim/MEGA

    Jose Rueda: jrrueda@us.es

    @param filename: name of the stl file
    """
    # --- Open and load the stil file
    a = open3d.io.read_triangle_mesh(stlfile)
    vertices = np.asarray(a.vertices)
    index = np.asarray(a.triangles)
    triangleNum = index.shape[0]

    conv_fac = 1.0
    if convert_mm_2_m:  # Catia files are usually in mm, hence convert to m
        conv_fac = 0.001

    # --- Write the vertices in the file
    # append stl vertice data to file that has information of kind of plate and
    # other comments in the first few lines already written
    f = open(outputfile, 'a')
    f.write(str(triangleNum) + '  ! Number of triangles\n')
    for i in range(triangleNum):
        for j in range(3):
            f.write(
                '%.9f %.9f %.9f \n' % (vertices[index[i, j], 0] * conv_fac,
                                       vertices[index[i, j], 1] * conv_fac,
                                       vertices[index[i, j], 2] * conv_fac))

    f.close()


def write_file_for_fortran_numpymesh(stlfile, outputfile,
                                     convert_mm_2_m: bool = False):
    """
    Transform .stl files into a format compatible with SINPA/iHIBPsim/MEGA
    using stl mesh

    Anton van Vuuren: avanvuuren@us.es

    works exactly like write_file_for_fortran()

    @param filename: name of the stl file (full path)
    """
    conv_fac = 1.0
    if convert_mm_2_m:  # Catia files are usually in mm, hence convert to m
        conv_fac = 0.001

    # --- Open and load the stil file
    mesh_obj = mesh.Mesh.from_file(stlfile)

    x1x2x3 = mesh_obj.x
    y1y2y3 = mesh_obj.y
    z1z2z3 = mesh_obj.z

    triangleNum = x1x2x3.shape[0]

    # --- Write the vertices in the file
    # append stl vertice data to file that has information of kind of plate and
    # other comments in the first few lines already written
    f = open(outputfile, 'a')
    f.write(str(triangleNum) + '  ! Number of triangles\n')
    for i in range(triangleNum):
        for j in range(3):
            f.write('%.9f %.9f %.9f \n' % (x1x2x3[i, j] * conv_fac,
                                           y1y2y3[i, j] * conv_fac,
                                           z1z2z3[i, j] * conv_fac))

    f.close()


def write_triangles_to_stl(geom: dict,
                           file_name_save: str = 'Test',
                           units: str = 'mm'):
    '''
    Function to store trianlges from Geometry object to stl format

    Anton van Vuuren: avanvuuren@us.es

    @param: Geometry object whose triangles will be stored to an stl file
    @param filename: name of the stl fileto be saved (.stl not needed)
    @param units: Units in which to savethe orbit positions.
    '''

    if geom['vertex'] is not None:
        return
    else:
        key_base = 'triangles'
    # See which data we need to plot
    key = key_base

    # Get the units:
    if units not in ['m', 'cm', 'mm']:
        raise Exception('Not understood units?')
    possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
    factor = possible_factors[units]

    data = np.zeros(geom['n'], dtype=mesh.Mesh.dtype)
    mesh_object = mesh.Mesh(data, remove_empty_areas=False)
    mesh_object.x[:] = np.reshape(geom[key][:, 0], (geom['n'], 3)) * factor
    mesh_object.y[:] = np.reshape(geom[key][:, 1], (geom['n'], 3)) * factor
    mesh_object.z[:] = np.reshape(geom[key][:, 2], (geom['n'], 3)) * factor
    mesh_object.save(file_name_save+'.stl')
