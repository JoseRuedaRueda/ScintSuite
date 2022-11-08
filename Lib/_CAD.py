"""
Rutines to prepare geometry files for the fortran codes starting from .stl file

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 0.5.9
"""
import logging
import numpy as np
import os
logger = logging.getLogger('ScintSuite.CAD')
try:
    from stl import mesh
except ImportError:
    logger.warning('10: stl not found. CAD file support limited')


def write_file_for_fortran_numpymesh(stlfile, outputfile,
                                     convert_mm_2_m: bool = False,
                                     input_units: str = 'mm',
                                     output_units: str = 'm'):
    """
    Transform .stl files into a format compatible with SINPA/iHIBPsim/MEGA
    using stl mesh

    Anton van Vuuren: avanvuuren@us.es

    Minor modeifications by Jose Rueda: jrrueda@us.es

    @param stlfile: name of the stl file (full path)
    @param outputfile: name of the file where to write the triangles
    @param convert_mm_2_m: flag to convert mm (assumed units of the input file)
        to m for the output file. Deprecated, plese use input_units and output_
        units instead

    Notice that the header of the file is not written, as this header is
    different from MEGAwall, iHIBPsim and SINPA
    """
    factors = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}
    conv_fac = 1.0
    if convert_mm_2_m:    # Catia files are usually in mm, hence convert to m
        conv_fac = 0.001  # old flag
    else:  # new system
        conv_fac = factors[input_units] / factors[output_units]

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


def read_triangle_file(fn_in: str):
    """
    Read and return in a dictionary a file containing the triangle information
    as written for the FORTRAN codes.

    Pablo Oyola - poyola@us.es

    @param fn_in: input filename to read.
    """

    if not os.path.isfile(fn_in):
        raise FileNotFoundError('Cannot open file %s!'%fn_in)

    # Let's start reading the data of the triangles.
    with open(fn_in, 'rt') as fid:
        n = int(fid.readline().strip())

        data = np.zeros((n, 3, 3))
        for ii in range(n):
            for jj in range(3):
                data[ii, jj, :] = np.array(fid.readline().strip().split())

    output = { 'n': n, 'data': data }

    return output


def triangles2stl(fn_in: str, fn_out: str, units: str = 'mm'):
    """
    Converting standard-simplified triangle data into STL files.

    Pablo Oyola - poyola@us.es

    @param fn_in: input filename with the triangle data for the FORTRAN codes.
    @param fn_out: output filename to write the STL file.
    @param units: output units to write the triangle file. To be chosen between
    m, mm and cm. Defaults to mm.
    """

    if not os.path.isfile(fn_in):
        raise FileNotFoundError('Cannot open file %s!'%fn_in)

    # Get the units:
    if units not in ['m', 'cm', 'mm']:
        raise Exception('Not understood units?')
    possible_factors = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}
    factor = possible_factors[units]

    # Reading from the file triangles.
    triangles = read_triangle_file(fn_in)

    # Creating the mesh class.
    data = np.zeros(triangles['n'], dtype=mesh.Mesh.dtype)
    mesh_object = mesh.Mesh(data, remove_empty_areas=False)
    mesh_object.x[:] = triangles['data'][..., 0] * factor
    mesh_object.y[:] = triangles['data'][..., 1] * factor
    mesh_object.z[:] = triangles['data'][..., 2] * factor

    if not fn_out.endswith('.stl'):
        fn_out += '.stl'
    mesh_object.save(fn_out)

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
