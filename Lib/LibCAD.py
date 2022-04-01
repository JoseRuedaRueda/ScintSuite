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


def write_file_for_fortran(stlfile, outputfile, convert_mm_2_m = False):
    """
    Transform .stl files into a format compatible with SINPA/iHIBPsim/MEGA

    Jose Rueda: jrrueda@us.es

    @param filename: name of the stl file (full path)
    """
    # --- Open and load the stil file
    a = open3d.io.read_triangle_mesh(stlfile)
    vertices = np.asarray(a.vertices)
    index = np.asarray(a.triangles)
    triangleNum = index.shape[0]
    
    conv_fac = 1.0
    if convert_mm_2_m:# Catia files are usually in mm, hence convert to m
        conv_fac =  0.001
    
    # --- Write the vertices in the file
    # append stl vertice data to file that has information of kind of plate and 
    # other comments in the first few lines already written
    f = open(outputfile, 'a')  
    f.write(str(triangleNum) + '  ! Number of triangles\n')
    for i in range(triangleNum):
        for j in range(3):
            f.write('%.3f %.3f %.3f \n' % (vertices[index[i, j], 0] * conv_fac,
                                           vertices[index[i, j], 1] * conv_fac,
                                           vertices[index[i, j], 2] * conv_fac))
    
    f.close() 
