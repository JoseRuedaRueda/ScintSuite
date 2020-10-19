"""@page Format_scint Format of the scintllator file
Format needed for the scintillator file

@section structure Structure of the file
    - One line with comments
    - The name of the plate: 'Name=$Name'
    - The number of vertices: 'N_vertices='
    - x,y,z of the vertices, separated by commas
    - A dummy line: 'Normal vector'
    - the coordinates of the actcual normal vector
    - A dummy line: Unit
    - the units in which the vertices are provided, cm, m, inch supported

@section example Example of file
\verbatim
# Plate file for FILDSIM.f90
Name=AUG_FILD1_SCINTILLATOR
N_vertices=7
-0.575,-0.17,-4.194
-0.575,-0.17,-0.894
-0.575,4.17,-0.894
-0.575,4.51,-1.684
-0.575,4.63,-2.684
-0.575,4.44,-3.472
-0.575,3.89,-4.194
Normal_vector
1.,0.,0.
Unit
cm
\endverbatim
"""