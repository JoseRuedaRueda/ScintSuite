"""@page Format_scint Format of the scintllator file
Format needed for the scintillator file

@section structure Structure of the file
    -# One line with comments
    -# The name of the plate: 'Name=$Name'
    -# The number of vertices: 'N_vertices='
    -# x,y,z of the vertices, separated by commas
    -# A dummy line: 'Normal vector'
    -# the coordinates of the actual normal vector
    -# A dummy line: Unit
    -# the units in which the vertices are provided, cm, m, inch supported

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

"""@page database Calibration database
Format and fields of the calibration database

@section fields fields present in the database
        -# camera: Camera model used
        -# shot1: Initial shot for which the calibration is valid
        -# shot2: Final shot for which the calibration is valid
        -# xshift: shift in the x direction to align the map
        -# yshift: shift in the y direction to align the map
        -# xscale: scale in the x direction the align the map
        -# yscale: scale in the y direction to align the map
        -# deg: angle to rotate and align the map
        -# cal_type: Type of calibration [PIX or IMAGE]
        -# diag_ID: Number identifying which fild/inpa/i-HIBP is the 
        calibration for

@section struc_database 
    -# Lines with comments. In principle the default number is 5, but this 
    number can be changes when we read the database, ie, any number can be 
    valid as far as this number is passed as input when the database object is 
    initialised
    -# Database, organised in rows, each row containing all data from one 
    calibration, starting from a calibration ID and followed by all the 
    fields in the order indicated above. See any example for more information
"""