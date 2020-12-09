"""@page Changes Changes repect FILD GUI
Main changes respect to FILD GUI

@section Tomography For Tomography and absolute flux
        -# area_pixel: The pixel are is no longer a number, but a matrix of
        size frame_shape -> for future distorted cases

@section struc_database
    -# Lines with comments. In principle the default number is 5, but this
    number can be changes when we read the database, ie, any number can be
    valid as far as this number is passed as input when the database object is
    initialised
    -# Database, organised in rows, each row containing all data from one
    calibration, starting from a calibration ID and followed by all the
    fields in the order indicated above. See any example for more information
"""
