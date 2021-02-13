"""@page Changes Changes respect FILD GUI
Main changes respect to FILD GUI

@section Tomography For Tomography and absolute flux
        -# area_pixel: The pixel are is no longer a number, but a matrix of
        size frame_shape -> for future distorted cases

@section struc_database Database files
    -# Lines with comments. In principle the default number is 5, but this
    number can be changes when we read the database, ie, any number can be
    valid as far as this number is passed as input when the database object is
    initialised
    -# Database, organised in rows, each row containing all data from one
    calibration, starting from a calibration ID and followed by all the
    fields in the order indicated above. See any example for more information

@section resolutions FILD resolution calculation
    -# By default, an adaptive bin width is used such that there are 4 bins
    in a standard deviation of the probability distribution. In this way
    we ensure to have enough bins to make the fitting without having too few
    counts in each bin (example, before for low gyroradii you needed like 0.01
    cm of bin width but if we use that bin width for large gyroradii, or we
    launch tens of thousand of markers or there was too much noise)
"""
