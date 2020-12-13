"""@page Changes Changes repect FILD GUI
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

@section png_files Normalization of the imported PNGs
    -# By default, idl read the pixel value of the file but python read it
    normalise to the maximum of the format; ie, imagine a pixel with 1200
    counts in a png writen in 16 bits format: python will load it as
    p=1200/2**16, while idl as p=1200. This only affect to the absolute value
    and have zero influence in the value from frame to frame so things should
    be fine. But if you want to match a time trace calculate with this suite
    with one calculated with the old one, just multiply by 2**12 (or the bit
    size of your png files!!)
"""
