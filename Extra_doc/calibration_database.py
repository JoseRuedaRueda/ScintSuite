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
    initialized
    -# Database, organized in rows, each row containing all data from one
    calibration, starting from a calibration ID and followed by all the
    fields in the order indicated above. See any example for more information
"""
