"""
Sublibrary to load and process video files

Jose RuedaRueda: jrrueda@us.es   &   Pablo Oyola - pablo.oyola@ipp.mpg.de

It contains several subpackages:
    - AuxFunction: Auxiliary functions such as the rgb2gray
    - BasicVideoObject: Contain the basic video object
    - CinFiles: routines to read cin files
    - PNGfiles: routines to read png files
    - Mp4files: routines to read mp4 files
    - TIFfiles: routines to read tif files
    - FILDVideoObject: Contain the child video object
    - VRTVideoObject: Contain the child video object

General user should not need each individual function of the package, just use
the BVO, FILDVideo, VRTvideo, guess_filename as defined below
"""
from Lib.LibVideo._BasicVideoObject import BVO
from Lib.LibVideo._FILDVideoObject import FILDVideo
from Lib.LibVideo._INPAVideoObject import INPAVideo
from Lib.LibVideo.VRTVideoObject import VRTVideo

from Lib.LibMachine import machine
if machine == 'AUG':
    from Lib.LibVideo._iHIBPvideoObject import iHIBPvideo
