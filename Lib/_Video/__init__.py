"""
Sublibrary to load and process video files.

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
# ---- Load the video objects
from Lib._Video._BasicVideoObject import BVO
from Lib._Video._FILDVideoObject import FILDVideo
from Lib._Video._INPAVideoObject import INPAVideo
from Lib._Video._VRTVideoObject import VRTVideo
# ---- Load the auxiliar libraries
import Lib._Video._PNGfiles as png
import Lib._Video._TIFfiles as tif
import Lib._Video._PCOfiles as pco

from Lib._Machine import machine as _machine
if _machine == 'AUG':
    from Lib._Video._iHIBPvideoObject import iHIBPvideo
    from Lib._Video._iHIBP_beam_camera import beam_camera as ihibp_beam_camera
