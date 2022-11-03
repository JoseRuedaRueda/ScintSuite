"""
Module to perform the mapping of the camera frames

Written by jrrueda@us.es, revised and polished by lvelarde@us.es

In principle, none of the routines in this module is intended to be called
directly by the user. They will be all called by the video object.

The strike map object present in this module is no longer the main strike map
since version 1.0.0 this is just a wrapper. Please use the _StrikeMap library
directly if you need a strike map.
"""
from Lib._Mapping._Calibration import CalParams, CalibrationDatabase, readTimeDependentCalibration
from Lib._Mapping._StrikeMap import StrikeMap
from Lib._Mapping._Common import *
from Lib._Mapping._FILD import *
from Lib._Mapping._INPA import remapAllLoadedFrames as remapAllLoadedFramesINPA
from Lib._Mapping._Scintillator import Scintillator
