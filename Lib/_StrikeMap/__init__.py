"""
Package to manage the Strike Map

Contains:
    - GeneralStrikeMap: Parent class, actually not intended to be initialised
    - FILDINPA_Smap: Parent class, actually not intended to be initialised
    - Fsmap: StrikeMap tailored for FILD diagnostic
    - Ismap: StrikeMap tailored for INPA diagnostic
"""

from Lib._StrikeMap._ParentStrikeMap import GeneralStrikeMap
from Lib._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap
from Lib._StrikeMap._FILD_StrikeMap import Fsmap
from Lib._StrikeMap._INPA_StrikeMap import Ismap
