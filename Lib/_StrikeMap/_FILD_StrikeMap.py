"""
FILD strike map

Contains the Strike Map object fully adapted to FILD
"""

from Lib._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap


class Fsmap(FILDINPA_Smap):
    """
    Strike Map object adapted to the FILD diagnostic

    Notice that most of the 'heavy' stuff is already in the parent, which is
    common for INPA and FILD. This is just a class for the finner detail,
    mostly related with plotting and visualization

    Jose Rueda Rueda: jrrueda@us.es

    """

    pass
