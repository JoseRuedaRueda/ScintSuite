"""
Strike map class

Jose Rueda: jrrueda@us.es
"""
from Lib._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap
from Lib._StrikeMap._INPA_StrikeMap import Ismap
import Lib.errors as errors
__all__ = ['StrikeMap']


def StrikeMap(id, file):
    """
    Just a wrapper to the Smap library for retrocompatibility

    If you are creating a new method or script, base it in the Smap object
    directly, do not use this method please

    @param id: diagnostic identification:
        -0, 'FILD': would be fild
        -1, 'INPA': would be INPA
    """
    if id == 0 or id.lower() == 'fild':
        return FILDINPA_Smap(file)
    elif id == 1 or id.lower() == 'inpa':
        return Ismap(file)
    else:
        raise errors.NotValidInput('Not understood diagnostic')
