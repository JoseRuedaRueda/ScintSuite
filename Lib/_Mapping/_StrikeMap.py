"""
Strike map class

Jose Rueda: jrrueda@us.es
"""
from Lib._StrikeMap._FILD_StrikeMap import Fsmap
from Lib._StrikeMap._INPA_StrikeMap import Ismap
import Lib.errors as errors
__all__ = ['StrikeMap']


def StrikeMap(id, file):
    """
    Just a wrapper to the Smap library for retro-compatibility

    If you are creating a new method or script, base it in the Smap object
    directly, do not use this method please

    @param id: diagnostic identification:
        -0, 'FILD': would be fild
        -1, 'INPA': would be INPA
    """
    if id == 0 or id == 'FILD':
        return Fsmap(file)
    elif id == 1 or id == 'INPA':
        return Ismap(file)
    else:
        raise errors.NotValidInput('Not understood diagnostic')
