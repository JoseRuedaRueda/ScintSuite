"""
Strike map class

Jose Rueda: jrrueda@us.es

This is deprecated and just left here for retrocompatibility.
Please if you were to implement something new, rely on the smap object
and not on this
"""
from ScintSuite.decorators import deprecated
from ScintSuite._StrikeMap._FILD_StrikeMap import Fsmap
from ScintSuite._StrikeMap._INPA_StrikeMap import Ismap
import ScintSuite.errors as errors
__all__ = ['StrikeMap']


@deprecated('Please call directly the smap library')
def StrikeMap(id, file):
    """
    Just a wrapper to the Smap library for retro-compatibility

    If you are creating a new method or script, base it in the Smap object
    directly, do not use this method please

    :param  id: diagnostic identification:
        -0, 'FILD': would be fild
        -1, 'INPA': would be INPA
    """
    if id == 0 or id == 'FILD':
        return Fsmap(file)
    elif id == 1 or id == 'INPA':
        return Ismap(file)
    else:
        raise errors.NotValidInput('Not understood diagnostic')
