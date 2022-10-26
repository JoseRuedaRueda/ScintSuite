"""Scintillator class."""
import numpy as np
from Lib._Scintillator._mainClass import Scintillator as Scint
from Lib.decorators import deprecated
__all__ = ['Scintillator']


# ------------------------------------------------------------------------------
# --- Scintillator object
# ------------------------------------------------------------------------------
@deprecated('Use the new object of the scintillator library')
def Scintillator(file: str, format = None,
                 material: str = 'TG-green'):
    """
    Wrapper to the new object, better call it directly
    """
    if format is None:
        if file.endswith('txt'):
            format='SINPA'
        else:
            format='FILDSIM'
    return Scint(file=file, format=format, material=material)
