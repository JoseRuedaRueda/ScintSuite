"""
Plotting module of the Scintillator Suite

Each object has its own plotting routines, this library just contain auxiliary
routines for axis beauty, customization, vessel, etc

Submodules:
    -1D_plotting
    -3D_plotting
    -ColorMaps: contain custom colormaps
    -Cursors: contain cursos to help analysis of data
    -Others: ECRH and fluxsurface
    -Settings: contain routines to set the default matplotlib behaviour and
        axis beauty
    -vessel: tokamak vessel
"""
from Lib._Plotting._1D_plotting import *
from Lib._Plotting._3D_plotting import *
from Lib._Plotting._ColorMaps import *
from Lib._Plotting._Cursors import *
from Lib._Plotting._Others import *
from Lib._Plotting._settings import *
from Lib._Plotting._vessel import *
