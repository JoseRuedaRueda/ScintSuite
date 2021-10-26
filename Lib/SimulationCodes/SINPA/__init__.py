"""
Routines to interact with the SINPA code

Jose Rueda: jrrueda@us.es

For a full documentation of SINPA please see:
https://gitlab.mpcdf.mpg.de/ruejo/SINPA

Introduced in version 0.6.0
"""
import Lib.SimulationCodes.SINPA.field as field
import Lib.SimulationCodes.SINPA.geometry as geometry
import Lib.SimulationCodes.SINPA.reading as read
# import Lib.SimulationCodes.SINPA.Plot as plt
import Lib.SimulationCodes.SINPA.execution as execution
from Lib.SimulationCodes.SINPA.LibStrike import Strikes as strikes
from Lib.SimulationCodes.SINPA.Orbits import OrbitClass as orbits
from Lib.SimulationCodes.Common.fields import fields as fieldObject
