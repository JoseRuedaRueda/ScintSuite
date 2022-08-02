"""
Routines to interact with the SINPA code

Jose Rueda: jrrueda@us.es

For a full documentation of SINPA please see:
https://gitlab.mpcdf.mpg.de/ruejo/SINPA

Introduced in version 0.6.0
"""
import Lib.SimulationCodes.SINPA._reading as read
import Lib.SimulationCodes.SINPA._execution as execution
from Lib.SimulationCodes.SINPA._Orbits import OrbitClass as orbits
import Lib.SimulationCodes.SINPA._geometry as geometry
from Lib.SimulationCodes.SINPA._INPA_strike_points import INPAStrikes
