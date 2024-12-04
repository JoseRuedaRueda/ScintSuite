"""
Routines to interact with the SINPA code

Jose Rueda: jrrueda@us.es

For a full documentation of SINPA please see:
https://gitlab.mpcdf.mpg.de/ruejo/SINPA

Introduced in version 0.6.0
"""
import ScintSuite.SimulationCodes.SINPA._reading as read
import ScintSuite.SimulationCodes.SINPA._execution as execution
from ScintSuite.SimulationCodes.SINPA._Orbits import OrbitClass as orbits
import ScintSuite.SimulationCodes.SINPA._geometry as geometry
from ScintSuite.SimulationCodes.SINPA._INPA_strike_points import INPAStrikes
import ScintSuite.SimulationCodes.SINPA._Forw_Mod_xarray as fmxarray