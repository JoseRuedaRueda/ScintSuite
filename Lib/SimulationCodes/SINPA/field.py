"""
Library to prepare and plot the input magnetic field to SINPA

Jose Rueda: jrrueda@us.es

Introduced in version 0.5.9
"""
import os
import f90nml
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import Lib.LibData as ssdat
import Lib.LibPlotting as ssplt
from Lib.LibPaths import Path
from Lib.LibMachine import machine
paths = Path(machine)


# -----------------------------------------------------------------------------
# --- Angles and orientation
# -----------------------------------------------------------------------------
def calculate_sinpa_angles(B: np.ndarray, geomID: str = 'Test'):
    """
    Calculate the zita and epsilon angles for SINPA

    Jose Rueda: jrrueda@us.es

    @param B: Magnetic field vector (full or just unit vector)
    @param geomID: ID of the geometry to be used

    @return zita: zita angle [deg]
    @return ipsilon: ipsilon angle [deg]
    """
    # --- Load the vectors
    filename = os.path.join(paths.SINPA, 'Geometry', geomID,
                            'ExtraGeometryParams.txt')
    nml = f90nml.read(filename)
    u1 = np.array(nml['ExtraGeometryParams']['u1'])
    u2 = np.array(nml['ExtraGeometryParams']['u2'])
    u3 = np.array(nml['ExtraGeometryParams']['u3'])

    # --- Calculate the zeta angle
    bmod = math.sqrt(np.sum(B * B))
    zita = math.acos(np.sum(B * u3) / bmod) * 180. / math.pi

    # --- Calculate the ipsilon angle
    ipsilon = math.atan2(np.sum(B * u2), np.sum(B * u1)) * 180. / math.pi
    return zita, ipsilon


def constructDirection(zita, ipsilon, geomID: str = 'Test'):
    """
    Calculate the zita and epsilon angles for SINPA

    Jose Rueda: jrrueda@us.es

    @param zita: zita angle [deg]
    @param ipsilon: ipsilon angle [deg]
    @param geomID: ID of the geometry to be used

    @return B: director vector
    """
    # --- Load the vectors
    filename = os.path.join(paths.SINPA, 'Geometry', geomID,
                            'ExtraGeometryParams.txt')
    nml = f90nml.read(filename)
    u1 = np.array(nml['ExtraGeometryParams']['u1'])
    u2 = np.array(nml['ExtraGeometryParams']['u2'])
    u3 = np.array(nml['ExtraGeometryParams']['u3'])

    direction = (math.cos(ipsilon * math.pi / 180.0) * u1
                 + math.sin(ipsilon * math.pi / 180.0) * u2) \
        * math.sin(zita * math.pi / 180.0) \
        + math.cos(zita * math.pi / 180.0) * u3
    return direction
