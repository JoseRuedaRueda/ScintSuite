"""
Plot SINPA geometry

Jose Rueda Rueda: jrrueda@us.es

Introduced in version 6.0.0
"""
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)
