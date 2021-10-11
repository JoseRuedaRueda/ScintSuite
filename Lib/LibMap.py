"""Module to remap the scintillator

It contains the routines to load and align the strike maps, as well as
perform the remapping. Contain the classes: Scintillator(), StrikeMap(),
CalibrationDataBase(), Calibration()
"""
# import time
import math
import datetime
import time
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interp
import Lib.LibPlotting as ssplt
import Lib.LibFILDSIM as ssFILDSIM
import Lib.LibUtilities as ssextra
from Lib.LibMachine import machine
import Lib.LibPaths as p
import Lib.LibIO as ssio
import Lib.LibData as ssdat
from tqdm import tqdm   # For waitbars
pa = p.Path(machine)
del p
