"""
Contain the class to prepare and apply the INPA instrument function

HIGHLY under development. Do not Use if you do not know what are you doing
"""
import os
import time
import logging
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.sparse import lil_matrix
from ScintSuite._SideFunctions import createGrid1D

# ---  Initialise the logger
logger = logging.getLogger('ScintSuite.INPAwf')
# Import optional modules
try:
    from pyLib._FBM_class import FBM
except ModuleNotFoundError:
    logger.warning('10: FIDASIM not found, not able to get INPAwf')
INPA_IF =2
# ------------------------------------------------------------------------------
# --- Main class
# # ------------------------------------------------------------------------------
# class INPAwf():
#     """
#     INPA weight function
#     """
#     def __init__():