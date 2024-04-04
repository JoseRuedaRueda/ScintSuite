"""
Custom maths models for ScintSuite

This file contains the custom models for the ScintSuite, for example, the rise
cosine model, the decay exponential model, etc.
"""
import math
import numpy as np
from lmfit import Model
from scipy import special
# ------------------------------------------------------------------------------
# %% Auxiliary functions
# ------------------------------------------------------------------------------
def _raised_cosine(x,amplitude,center,sigma,gamma):
    """
    Raised cosine model.
    
    See https://doi.org/10.1088/1361-6587/ad268f for full details
    
    :param x: x values where to evaluate the model
    :param amplitude: amplitude of the model
    :param center: center of the model
    :param sigma: width parameter of the model
    :param gamma: decay parameter of the model
    
    """
    # Only defined inside a sigma interval
    if np.abs(x-center)>sigma:
        return 0.0
    cosine_part = (1.0+np.cos((x-center)/sigma*math.pi))/2.0/sigma
    error_part = 1.0 + special.erf((x-center)/math.sqrt(2.0)*gamma)
    return amplitude*cosine_part*error_part

# ------------------------------------------------------------------------------
# %% Models
# ------------------------------------------------------------------------------
def RaisedCosine():
    """
    Raised cosine model.
    
    See https://doi.org/10.1088/1361-6587/ad268f for full details
    """
    
    return Model(_raised_cosine)