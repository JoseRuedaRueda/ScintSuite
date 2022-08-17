"""
Merit functions
"""
import numpy as np

def residual(ypred, ytrue):
    """
    Calculate the residual

    Jose Rueda: jrrueda@us.es

    Calculate the sum of the absolute difference between ypred and ytrue
    """
    return np.sum(abs(ypred - ytrue))