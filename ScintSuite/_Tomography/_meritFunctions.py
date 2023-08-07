"""
Merit functions
"""
import numpy as np

def residual(ypred: np.ndarray, ytrue: np.ndarray):
    """
    Calculate the residual

    Jose Rueda: jrrueda@us.es

    Calculate the sum of the absolute difference between ypred and ytrue

    :param ypred: (np.ndarray) Predicted values
    :param ytrue: (np.ndarray) True values

    :Example:
    >>> ypred = np.random.rand(10)
    >>> ytrue = np.random.rand(10)
    >>> residual(ypred, ytrue)
    """
    return np.sum(abs(ypred - ytrue))