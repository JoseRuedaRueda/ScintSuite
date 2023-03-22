"""
Solvers fortomography.
"""
import logging
import numpy as np
logger = logging.getLogger('ScintSuite.Tomography.Solvers')
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    logger.info('Sklearn intel patched')
except ModuleNotFoundError:
    logger.warning('Sklearn intel not patched, tomography will be slow')
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.optimize import nnls     # Non negative least squares
from Lib._Tomography._meritFunctions import residual
from sklearn.linear_model import ElasticNet  # ElaticNet


# -----------------------------------------------------------------------------
# --- SOLVERS AND REGRESSION ALGORITHMS
# -----------------------------------------------------------------------------
def ols(X, y):
    """
    Perform an OLS inversion using the analytical solution.

    Jose Rueda: jrrueda@us.es

    :param  X: Design matrix
    :param  y: signal
    :return beta: best fit coefficients

    :return MSE: Mean squared error
    :return r2: R2 score
    """
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    res = residual(y_pred, y)
    return beta, MSE, res, r2


def nnlsq(X, y, **kargs):
    """
    Perform a non-negative least squares inversion using scipy.

    Jose Rueda: jrrueda@us.es

    :param  X: Design matrix
    :param  y: signal
    :param  param: dictionary with options for the nnls solver (see scipy)

    :return beta: best fit coefficients
    :return MSE: Mean squared error
    :return r2: R2 score
    """
    beta, dummy = nnls(X, y, **kargs)
    y_pred = X @ beta
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    # yes the dummy output is already the residual, but for coherency with the
    # other methods, I call this residual function
    res = residual(y_pred, y)
    return beta, MSE, res, r2


def tikhonov0(X, y, alpha, weight=None, **kargs):
    """
    Perform a Ridge (0th Tikhonov) regression.

    Jose Rueda: jrrueda@us.es

    :param  X: Design matrix
    :param  y: signal
    :param  alpha: hyperparameter
    :param  weight: weight of the samples, for the regression
    :param  *kargs: arguments for the Ridge regressor

    :return ridge.coef_: best fit coefficients
    :return MSE: Mean squared error
    :return r2: R2 score
    """
    ridge = Ridge(alpha, **kargs)
    ridge.fit(X, y, sample_weight=weight)
    y_pred = ridge.predict(X)
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    res = residual(y_pred, y)
    return ridge.coef_, MSE, res, r2


def nntikhonov0(X, y, alpha, **kargs):
    """
    Perform a non-negative Ridge inversion

    :param  X: Design matrix
    :param  y: signal
    :param  alpha: hyperparameter
    :param  param. dictionary with extra parameters for scipy.nnls
    :return ridge.coef_: best fit coefficients
    :return MSE: Mean squared error
    :return r2: R2 score
    """
    # Auxiliar arrays:
    n1, n2 = X.shape
    L = np.eye(n2)
    GalphaL0 = np.zeros((n2, 1))
    # Extended design matrix
    WalphaL = np.vstack((X, np.sqrt(alpha) * L))
    GalphaL = np.vstack((y[:, np.newaxis], GalphaL0)).squeeze()
    # Non-negative ols solution:
    beta, dummy = nnls(WalphaL, GalphaL, **kargs)
    y_pred = X @ beta
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    res = residual(y_pred, y)
    return beta, MSE, res, r2


def Elastic_Net(X, y, alpha, l1_ratio=0.05, positive=True, max_iter=1000,
                **kargs):
    """
    Wrap for the elastic net function

    Jose Rueda: jrrueda@us.es

    :param  X: Design matrix
    :param  y: signal
    :param  alpha: hyperparameter
    :param  l1_ratio: hyperparameter of the ElasticNet
    :param  positive: flag to force positive coefficients

    :return reg.coef_: best fit coefficients
    :return MSE: Mean squared error
    :return r2: R2 score
    """
    # --- Initialise the regresor
    reg = ElasticNet(alpha=alpha, positive=positive, l1_ratio=l1_ratio,
                     max_iter=max_iter)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    res = residual(y_pred, y)
    return reg.coef_, MSE, res, r2
