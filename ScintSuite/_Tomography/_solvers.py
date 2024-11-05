"""
Solvers for tomography reconstructions.
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
from ScintSuite._Tomography._meritFunctions import residual
from sklearn.linear_model import ElasticNet  # ElaticNet
from scipy.sparse import diags
from scipy.sparse.linalg import inv
from scipy.optimize import minimize, Bounds

# -----------------------------------------------------------------------------
# --- SOLVERS AND REGRESSION ALGORITHMS
# -----------------------------------------------------------------------------
def ols(X: np.ndarray, y: np.ndarray):
    """
    Perform an OLS inversion using the analytical solution.

    Jose Rueda: jrrueda@us.es

    :param  X: Design matrix
    :param  y: signal

    :return beta: best fit coefficients
    :return MSE: Mean squared error
    :return res: Residual
    :return r2: R2 score

    :Example:
    >>> X = np.random.rand(10, 5)
    >>> y = np.random.rand(10)
    >>> beta, MSE, res, r2 = ols(X, y)
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
    :return res: Residual
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


def nntikhonov0projective(X, y, alpha, weight=None, **kargs):
    """
    Perform a Ridge (0th Tikhonov) regression and project in the positive space

    Jose Rueda: jruedaru@uci.edu
    
    Yes, this is just a fancy name for a common 0th Tikhonov regression with
    sol[sol<0] = 0

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
    ridge.coef_[ridge.coef_ < 0] = 0.0
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


# -----------------------------------------------------------------------------
# --- Maximum entropy regularization
# -----------------------------------------------------------------------------
class EntropyFunctional:
    """
    Auxiliary class to perform maximum entropy regularization.
    
    Transated from Julia code written by Luke Stagner:
    https://github.com/lstagner/PPCF-58-4-2016/blob/master/PPCF16/src/entropy.jl
    
    :param A: Design matrix
    :param b: Signal vector
    :param alpha: hyperparameter (weight of the entropy)
    :param d: default solution
    :param tol: tolerance for the miminization
    
    """
    
    def __init__(self, A, b, alpha=1e0, d=None,
                 tol: float = 1.0e-5):
        self.A = A
        self.b = b
        self.m = None
        self.alpha = np.atleast_1d(np.array([alpha]))
        self.d = d if d is not None else np.mean(b) / np.mean(self.A, axis=0)
        self.tol = tol
        
    def _objective(self, x):
        # define the objective function to be minimized
        entropy = -self.alpha[0] * np.sum(x - self.d - x*np.log(x/self.d))
        residual = 0.5 * np.sum((self.A @ x - self.b)**2)
        return entropy + residual

    def solve(self):
        # solve the optimization problem
        bounds = Bounds([0]*len(self.d), [np.inf]*len(self.d))
        cons = [{'type': 'ineq', 'fun': lambda x: x}]
        self.m = minimize(self._objective, self.d, 
                          bounds=bounds, constraints=cons, tol=self.tol,
                          options={'maxiter': 500})
        beta = self.m.x
        ypredict = self.A @ beta
        MSE = mean_squared_error(self.b, ypredict)
        r2 = r2_score(self.b, ypredict)
        res = residual(ypredict, self.b)
        return self.m.x, MSE, res, r2
