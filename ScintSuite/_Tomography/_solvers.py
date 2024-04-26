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
from scipy.sparse import linalg as spla
from scipy.sparse import issparse
from random import randint
import time

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

def kaczmarz_solve(W, y, x0, maxiter, damp, tol, relaxParam = 1,**kargs):
        """
        Perform kaczmarz algorithm.

        Marina Jimenez Comez: mjimenez37@us.es

        :param  W: Design matrix
        :param  y: signal
        :param  maxiter: number of iterations where the output is saved
        :param  *kargs: arguments for the kaczmarz algorithm

        :return MSE: Mean squared error
        :return r2: R2 score
        """
        # Start time
        timeStart = time.time()
        timeEnd = np.zeros(len(maxiter))

        # Initialize matrix to return
        xk_output = np.zeros((x0.shape[0], len(maxiter)))

        # Residual of initial guess
        rk = y - W @ x0

        # Initialization before iterations.
        k = 0   # Iteration counter.
        maxK = maxiter.max() # Maximum iterations from all the ksteps to return
        xk = x0 # Use initial vector.
        
        if issparse(W):
            normWi = spla.norm(W, axis=1)**2
        else:
            normWi = np.linalg.norm(W, axis=1)**2

        # Set row order
        I = [i for i in range(len(normWi)) if normWi[i] > 0]
        I = np.random.permutation(I)

        # Apply damping
        normWi += damp * np.max(normWi)

        # Starting loop
        stop_loop = False
        num_exec = 0 # number of times that an element of maxiter is reached

        while not stop_loop:
            k += 1            

            # The Kaczmarz sweep
            for ri in I:
                wi = W[ri,:]
                update = (relaxParam*(y[ri]- wi @ xk)/normWi[ri])*wi
                xk += update
                xk[xk<0]=0

            # Save the output and end time
            if k in maxiter:
                xk_output[:,num_exec] = xk
                timeEnd[num_exec] = time.time()
                num_exec += 1
                

            rk = y - W @ xk
            stop_loop = stopping_criterion(k, rk, maxK, tol)
        

        # Calculated performance metrics
        MSE = np.zeros(len(maxiter))
        r2 = np.zeros(len(maxiter))
        res = np.zeros(len(maxiter))
        duration = np.zeros(len(maxiter))
        for i in range(len(maxiter)):
            y_pred = W @ xk_output[:,i]
            MSE[i] = mean_squared_error(y, y_pred)
            r2[i] = r2_score(y,y_pred)
            res[i] = residual(y_pred, y)
            duration[i] = timeEnd[i] - timeStart


        return xk_output, MSE, res, r2, duration

def coordinate_descent_solve(W, y, x0, maxiter, damp, tol, relaxParam = 1,**kargs):
        """
        Perform coordinate descent algorithm.

        Marina Jimenez Comez: mjimenez37@us.es

        :param  X: Design matrix
        :param  y: signal
        :param  maxiter: maximum number of iterations to perform
        :param  *kargs: arguments for the coordinate descent algorithm

        :return MSE: Mean squared error
        :return r2: R2 score
        """

        # Start time
        timeStart = time.time()
        timeEnd = np.zeros(len(maxiter))

        # Initialize matrix to return
        xk_output = np.zeros((x0.shape[0], len(maxiter)))

        # Residual of initial guess
        rk = y - W @ x0

        # Initialization before iterations.
        k = 0   # Iteration counter.
        maxK = maxiter.max() # Maximum iterations from all the ksteps to return
        xk = x0 # Use initial vector.
        n,m = W.shape

        # Calculate norm of each column
        if issparse(W):
            normWj = spla.norm(W, axis=0)**2
        else:
            normWj = np.linalg.norm(W, axis=0)**2

        # Set column order
        J = [j for j in range(len(normWj)) if normWj[j] > 0]

        # Apply damping
        normWj += damp * np.max(normWj)

        # Set additional CART "flagging" parameters.
        F = np.ones(n)       # Vector of logical "flags."
        Nflag = np.zeros(n)

        # Starting loop
        stop_loop = False
        num_exec = 0 # number of times that an element of maxiter is reached
        Numflag = np.round(maxK/4)
        kbegin = 10
        THR = 1e-4

        while not stop_loop:
            k += 1
            
            for j in J:
                mm = np.max(np.abs(xk))

                if F[j] == 1:
                    wj = W[:,j]
                    update = relaxParam*(np.transpose(wj) @ rk/normWj[j])
                    xk[j] += update
                    xk[xk<0]=0
                    rk = rk - update * wj

                    if k >= kbegin and np.abs(update) < THR*mm:
                        F[j] = 0
                        Nflag[j] = 1

                else:
                    if Nflag[j]< randint(1,Numflag):
                        Nflag[j] += 1
                    else:
                        F[j] = 1
                
            
            # xk[xk<0]=0

            
        
        # Save the output and end time
            if k in maxiter:
                xk_output[:,num_exec] = xk
                timeEnd[num_exec] = time.time()
                num_exec += 1
                

            stop_loop = stopping_criterion(k, rk, maxK, tol)
        

        # Calculated performance metrics
        MSE = np.zeros(len(maxiter))
        r2 = np.zeros(len(maxiter))
        res = np.zeros(len(maxiter))
        duration = np.zeros(len(maxiter))
        for i in range(len(maxiter)):
            y_pred = W @ xk_output[:,i]
            MSE[i] = mean_squared_error(y, y_pred)
            r2[i] = r2_score(y,y_pred)
            res[i] = residual(y_pred, y)
            duration[i] = timeEnd[i] - timeStart


        return xk_output, MSE, res, r2, duration

def cimmino_solve(W, y, x0, maxiter, damp, tol: float, 
                  relaxParam: float = 1.0,**kargs):
        """
        Perform kaczmarz algorithm.

        Marina Jimenez Comez: mjimenez37@us.es

        :param  W: Design matrix
        :param  y: signal
        :param  maxiter: maximum number of iterations to perform
        :param  *kargs: arguments for the kaczmarz algorithm

        :return MSE: Mean squared error
        :return r2: R2 score
        """

        # Start time
        timeStart = time.time()
        timeEnd = np.zeros(len(maxiter))

        # Initialize matrix to return
        xk_output = np.zeros((x0.shape[0], len(maxiter)))

        # Transpose of the matrix
        W_transp = np.transpose(W)

        # Residual of initial guess
        rk = y - W @ x0

        # Initialization before iterations.
        k = 0   # Iteration counter.
        maxK = maxiter.max() # Maximum iterations from all the ksteps to return
        xk = x0 # Use initial vector.
        
        # Calculate norm of each column
        if issparse(W):
            normWi = spla.norm(W, axis=1)**2
        else:
            normWi = np.linalg.norm(W, axis=1)**2

        # Apply damping
        m = W.shape[0]
        normWi = normWi * m + damp * np.max(normWi)

        # Obtain the inverse of the elements of normXj
        inverse_normWi = np.reciprocal(normWi)

        # Create a diagonal matrix with the inverse norms on the diagonal
        M = np.diag(inverse_normWi)

        # Create the identity matrix n x n
        D = np.eye(W.shape[1])

        # Starting loop
        stop_loop = False
        num_exec = 0 # number of times that an element of maxiter is reached

        while not stop_loop:
            k += 1
  
            Mrk = M @ rk
            WTMrk = W_transp @ Mrk
            update = relaxParam*(D @ WTMrk)
            xk += update
            xk[xk<0]=0
        
            # Save the output and end time
            if k in maxiter:
                xk_output[:,num_exec] = xk
                timeEnd[num_exec] = time.time()
                num_exec += 1
                

            rk = y - W @ xk
            stop_loop = stopping_criterion(k, rk, maxK, tol)
        

        # Calculated performance metrics
        MSE = np.zeros(len(maxiter))
        r2 = np.zeros(len(maxiter))
        res = np.zeros(len(maxiter))
        duration = np.zeros(len(maxiter))
        for i in range(len(maxiter)):
            y_pred = W @ xk_output[:,i]
            MSE[i] = mean_squared_error(y, y_pred)
            r2[i] = r2_score(y,y_pred)
            res[i] = residual(y_pred, y)
            duration[i] = timeEnd[i] - timeStart


        return xk_output, MSE, res, r2, duration


def stopping_criterion(k, rk, maxiter, tol):
        """Check if the iteration should terminate.

        :param k: int. The number of iterations that have passed.
        :param xk: (n,) array. The current iterate of the Kaczmarz algorithm.
        :param maxiter: Maximum number of iterations
        :param tol: Permited tolerance

        :return stop: bool. True if the iteration should be terminated.
        """
        if k >= maxiter:
            return True

        if tol is not None:
            residual_norm = np.linalg.norm(rk)

            if residual_norm < tol:
                return True

        return False


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
