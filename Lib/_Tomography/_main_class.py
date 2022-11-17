"""
Main class to perform tomographic inversions

Jose Rueda-Rueda: jrrueda@us.es

"""
import logging
import numpy as np
import xarray as xr
import Lib.errors as errors
import Lib._Tomography._martix_collapse as matrix
import Lib._Tomography._solvers as solvers
from tqdm import tqdm


# --- Auxiliary objects
logger = logging.getLogger('ScintSuite.Tomography')


class Tomography():
    """
    Main class of the tomography

    Jose Rueda-Rueda: jrrueda@us.es

    Public methods:
        -nnlsq: Perform an nnlsq regression

    Private methods:

    Properties:

    Main attributes:

    """
    def __init__(self, W, s, normalise: bool = True):
        """
        Initialise the class

        :param  W: xr.DataArray with the instrument func created
        :param  s: signal DataArray, extracted from the remap

        Notice that both should share the scintillator grid
        """
        # if np.sum(W.shape[0:2] != s.shape):
        #     print(W.shape[0:2], s.shape[0:2])
        #     raise errors.NotValidInput('revise input shape')
        self.W = W
        self.s = s
        self.inversion = {}
        # --- Now collapse the signal and the weight function
        logger.info('Collapsing W: ')
        self.W2D = matrix.collapse_array4D(self.W.values)
        logger.info('Collapsing Signal')
        self.s1D = matrix.collapse_array2D(self.s.values.squeeze())
        # self.W2D, self.s1D = self._collapseWandS()
        # --- Now normalise them (optional)
        self.norms = {
            's': self.s1D.max(),
            'W': self.W2D.max()
        }
        if normalise:
            self.s1D /= self.norms['s']
            self.W2D /= self.norms['W']

    def nnlsq(self, **kargs):
        """
        Perform an nnlsq inversion

        :param  kargs: optional arguments for the scipy non-negative solver
        """
        logger.info('Performing non-negative least squares regression')
        beta, MSE, res, r2 = solvers.nnlsq(self.W2D, self.s1D, **kargs)
        self.inversion['nnlsq'] = xr.Dataset()
        # -- Reshape the coefficients
        beta_shaped = matrix.restore_array2D(beta, self.W.shape[2],
                                             self.W.shape[3])
        # -- save it in the xr.Dataset()
        self.inversion['nnlsq']['F'] = xr.DataArray(
                beta_shaped, dims=('xp', 'yp'),
                coords={'xp': self.W['xp'], 'yp': self.W['yp']}
        )
        self.inversion['nnlsq']['F'].attrs['long_name'] = 'FI distribution'

        self.inversion['nnlsq']['MSE'] = xr.DataArray(MSE)
        self.inversion['nnlsq']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['nnlsq']['residual'] = xr.DataArray(res)
        self.inversion['nnlsq']['residual'].attrs['long_name'] = 'Residual'

        self.inversion['nnlsq']['r2'] = xr.DataArray(r2)
        self.inversion['nnlsq']['residual'].attrs['long_name'] = '$r^2$'

    def tikhonov0(self, alpha, weights=None, *kargs):
        """
        Perform a 0th order Tikonov regularized regression

        Jose Rueda-Rueda: jrrueda@us.es

        :param  alpha: hyperparameter. Can be a number (single regression) or
            a list or array. In this latter case, the regression will be done
            for each value in the list (array)
        :param  weights: weights, placeholder for the future
        :param  **kargs: extra arguments to initialise skitlearn ridge regressor
        """
        # --- Ensure we have an array or iterable:
        if isinstance(alpha, (list, np.ndarray)):
            alp = alpha
        else:
            alp = np.array([alpha])
        n_alpha = len(alp)
        # --- Initialise the variables
        npoints, nfeatures = self.W2D.shape
        beta = np.zeros((nfeatures, n_alpha))
        MSE = np.zeros(n_alpha)
        r2 = np.zeros(n_alpha)
        res = np.zeros(n_alpha)
        # --- Perform the regression
        logger.info('Performing 0th- Tikhonov regularization')
        for i in tqdm(range(n_alpha)):
            beta[:, i], MSE[i], res[i], r2[i] = \
                solvers.tikhonov0(self.W2D, self.s1D, alp[i], weights=weights
                                  **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping the distribution')
        beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha))
        for i in range(n_alpha):
            beta_shaped[..., i]  = \
                matrix.restore_array2D(beta[:,i], self.W.shape[2],
                                       self.W.shape[3])
        # --- Save it in the dataset
        self.inversion['tikhonov0'] = xr.Dataset()
        self.inversion['tikhonov0']['F'] = xr.DataArray(
                beta_shaped, dims=('xp', 'yp', 'alpha'),
                coords={'xp': self.W['xp'], 'yp': self.W['yp'],
                        'alpha': alp}
        )
        self.inversion['tikhonov0']['F'].attrs['long_name'] = 'FI distribution'

        self.inversion['tikhonov0']['alpha'].attrs['long_name'] = '$\\alpha$'

        self.inversion['tikhonov0']['MSE'] = xr.DataArray(MSE, dims='alpha')
        self.inversion['tikhonov0']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['tikhonov0']['residual'] = xr.DataArray(res,dims='alpha')
        self.inversion['tikhonov0']['residual'].attrs['long_name'] = 'Residual'

        self.inversion['tikhonov0']['r2'] = xr.DataArray(r2, dims='alpha')
        self.inversion['tikhonov0']['residual'].attrs['long_name'] = '$r^2$'

    def nntikhonov0(self, alpha, **kargs):
        """
        Perform a 0th order Tikonov regularized non-negative regression

        Jose Rueda-Rueda: jrrueda@us.es

        :param  alpha: hyperparameter. Can be a number (single regression) or
            a list or array. In this latter case, the regression will be done
            for each value in the list (array)
        :param  weights: weights, placeholder for the future
        :param  **kargs: extra arguments for the nnlsqr solver
        """
        # --- Ensure we have an array or iterable:
        if isinstance(alpha, (list, np.ndarray)):
            alp = alpha
        else:
            alp = np.array([alpha])
        n_alpha = len(alp)
        # --- Initialise the variables
        npoints, nfeatures = self.W2D.shape
        beta = np.zeros((nfeatures, n_alpha))
        MSE = np.zeros(n_alpha)
        r2 = np.zeros(n_alpha)
        res = np.zeros(n_alpha)
        # --- Perform the regression
        logger.info('Performing 0th- Tikhonov regularization')
        for i in tqdm(range(n_alpha)):
            beta[:, i], MSE[i], res[i], r2[i] = \
                solvers.nntikhonov0(self.W2D, self.s1D, alp[i], **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping the distribution')
        beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha))
        for i in range(n_alpha):
            beta_shaped[..., i]  = \
                matrix.restore_array2D(beta[:,i], self.W.shape[2],
                                       self.W.shape[3])
        # --- Save it in the dataset
        self.inversion['nntikhonov0'] = xr.Dataset()
        self.inversion['nntikhonov0']['F'] = xr.DataArray(
                beta_shaped, dims=('xp', 'yp', 'alpha'),
                coords={'xp': self.W['xp'], 'yp': self.W['yp'],
                        'alpha': alp}
        )
        self.inversion['nntikhonov0']['F'].attrs['long_name'] =\
            'FI distribution'

        self.inversion['nntikhonov0']['alpha'].attrs['long_name'] =\
            '$\\alpha$'

        self.inversion['nntikhonov0']['MSE'] = xr.DataArray(MSE, dims='alpha')
        self.inversion['nntikhonov0']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['nntikhonov0']['residual'] = \
            xr.DataArray(res, dims='alpha')
        self.inversion['nntikhonov0']['residual'].attrs['long_name'] =\
            'Residual'

        self.inversion['nntikhonov0']['r2'] = xr.DataArray(r2, dims='alpha')
        self.inversion['nntikhonov0']['residual'].attrs['long_name'] = '$r^2$'






