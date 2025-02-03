"""
Main class to perform tomographic inversions.

Jose Rueda-Rueda: jrrueda@us.es

"""
import os
import json
import logging
import tarfile
import numpy as np
import xarray as xr
import ScintSuite.errors as errors
import matplotlib.pyplot as plt
import ScintSuite._Tomography._martix_collapse as matrix
import ScintSuite._Tomography._solvers as solvers
from tqdm import tqdm
from scipy import linalg
from ScintSuite.version_suite import exportVersion
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider


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
    def __init__(self, W=None, s=None, normalise: bool = True,
                 folder: str = None):
        """
        Initialise the class

        :param  W: xr.DataArray with the instrument func created
        :param  s: signal DataArray, extracted from the remap
        :param normalise: if true, signal and W will be normalised to 1
        :param folder: if not None, s and W inputs will be ignored and data will
            be load form the folder (it is supposed to be fille with the files
            created by the export method())


        Notes:
            - Both should share the scintillator grid
        """
        if folder is None:
            if np.sum(W.shape[0:2] != s.shape):
                print(W.shape[0:2], s.shape[0:2])
                raise errors.NotValidInput('revise input shape')
            self.W = W
            self.s = s
            self.inversion = {}
            # --- Get the shape of W
            self.Wndims = len(W.shape)
            self.sndims = len(s.shape)
            # --- Now collapse the signal and the weight function
            logger.info('Collapsing W: ')
            if self.Wndims == 4:
                self.W2D = matrix.collapse_array4D(self.W.values)
            elif self.Wndims == 5:
                self.W2D = matrix.collapse_array5D(self.W.values)
            else:
                raise errors.NotValidInput('W should be 4 or 5D')
                
            logger.info('Collapsing Signal')
            self.s1D = matrix.collapse_array2D(self.s.values.squeeze())
            # self.W2D, self.s1D = self._collapseWandS()
            # --- Now normalise them (optional)
            self.norms = {
                's': self.s1D.max(),
                'W': self.W2D.max(),
                'normalised': np.array([0]),
            }
            if normalise:
                self.s1D /= self.norms['s']
                self.W2D /= self.norms['W']
                self.norms['normalised'][0] = 1
            else:
                self.norms['normalised'][0] = 0
            self.folder = None
        else:
            if W is not None or s is not None:
                logger.warning('30: Folder argument present, ignorig W and s.')
            logger.info('Reading results from %s', folder)
            self.W = xr.load_dataarray(os.path.join(folder, 'WeightFunc.nc'))
            self.s = xr.load_dataarray(os.path.join(folder, 'Signal.nc'))
            self.inversion = {}
            try:
                self.norms = json.load(open(os.path.join(folder, 'norms.json')))
                needGuess = False
            except FileNotFoundError:
                text = 'Old tomography files, not present norms.' +\
                    'it will be assumed data was normalised'
                logger.warning(text)
                self.norms = {}
                needGuess = True
            # --- Get the shape of W
            self.Wndims = len(self.W.shape)
            self.sndims = len(self.s.shape)
            # --- Now collapse the signal and the weight function
            logger.info('Collapsing W ')
            if self.Wndims == 4:
                self.W2D = matrix.collapse_array4D(self.W.values)
            elif self.Wndims == 5:
                self.W2D = matrix.collapse_array5D(self.W.values)
            else:
                raise errors.NotValidInput('W should be 4 or 5D')
            logger.info('Collapsing Signal')
            self.s1D = matrix.collapse_array2D(self.s.values.squeeze())
            if needGuess:
                self.norms = {
                    's': self.s1D.max(),
                    'W': self.W2D.max(),
                    'normalised': np.array([1]),
                }
            # --- Now load the inversions
            supportedFiles = ['nnelasticnet.nc', 'nntikhonov0.nc',
                              'tikhonov0.nc', 'nntikhonov0projective.nc',
                              'nnlsq.nc', 'maxEntropy.nc']
            for file in supportedFiles:
                filename = os.path.join(folder, file)
                if os.path.isfile(filename):
                    key = file.split('.')[0]
                    logger.info('reading %s', filename)
                    self.inversion[key] = xr.load_dataset(filename)
            self.folder = folder

    # -------------------------------------------------------------------------
    # %% Weight decomposition and manipulation block
    # -------------------------------------------------------------------------
    def svdWeightMatrix(self, **kargs) -> xr.Dataset:
        """
        """
        # ---- Calculate the decomposition mamtrices
        U, s, Vh = linalg.svd(self.W2D)
        # ---- Allocate the output matrices
        ns = s.size
        Ureshaped = np.zeros((ns, self.W.shape[0], self.W.shape[1]))
        Vreshaped = np.zeros((ns, self.W.shape[2], self.W.shape[3]))
        for js in range(ns):
            Ureshaped[js, :, :] =\
                matrix.restore_array2D(U[:, js], self.W.shape[0], 
                                       self.W.shape[1])            
                                       
            Vreshaped[js, :, :] =\
                matrix.restore_array2D(Vh[js, :], self.W.shape[2], 
                                       self.W.shape[3])
        # The transpose is because we need the matrix V, and the scipy output is
        # the v transposed.
        out = xr.Dataset()
        out['U'] = xr.DataArray(Ureshaped, dim=('sigma', 'xs', 'ys'),
                                coords={'sigma': s, 'xs': self.W.xs,
                                        'ys': self.W.ys})
        out['V'] = xr.DataArray(Vreshaped, dims=('sigma', 'x', 'y'),
                                coords={'sigma': s, 'x': self.W.x, 'y': self.W.y})
        return out

    
    def nnlsq(self, **kargs) -> None:
        """
        Perform an nnlsq inversion

        :param  kargs: optional arguments for the scipy non-negative solver
        """
        logger.info('Performing non-negative least squares regression')
        beta, MSE, res, r2 = solvers.nnlsq(self.W2D, self.s1D, **kargs)
        self.inversion['nnlsq'] = xr.Dataset()
        # -- Reshape the coefficients
        if self.Wndims == 4:
            beta_shaped = matrix.restore_array2D(beta, self.W.shape[2],
                                                self.W.shape[3])
        else:
            beta_shaped = matrix.restore_array3D(beta, self.W.shape[2],
                                                self.W.shape[3],
                                                self.W.shape[4])
        # -- save it in the xr.Dataset()
        if self.Wndims == 4:
            self.inversion['nnlsq']['F'] = xr.DataArray(
                beta_shaped, dims=('x', 'y'),
                coords={'x': self.W['x'], 'y': self.W['y']}
            )
        else:
            self.inversion['nnlsq']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'z'),
                    coords={'x': self.W['x'], 'y': self.W['y'],
                            'z': self.W['z']}
            )
        self.inversion['nnlsq']['F'].attrs['long_name'] = 'FI distribution'

        self.inversion['nnlsq']['MSE'] = xr.DataArray(MSE)
        self.inversion['nnlsq']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['nnlsq']['residual'] = xr.DataArray(res)
        self.inversion['nnlsq']['residual'].attrs['long_name'] = 'Residual'

        self.inversion['nnlsq']['r2'] = xr.DataArray(r2)
        self.inversion['nnlsq']['residual'].attrs['long_name'] = '$r^2$'

    def kaczmarz_solve(self, x0, iterations, window = None, 
                       damp = None, tol = None, 
                       relaxParam = 1, **kargs) -> None:
        """
        Perform kaczmarz algorithm

        Marina Jimenez Comez: mjimenez37@us.es

        :param iterations: number of iterations that the user wants the 
        algorithmn to perform
        :param  **kargs: extra arguments 
        """
        # --- Ensure we have an array or iterable:
        if isinstance(iterations, (list, np.ndarray)):
            numIter = iterations
        else:
            numIter = np.array([iterations])
        n_execution = len(numIter)
        # --- Perform the algorithm
        logger.info('Performing kaczmarz algorithm')
        x_hat, MSE, res, r2, time = \
                solvers.kaczmarz_solve(self.W2D, self.s1D, x0, maxiter = numIter,
                                       damp = damp, tol = tol, 
                                       relaxParam = relaxParam,
                                        x_coord = self.W['x'], 
                                        y_coord = self.W['y'],
                                        window = window, 
                                        **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping prediction')
        x_hat_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_execution))
        for i in range(n_execution):
            x_hat_shaped[..., i] = \
                matrix.restore_array2D(x_hat[:, i], self.W.shape[2],
                                       self.W.shape[3])
        # --- Save it in the dataset
        self.inversion['kaczmarz'] = xr.Dataset()
        self.inversion['kaczmarz']['F'] = xr.DataArray(
                x_hat_shaped, dims=('x', 'y','alpha'),
                coords={'x': self.W['x'], 'y': self.W['y'], 'alpha': numIter}
        )
        self.inversion['kaczmarz']['F'].attrs['long_name'] = 'FI distribution'
        self.inversion['kaczmarz']['alpha'].attrs['long_name'] = 'alpha'

        self.inversion['kaczmarz']['MSE'] = xr.DataArray(MSE, dims='alpha')
        self.inversion['kaczmarz']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['kaczmarz']['residual'] = xr.DataArray(res,dims='alpha')
        self.inversion['kaczmarz']['residual'].attrs['long_name'] = 'Residual'

        self.inversion['kaczmarz']['r2'] = xr.DataArray(r2, dims='alpha')
        self.inversion['kaczmarz']['residual'].attrs['long_name'] = '$r^2$'

        self.inversion['kaczmarz']['time'] = xr.DataArray(time, dims='alpha')
        self.inversion['kaczmarz']['time'].attrs['long_name'] = '$t (s)$'

    def coordinate_descent_solve(self, x0, iterations = None, window = None,
                                 pitch_map = None, gyro_map = None,
                                 resolution = False, peak_amp = 0.15,
                                  damp = None, tol = None, 
                                  relaxParam = 1, control_iters = 100,
                                    **kargs) -> None:
        """
        Perform coordinate descent algorithm

        Marina Jimenez Comez: mjimenez37@us.es

        :param iterations: number of iterations that the user wants the 
        algorithmn to perform
        :param  **kargs: extra arguments 
        """

        # --- Ensure we have an array or iterable:
        n_execution = 1
        if iterations is not None:
            if not isinstance(iterations, (list, np.ndarray)):
                iterations = np.array([iterations])
            n_execution = len(iterations)

        # --- Perform the algorithm
        logger.info('Performing coordinate descent algorithm')
        x_hat, MSE, res, r2, time, alphas = \
                solvers.coordinate_descent_solve(self.W2D, self.s1D, x0,
                                                  maxiter = iterations,
                                                  pitch_map = pitch_map,
                                                  gyro_map = gyro_map,
                                                  resolution = resolution,
                                                  peak_amp = peak_amp,
                                                  damp = damp, tol = tol, 
                                                  relaxParam = relaxParam, 
                                                  control_iters = control_iters,
                                                  x_coord = self.W['x'], 
                                                  y_coord = self.W['y'],
                                                  window = window,
                                                  **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping prediction')
        x_hat_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_execution))
        for i in range(n_execution):
            x_hat_shaped[..., i] = \
                matrix.restore_array2D(x_hat[:, i], self.W.shape[2],
                                       self.W.shape[3])
        # --- Save it in the dataset
        self.inversion['descent'] = xr.Dataset()
        self.inversion['descent']['F'] = xr.DataArray(
                x_hat_shaped, dims=('x', 'y','alpha'),
                coords={'x': self.W['x'], 'y': self.W['y'], 'alpha': alphas}
        )
        self.inversion['descent']['F'].attrs['long_name'] = 'FI distribution'
        self.inversion['descent']['alpha'].attrs['long_name'] = 'alpha'

        self.inversion['descent']['MSE'] = xr.DataArray(MSE, dims='alpha')
        self.inversion['descent']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['descent']['residual'] = xr.DataArray(res,dims='alpha')
        self.inversion['descent']['residual'].attrs['long_name'] = 'Residual'

        self.inversion['descent']['r2'] = xr.DataArray(r2, dims='alpha')
        self.inversion['descent']['residual'].attrs['long_name'] = '$r^2$'

        self.inversion['descent']['time'] = xr.DataArray(time, dims='alpha')
        self.inversion['descent']['time'].attrs['long_name'] = '$t (s)$'


    def cimmino_solve(self, x0, iterations, window = None,
                       damp = None, tol = None, 
                       relaxParam = 1, **kargs) -> None:
        """
        Perform cimmino algorithm

        Marina Jimenez Comez: mjimenez37@us.es

        :param iterations: number of iterations that the user wants the 
        algorithmn to perform
        :param  **kargs: extra arguments 
        """

        # --- Ensure we have an array or iterable:
        if isinstance(iterations, (list, np.ndarray)):
            numIter = iterations
        else:
            numIter = np.array([iterations])
        n_execution = len(numIter)
        # --- Perform the algorithm
        logger.info('Performing cimmino algorithm')
        x_hat, MSE, res, r2, time = \
                solvers.cimmino_solve(self.W2D, self.s1D, x0,
                                                  maxiter = numIter,
                                                  damp = damp, tol = tol, 
                                                  relaxParam = relaxParam, 
                                                  x_coord = self.W['x'], 
                                                  y_coord = self.W['y'],
                                                  window = window,
                                                  **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping prediction')
        x_hat_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_execution))
        for i in range(n_execution):
            x_hat_shaped[..., i] = \
                matrix.restore_array2D(x_hat[:, i], self.W.shape[2],
                                       self.W.shape[3])
        # --- Save it in the dataset
        self.inversion['cimmino'] = xr.Dataset()
        self.inversion['cimmino']['F'] = xr.DataArray(
                x_hat_shaped, dims=('x', 'y','alpha'),
                coords={'x': self.W['x'], 'y': self.W['y'], 'alpha': numIter}
        )
        self.inversion['cimmino']['F'].attrs['long_name'] = 'FI distribution'
        self.inversion['cimmino']['alpha'].attrs['long_name'] = 'alpha'

        self.inversion['cimmino']['MSE'] = xr.DataArray(MSE, dims='alpha')
        self.inversion['cimmino']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['cimmino']['residual'] = xr.DataArray(res,dims='alpha')
        self.inversion['cimmino']['residual'].attrs['long_name'] = 'Residual'

        self.inversion['cimmino']['r2'] = xr.DataArray(r2, dims='alpha')
        self.inversion['cimmino']['residual'].attrs['long_name'] = '$r^2$'

        self.inversion['cimmino']['time'] = xr.DataArray(time, dims='alpha')
        self.inversion['cimmino']['time'].attrs['long_name'] = '$t (s)$'


    def tikhonov0(self, alpha, weights=None, **kargs) -> None:
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
                solvers.tikhonov0(self.W2D, self.s1D, alp[i], weight=weights,
                                  **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping the distribution')
        if self.Wndims == 4:
            beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha))
            for i in range(n_alpha):
                beta_shaped[..., i] = \
                    matrix.restore_array2D(beta[:, i], self.W.shape[2],
                                           self.W.shape[3])
        elif self.Wndims == 5:
            beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3],
                                  self.W.shape[4], n_alpha))
            for i in range(n_alpha):
                beta_shaped[..., i] = \
                    matrix.restore_array3D(beta[:, i], self.W.shape[2],
                                           self.W.shape[3], self.W.shape[4])

        # --- Save it in the dataset
        self.inversion['tikhonov0'] = xr.Dataset()
        if self.Wndims == 4:
            self.inversion['tikhonov0']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'alpha'),
                    coords={'x': self.W['x'], 'y': self.W['y'],
                            'alpha': alp}
            )
        else:
            self.inversion['tikhonov0']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'z', 'alpha'),
                    coords={'x': self.W['x'], 'y': self.W['y'], 'z': self.W['z'],
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

    def nntikhonov0(self, alpha, **kargs) -> None:
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
        logger.info('Performing 0th-nnTikhonov regularization')
        for i in tqdm(range(n_alpha)):
            beta[:, i], MSE[i], res[i], r2[i] = \
                solvers.nntikhonov0(self.W2D, self.s1D, alp[i], **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping the distribution')
        if self.Wndims == 4:
            beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha))
            for i in range(n_alpha):
                beta_shaped[..., i]  = \
                    matrix.restore_array2D(beta[:,i], self.W.shape[2],
                                        self.W.shape[3])
        else:
            beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3],
                                  self.W.shape[4], n_alpha))
            for i in range(n_alpha):
                beta_shaped[..., i] = \
                    matrix.restore_array3D(beta[:, i], self.W.shape[2],
                                           self.W.shape[3], self.W.shape[4])
        # --- Save it in the dataset
        self.inversion['nntikhonov0'] = xr.Dataset()
        if self.Wndims == 4:
            self.inversion['nntikhonov0']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'alpha'),
                    coords={'x': self.W['x'], 'y': self.W['y'],
                            'alpha': alp}
            )
        else:
            self.inversion['nntikhonov0']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'z', 'alpha'),
                    coords={'x': self.W['x'], 'y': self.W['y'], 'z': self.W['z'],
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
        
    def nntikhonov0projective(self, alpha, **kargs) -> None:
        """
        Perform a 0th order Tikonov regularized regression followed by a 
        projection
        
        NOTE: The projection is done before calculating the MSE and R2

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
        logger.info('Performing 0th-nnTikhonovprojective regularization')
        for i in tqdm(range(n_alpha)):
            beta[:, i], MSE[i], res[i], r2[i] = \
                solvers.nntikhonov0projective(self.W2D, self.s1D, alp[i], **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping the distribution')
        if self.Wndims == 4:
            beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha))
            for i in range(n_alpha):
                beta_shaped[..., i]  = \
                    matrix.restore_array2D(beta[:,i], self.W.shape[2],
                                        self.W.shape[3])
        else:
            beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3],
                                  self.W.shape[4], n_alpha))
            for i in range(n_alpha):
                beta_shaped[..., i] = \
                    matrix.restore_array3D(beta[:, i], self.W.shape[2],
                                           self.W.shape[3], self.W.shape[4])
        # --- Save it in the dataset
        self.inversion['nntikhonov0projective'] = xr.Dataset()
        if self.Wndims == 4:
            self.inversion['nntikhonov0projective']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'alpha'),
                    coords={'x': self.W['x'], 'y': self.W['y'],
                            'alpha': alp}
            )
        else:
            self.inversion['nntikhonov0projective']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'z', 'alpha'),
                    coords={'x': self.W['x'], 'y': self.W['y'], 'z': self.W['z'],
                            'alpha': alp}
            )
        self.inversion['nntikhonov0projective']['F'].attrs['long_name'] =\
            'FI distribution'

        self.inversion['nntikhonov0projective']['alpha'].attrs['long_name'] =\
            '$\\alpha$'

        self.inversion['nntikhonov0projective']['MSE'] = xr.DataArray(MSE, dims='alpha')
        self.inversion['nntikhonov0projective']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['nntikhonov0projective']['residual'] = \
            xr.DataArray(res, dims='alpha')
        self.inversion['nntikhonov0projective']['residual'].attrs['long_name'] =\
            'Residual'

        self.inversion['nntikhonov0projective']['r2'] = xr.DataArray(r2, dims='alpha')
        self.inversion['nntikhonov0projective']['residual'].attrs['long_name'] = '$r^2$'
        
    def nnElasticNet(self, alpha, l1_ratio, **kargs) -> None:
        """
        Perform a 0th order Tikonov regularized non-negative regression

        Jose Rueda-Rueda: jrrueda@us.es

        :param  alpha: hyperparameter. Can be a number (single regression) or
            a list or array. In this latter case, the regression will be done
            for each value in the list (array)
        :param  l1_ratio: weight of L2 to L1 norm
        :param  **kargs: extra arguments for the nnlsqr solver
        """
        # --- Ensure we have an array or iterable:
        if isinstance(alpha, (list, np.ndarray)):
            alp = alpha
        else:
            alp = np.array([alpha])
        n_alpha = len(alp)
        # Now for the L1 ratio
        if isinstance(l1_ratio, (list, np.ndarray)):
            l1 = l1_ratio
        else:
            l1 = np.array([l1_ratio])
        n_l1 = len(l1)
        # --- Initialise the variables
        npoints, nfeatures = self.W2D.shape
        beta = np.zeros((nfeatures, n_alpha, n_l1))
        MSE = np.zeros((n_alpha, n_l1))
        r2 = np.zeros((n_alpha, n_l1))
        res = np.zeros((n_alpha, n_l1))
        # --- Perform the regression
        logger.info('Performing nnElasticNet regularization')
        for i in tqdm(range(n_alpha)):
            for j in range(n_l1):
                beta[:, i, j], MSE[i, j], res[i, j], r2[i, j] = \
                    solvers.Elastic_Net(self.W2D, self.s1D, alp[i], l1[j],
                                        positive=True, **kargs)
        # --- reshape the coefficients
        logger.info('Reshaping the distribution')
        if self.Wndims == 4:
            beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha, n_l1))
            for i in range(n_alpha):
                for j in range(n_l1):
                    beta_shaped[..., i, j] = \
                        matrix.restore_array2D(beta[:, i, j], self.W.shape[2],
                                            self.W.shape[3])
        elif self.Wndims == 5:
            beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3],
                                  self.W.shape[4], n_alpha, n_l1))
            for i in range(n_alpha):
                for j in range(n_l1):
                    beta_shaped[..., i, j] = \
                        matrix.restore_array3D(beta[:, i, j], self.W.shape[2],
                                            self.W.shape[3], self.W.shape[4])
        else:
            raise errors.NotValidInput('W should be 4 or 5D')
        # --- Save it in the dataset
        self.inversion['nnelasticnet'] = xr.Dataset()
        if self.Wndims == 4:
            self.inversion['nnelasticnet']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'alpha', 'l1'),
                    coords={'x': self.W['x'], 'y': self.W['y'],
                            'alpha': alp, 'l1': l1}
            )
        else:
            self.inversion['nnelasticnet']['F'] = xr.DataArray(
                    beta_shaped, dims=('x', 'y', 'z', 'alpha', 'l1'),
                    coords={'x': self.W['x'], 'y': self.W['y'], 'z': self.W['z'],
                            'alpha': alp, 'l1': l1}
            )
        self.inversion['nnelasticnet']['F'].attrs['long_name'] =\
            'FI distribution'

        self.inversion['nnelasticnet']['alpha'].attrs['long_name'] =\
            '$\\alpha$'

        self.inversion['nnelasticnet']['l1'].attrs['long_name'] =\
            '$l_1$'

        self.inversion['nnelasticnet']['MSE'] = \
            xr.DataArray(MSE, dims=('alpha', 'l1'))
        self.inversion['nnelasticnet']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['nnelasticnet']['residual'] = \
            xr.DataArray(res, dims=('alpha', 'l1'))
        self.inversion['nnelasticnet']['residual'].attrs['long_name'] =\
            'Residual'

        self.inversion['nnelasticnet']['r2'] =\
            xr.DataArray(r2, dims=('alpha', 'l1'))
        self.inversion['nnelasticnet']['residual'].attrs['long_name'] = '$r^2$'

    def maximumEntropy(self, alpha, d=None,  **kargs) -> None:
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
        if d is not None:
            d2 = matrix.collapse_array2D(d)
        else:
            d2 = d
        # --- Perform the regression
        logger.info('Performing maximum entropy regularization')
        for i in tqdm(range(n_alpha)):
            reg = solvers.EntropyFunctional(
                self.W2D, self.s1D, alpha=alp[i], d=d2)
            beta[:, i], MSE[i], res[i], r2[i] = reg.solve()
        # --- reshape the coefficients and predict the MSE
        logger.info('Reshaping the distribution')
        beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha))
        for i in range(n_alpha):
            beta_shaped[..., i]  = \
                matrix.restore_array2D(beta[:,i], self.W.shape[2],
                                       self.W.shape[3])
        # --- Predict the solutions and get the MSE:
        
        # --- Save it in the dataset
        self.inversion['maxEntropy'] = xr.Dataset()
        self.inversion['maxEntropy']['F'] = xr.DataArray(
                beta_shaped, dims=('x', 'y', 'alpha'),
                coords={'x': self.W['x'], 'y': self.W['y'],
                        'alpha': alp}
        )
        self.inversion['maxEntropy']['F'].attrs['long_name'] =\
            'FI distribution'

        self.inversion['maxEntropy']['alpha'].attrs['long_name'] =\
            '$\\alpha$'

        self.inversion['maxEntropy']['MSE'] = xr.DataArray(MSE, dims='alpha')
        self.inversion['maxEntropy']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['maxEntropy']['residual'] = \
            xr.DataArray(res, dims='alpha')
        self.inversion['maxEntropy']['residual'].attrs['long_name'] =\
            'Residual'

        self.inversion['maxEntropy']['r2'] = xr.DataArray(r2, dims='alpha')
        self.inversion['maxEntropy']['residual'].attrs['long_name'] = '$r^2$'

    def calculateLcurves(self, reconstructions =  None) -> None:
        """
        Calculate the L curve
        
        jose rueda: jrrueda@us.es
        
        :param reconstructions: list with strings with the names of the
            reconstructions for the calculation of the L curve. If None, all
            reconstructions in the 'inversion' attribute wil be used
        """
        if reconstructions is None:
            reconstructions = self.inversion.keys()
        for k in reconstructions:
            x = np.log10(self.inversion[k].MSE)
            y = np.log10(np.sqrt((self.inversion[k].F)**2).sum(dim=('x', 'y')))
            # It can be the case of elastic net, which has a second hyper param
            if len(x.shape) == 1:
                curv = self._calccurvature(x,y)
                self.inversion[k]['curvature'] = \
                    xr.DataArray(curv, dims='alpha')
                self.inversion[k]['curvature'].attrs['long_name'] = \
                    'Curvature of the L curve'
            if len(x.shape) == 2:
                curvature = np.zeros(x.shape)
                for i in range(self.inversion[k].l1.size):
                    curvature[:, i] = self._calccurvature(x.isel(l1=i),
                                                          y.isel(l1=i))
                self.inversion[k]['curvature'] = \
                    xr.DataArray(curvature, dims=('alpha', 'l1'))
                self.inversion[k]['curvature'].attrs['long_name'] = \
                    'Curvature of the L curve'
    
    def _calccurvature(self, x, y) -> np.ndarray:
        # perform an interpolation, with a lot of points, to avoid the noise
        dx = np.gradient(x, x)  # first derivatives
        dy = np.gradient(y, x)
        d2x = np.gradient(dx, x)  # second derivatives
        d2y = np.gradient(dy, x)
        return np.abs(d2y) / (np.sqrt(1 + dy ** 2)) ** 1.5  # curvature
    
    def calculateBiasAndTotalRatio(self, trueSolution: xr.DataArray,
                                reconstructions: list =  None) -> None:
        """
        Calculate the bias and the total ratio of FI to the true solution
        
        jose rueda: jrrueda@us.es

        :param trueSolution: xr.DataArray with the true solution
        :param reconstructions: list with strings with the names of the
            inversion methods for the calculation of the bias. If None, all
            present in the object will be used
        
        """
        # --- Check the input
        if reconstructions is None:
            reconstructions = self.inversion.keys()
        # --- interpolate the trueSolution in the grid
        
        # --- Calculate the bias
        for k in reconstructions:
            # ---- interpolate the true solution
            # I make it here in case each iversion method was reconstructed
            # using a different grid
            trueSolInterp = trueSolution.interp(x=self.inversion[k].F.x,
                                                y=self.inversion[k].F.y)
            # deltaE = trueSolution.y[1] - trueSolution.y[0]
            # deltaE2 = self.inversion[k].F.y[1] - self.inversion[k].F.y[0]
            # First invert the normalization
            if self.norms['normalised'][0]:
                F = self.inversion[k].F * self.norms['s'] / self.norms['W']
                factor = self.norms['s'] / self.norms['W']
            else:
                F = self.inversion[k].F
                factor = 1.0
            deltaX = self.inversion[k].F.x[1] - self.inversion[k].F.x[0]
            deltaY = self.inversion[k].F.y[1] - self.inversion[k].F.y[0]
            Omega = deltaX.values * deltaY.values
            factor /= Omega
            F /= Omega
            # Now calculate the bias
            bias = F - trueSolInterp
            # Store the bias in place
            self.inversion[k]['bias'] = bias
            self.inversion[k]['bias'].attrs['long_name'] = 'Bias'
            # Now calculate the total ratio
            totalRatio = F.sum(('x', 'y'), skipna=True) \
                / trueSolInterp.sum(('x', 'y'), skipna=True)
            self.inversion[k]['totalRatio'] = totalRatio
            self.inversion[k]['totalRatio'].attrs['long_name'] = \
                'Total ratio of FI to the true solution'
            self.inversion[k]['desnormalizationFactor'] = factor
            

    # ------------------------------------------------------------------------
    # %% Potting block
    # ------------------------------------------------------------------------
    def plotLcurve(self, inversion: str='maxEntropy', ax=None,
                   line_params: dict = {}) -> plt.Axes:
        """
        Plot the L curve and the MSE

        Jose Rueda: jrrueda@us.es

        :param inversion: name of the inversion to plot.
        :param ax: axes to plot the data, should be an array of 2 axis, in the 
            first one, the L curve will be plotted, in the second one, the
            curvature of the L curve will be plotted. If the axis are not
            created, the labels will not be changed.
        
        :return ax: axes where the data has been plotted, list of 2 axes
        """
        # ---- Initialise the settings
        line_options = {
            'lw': 0.75,
            'marker': '+',
            'ms': 3,
        }
        line_options.update(line_params)
        if 'label' not in line_options.keys():
            line_options['label'] = inversion
        if ax is None:
            fig, ax = plt.subplots(2,2)
            ax[1, 0].set_xlabel('MSE')
            ax[1, 0].set_ylabel('k')
            ax[0, 0].set_ylabel('|FI|')            
            ax[0, 0].set_xscale('log')
            ax[1, 0].set_xscale('log')
            ax[1, 0].set_yscale('log')
            ax[0, 0].set_yscale('log')
            ax[1, 1].set_xlabel('Alpha')
            ax[1, 1].set_xscale('log')
            ax[0, 1].set_xscale('log')
            ax[1, 1].set_ylabel('k')
            ax[0, 1].set_ylabel('Residual')
        # ---- Plot the data
        ax[0, 0].plot(self.inversion[inversion].MSE,
                   np.sqrt((self.inversion[inversion].F)**2).sum(dim=('x', 'y')),
                   **line_options)
        ax[1, 0].plot(self.inversion[inversion].MSE,
                   self.inversion[inversion].curvature,
                   **line_options)        
        # ---- Plot the data
        ax[0, 1].plot(self.inversion[inversion].alpha,
                   self.inversion[inversion].residual,
                   **line_options)
        ax[1, 1].plot(self.inversion[inversion].alpha,
                   self.inversion[inversion].curvature,
                   **line_options)
        ax[0, 0].get_figure().show()
        return ax

    def GUIprofiles(self, inversion: str= 'nntikhonov0', true_solution= None, jl1=None):
        """
        Plot the profiles of the inversion

        Jose Rueda:
        """
        # Check input
        if inversion not in self.inversion.keys():
            raise ValueError(f'The input {inversion} is not stored in the data.')
        if jl1 is None:
            data = self.inversion[inversion].F.copy()
            curvature = self.inversion[inversion].curvature
            MSE = self.inversion[inversion].MSE
        else:
            data = self.inversion[inversion].F.isel(l1=jl1).copy()
            curvature = self.inversion[inversion].isel(l1=jl1).curvature
            MSE = self.inversion[inversion].isel(l1=jl1).MSE
        logAlpha = np.log10(self.inversion[inversion].alpha.values)
        # logL1 = np.log10(self.inversion[inversion].L1)
        # renormalize the data
        if self.norms['normalised'][0]:
            factor = self.norms['s'] / self.norms['W']
        else:
            factor = 1.0
        deltaX = data.x[1] - data.x[0]
        deltaY = data.y[1] - data.y[0]
        Omega = deltaX.values * deltaY.values
        factor /= Omega
        data *= factor
        # Now calculate the profiles
        Eprof = data.sum('x')
        Rprof = data.sum('y')
        if true_solution is not None:
            # Interpolate the true solution in the grid
            trueSolInterp = true_solution.interp(x=data.x, y=data.y)
            # get the profiles
            EprofTrue = trueSolInterp.sum('x')
            RprofTrue = trueSolInterp.sum('y')
        else:
            EprofTrue = None
            RprofTrue = None
        Total = np.sqrt((data**2).sum(dim=('x','y')))
        fig, ax = plt.subplots(2,2)
        if true_solution is not None:
            ax[0,0].plot(EprofTrue.y, EprofTrue, '--k', label='True',)
            ax[0,0].set_title('Gyroscalar profile')
            ax[0,0].set_xlabel('Gyroscalar')
            ax[0,0].set_ylabel('Counts')
            ax[1,0].plot(RprofTrue.x, RprofTrue, '--k', label='True',)
            ax[1,0].set_title('Pitch profile')
            ax[1,0].set_xlabel('Pitch')
            ax[1,0].set_ylabel('Counts')
        ax[0, 1].plot(MSE, Total)
        ax[0,1].set_title('L-curve plot')
        ax[0, 1].set_xlabel('MSE')
        ax[0, 1].set_ylabel('Norm of reconstructed signal')
        ax[1, 1].plot(self.inversion[inversion].alpha, curvature) 
        ax[1, 1].set_title('Curvature plot')
        ax[1, 1].set_xlabel('Hyperparameter')
        ax[1, 1].set_ylabel('Curvature')
        ax[0, 1].set_xscale('log')
        ax[0, 1].set_yscale('log')       
        ax[1, 1].set_xscale('log')
        ax[1, 1].set_yscale('log')
        # We will now check how many slider-variables are actually
        # varying.
        slider_vars = ['alpha']
        lineE, = ax[0, 0].plot(Eprof.y, Eprof.isel(alpha=0))
        lineR, = ax[1, 0].plot(Rprof.x, Rprof.isel(alpha=0))
        pointRes, = ax[0, 1].plot(MSE[0], Total[0], 'o')
        pointCurv, = ax[1, 1].plot(self.inversion[inversion].alpha[0],
                                   curvature[0], 'o')
        # Base updater.
        def base_updater(slider_var: str, val):
            # Updating the plotter.
            slider_vars_val[slider_var] = 10.0**val
            E2plot = Eprof.sel(method='nearest',
                               **slider_vars_val).values            
            R2plot = Rprof.sel(method='nearest',
                               **slider_vars_val).values
            if slider_var == 'alpha':
                ialpha = np.argmin(np.abs(Eprof.alpha.values - slider_vars_val[slider_var]))
                ax[0,1].set_title('ialpha: %i'% ialpha)
            lineE.set_ydata(E2plot)
            lineR.set_ydata(R2plot)
            pointRes.set_xdata(MSE.sel(method='nearest',
                               **slider_vars_val).values)
            pointRes.set_ydata(Total.sel(method='nearest',**slider_vars_val).values)
            pointCurv.set_xdata(self.inversion[inversion].alpha.sel(method='nearest',
                                 **slider_vars_val).values)
            pointCurv.set_ydata(curvature.sel(method='nearest',
                                    **slider_vars_val).values)
            ax[0, 0].relim()
            ax[0, 0].autoscale_view()
            ax[1, 0].relim()
            ax[1, 0].autoscale_view()
            


        # Creating the axes divider.
        ax_div = make_axes_locatable(ax[0, 1])
        axes_sliders = list()
        slider_vars_val = dict()
        sliders = list()
        copy_var = slider_vars.copy()
        for ii, ivar in enumerate(slider_vars):
            iax = ax_div.append_axes('bottom', '5%', pad='15%')
            axes_sliders.append(iax)
            islider = Slider(
                ax=iax,
                label='%s' % slider_vars[ii],
                valstep=logAlpha,
                valinit=logAlpha[0],
                valmin=logAlpha.min(),
                valmax=logAlpha.max()
            )
            islider.on_changed(lambda val: base_updater(copy_var[ii], val))

            slider_vars_val[ivar] = logAlpha[0]
            sliders.append(islider)


        return ax, [lineE, lineR], sliders, Eprof, Rprof, EprofTrue, RprofTrue
    
    def GUIprofilesInversion2D(self, inversion: str= 'nntikhonov0', true_solution= None, jl1=None):
        """
        Plot the profiles of the inversion and the inversion in 2D in a subplot.

        Jose Rueda:
        """
        # Check input
        if inversion not in self.inversion.keys():
            raise ValueError(f'The input {inversion} is not stored in the data.')
        if jl1 is None:
            data = self.inversion[inversion].F.copy()
            MSE = self.inversion[inversion].MSE
        else:
            data = self.inversion[inversion].F.isel(l1=jl1).copy()
            MSE = self.inversion[inversion].isel(l1=jl1).MSE
        logAlpha = np.log10(self.inversion[inversion].alpha.values)
        # logL1 = np.log10(self.inversion[inversion].L1)
        # renormalize the data
        if self.norms['normalised'][0]:
            factor = self.norms['s'] / self.norms['W']
        else:
            factor = 1.0
        deltaX = data.x[1] - data.x[0]
        deltaY = data.y[1] - data.y[0]
        Omega = deltaX.values * deltaY.values
        factor /= Omega
        data *= factor
        # Now calculate the profiles
        Eprof = data.sum('x')
        Rprof = data.sum('y')
        # Interpolate the true solution in the grid
        if true_solution is not None:
            trueSolInterp = true_solution.interp(x=data.x, y=data.y)
            # get the profiles
            EprofTrue = trueSolInterp.sum('x')
            RprofTrue = trueSolInterp.sum('y')
            weHaveTrue = True
        else:
            weHaveTrue = False

        Total = np.sqrt(self.inversion[inversion].F**2).sum(dim=('x','y'))
        fig, ax = plt.subplots(2,2)
        if weHaveTrue:
            ax[0,0].plot(EprofTrue.y, EprofTrue, '--k', label='True',)
            ax[1,0].plot(RprofTrue.x, RprofTrue, '--k', label='True',)
        ax[0, 1].plot(self.inversion[inversion].MSE,
                      Total)
        # ax[1, 1].plot(self.inversion[inversion].alpha,
        #               self.inversion[inversion].curvature) 
        ax[0, 1].set_xscale('log')
        ax[0, 1].set_yscale('log')       
        # ax[1, 1].set_xscale('log')
        # ax[1, 1].set_yscale('log')
        # We will now check how many slider-variables are actually
        # varying.
        slider_vars = ['alpha']
        lineE, = ax[0, 0].plot(Eprof.y, Eprof.isel(alpha=0))
        lineR, = ax[1, 0].plot(Rprof.x, Rprof.isel(alpha=0))
        pointRes, = ax[0, 1].plot(MSE[0],
                                  Total[0], 'o')
        img = data.isel(alpha=0).T.plot.imshow(ax=ax[1,1])
        # Base updater.
        def base_updater(slider_var: str, val):
            # Updating the plotter.
            slider_vars_val[slider_var] = 10.0**val
            E2plot = Eprof.sel(method='nearest',
                               **slider_vars_val).values            
            R2plot = Rprof.sel(method='nearest',
                               **slider_vars_val).values
            if slider_var == 'alpha':
                ialpha = np.argmin(np.abs(Eprof.alpha.values - slider_vars_val[slider_var]))
                ax[0, 1].set_title('ialpha: %i'% ialpha)
                ax[1, 1].set_title('alpha: %.2e' % Eprof.alpha[ialpha])
            lineE.set_ydata(E2plot)
            lineR.set_ydata(R2plot)
            pointRes.set_xdata(MSE.sel(method='nearest',
                               **slider_vars_val).values)
            pointRes.set_ydata(Total.sel(method='nearest',**slider_vars_val).values)

            img.set_data(data.sel(method='nearest',**slider_vars_val ).values.T)
            
            ax[0, 0].relim()
            ax[0, 0].autoscale_view()
            ax[1, 0].relim()
            ax[1, 0].autoscale_view()
            


        # Creating the axes divider.
        ax_div = make_axes_locatable(ax[0, 1])
        axes_sliders = list()
        slider_vars_val = dict()
        sliders = list()
        copy_var = slider_vars.copy()
        for ii, ivar in enumerate(slider_vars):
            iax = ax_div.append_axes('bottom', '5%', pad='15%')
            axes_sliders.append(iax)
            islider = Slider(
                ax=iax,
                label='%s' % slider_vars[ii],
                valstep=logAlpha,
                valinit=logAlpha[0],
                valmin=logAlpha.min(),
                valmax=logAlpha.max()
            )
            islider.on_changed(lambda val: base_updater(copy_var[ii], val))

            slider_vars_val[ivar] = logAlpha[0]
            sliders.append(islider)


        return ax, [lineE, lineR], sliders
    
    def plot_MSE_error(self, inverters = ['descent', 'kaczmarz', 
                                                'cimmino'], ax=None,
                    plot_params: dict = {}) -> plt.Axes:
            """
            Plot the MSE
    
            Marina Jimenez Comez: mjimenez37@us.es

            :param inversion: name of the inversions to plot.
            :param ax: axes to plot the data.

            :return ax: axes where the data has been plotted
            """
            # ---- Initialise the settings
            if ax is None:
                fig, ax = plt.subplots()
            if 'label_size' in plot_params.keys():
                label_size = plot_params['label_size']
            else:
                label_size = 16

            true_norm = (self.s**2).sum(dim=('xs','ys')) # It's squared
            for inv in inverters:
                MSE = self.inversion[inv].MSE
                error = MSE/ true_norm
                ax.plot(self.inversion[inv].alpha, error, '-o', label=inv)

            ax.set_title('Normalized MSE vs iterations', fontsize=label_size)
            ax.set_xlabel('k', fontsize=label_size)
            ax.set_ylabel('Error', fontsize=label_size)
            ax.legend(inverters)

            return ax
    
    def plot_synthetic_error(self, x_syntheticXR, 
                             inverters = ['descent', 'kaczmarz', 'cimmino'], 
                             ax=None,
                    plot_params: dict = {}) -> plt.Axes:
            """
            Plot the error with respect to the synthetic data
    
            Marina Jimenez Comez: mjimenez37@us.es

            :param x_syntheticXR: synthetic data at the pinhole plane
            :param inversion: names of the inversions to plot.
            :param ax: axes to plot the data.

            :return ax: axes where the data has been plotted
            """
            # ---- Initialise the settings
            if ax is None:
                fig, ax = plt.subplots()
            if 'label_size' in plot_params.keys():
                label_size = plot_params['label_size']
            else:
                label_size = 16
            
            norm = 1
            if self.norms['normalised'][0] ==1:
                norm = self.norms['s']/self.norms['W']

            for inv in inverters:
                x_hat = self.inversion[inv].F.copy()*norm
                MSE = np.sqrt(((x_hat-x_syntheticXR)**2).sum(dim=('x','y')))
                true_norm = np.sqrt((x_syntheticXR**2).sum(dim=('x','y')))
                error = MSE/ true_norm
                ax.plot(self.inversion[inv].alpha, error, '-o', label=inv)

            ax.set_title('Error vs iterations', fontsize=label_size)
            ax.set_xscale('log')
            ax.set_xlabel('k', fontsize=label_size)
            ax.set_ylabel('Error', fontsize=label_size)
            ax.legend(inverters)

            return ax
    
    def plot_computational_time(self, inverters = ['descent', 'kaczmarz',
                                                'cimmino'], ax=None,
                    plot_params: dict = {}) -> plt.Axes:
            """
            Plot the computational time
    
            Marina Jimenez Comez: mjimenez37@us.es

            :param inversion: name of the inversion to plot.
            :param ax: axes to plot the data.

            :return ax: axes where the data has been plotted
            """
            # ---- Initialise the settings
            if ax is None:
                fig, ax = plt.subplots()
            if 'label_size' in plot_params.keys():
                label_size = plot_params['label_size']
            else:
                label_size = 16

            for inv in inverters:
                time = self.inversion[inv].time
                ax.plot(self.inversion[inv].alpha, time, '-o', label=inv)

            ax.set_title('Computational time vs iterations', fontsize=label_size)
            ax.set_xlabel('k', fontsize=label_size)
            ax.set_ylabel('Time [s]', fontsize=label_size,)
            ax.legend(inverters)

            return ax


    # ------------------------------------------------------------------------
    # %% Export block
    # ------------------------------------------------------------------------
    def export(self, folder: str, inversionKeys: list = None,
               createTar: bool = False) -> str:
        """
        Export the tomography data into a folder

        Jose Rueda: jrrueda@us.es

        :param folder: folder where the data will be saved
        :param inversionKeys: list of strings with the names of the inversion
            methods to be saved. If None, all will be saved
        :param createTar: if True, a TAR file will be created with all the
            exported files

        :return folder: folder where the data has been saved
        """
        logger.info('Saving results in: %s', folder)
        filesSaved = []
        if inversionKeys is None:
            inversionKeys = self.inversion.keys()
        # Export the different inversion results
        for k, inversion in self.inversion.items():
            if k not in inversionKeys:
                continue
            fileToSave = os.path.join(folder, k + '.nc')
            logger.info('Saving: %s', fileToSave)
            inversion.to_netcdf(fileToSave, format='NETCDF4')
            filesSaved.append(fileToSave)
        # Export the signal
        fileToSave = os.path.join(folder, 'Signal.nc')
        self.s.to_netcdf(fileToSave, format='NETCDF4')
        filesSaved.append(fileToSave)
        # Export the W
        fileToSave = os.path.join(folder, 'WeightFunc.nc')
        self.W.to_netcdf(fileToSave, format='NETCDF4')
        filesSaved.append(fileToSave)
        # Export the normalizations
        fileToSave = os.path.join(folder, 'norms.json')
        json.dump({k:v.tolist() for k,v in self.norms.items()},
                  open(fileToSave, 'w' ))
        filesSaved.append(fileToSave)
        # Export the suite version
        fileToSave = os.path.join(folder, 'SuiteVersion.txt')
        exportVersion(fileToSave)
        filesSaved.append(fileToSave)
        # Compress into a TAR:
        if createTar:
            tarFile = os.path.join(folder, 'Complete.tar')
            tar = tarfile.open(name=tarFile, mode='w')
            for f in filesSaved:
                tar.add(f, arcname=os.path.split(f)[-1])
            tar.close()
        return folder
