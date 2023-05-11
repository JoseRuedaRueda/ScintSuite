"""
Main class to perform tomographic inversions.

Jose Rueda-Rueda: jrrueda@us.es

"""
import os
import logging
import tarfile
import numpy as np
import xarray as xr
import Lib.errors as errors
import matplotlib.pyplot as plt
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
        else:
            if W is not None or s is not None:
                logger.warning('30: Folder argument present, ignorig W and s.')
            logger.info('Reading results from %s', folder)
            self.W = xr.load_dataarray(os.path.join(folder, 'WeightFunc.nc'))
            self.s = xr.load_dataarray(os.path.join(folder, 'Signal.nc'))
            self.inversion = {}
            # --- Now collapse the signal and the weight function
            logger.info('Collapsing W: ')
            self.W2D = matrix.collapse_array4D(self.W.values)
            logger.info('Collapsing Signal')
            self.s1D = matrix.collapse_array2D(self.s.values.squeeze())
            # --- Now load the inversions
            supportedFiles = ['nnelasticnet.nc', 'nntikhonov0.nc',
                              'tikhonov0.nc',
                              'nnlsq.nc', 'maxEntropy.nc']
            for file in supportedFiles:
                filename = os.path.join(folder, file)
                if os.path.isfile(filename):
                    key = file.split('.')[0]
                    logger.info('reading %s', filename)
                    self.inversion[key] = xr.load_dataset(filename)


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
                beta_shaped, dims=('x', 'y'),
                coords={'x': self.W['x'], 'y': self.W['y']}
        )
        self.inversion['nnlsq']['F'].attrs['long_name'] = 'FI distribution'

        self.inversion['nnlsq']['MSE'] = xr.DataArray(MSE)
        self.inversion['nnlsq']['MSE'].attrs['long_name'] = 'MSE'

        self.inversion['nnlsq']['residual'] = xr.DataArray(res)
        self.inversion['nnlsq']['residual'].attrs['long_name'] = 'Residual'

        self.inversion['nnlsq']['r2'] = xr.DataArray(r2)
        self.inversion['nnlsq']['residual'].attrs['long_name'] = '$r^2$'

    def tikhonov0(self, alpha, weights=None, **kargs):
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
        beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha))
        for i in range(n_alpha):
            beta_shaped[..., i] = \
                matrix.restore_array2D(beta[:, i], self.W.shape[2],
                                       self.W.shape[3])
        # --- Save it in the dataset
        self.inversion['tikhonov0'] = xr.Dataset()
        self.inversion['tikhonov0']['F'] = xr.DataArray(
                beta_shaped, dims=('x', 'y', 'alpha'),
                coords={'x': self.W['x'], 'y': self.W['y'],
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
        logger.info('Performing 0th-nnTikhonov regularization')
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
                beta_shaped, dims=('x', 'y', 'alpha'),
                coords={'x': self.W['x'], 'y': self.W['y'],
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

    def nnElasticNet(self, alpha, l1_ratio, **kargs):
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
        beta_shaped = np.zeros((self.W.shape[2], self.W.shape[3], n_alpha, n_l1))
        for i in range(n_alpha):
            for j in range(n_l1):
                beta_shaped[..., i, j] = \
                    matrix.restore_array2D(beta[:, i, j], self.W.shape[2],
                                           self.W.shape[3])
        # --- Save it in the dataset
        self.inversion['nnelasticnet'] = xr.Dataset()
        self.inversion['nnelasticnet']['F'] = xr.DataArray(
                beta_shaped, dims=('x', 'y', 'alpha', 'l1'),
                coords={'x': self.W['x'], 'y': self.W['y'],
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

    def maximumEntropy(self, alpha, d=None,  **kargs):
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

    def calculateLcurves(self, reconstructions =  None):
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
            x = self.inversion[k].MSE
            y = self.inversion[k].F.sum(dim=('x', 'y'))
            # It can be the case of elastic net, which has a second hyper param
            if len(x.shape) == 1:
                curv = self._calccurvature(x,y)
                self.inversion[k]['curvature'] = \
                    xr.DataArray(curv, dims='alpha')
                self.inversion[k]['residual'].attrs['long_name'] = \
                    'Curvature of the L curve'
            if len(x.shape) == 2:
                curvature = np.zeros(x.shape)
                for i in range(self.inversion[k].l1.size):
                    curvature[:, i] = self._calccurvature(x.isel(l1=i),
                                                          y.isel(l1=i))
                self.inversion[k]['curvature'] = \
                    xr.DataArray(curvature, dims=('alpha', 'l1'))
                self.inversion[k]['residual'].attrs['long_name'] = \
                    'Curvature of the L curve'
    def _calccurvature(self, x, y):
        # perform an interpolation, with a lot of points, to avoid the noise
        dx = np.gradient(x, x)  # first derivatives
        dy = np.gradient(y, x)
        d2x = np.gradient(dx, x)  # second derivatives
        d2y = np.gradient(dy, x)
        return np.abs(d2y) / (np.sqrt(1 + dy ** 2)) ** 1.5  # curvature
    
    # def _calccurvatureSurface(sef, x, y)
    # ------------------------------------------------------------------------
    # %% Potting block
    # ------------------------------------------------------------------------
    def plotLcurve(self, inversion: str='maxEntropy', ax=None):
        if ax is None:
            fig, ax = plt.subplots(2)
        ax[0].plot(self.inversion[inversion].MSE,
                   self.inversion[inversion].F.sum(dim=('x','y')))
        ax[1].plot(self.inversion[inversion].MSE,
                   self.inversion[inversion].curvature)

    def export(self, folder: str, createTar: bool = False):
        """
        Export the tomography data into a folder

        :param folder: DESCRIPTION
        :type folder: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        #TODO :add norms and suite version

        """
        logger.info('Saving results in: %s', folder)
        filesSaved = []
        # Export the different inversion results
        for k, inversion in self.inversion.items():
            fileToSave = os.path.join(folder, k + '.nc')
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
        # Compress into a TAR:
        if createTar:
            tarFile = os.path.join(folder, 'Complete.tar')
            tar = tarfile.open(name=tarFile, mode='w')
            for f in filesSaved:
                tar.add(f, arcname=os.path.split(f)[-1])
            tar.close()
