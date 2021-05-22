"""
Contains the routines to perform the tomographic inversion

NOTICE: I consider the creatation of the transfer matrix as an issue of the
synthetic codes (INPASIM, FILDSIM, i/HIBPSIM) therefore the routines which
create these matries are placed are their corresponding libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import Lib.LibFILDSIM as ssfildsim
import Lib.LibMap as ssmapping
import Lib.LibIO as ssio
from scipy import ndimage        # To denoise the frames
from scipy.io import netcdf                # To export remap data
from tqdm import tqdm            # For waitbars
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.optimize import nnls     # Non negative least squares
from sklearn.linear_model import ElasticNet  # ElaticNet
from Lib.version_suite import version
try:
    import lmfit as lm
except ModuleNotFoundError:
    print('You cannot use the L curve in tomography')


# -----------------------------------------------------------------------------
# --- Auxiliar functions
# -----------------------------------------------------------------------------
def residual(ypred, ytrue):
    """
    Calculate the residual

    Jose Rueda: jrrueda@us.es

    Calculate the sum of the absolute difference between ypred and ytrue
    """
    return np.sum(abs(ypred - ytrue))


# -----------------------------------------------------------------------------
# --- SOLVERS AND REGRESSION ALGORITHMS
# -----------------------------------------------------------------------------
def OLS_inversion(X, y):
    """
    Perform an OLS inversion using the analytical solution

    Jose Rueda: jrrueda@us.es

    @param X: Design matrix
    @param y: signal
    @return beta: best fit coefficients
    @return MSE: Mean squared error
    @return r2: R2 score
    """
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    res = residual(y_pred, y)
    return beta, MSE, res, r2


def nnols_inversion(X, y, param: dict = {}):
    """
    Perform a non-negative least squares inversion using scipy

    Jose Rueda: jrrueda@us.es

    @param X: Design matrix
    @param y: signal
    @param param: dictionary with options for the nnls solver (see scipy)
    @return beta: best fit coefficients
    @return MSE: Mean squared error
    @return r2: R2 score
    """
    beta, dummy = nnls(X, y, **param)
    y_pred = X @ beta
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    res = residual(y_pred, y)
    return beta, MSE, res, r2


def Ridge_inversion(X, y, alpha):
    """
    Perform a Ridge (0th Tikhonov) regression

    Jose Rueda: jrrueda@us.es

    @param X: Design matrix
    @param y: signal
    @param alpha: hyperparameter
    @return ridge.coef_: best fit coefficients
    @return MSE: Mean squared error
    @return r2: R2 score
    """
    ridge = Ridge(alpha)
    ridge.fit(X, y)
    y_pred = ridge.predict(X)
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    res = residual(y_pred, y)
    return ridge.coef_, MSE, res, r2


def Ridge_scan(X, y, alpha_min: float, alpha_max: float, n_alpha: int = 20,
               log_spaced: bool = True, plot: bool = True,
               line_param: dict = {'linewidth': 1.5},
               FS: float = 14):
    """
    Scan the slpha parameters to find the best hyper-parameter

    Jose Rueda: jrrueda@us.es

    @param X: Design matrix
    @param y: signal
    @param alpha_min: minimum value for the hyper-parameter scan
    @param alpha_max: maximum value for the hyper-parameter scan
    @param n_alpha: number of points in the scan
    @param log_spaced: if true, points will be logspaced
    @param line_param: dictionary with the line plotting parameters
    @param FS: FontSize
    @return out: Dictionay with fields:
        -# beta: array of coefficients [nfeatures, nalphas]
        -# MSE: arrays of MSEs
        -# r2: arrays of r2
        -# residual: arrays of residual
        -# norm: norm of the coefficients
        -# alpha: Used hyperparameters
    @return figures: Dictionay with the figures created by the method:
        -# Merit: r2 and MSE vs hyperparam
        -# L_curve: f norm vs residual
    """
    # --- Initialise the variables
    npoints, nfeatures = X.shape
    beta = np.zeros((nfeatures, n_alpha))
    MSE = np.zeros(n_alpha)
    r2 = np.zeros(n_alpha)
    res = np.zeros(n_alpha)

    if log_spaced:
        alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    else:
        alpha = np.linspace(alpha_min, alpha_max, n_alpha)
    # --- Perform the scan
    print('Performing regression')
    for i in tqdm(range(n_alpha)):
        beta[:, i], MSE[i], res[i], r2[i] = Ridge_inversion(X, y, alpha[i])

    # --- Plot if needed:
    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        # Plot the r2:
        ax1.plot(alpha, r2, **line_param)
        ax1.grid(True, which='minor', linestyle=':')
        ax1.minorticks_on()
        ax1.grid(True, which='major')
        ax1.set_ylabel('$r^2$', fontsize=FS)

        # Plot the MSE
        ax2.plot(alpha, MSE, **line_param)
        ax2.grid(True, which='minor', linestyle=':')
        ax2.minorticks_on()
        ax2.grid(True, which='major')
        ax2.set_ylabel('MSE', fontsize=FS)
        ax2.set_xlabel('$\\alpha$')

        # Plot the modulus versus the residual
        fig2, ax = plt.subplots()
        y = np.sum(np.sqrt(beta**2), axis=0)
        ax.plot(res, y, **line_param)
        ax.grid(True, which='minor', linestyle=':')
        ax.minorticks_on()
        ax.grid(True, which='major')
        ax.set_ylabel('$|F| [a.u.]$', fontsize=FS)
        ax.set_xlabel('Residual')
        if log_spaced:
            ax.set_xscale('log')
        figures = {
            'Merit': fig,
            'L_curve': fig2
        }
    else:
        figures = {}
    out = {
        'beta': beta,
        'MSE': MSE,
        'residual': res,
        'norm': np.sum(np.sqrt(beta**2), axis=0),
        'r2': r2,
        'alpha': alpha,
    }
    return out, figures


def nnRidge(X, y, alpha, param: dict = {}):
    """
    Perfom a non-negative Ridge inversion

    @param X: Design matrix
    @param y: signal
    @param alpha: hyperparameter
    @param param. dictionary with extra parameters for scipy.nnls
    @return ridge.coef_: best fit coefficients
    @return MSE: Mean squared error
    @return r2: R2 score
    """
    # Auxiliar arrays:
    n1, n2 = X.shape
    L = np.eye(n2)
    GalphaL0 = np.zeros((n2, 1))
    # Extended design matrix
    WalphaL = np.vstack((X, np.sqrt(alpha) * L))
    GalphaL = np.vstack((y[:, np.newaxis], GalphaL0)).squeeze()
    # Non-negative ols solution:
    beta, dummy = nnls(WalphaL, GalphaL, **param)
    y_pred = X @ beta
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    res = residual(y_pred, y)
    return beta, MSE, res, r2


def nnRidge_scan(X, y, alpha_min: float, alpha_max: float, n_alpha: int = 20,
                 log_spaced: bool = True, plot: bool = True,
                 line_param: dict = {'linewidth': 1.5},
                 FS: float = 14):
    """
    Scan the alpha parameters to find the best hyper-parameter (nnRidge)

    Jose Rueda: jrrueda@us.es

    @param X: Design matrix
    @param y: signal
    @param alpha_min: minimum value for the hyper-parameter scan
    @param alpha_max: maximum value for the hyper-parameter scan
    @param n_alpha: number of points in the scan
    @param log_spaced: if true, points will be logspaced
    @param line_param: dictionary with the line plotting parameters
    @param FS: FontSize
    @return out: Dictionay with fields:
        -# beta: array of coefficients [nfeatures, nalphas]
        -# MSE: arrays of MSEs
        -# r2: arrays of r2
        -# residual: arrays of residual
        -# norm: norm of the coefficients
        -# alpha: Used hyperparameters
    @return figures: Dictionay with the figures created by the method:
        -# Merit: r2 and MSE vs hyperparam
        -# L_curve: f norm vs residual
    """
    # --- Initialise the variables
    npoints, nfeatures = X.shape
    beta = np.zeros((nfeatures, n_alpha))
    MSE = np.zeros(n_alpha)
    r2 = np.zeros(n_alpha)
    res = np.zeros(n_alpha)

    if log_spaced:
        alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    else:
        alpha = np.linspace(alpha_min, alpha_max, n_alpha)
    # --- Perform the scan
    print('Performing regression')
    for i in tqdm(range(n_alpha)):
        beta[:, i], MSE[i], res[i], r2[i] = nnRidge(X, y, alpha[i])

    # --- Plot if needed:
    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        # Plot the r2:
        ax1.plot(alpha, r2, **line_param)
        ax1.grid(True, which='minor', linestyle=':')
        ax1.minorticks_on()
        ax1.grid(True, which='major')
        ax1.set_ylabel('$r^2$', fontsize=FS)

        # Plot the MSE
        ax2.plot(alpha, MSE, **line_param)
        ax2.grid(True, which='minor', linestyle=':')
        ax2.minorticks_on()
        ax2.grid(True, which='major')
        ax2.set_ylabel('MSE', fontsize=FS)
        ax2.set_xlabel('$\\alpha$')

        # Plot the modulus versus the residual
        fig2, ax = plt.subplots()
        y = np.sum(np.sqrt(beta**2), axis=0)
        ax.plot(res, y, **line_param)
        ax.grid(True, which='minor', linestyle=':')
        ax.minorticks_on()
        ax.grid(True, which='major')
        ax.set_ylabel('$|F| [a.u.]$', fontsize=FS)
        ax.set_xlabel('Residual')
        if log_spaced:
            ax.set_xscale('log')
        figures = {
            'Merit': fig,
            'L_curve': fig2
        }
    else:
        figures = {}
    out = {
        'beta': beta,
        'MSE': MSE,
        'residual': res,
        'norm': np.sum(np.sqrt(beta**2), axis=0),
        'r2': r2,
        'alpha': alpha,
    }
    return out, figures


def Elastic_Net(X, y, alpha, l1_ratio=0.05, positive=True, max_iter=1000):
    """
    Wrap for the elastic net function

    Jose Rueda: jrrueda@us.es

    @param X: Design matrix
    @param y: signal
    @param alpha: hyperparameter
    @param l1_ratio: hyperparameter of the ElasticNet
    @param positive: flag to force positive coefficients
    @return reg.coef_: best fit coefficients
    @return MSE: Mean squared error
    @return r2: R2 score
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


def Elastic_net_scan(X, y, alpha_min: float, alpha_max: float,
                     n_alpha: int = 20, log_spaced: bool = True,
                     plot: bool = True, line_params: dict = {'linewidth': 1.5},
                     FS: float = 14, l1_ratio: float = 0.05,
                     positive: bool = True, max_iter=2000):
    """
    Scan the slpha parameters to find the best hyper-parameter

    Jose Rueda: jrrueda@us.es

    @param X: Design matrix
    @param y: signal
    @param alpha_min: minimum value for the hyper-parameter scan
    @param alpha_max: maximum value for the hyper-parameter scan
    @param n_alpha: number of points in the scan
    @param log_spaced: if true, points will be logspaced
    @param line_param: dictionary with the line plotting parameters
    @param FS: FontSize
    @return out: Dictionay with fields:
        -# beta: array of coefficients [nfeatures, nalphas]
        -# MSE: arrays of MSEs
        -# r2: arrays of r2
        -# residual: arrays of residual
        -# norm: norm of the coefficients
        -# alpha: Used hyperparameters
        -# l1_ratio: l1 hyperparameter (ratio LASSO Ridge)
    @return figures: Dictionay with the figures created by the method:
        -# Merit: r2 and MSE vs hyperparam
        -# L_curve: f norm vs residual
        -# L_curve_alpha: f norm vs hyperparameter
    """
    # --- Initialise the variables
    npoints, nfeatures = X.shape
    beta = np.zeros((nfeatures, n_alpha))
    MSE = np.zeros(n_alpha)
    r2 = np.zeros(n_alpha)
    res = np.zeros(n_alpha)

    if log_spaced:
        alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    else:
        alpha = np.linspace(alpha_min, alpha_max, n_alpha)
    # --- Perform the scan
    print('Performing regression')
    for i in tqdm(range(n_alpha)):
        beta[:, i], MSE[i], res[i], r2[i] = Elastic_Net(X, y, alpha[i],
                                                        l1_ratio=l1_ratio,
                                                        positive=positive,
                                                        max_iter=max_iter)

    # --- Plot if needed:
    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        # Plot the r2:
        ax1.plot(alpha, r2, **line_params)
        ax1.grid(True, which='minor', linestyle=':')
        ax1.minorticks_on()
        ax1.grid(True, which='major')
        ax1.set_ylabel('$r^2$', fontsize=FS)

        # Plot the MSE
        ax2.plot(alpha, MSE, **line_params)
        ax2.grid(True, which='minor', linestyle=':')
        ax2.minorticks_on()
        ax2.grid(True, which='major')
        ax2.set_ylabel('MSE', fontsize=FS)
        ax2.set_xlabel('$\\alpha$')

        # Plot the modulus versus the residual
        fig2, ax = plt.subplots()
        y = np.sum(np.sqrt(beta**2), axis=0)
        ax.plot(res, y, **line_params)
        ax.grid(True, which='minor', linestyle=':')
        ax.minorticks_on()
        ax.grid(True, which='major')
        ax.set_ylabel('$|F| [a.u.]$', fontsize=FS)
        ax.set_xlabel('Residual')
        if log_spaced:
            ax.set_xscale('log')

        fig3, ax3 = plt.subplots()
        ax3.plot(alpha, y, **line_params)
        ax3.grid(True, which='minor', linestyle=':')
        ax3.minorticks_on()
        ax3.grid(True, which='major')
        ax3.set_ylabel('$|F| [a.u.]$', fontsize=FS)
        ax3.set_xlabel('\\alpha')
        if log_spaced:
            ax3.set_xscale('log')
        figures = {
            'Merit': fig,
            'L_curve': fig2,
            'L_curve_alpha': fig3
        }
    else:
        figures = {}
    out = {
        'beta': beta,
        'MSE': MSE,
        'residual': res,
        'norm': np.sum(np.sqrt(beta**2), axis=0),
        'r2': r2,
        'alpha': alpha,
        'l1_ratio': l1_ratio
    }

    return out, figures


# -----------------------------------------------------------------------------
# --- L-Curve
# -----------------------------------------------------------------------------
def L_curve_modelling(x, a1, b1, x1, a2, x2, a3):
    """
    Auxiliar function to fit the L curve

    Jose Rueda: jrrueda@us.es

    The L-curve, in log-log, scale is approximated by three linar segments.
    Please find the complete details of this approximation in the scan:
    'L_curve_fitting' inside the extra_doc folder
    """
    line1 = a1 * x + b1
    line2 = a2 * x + x1 * (a1 - a2) + b1
    line3 = a3 * x + x2 * (a2 - a3) + x1 * (a1 - a2) + b1
    w = 0.1
    y = line1 * 0.5 * (1 - np.tanh((x - x1)/w)) \
        + line2 * 0.5 * (1 + np.tanh((x - x1)/w)) \
        * 0.5 * (1 - np.tanh((x - x2)/w))\
        + line3 * 0.5 * (1 - np.tanh((x - x2)/w))
    return y


def L_curve_fit(norm, residual, a1_min=-1000, a1_max=0,
                a2_min=-1000, a2_max=0,
                a3_min=-10000, a3_max=0, plot: bool = True):
    """
    Create the lmfit model to fit the distribution

    Jose Rueda: jrrueda@us.es

    @param residual: residual of the tomography scan
    @param norm: norm of the coefficients of the tomography scan
    """
    x_fit = np.log(residual)
    min_x = x_fit.min()
    x_fit -= min_x
    x_max = x_fit.max()
    y_fit = np.log(norm)
    # --- Create the model:
    model = lm.Model(L_curve_modelling)
    # --- Create the parameters:
    # - estimate the initial points for x1, x2: just assume that the three
    # lines has the same x-length
    x1_ini = 0.33 * x_max
    x2_ini = 0.66 * x_max
    # Estimate the initial points for b1:
    b1_ini = np.log(norm.max())
    # Estimate the initial points for the slopes
    a1_ini = (y_fit.min() - y_fit.max()) / x_max
    a2_ini = (y_fit.min() - y_fit.max()) / x_max
    a3_ini = (y_fit.min() - y_fit.max()) / x_max
    # Fill the parameters
    params = lm.Parameters()
    params.add_many(
        ('a1', a1_ini, True, a1_min, a1_max, None, None),
        ('b1', b1_ini, True, b1_ini*0.5, b1_ini*1.5, None, None),
        ('x1', x1_ini, True, 0, x_max, None, None),
        ('a2', a2_ini, True, a2_min, a2_max, None, None),
        ('x2', x2_ini, True, 0, x_max, None, None),
        ('a3', a3_ini, True, a3_min, a3_max, None, None)
    )
    # Perform the fit

    result = model.fit(y_fit, params, x=x_fit, max_nfev=150000)
    x1 = min_x + result.params['x1'].value
    opt_residual = np.exp(x1)
    if plot:
        result.plot_fit()
    return result, opt_residual


# -----------------------------------------------------------------------------
# --- FILD TOMOGRAPHY
# -----------------------------------------------------------------------------
def prepare_X_y_FILD(frame, smap, s_opt: dict, p_opt: dict,
                     verbose: bool = True, plt_frame: bool = False,
                     LIMIT_REGION_FCOL: bool = True,
                     efficiency=None, median_filter=True,
                     filter_option: dict = {'size': 4}):
    """
    Prepare the arrays to perform the tomographic inversion in FILD

    Jose Rueda: jrrueda@us.es

    @param    frame: camera frame (in photons /s) you can put directly the
    camera frame if you want a.u
    @param    smap: Scintillator map, with resolution calculated
    @param    s_opt: Parameters for the scintillator grid
    @param    p_opt: Parameters for the pinhole grid
    @param    verbose: Print some notes on the console
    @param    plt_frame: Plot the frame and noise suppressed frame (todo)
    @param    LIMIT_REGION_FCOL: Limit the pinhole grid to points with fcol>0
    @param    efficiency: efficiency dictionary
    @param    median_filter: apply median filter to the remap frame
    @param    filter options: options for the median filter, for the remap

    @return   signal1D:  Signal filtered and reduced in 1D array
    @return   W2D: Weight function compressed as 2D
    """
    print('.--. ... ..-. -')
    print('Preparing W and the measurement')
    # --- create the grids
    nr = int((s_opt['rmax'] - s_opt['rmin']) / s_opt['dr'])
    nnp = int((s_opt['pmax'] - s_opt['pmin']) / s_opt['dp'])
    redges = s_opt['rmin'] - s_opt['dr']/2 + np.arange(nr+2) * s_opt['dr']
    pedges = s_opt['pmin'] - s_opt['dp']/2 + np.arange(nnp+2) * s_opt['dp']

    scint_grid = {'nr': nr + 1, 'np': nnp + 1,
                  'r': 0.5 * (redges[:-1] + redges[1:]),
                  'p': 0.5 * (pedges[:-1] + pedges[1:])}

    nr = int((p_opt['rmax'] - p_opt['rmin']) / p_opt['dr'])
    nnp = int((p_opt['pmax'] - p_opt['pmin']) / p_opt['dp'])
    redges = p_opt['rmin'] - p_opt['dr']/2 + np.arange(nr+2) * p_opt['dr']
    pedges = p_opt['pmin'] - p_opt['dp']/2 + np.arange(nnp+2) * p_opt['dp']
    pin_grid = {'nr': nr + 1, 'np': nnp + 1,
                'r': 0.5 * (redges[:-1] + redges[1:]),
                'p': 0.5 * (pedges[:-1] + pedges[1:])}
    # Note: In the original IDL implementation, the frame was denoised with the
    # median filter at thi point of the routine. Here the frame it supposed to
    # be filtered and denoised with the routines of the video class BEFORE
    # calling the tomography reconstruction. In this way routines are not
    # duplicated and all the new implementations of filters can be used

    # --- Remap the frame
    rep_frame, r, p = ssmapping.remap(smap, frame, x_min=s_opt['pmin'],
                                      x_max=s_opt['pmax'],
                                      delta_x=s_opt['dp'],
                                      y_min=s_opt['rmin'],
                                      y_max=s_opt['rmax'],
                                      delta_y=s_opt['dr'])
    # Just transpose the frame. (To have the W in the same ijkl order of the
    # old IDL-MATLAB implementation)
    rep_frame = rep_frame.T
    if median_filter:
        print('..-. .. .-.. - . .-. .. -. --.')
        'Applying median filter to remap frame'
        rep_frame = ndimage.median_filter(rep_frame, **filter_option)

    # --- Limit the grid
    if LIMIT_REGION_FCOL:
        print('Grid Definition --> Limiting to regions where FCOL>0')
        # Find gyr and pitch with fcol>0: Only the ones presents in the
        # strike points file has fcol>0
        minr = smap.strike_points['gyroradius'].min()
        maxr = smap.strike_points['gyroradius'].max()
        minp = smap.strike_points['pitch'].min()
        maxp = smap.strike_points['pitch'].max()
        # Select only those values on the scint grid
        flags_r = (pin_grid['r'] > minr) * (pin_grid['r'] < maxr)
        flags_p = (pin_grid['p'] > minp) * (pin_grid['p'] < maxp)

        pgrid = {'nr': np.sum(flags_r), 'np': np.sum(flags_p),
                 'r': pin_grid['r'][flags_r], 'p': pin_grid['p'][flags_p]}
    else:
        pgrid = pin_grid

    if verbose:
        print('---')
        print('PINHOLE GRID')
        print(pgrid['nr'], ' x ', pgrid['np'])
        print('Gyroradius (cm):', pgrid['r'].min(), pgrid['r'].max())
        print('Pitch Angle (degree)', pgrid['p'].min(), pgrid['p'].max())
        print('---')
        print('SCINTILLATOR GRID')
        print(scint_grid['nr'], ' x ', scint_grid['np'])
        print('Gyroradius (cm):', scint_grid['r'].min(), scint_grid['r'].max())
        print('Pitch Angle (degree)',
              scint_grid['p'].min(), scint_grid['p'].max())
        print('---')

    # --- Build transfer function
    W4D, W2D = ssfildsim.build_weight_matrix(smap, scint_grid['r'],
                                             scint_grid['p'], pgrid['r'],
                                             pgrid['p'], efficiency)

    # --- Collapse signal into 1D
    signal1D = np.zeros(scint_grid['nr'] * scint_grid['np'])
    for irs in range(scint_grid['nr']):
        for ips in range(scint_grid['np']):
            signal1D[irs * scint_grid['np'] + ips] = rep_frame[irs, ips]

    return signal1D, W2D, W4D, scint_grid, pgrid, rep_frame


# -----------------------------------------------------------------------------
# --- Import/Export tomography
# -----------------------------------------------------------------------------
def export_tomography(name, data):
    """
    Function in beta phase

    """
    if name is None:
        name = ssio.ask_to_save()
    print('Saving results in: ', name)
    with netcdf.netcdf_file(name, 'w') as f:
        f.history = 'Done with version ' + version
        # --- Create the dimenssions:
        pxf, pyf = data['frame'].shape
        nr_scint = data['sg']['nr']
        np_scint = data['sg']['np']
        nr_pin = data['sg']['nr_pin']
        np_pin = data['sg']['np_pin']
        nalpha = len(data['MSE'])
        f.createDimension('pxf', pxf)
        f.createDimension('pyf', pyf)
        f.createDimension('nr_scint', nr_scint)
        f.createDimension('np_scint', np_scint)
        f.createDimension('nr_pin', nr_pin)
        f.createDimension('np_pin', np_pin)
        f.createDimension('nalpha', nalpha)

        # --- Save the camera frame:
        var = f.createVariable('frame', 'float64', ('pxf', 'pyf'))
        var[:] = data['frame']
        var.units = '#'
        var.long_name = 'Camera frame'
        var.short_name = 'Counts'

        # --- Remap
        var = f.createVariable('remap', 'float64', ('nr_scint', 'np_scint'))
        var[:] = data['remap']
        var.units = 'a.u.'
        var.long_name = 'Remaped frame'
        var.short_name = 'Remap'

        # --- Inverted frames:
        var = f.createVariable('tomoFrames', 'float64', ('nr_pin', 'np_pin',
                                                         'nalpha'))
        var[:] = data['tomoFrames']
        var.units = 'a.u.'
        var.long_name = 'Tomographic inversion frame'
        var.short_name = 'Pinhole frames'

        # --- figures of merit:
        var = f.createVariable('norm', 'float64', ('nalpha'))
        var[:] = data['norm']
        var.units = 'a.u.'
        var.long_name = 'Norm of the pinhole distribution'
        var.short_name = '|F|'

        # --- MSE:
        var = f.createVariable('MSE', 'float64', ('nalpha'))
        var[:] = data['MSE']
        var.units = 'a.u.'
        var.long_name = 'Mean Squared Error'
        var.short_name = 'MSE'

        # --- grid:
        var = f.createVariable('rpin', 'float64', ('nr_pin'))
        var[:] = data['sg']['r']
        var.units = 'cm'
        var.long_name = 'Gyroradius at pinhole'
        var.short_name = '$r_l$'

        var = f.createVariable('ppin', 'float64', ('np_pin'))
        var[:] = data['sg']['p']
        var.units = ' '
        var.long_name = 'Pitch at pinhole'
        var.short_name = 'Pitch'

        var = f.createVariable('rscint', 'float64', ('nr_scint'))
        var[:] = data['sg']['r']
        var.units = 'cm'
        var.long_name = 'Gyroradius at scintillator'
        var.short_name = '$r_l$'

        var = f.createVariable('pscint', 'float64', ('nr_scint'))
        var[:] = data['sg']['r']
        var.units = ' '
        var.long_name = 'Pitch at scintillator'
        var.short_name = 'Pitch'
