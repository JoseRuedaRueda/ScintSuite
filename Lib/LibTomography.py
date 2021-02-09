"""
Contains the routines to perform the tomographic inversion

NOTICE: I consider the creatation of the transfer matrix as an issue of the
synthetic codes (INPASIM, FILDSIM, i/HIBPSIM) therefore the routines which
create these matrix are placed are their corresponding libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import LibFILDSIM as ssfildsim
import LibMap as ssmapping
from scipy import ndimage        # To denoise the frames
from tqdm import tqdm            # For waitbars
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.optimize import nnls     # Non negative least squares
from sklearn.linear_model import ElasticNet  # ElaticNet


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
    return beta, MSE, r2


def nnls_inversion(X, y, param: dict = {}):
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
    return beta, MSE, r2


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
    return ridge.coef_, MSE, r2


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
    @return beta: array of coefficients [nfeatures, nalphas]
    @return MSE: arrays of MSEs
    @return r2: arrays of r2
    """
    # --- Initialise the variables
    npoints, nfeatures = X.shape
    beta = np.zeros((nfeatures, n_alpha))
    MSE = np.zeros(n_alpha)
    r2 = np.zeros(n_alpha)

    if log_spaced:
        alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    else:
        alpha = np.linspace(alpha_min, alpha_max, n_alpha)
    # --- Perform the scan
    for i in tqdm(range(n_alpha)):
        beta[:, i], MSE[i], r2[i] = Ridge_inversion(X, y, alpha[i])

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

        # Plot the modulus versus the MSE
        fig2, ax = plt.subplots()
        y = np.sum(np.sqrt(beta**2), axis=0)
        ax.plot(MSE, y, **line_param)
        ax.grid(True, which='minor', linestyle=':')
        ax.minorticks_on()
        ax.grid(True, which='major')
        ax.set_ylabel('$|F| [a.u.]$', fontsize=FS)
        ax.set_xlabel('MSE')
        if log_spaced:
            ax2.set_xscale('log')
    return beta, MSE, r2, alpha


def Elastic_Net(X, y, alpha, l1_ratio=0.05, positive=True):
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
    reg = ElasticNet(alpha=alpha, positive=positive, l1_ratio=l1_ratio)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    MSE = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return reg.coef_, MSE, r2


def Elastic_net_scan(X, y, alpha_min: float, alpha_max: float,
                     n_alpha: int = 20, log_spaced: bool = True,
                     plot: bool = True, line_param: dict = {'linewidth': 1.5},
                     FS: float = 14, l1_ratio: float = 0.05,
                     positive: bool = True):
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
    @return beta: array of coefficients [nfeatures, nalphas]
    @return MSE: arrays of MSEs
    @return r2: arrays of r2
    """
    # --- Initialise the variables
    npoints, nfeatures = X.shape
    beta = np.zeros((nfeatures, n_alpha))
    MSE = np.zeros(n_alpha)
    r2 = np.zeros(n_alpha)

    if log_spaced:
        alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    else:
        alpha = np.linspace(alpha_min, alpha_max, n_alpha)
    # --- Perform the scan
    for i in tqdm(range(n_alpha)):
        beta[:, i], MSE[i], r2[i] = Elastic_Net(X, y, alpha[i],
                                                l1_ratio=l1_ratio,
                                                positive=positive)

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

        # Plot the modulus versus the MSE
        fig2, ax = plt.subplots()
        y = np.sum(np.sqrt(beta**2), axis=0)
        ax.plot(MSE, y, **line_param)
        ax.grid(True, which='minor', linestyle=':')
        ax.minorticks_on()
        ax.grid(True, which='major')
        ax.set_ylabel('$|F| [a.u.]$', fontsize=FS)
        ax.set_xlabel('MSE')
        if log_spaced:
            ax2.set_xscale('log')
    return beta, MSE, r2, alpha


# -----------------------------------------------------------------------------
# --- L-Curve
# -----------------------------------------------------------------------------
def L_curve(beta, MSE):
    """
    Calculate the L curve and its derivative

    Jose Rueda: jrrueda@us.es

    @param beta: matrix with the coefficient generated by any of the regressors
    @param MSE: MSE calculated by any of the avobe functions
    """
    # --- Norm of the distribution
    y = np.sqrt(np.sum(beta, axis=0))

    # --- First derivative
    y1 = np.gradient(y, MSE)

    # --- second derivative
    y2 = np.gradient(y, MSE)

    return y1, y2


# -----------------------------------------------------------------------------
# --- FILD TOMOGRAPHY
# -----------------------------------------------------------------------------
def prepare_X_y_FILD(frame, smap, s_opt: dict, p_opt: dict,
                     verbose: bool = True, plt_frame: bool = False,
                     LIMIT_REGION_FCOL: bool = True, denoise: bool = True,
                     efficiency: dict = {}):
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
    @param    denoise: apply median filter to the frame
    @param    efficiency: efficiency dictionary
    @return   signal1D:  Signal filtered and reduced in 1D array
    @return   W2D: Weight function compressed as 2D
    """
    print('.--. ... ..-. -')
    print('Preparing W and the measurement')
    # --- create the grids
    redges = np.arange(start=s_opt['rmin'], stop=s_opt['rmax'],
                       step=s_opt['dr'])
    pedges = np.arange(start=s_opt['pmin'], stop=s_opt['pmax'],
                       step=s_opt['dp'])
    scint_grid = {'nr': len(redges) - 1, 'np': len(pedges) - 1,
                  'r': 0.5 * (redges[0:-1] + redges[1:]),
                  'p': 0.5 * (pedges[0:-1] + pedges[1:])}

    redges = np.arange(start=p_opt['rmin'], stop=p_opt['rmax'],
                       step=p_opt['dr'])
    pedges = np.arange(start=p_opt['pmin'], stop=p_opt['pmax'],
                       step=p_opt['dp'])
    pin_grid = {'nr': len(redges) - 1, 'np': len(pedges) - 1,
                'r': 0.5 * (redges[0:-1] + redges[1:]),
                'p': 0.5 * (pedges[0:-1] + pedges[1:])}
    # --- Denoise the frame with the median filter
    if denoise:
        frame2 = ndimage.median_filter(frame, 3)
        if verbose:
            print('Median filter applied to the frame')
            print('Previous max: ', frame.max(), ' min: ', frame.min())
            print('New max: ', frame2.max(), ' min: ', frame2.min())
    # --- Remap the frame
    rep_frame, r, p = ssmapping.remap(smap, frame2, x_min=s_opt['pmin'],
                                      x_max=s_opt['pmax'],
                                      delta_x=s_opt['dp'],
                                      y_min=s_opt['rmin'],
                                      y_max=s_opt['rmax'],
                                      delta_y=s_opt['dr'])
    rep_frame = rep_frame.T

    # --- Limit the grid
    if LIMIT_REGION_FCOL:
        print('Grid Definition --> Limiting to regions where FCOL>0')
        # Find gyr and pitch with fcol>0
        flags = smap.collimator_factor > 0.
        dummy_r = smap.gyroradius[flags]
        dummy_p = smap.pitch[flags]
        unique_r = np.unique(dummy_r)
        unique_p = np.unique(dummy_p)
        minr = unique_r.min()
        maxr = unique_r.max()
        minp = unique_p.min()
        maxp = unique_p.max()
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
    W4D = ssfildsim.build_weight_matrix(smap, scint_grid['r'], scint_grid['p'],
                                        pgrid['r'], pgrid['p'], efficiency)

    # --- Collapse Weight function
    W2D = np.zeros((scint_grid['nr'] * scint_grid['np'],
                   pgrid['nr'] * pgrid['np']))
    ## todo make this with an elegant numpy reshape, not manually
    print('Reshaping W: ')
    for irs in tqdm(range(scint_grid['nr'])):
        for ips in range(scint_grid['np']):
            for irp in range(pgrid['nr']):
                for ipp in range(pgrid['np']):
                    W2D[irs * scint_grid['np'] + ips, irp * pgrid['np'] + ipp]\
                        = W4D[irs, ips, irp, ipp]
    # --- Collapse signal into 1D
    signal1D = np.zeros(scint_grid['nr'] * scint_grid['np'])
    print(signal1D.shape)
    print(rep_frame.T.shape)
    print(scint_grid['nr'])
    print(scint_grid['np'])
    for irs in range(scint_grid['nr']):
        for ips in range(scint_grid['np']):
            signal1D[irs * scint_grid['np'] + ips] = rep_frame[irs, ips]

    return signal1D, W2D, W4D, scint_grid, pgrid
