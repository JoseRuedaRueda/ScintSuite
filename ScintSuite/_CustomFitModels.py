"""
Custom models to perform fits with lmfit


Jose Rueda Rueda: jrrueda@us.es

Introduced in version
"""
import math
import lmfit
import numpy as np
from lmfit.models import ExpressionModel
from lmfit import Model
from scipy import special
from  scipy.special import loggamma

# -----------------------------------------------------------------------------
# %% Bivariant Gaussian
# -----------------------------------------------------------------------------
# Model taken from
# https://mathworld.wolfram.com/BivariateNormalDistribution.html
# just a scale constant added
fun = 'amp / (2.0*pi*sx*sy*sqrt(1-rho**2)) * ' + \
    'exp(-((x-mux)**2/sx**2 + (y-muy)**2/sy**2 - ' +\
    '2.0*rho*(x-mux)*(y-muy)/sx/sy)/(2.0*(1-rho**2)))'
BivariateNormalDistribution = ExpressionModel(fun, independent_vars=['x', 'y'])

def guessParamsBivariateNormalDistribution(x, y, z)->lmfit.parameter.Parameters:
    """
    Guess the parameters for the bivariate normal distribution

    formula of weiugthed standard deviation taken from nist:
    https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    assume that N' is much larger than 1, so the denominator simplifies
    """
    # ---- Calculate the weighted means
    mux = (x*z).sum() / z.sum()
    muy = (y*z).sum() / z.sum()
    # ---- Calculate the weighted std
    sx = np.sqrt((z * (x - mux)**2).sum() / z.sum())
    sy = np.sqrt((z * (y - muy)**2).sum() / z.sum())
    # ---- Calculate the amplitude
    amp = z.sum()
    if amp > 0:
        maxamp = 10*amp
        minamp = 0.0
    else:
        minamp = 10*amp
        maxamp = 0.0
    # ---- correlation
    # For the correlation, just leave it to zero, and the model will fit
    rho = 0
    # ---- Create params
    # init some dummy values
    p = BivariateNormalDistribution.make_params(amp=7.0e8, sx=0.05, sy=5,
                                                mux=1.7, muy=60.0, rho=0.5)
    # fill the proper values
    p['rho'].set(0.0, min=-1.0, max=1.0)
    p['amp'].set(amp, min=minamp, max=maxamp)
    p['sx'].set(sx, min=0.1*sx, max=10*sx)
    p['sy'].set(sy, min=0.1*sy, max=10*sy)
    p['muy'].set(muy, min=0.5*muy, max=1.5*muy)
    p['mux'].set(mux, min=0.5*mux, max=1.5*mux)
    return p

# -----------------------------------------------------------------------------
# %% Multiple Gaussian
# -----------------------------------------------------------------------------
def multiGaussian(n: int) -> lmfit.model.CompositeModel:
    """
    Create a composite model with n gaussian functions

    :param n : (int) Number of gaussian functions to include in the composite model
    
    :return out: lmfit.models.CompositeModel Composite model with n gaussian functions
    """
    # ---- Create the composite model
    compositeModel = lmfit.models.GaussianModel(prefix='g0_')
    for i in range(n-1):
        prefix = 'g' + str(i+1) + '_'
        compositeModel = \
            compositeModel + lmfit.models.GaussianModel(prefix=prefix)
    return compositeModel

# ------------------------------------------------------------------------------
# %% Raised Cosine
# ------------------------------------------------------------------------------
# Auxiliary function
def _raised_cosine(x,amplitude,center,sigma,beta):
    """
    Raised cosine model.
    
    See https://doi.org/10.1088/1361-6587/ad268f for full details
    
    :param x: x values where to evaluate the model
    :param amplitude: amplitude of the model
    :param center: center of the model
    :param sigma: width parameter of the model
    :param beta: decay like parameter of the model
    
    To avoid the non linearity, gamma/sigma in the erro function, which is more
    difficult to fit, I define beta = gamma/sigma. This way, the model is
    easier to fit. The gamma parameter for the model can be easily obtained 
    from here

    """
    # Only defined inside a sigma interval
    cosine_part = (1.0+np.cos((x-center)/sigma*math.pi))/2.0/sigma
    cosine_part[np.abs(x-center)>sigma] = 0.0
    error_part = 1.0 + special.erf((x-center)/math.sqrt(2.0)*beta)
    # print((x-center)/sigma/math.sqrt(2.0)*gamma)
    # print(error_part)
    return amplitude*cosine_part*error_part

# lmfit model
def RaisedCosine():
    """
    Raised cosine model.
    
    See https://doi.org/10.1088/1361-6587/ad268f for full details
    """
    
    return Model(_raised_cosine)

# -----------------------------------------------------------------------------
# %% Weigner Semicircle
# -----------------------------------------------------------------------------
def _wignerse(x,amplitude,center,sigma):
    """
    Wigner's semicircle paper.
    
    See https://doi.org/10.1088/1361-6587/ad268f for full details
    
    :param x: x values where to evaluate the model
    :param amplitude: amplitude of the model
    :param center: center of the model
    :param sigma: width parameter of the model
    """
    out = np.zeros_like(x)
    y = x-center
    mask = np.abs(y)**2 < sigma**2
    out[mask] = 2.0 * amplitude * np.sqrt(sigma**2 - y[mask]**2) / math.pi / sigma**2
    return out

def WignerSemicircle():
    """
    Wigner's semicircle paper.
    
    See https://doi.org/10.1088/1361-6587/ad268f for full details
    """
    return Model(_wignerse)

# -----------------------------------------------------------------------------
# %% Parse model names
# -----------------------------------------------------------------------------

def parseModelNames(name:str)->Model:
    """
    Parse the name of the model and return the proper model

    :param name: (str) Name of the model to be parsed

    :return out: lmfit.Model Model corresponding to the name
    """
    if name.lower() == 'gaussian' or name.lower() == 'gauss':
        return lmfit.models.GaussianModel()
    elif name.lower() == 'bivariategaussian':
        return BivariateNormalDistribution
    elif name.lower() == 'multigaussian':
        return multiGaussian
    elif name.lower() == 'sgauss' or name.lower() == 'skewedgaussian':
        return lmfit.models.SkewedGaussianModel()
    elif name.lower() == 'raisedcosine':
        return RaisedCosine()
    elif name.lower() == 'wignersemicircle':
        return WignerSemicircle()
    else:
        raise ValueError(f'Unknown model name: {name}')
    
    
# -----------------------------------------------------------------------------
# %% Poisson distribution
# -----------------------------------------------------------------------------
def _log_poisson(x, mu):
    """
    Logarithm of the Poisson distribution

    :param x: (array) x values
    :param mu: (float) mean of the distribution

    :return out: (array) log of the Poisson distribution
    """
    return -mu + x*np.log(mu) - loggamma(x+1)

def log_poisson() -> lmfit.models.ExpressionModel:
    """
    Create a Poisson distribution model

    :return out: (lmfit.model.ExpressionModel) Poisson distribution model
    """
    return ExpressionModel(_log_poisson, independent_vars=['x'], name='log_poisson')