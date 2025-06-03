"""
Custom models to perform fits with lmfit


Jose Rueda Rueda: jrrueda@us.es

Introduced in version
"""
import lmfit
import numpy as np
from lmfit.models import ExpressionModel
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